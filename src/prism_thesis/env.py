"""MySQL 执行环境。

做法与官方 db_bench 一致：每道题建一个与表同名的数据库，建表并写入初始数据；
智能体的 SQL 在该数据库中执行，返回 str(结果行列表) 或错误信息；题目结束后删库。

与官方实现的差别（为隔离与可复现，所有实验条件相同）：
  1. 智能体的 SQL 以一个只拥有本题数据库增删改查权限的账号执行，无法访问其他数据库或修改表结构；
  2. 每次操作只允许一条 SQL 语句（官方按分号拆分后依次执行），多条语句返回错误且不执行；
  3. 单条语句的执行时间上限为 statement_timeout_ms（防止失控查询拖住实验）。
MySQL 运行在 Docker 容器中（镜像版本在配置中固定），也可以连接已有的 MySQL 服务。
"""
from __future__ import annotations

import secrets
import subprocess
import time
from dataclasses import dataclass
from typing import Optional

import pymysql

from .data import Task

MULTI_STATEMENT_ERROR = 'Error: exactly one SQL statement is allowed per operation; nothing was executed.'


@dataclass
class ExecResult:
    ok: bool
    text: str                     # 返回给智能体的观察文本
    error_code: Optional[int]     # MySQL 错误号；非 MySQL 错误时为 None
    kind: str = 'ok'              # ok / mysql_error / multi_statement / empty


class InfrastructureError(RuntimeError):
    """数据库连接或传输故障（不是智能体的 SQL 引起的），该次运行记为基础设施错误，续跑时重做。"""


def is_connection_error(e: Exception) -> bool:
    """MySQL 客户端错误号 2000–2999（如 2006 服务已断开、2013 查询中连接丢失）与接口错误属于基础设施故障；
    服务端返回的语法、权限、数据、超时（3024）等错误属于智能体的操作结果。"""
    if isinstance(e, pymysql.err.InterfaceError):
        return True
    code = e.args[0] if e.args and isinstance(e.args[0], int) else None
    return code is None or code == 0 or 2000 <= code < 3000


def split_statements(sql: str) -> list[str]:
    """按分号拆分语句，按 MySQL 默认 SQL 模式（未设置 NO_BACKSLASH_ESCAPES，本环境不修改 sql_mode）的词法
    忽略字符串、带引号标识符与注释中的分号：单引号与双引号字符串支持反斜杠转义和成对引号；反引号标识符
    支持成对反引号；注释为 #、“-- ”（后跟空白或行尾）与 /* */。只含注释或空白的片段不算语句。"""
    parts, buf, n, i = [], [], len(sql), 0
    has_code = False

    def flush():
        nonlocal buf, has_code
        if has_code:
            parts.append(''.join(buf).strip())
        buf, has_code = [], False

    while i < n:
        ch = sql[i]
        if ch in ("'", '"', '`'):
            j = i + 1
            while j < n:
                if sql[j] == '\\' and ch != '`':
                    j += 2
                    continue
                if sql[j] == ch:
                    if j + 1 < n and sql[j + 1] == ch:
                        j += 2
                        continue
                    break
                j += 1
            buf.append(sql[i:j + 1])
            has_code = True
            i = j + 1
        elif ch == '#' or (sql.startswith('--', i) and (i + 2 == n or sql[i + 2] in ' \t\r\n')):
            j = sql.find('\n', i)
            j = n if j < 0 else j
            buf.append(sql[i:j])
            i = j
        elif sql.startswith('/*', i):
            j = sql.find('*/', i + 2)
            j = n if j < 0 else j + 2
            buf.append(sql[i:j])
            i = j
        elif ch == ';':
            flush()
            i += 1
        else:
            buf.append(ch)
            has_code = has_code or not ch.isspace()
            i += 1
    flush()
    return parts


def init_sql(task: Task) -> list[str]:
    """建库、建表、写入初始数据的语句（与官方 _build_init_sql 的结构相同）。"""
    cols = ','.join(f'`{c.name}` {c.type}' for c in task.columns)
    names = ','.join(f'`{c.name}`' for c in task.columns)

    def lit(v):
        return "'" + str(v).replace("'", "''") + "'"

    values = ','.join('(' + ','.join(lit(v) for v in row) + ')' for row in task.rows)
    db = task.table
    return [
        f'DROP DATABASE IF EXISTS `{db}`',
        f'CREATE DATABASE `{db}`',
        f'CREATE TABLE `{db}`.`{task.table}` ({cols})',
        f'INSERT INTO `{db}`.`{task.table}` ({names}) VALUES {values}',
    ]


def md5_sql(task: Task) -> str:
    """官方修改题判分所用的表哈希查询。"""
    names = ','.join(f'`{c.name}`' for c in task.columns)
    return ('select md5(group_concat(rowhash order by rowhash)) as hash '
            f"from( SELECT substring(MD5(CONCAT_WS(',', {names})), 1, 5) AS rowhash "
            f'FROM `{task.table}`) as sub')


class MySQLServer:
    """一个 MySQL 服务。docker=True 时由本程序启动并管理 Docker 容器；否则连接已有服务。"""

    def __init__(self, image: str = 'mysql:8.0.36', host: str = '127.0.0.1', port: int = 13306,
                 user: str = 'root', password: Optional[str] = None, docker: bool = True,
                 name: Optional[str] = None):
        self.image, self.host, self.port, self.user, self.docker = image, host, port, user, docker
        self.password = password if password is not None else secrets.token_hex(8)
        self.name = name or f'prism-mysql-{port}'

    def start(self, timeout: float = 180) -> None:
        if self.docker:
            subprocess.run(['docker', 'rm', '-f', self.name], capture_output=True)
            # 不用 --rm：容器启动失败退出时保留下来，才能读到它的日志；stop() 负责删除
            subprocess.run(['docker', 'run', '-d', '--name', self.name,
                            '-e', f'MYSQL_ROOT_PASSWORD={self.password}',
                            '-p', f'{self.host}:{self.port}:3306', self.image],
                           check=True, capture_output=True)
        deadline = time.time() + timeout
        while True:
            try:
                self.connect().close()
                return
            except pymysql.MySQLError:
                if self.docker and not self._running():
                    raise RuntimeError(f'MySQL 容器 {self.name} 启动后退出：\n{self._logs()}')
                if time.time() > deadline:
                    detail = f'\n容器日志最后几行：\n{self._logs()}' if self.docker else ''
                    raise TimeoutError(f'MySQL at {self.host}:{self.port} not ready after {timeout}s{detail}')
                time.sleep(2)

    def _running(self) -> bool:
        out = subprocess.run(['docker', 'inspect', '-f', '{{.State.Running}}', self.name],
                             capture_output=True, text=True)
        return out.stdout.strip() == 'true'

    def _logs(self, n: int = 15) -> str:
        out = subprocess.run(['docker', 'logs', '--tail', str(n), self.name], capture_output=True, text=True)
        return (out.stdout + out.stderr).strip() or '（无日志）'

    def stop(self) -> None:
        if self.docker:
            subprocess.run(['docker', 'rm', '-f', self.name], capture_output=True)

    def connect(self, user: Optional[str] = None, password: Optional[str] = None,
                database: Optional[str] = None, read_timeout: Optional[int] = None):
        return pymysql.connect(host=self.host, port=self.port, user=user or self.user,
                               password=self.password if password is None else password,
                               database=database, autocommit=True, charset='utf8mb4',
                               read_timeout=read_timeout)

    def __enter__(self) -> 'MySQLServer':
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()


class TaskSession:
    """一道题的数据库会话：准备数据、执行智能体 SQL、计算判分所需的表哈希、清理。"""

    def __init__(self, server: MySQLServer, task: Task, agent_user: str = 'agent',
                 statement_timeout_ms: int = 10000):
        self.server, self.task = server, task
        self.agent_user = agent_user
        self.agent_password = secrets.token_hex(8)
        self.statement_timeout_ms = statement_timeout_ms

    def __enter__(self) -> 'TaskSession':
        root = self.server.connect()
        try:
            with root.cursor() as cur:
                for stmt in init_sql(self.task):
                    cur.execute(stmt)
                cur.execute(f"DROP USER IF EXISTS '{self.agent_user}'@'%'")
                cur.execute(f"CREATE USER '{self.agent_user}'@'%' IDENTIFIED BY '{self.agent_password}'")
                cur.execute(f"GRANT SELECT, INSERT, UPDATE, DELETE ON `{self.task.table}`.* "
                            f"TO '{self.agent_user}'@'%'")
        finally:
            root.close()
        self.conn = self.server.connect(self.agent_user, self.agent_password, self.task.table,
                                        read_timeout=max(30, self.statement_timeout_ms // 1000 * 3))
        with self.conn.cursor() as cur:
            cur.execute(f'SET SESSION max_execution_time = {int(self.statement_timeout_ms)}')
        return self

    def execute(self, sql: str) -> ExecResult:
        stmts = split_statements(sql)
        if not stmts:
            return ExecResult(False, 'Error: the SQL statement is empty.', None, 'empty')
        if len(stmts) != 1:
            return ExecResult(False, MULTI_STATEMENT_ERROR, None, 'multi_statement')
        try:
            with self.conn.cursor() as cur:
                cur.execute(stmts[0])
                rows = cur.fetchall()
            return ExecResult(True, str([tuple(r) for r in rows]), None, 'ok')
        except pymysql.MySQLError as e:
            if is_connection_error(e):
                raise InfrastructureError(f'数据库连接故障：{e!r}') from e
            code = e.args[0] if e.args and isinstance(e.args[0], int) else None
            msg = e.args[1] if len(e.args) > 1 else str(e)
            text = f'Error {code}: {msg}' if code is not None else f'Error: {msg}'
            return ExecResult(False, text, code, 'mysql_error')

    def table_md5(self) -> str:
        """当前表的官方哈希；空表时为 'None'（与官方从 str(结果) 中解析的结果一致）。"""
        root = self.server.connect(database=self.task.table)
        try:
            with root.cursor() as cur:
                cur.execute(md5_sql(self.task))
                (value,) = cur.fetchone()
        finally:
            root.close()
        return str(value)

    def __exit__(self, *exc) -> None:
        try:
            self.conn.close()
        finally:
            root = self.server.connect()
            try:
                with root.cursor() as cur:
                    cur.execute(f'DROP DATABASE IF EXISTS `{self.task.table}`')
                    cur.execute(f"DROP USER IF EXISTS '{self.agent_user}'@'%'")
            finally:
                root.close()

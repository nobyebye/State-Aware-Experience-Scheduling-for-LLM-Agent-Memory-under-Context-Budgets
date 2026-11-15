"""操作系统环境：LifelongAgentBench 的 os_interaction 任务，在 Docker 中的 Ubuntu 容器里用 bash 完成。

与官方实现一致之处：每道题一个新容器；先执行题目的初始化脚本；智能体每轮提交一段 bash 脚本，由
`bash -c` 非交互执行（每次是新的 shell，工作目录与变量不保留，文件系统保留）；结束后执行题目的评测
命令，退出码为 0 即成功。
与官方实现不同之处（所有实验条件相同）：动作格式与本研究其他环境统一为 "Action: Operation" + ```bash
代码块 / "Action: Answer"；容器不联网并限制内存与进程数；每条命令有超时；观察按字符数截断。

数据由 scripts/pub/prepare_os.py 生成（data/os/tasks.jsonl 与 splits.json）。
规则、命令说明与词表在看到任何运行结果之前写定。
"""
from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

from ..env import ExecResult, InfrastructureError
from ..llm import ScriptedLLM
from ..parser import Parsed
from ..policies.base import TaskFeatures
from .base import Environment

ROOT = Path(__file__).resolve().parents[3]
SPLITS = ('build', 'search', 'calib', 'test', 'reserve')
RUNNERS = {'bash': ['bash', '-c'], 'python3': ['python3', '-c'], 'python': ['python3', '-c']}


@dataclass(frozen=True)
class OsTask:
    task_id: int
    group_id: str
    split: str
    instruction: str
    init: Optional[tuple]            # (command_name, script) 或 None
    ground_truth: Optional[tuple]    # 参考解法，只用于核对数据与脚本化演练，不进入提示
    evaluation: tuple                # (command_name, script)：退出码 0 即成功
    skills: tuple

    is_query = False


CORE = (
    ('role', "You complete the user's task by running bash commands on an Ubuntu system as root."),
    ('op_format', 'To run commands output exactly:\nAction: Operation\n```bash\nYOUR_COMMANDS\n```'),
    ('answer_action', 'When the task is complete output exactly:\nAction: Answer\nFinal Answer: done'),
)
RULES = (
    ('observe', 'Use actual tool observations.'),
    ('fresh_shell', 'Each operation runs in a new non-interactive shell: the working directory and shell '
                    'variables are not kept between operations, but files are.'),
    ('noninteractive', 'Do not use interactive programs (vim, nano, less, top) or commands that wait for input.'),
    ('paths', 'Use absolute paths, and create missing parent directories before writing files.'),
    ('no_sudo', 'You are already root; do not use sudo.'),
    ('on_error', 'If a command fails, read the error message and fix its cause instead of repeating the command.'),
    ('verify', 'Before finishing, check the result with a read-only command such as ls -l, cat, id or getent.'),
    ('rounds', None),
)
COMMANDS = (
    ('useradd', 'useradd [-m] [-s SHELL] [-g GROUP] [-G GROUPS] NAME: create a user', ('user', 'account')),
    ('usermod', 'usermod [-aG GROUPS] [-s SHELL] [-d HOME] NAME: change a user', ('user', 'member', 'shell')),
    ('userdel', 'userdel [-r] NAME: delete a user', ('delete user', 'remove user', 'remove the user')),
    ('groupadd', 'groupadd [-g GID] NAME: create a group', ('group',)),
    ('addgroup', 'addgroup NAME: create a group (Debian helper)', ('group',)),
    ('gpasswd', 'gpasswd -a USER GROUP / -d USER GROUP: add or remove a group member', ('member', 'group')),
    ('chpasswd', "echo 'USER:PASSWORD' | chpasswd: set a password", ('password',)),
    ('id', 'id NAME: show uid, gid and groups', ('user', 'group')),
    ('getent', 'getent passwd|group NAME: look up a user or group', ('user', 'group')),
    ('chmod', 'chmod [-R] MODE PATH: change permissions', ('permission', 'executable', 'readable', 'writable', 'mode')),
    ('chown', 'chown [-R] USER[:GROUP] PATH: change owner', ('owner', 'ownership', 'owned')),
    ('chgrp', 'chgrp [-R] GROUP PATH: change group owner', ('group owner', 'group ownership')),
    ('mkdir', 'mkdir -p PATH: create directories', ('directory', 'folder')),
    ('touch', 'touch PATH: create an empty file or update its time', ('empty file', 'create a file', 'timestamp')),
    ('cp', 'cp [-r] [-p] SRC DST: copy', ('copy', 'backup')),
    ('mv', 'mv SRC DST: move or rename', ('move', 'rename')),
    ('rm', 'rm [-r] [-f] PATH: remove', ('remove', 'delete', 'clean')),
    ('ln', 'ln -s TARGET LINK: create a symbolic link', ('link', 'symlink')),
    ('find', 'find DIR [-name PATTERN] [-type f|d] [-exec CMD {} +]: search files', ('find', 'all files', 'search')),
    ('grep', 'grep [-r] [-i] [-c] [-E] PATTERN FILE: search text', ('contain', 'match', 'search', 'lines')),
    ('sed', "sed -i 's/OLD/NEW/g' FILE: edit text in place", ('replace', 'substitute', 'edit')),
    ('awk', "awk -F SEP '{print $N}' FILE: process columns", ('column', 'field')),
    ('sort', 'sort [-n] [-r] [-u] [-k N] FILE: sort lines', ('sort', 'order')),
    ('uniq', 'uniq [-c]: collapse adjacent duplicate lines', ('unique', 'duplicate')),
    ('wc', 'wc -l|-w|-c FILE: count lines, words or bytes', ('count', 'number of')),
    ('head', 'head -n N FILE: first lines', ('first',)),
    ('tail', 'tail -n N FILE: last lines', ('last',)),
    ('tee', 'CMD | tee [-a] FILE: write output to a file', ('append', 'write', 'save')),
    ('tar', 'tar -czf ARCHIVE.tar.gz -C DIR . / tar -xzf ARCHIVE -C DIR: archive', ('archive', 'compress', 'tar',
                                                                                    'extract')),
    ('stat', "stat -c '%a %U %G' PATH: show mode, owner and group", ('permission', 'owner')),
)
COMMAND_HEADER = 'Useful commands:'
RULE_IDS = tuple(u for u, _ in RULES)
COMMAND_IDS = frozenset(f'cmd:{c}' for c, _, _ in COMMANDS)
CORE_IDS = frozenset(u for u, _ in CORE) | {'task'}
ALL_IDS = CORE_IDS | frozenset(RULE_IDS) | COMMAND_IDS
SELECTABLE_GROUPS = RULE_IDS + ('cmd_predicted', 'cmd_other')
_TEXT = dict(CORE + RULES) | {f'cmd:{c}': usage for c, usage, _ in COMMANDS}

ERROR_KINDS = (
    ('no_command', 'command not found'), ('perm', 'permission denied'), ('perm', 'operation not permitted'),
    ('no_file', 'no such file or directory'), ('usage', 'invalid option'), ('usage', 'unrecognized option'),
    ('usage', 'missing operand'), ('usage', 'usage:'), ('exists', 'already exists'), ('exists', 'file exists'),
    ('no_entity', 'does not exist'), ('no_entity', 'no such user'), ('no_entity', 'unknown user'),
    ('no_entity', 'invalid user'), ('no_entity', 'no such group'), ('syntax', 'syntax error'),
)
RECOVERY = {
    'no_command': ('cmd_predicted', 'cmd_other'), 'usage': ('cmd_predicted', 'cmd_other'),
    'syntax': ('fresh_shell', 'cmd_predicted', 'cmd_other'),
    'timeout': ('noninteractive', 'fresh_shell'), 'perm': ('no_sudo', 'paths'),
    'no_file': ('paths', 'fresh_shell'), 'exists': ('on_error', 'verify'), 'no_entity': ('on_error', 'verify'),
}

_ACT = re.compile(r'^Action:[ \t]*([^\r\n]*)$', re.MULTILINE)
_BLOCK = re.compile(r'```(?:bash|sh|shell|console)?[ \t]*\n([\s\S]*?)\n[ \t]*```', re.IGNORECASE)
_STRICT = re.compile(r'```bash\n([\s\S]*?)\n```')


def first_action(response: str) -> str:
    """只保留第一个动作并规范化：结束类标签（Answer / finish / done）统一为 Answer；
    其他标签后跟代码块（bash / sh / 无语言标记）统一为 Operation + ```bash。认不出的回复保持原样。"""
    m = _ACT.search(response)
    if m is None:
        return response
    label = m.group(1).strip().lower()
    rest = response[m.end():]
    nxt = _ACT.search(rest)
    scope = rest[:nxt.start()] if nxt else rest
    if label in ('answer', 'finish', 'done'):
        return 'Action: Answer\nFinal Answer: done'
    block = _BLOCK.search(scope)
    if block is None or not block.group(1).strip():
        return response
    return 'Action: Operation\n```bash\n' + block.group(1).strip() + '\n```'


def parse(response: str) -> Parsed:
    m = re.search(r'Action: (Operation|Answer)', response)
    if m is None:
        return Parsed('invalid', None, 'no_action')
    if m.group(1) == 'Answer':
        return Parsed('answer', 'done')
    block = _STRICT.search(response)
    if block is None:
        return Parsed('invalid', None, 'no_bash_block')
    return Parsed('operation', block.group(1).strip())


def error_kind(exit_code: int, output: str) -> str:
    if exit_code in (124, 137):
        return 'timeout'
    if exit_code == 127:
        return 'no_command'
    low = output.lower()
    return next((k for k, pat in ERROR_KINDS if pat in low), 'other')


def first_command(script: str) -> str:
    for word in re.findall(r'[A-Za-z_./][\w./-]*=?\S*', script or ''):
        if '=' in word or word in ('sudo', 'env', 'set', 'cd', 'export'):
            continue
        return word.rsplit('/', 1)[-1]
    return ''


def _docker(args: list, timeout: float = 120, input: Optional[str] = None) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(['docker', *args], capture_output=True, timeout=timeout, input=input,
                              text=True, errors='replace')
    except FileNotFoundError as e:
        raise InfrastructureError('找不到 docker 命令') from e
    except subprocess.TimeoutExpired as e:
        raise InfrastructureError(f'docker {args[0]} 超时') from e


class OsBackend:
    """确认镜像可用（没有时按配置的 Dockerfile 构建，否则拉取）；清理本线程遗留的容器。"""

    def __init__(self, image: str, dockerfile: Optional[Path], name: str):
        self.image, self.dockerfile, self.name = image, dockerfile, name

    def start(self, timeout: float = 600) -> None:
        if _docker(['image', 'inspect', self.image]).returncode == 0:
            return
        if self.dockerfile is not None:
            r = _docker(['build', '-t', self.image, '-f', str(self.dockerfile), str(self.dockerfile.parent)], timeout)
        else:
            r = _docker(['pull', self.image], timeout)
        if r.returncode != 0:
            raise InfrastructureError(f'镜像 {self.image} 不可用：{r.stderr[-300:]}')

    def stop(self) -> None:
        try:
            _docker(['rm', '-f', self.name], 60)
        except InfrastructureError:
            pass


class OsSession:
    def __init__(self, image: str, name: str, task: OsTask, timeout_s: int, obs_chars: int,
                 memory: str = '512m', pids: int = 256):
        self.image, self.name, self.task = image, name, task
        self.timeout_s, self.obs_chars, self.memory, self.pids = int(timeout_s), obs_chars, memory, pids

    def __enter__(self) -> 'OsSession':
        _docker(['rm', '-f', self.name], 60)
        r = _docker(['run', '-d', '--rm', '--name', self.name, '--network', 'none', '--memory', self.memory,
                     '--pids-limit', str(self.pids), self.image, 'sleep', 'infinity'])
        if r.returncode != 0:
            raise InfrastructureError(f'容器启动失败：{r.stderr[-300:]}')
        if self.task.init is not None:
            code, out = self.run(self.task.init, timeout_s=120)
            if code != 0:
                self.__exit__()
                raise InfrastructureError(f'题目 {self.task.task_id} 的初始化脚本失败（退出码 {code}）：{out[-300:]}')
        return self

    def __exit__(self, *exc) -> None:
        _docker(['rm', '-f', self.name], 60)

    def run(self, item: tuple, timeout_s: Optional[int] = None) -> tuple[int, str]:
        """在容器内执行 (command_name, script)；返回 (退出码, 合并的标准输出与标准错误)。"""
        name, script = item
        if name not in RUNNERS:
            raise InfrastructureError(f'不支持的命令类型：{name}')
        t = int(timeout_s or self.timeout_s)
        r = _docker(['exec', self.name, 'timeout', '-k', '2', str(t), *RUNNERS[name], script], timeout=t + 30)
        if r.returncode in (125, 126) and 'No such container' in (r.stderr or ''):
            raise InfrastructureError(f'容器 {self.name} 已不存在')
        return r.returncode, (r.stdout or '') + (r.stderr or '')

    def clip(self, text: str) -> str:
        text = text.strip('\n')
        if len(text) <= self.obs_chars:
            return text
        half = self.obs_chars // 2
        return f'{text[:half]}\n... ({len(text) - 2 * half} characters omitted) ...\n{text[-half:]}'

    def execute(self, script: str) -> ExecResult:
        if not script.strip():
            return ExecResult(False, 'Error: the command is empty.', None, 'empty')
        code, out = self.run(('bash', script))
        out = self.clip(out)
        if code == 0:
            return ExecResult(True, f'The output of the OS:\n\n{out}' if out else 'The output of the OS is empty.',
                              None, 'ok')
        kind = error_kind(code, out)
        head = f'The command was stopped after {self.timeout_s} seconds.' if kind == 'timeout' \
            else f'The command failed with exit code {code}.'
        return ExecResult(False, f'{head}\n\n{out}' if out else head, code, kind)

    def evaluate(self) -> int:
        return self.run(self.task.evaluation, timeout_s=60)[0]


class OsEnv(Environment):
    name = 'os'
    task_types = ('modify',)
    sealed_splits = frozenset({'test'})
    memory_splits = ('build',)
    obs_label = 'OS'
    core_ids = CORE_IDS
    selectable_groups = SELECTABLE_GROUPS
    insight_system = ('You analyse attempts of an agent that completes system administration tasks by running '
                      'bash commands on an Ubuntu system and then declares the task done.')
    insight_prompt = ('Below are pairs of attempts at operating system tasks. In each pair, the first attempt failed '
                      'and the second attempt succeeded on the same task.\n\n{pairs}\n\nWrite at most {n} general '
                      "rules that would help the agent succeed on similar future tasks. Each rule must be a single "
                      "line starting with '- ', must not mention specific file names, user names or values, and must "
                      'apply to other tasks as well. Output only the rules.')

    def __init__(self, cfg: dict):
        o = cfg.get('os', {})
        d = Path(o.get('data_dir', 'data/os'))
        self.data_dir = d if d.is_absolute() else ROOT / d
        self.image = o.get('image', 'prism-os:24.04')
        df = o.get('dockerfile', 'docker/os/Dockerfile')
        self.dockerfile = None if not df else (Path(df) if Path(df).is_absolute() else ROOT / df)
        self.timeout_s = int(o.get('timeout_s', 20))
        self.obs_chars = int(o.get('obs_chars', 2000))
        self.memory, self.pids = o.get('memory', '512m'), int(o.get('pids_limit', 256))
        self.container_prefix = o.get('container_prefix', 'prism-os')
        self.max_rounds = int(cfg['agent']['max_rounds'])
        self._features = lru_cache(maxsize=4096)(self._task_features)

    # ---- 数据 ----
    def load_tasks(self, split=None, include_excluded=False):
        meta = {m['task_id']: m for m in
                json.loads((self.data_dir / 'splits.json').read_text(encoding='utf-8'))['tasks']}
        wanted = None if split is None else ((split,) if isinstance(split, str) else tuple(split))
        out = []
        for line in (self.data_dir / 'tasks.jsonl').read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            m = meta[r['task_id']]
            if (m['excluded'] and not include_excluded) or (wanted is not None and m['split'] not in wanted):
                continue
            out.append(task_from_row(r, m['group_id'], m['split']))
        return sorted(out, key=lambda t: t.task_id)

    def data_files(self):
        files = [self.data_dir / 'tasks.jsonl', self.data_dir / 'splits.json']
        return files + ([self.dockerfile] if self.dockerfile is not None and self.dockerfile.exists() else [])

    def fingerprint(self):
        return {'env': self.name, 'image': self.image, 'timeout_s': self.timeout_s, 'obs_chars': self.obs_chars,
                'memory': self.memory, 'pids_limit': self.pids}

    # ---- 单元 ----
    def all_ids(self, task) -> frozenset:
        return ALL_IDS

    def unit_text(self, uid, task) -> str:
        if uid == 'task':
            return task.instruction
        if uid == 'rounds':
            return f'You have at most {self.max_rounds} rounds, and declaring the task done takes one of them.'
        return _TEXT[uid]

    def render_rules(self, unit_ids, task) -> str:
        ids = set(unit_ids)
        lines = [self.unit_text(u, task) for u, _ in CORE + RULES if u in ids]
        cmds = [f'- {usage}' for c, usage, _ in COMMANDS if f'cmd:{c}' in ids]
        if cmds:
            lines.append(COMMAND_HEADER + '\n' + '\n'.join(cmds))
        return '\n'.join(lines)

    def render_system(self, unit_ids, task) -> str:
        return self.render_rules(unit_ids, task)

    def render_first_user(self, unit_ids, task) -> str:
        return task.instruction

    # ---- 特征 ----
    def _task_features(self, task) -> TaskFeatures:
        text = task.instruction.lower()
        words = set(re.findall(r'[a-z0-9_]+', text))
        pred = frozenset(c for c, _, kws in COMMANDS if c in words or any(k in text for k in kws))
        groups = {g: frozenset({g}) for g in RULE_IDS}
        groups['cmd_predicted'] = frozenset(f'cmd:{c}' for c in pred)
        groups['cmd_other'] = COMMAND_IDS - groups['cmd_predicted']
        return TaskFeatures('modify', pred, groups, env=self)

    def task_features(self, task) -> TaskFeatures:
        return self._features(task)

    # ---- 交互 ----
    def first_action(self, response):
        return first_action(response)

    def parse(self, response):
        return parse(response)

    def grade(self, task, session, status, answer):
        code = session.evaluate()
        return code == 0, {'evaluation_exit_code': code}

    def recovery_groups(self, last_exec):
        if last_exec is None or last_exec.ok:
            return frozenset()
        return frozenset(RECOVERY.get(last_exec.kind, SELECTABLE_GROUPS))

    def error_key(self, result):
        return None if result is None or result.ok else result.kind

    def signature(self, result, action):
        key = self.error_key(result)
        return None if key is None else f'{key}|{first_command(action or "")}:'

    def expert_drop(self, phase, task_type):
        """人工规则（看到运行结果之前写定）：初始只发预测会用到的命令说明；成功执行后省略 shell 用法类规则
        与全部命令说明，只保留“依据观察”“结束前核对”和轮数限制。"""
        if phase == 'after_error':
            return ()
        if phase == 'initial':
            return ('cmd_other',)
        return ('fresh_shell', 'noninteractive', 'paths', 'no_sudo', 'on_error', 'cmd_predicted', 'cmd_other')

    # ---- 记忆 ----
    def example_text(self, task, actions, answer, max_answer_chars):
        actions = [actions] if isinstance(actions, str) else actions
        ops = ''.join(f'Action: Operation\n```bash\n{a}\n```\n' for a in actions)
        return f'Example task: {task.instruction}\n{ops}Action: Answer\nFinal Answer: done'

    def repair_text(self, error_line, failed, fixed):
        return f'Fix for a similar error: {error_line}\nFailed commands:\n{failed}\nCorrected commands:\n{fixed}'

    # ---- 运行 ----
    def container_name(self, idx) -> str:
        return f'{self.container_prefix}-{idx}'

    def make_backend(self, cfg, idx):
        return OsBackend(self.image, self.dockerfile, self.container_name(idx))

    def open_session(self, backend, task, idx, cfg):
        return OsSession(self.image, self.container_name(idx), task, self.timeout_s, self.obs_chars,
                         self.memory, self.pids)

    def scripted(self, kind, task):
        if kind not in ('gold', 'gold_after_error'):
            raise KeyError(kind)
        name, script = task.ground_truth
        if name != 'bash':
            script = f"python3 - <<'PRISM_EOF'\n{script}\nPRISM_EOF"

        def op(s):
            return f'I will run the commands.\nAction: Operation\n```bash\n{s}\n```'

        def reply(messages):
            n = sum(m['role'] == 'assistant' for m in messages)
            if kind == 'gold_after_error':
                if n == 0:
                    return op('ls /no/such/directory')
                n -= 1
            return op(script) if n == 0 else 'Action: Answer\nFinal Answer: done'
        return ScriptedLLM(reply, model=f'scripted-{kind}')

    def record_extras(self, task, features) -> dict:
        return {'task_type_pred': features.task_type, 'is_query': False,
                'predicted_commands': sorted(features.predicted_functions), 'skills': list(task.skills)}


def _item(d) -> Optional[tuple]:
    if not d:
        return None
    return (d.get('command_name') or 'bash', d.get('script') or '')


def task_from_row(r: dict, group_id: str, split: str) -> OsTask:
    return OsTask(r['task_id'], group_id, split, r['instruction'], _item(r.get('init')),
                  _item(r.get('ground_truth')), _item(r['evaluation']), tuple(r.get('skills') or ()))

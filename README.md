# 面向大语言模型智能体记忆的状态感知经验调度方法：实验代码

本目录是按开题报告（docs/开题报告_经验调度_20261025.docx）编写的全部实验代码：经验记忆库的构建与检索、
经验条目与指令单元的联合调度（PRISM），以及对比方法、消融与统计分析。
旧仓库（E:\硕士毕业论文）的代码不再使用，只沿用公开数据集原始文件与固定的题目划分。

## 目录

| 路径 | 内容 |
|---|---|
| `docs/` | 研究计划与第一至四步文档 |
| `analysis/` | 零调用分析：第一步问题分析、第四步可行性估算、功效模拟（共用代码在 `analysis/common.py`） |
| `data/db_bench.parquet` | LifelongAgentBench 数据库任务公开原始文件（500 题） |
| `data/splits.json` | 题目划分（构建集 / 开发集 / 测试集）、题目组与已批准的排除题目 |
| `data/tokenizer/` | Llama-3.1 分词器，用于本地 token 计数 |
| `src/prism_thesis/data.py` | 读取题目 |
| `src/prism_thesis/env.py` | MySQL 执行环境（Docker，每题独立建库，智能体只有本题数据的增删改查权限） |
| `src/prism_thesis/evaluate.py` | 判分，按官方 DirectTypeAnswerValidator 规则重新实现 |
| `src/prism_thesis/prompts.py` | 上下文单元（系统指令拆成可整体保留或省略的单元） |
| `src/prism_thesis/parser.py` | 解析智能体动作（官方规则）与第一个动作的截取 |
| `src/prism_thesis/features.py` | 任务类型、函数预测、交互状态识别（规则） |
| `src/prism_thesis/agent.py` | 智能体主循环，所有条件共用 |
| `src/prism_thesis/memory/` | 经验记忆库：条目与文件（store）、BM25 索引（bm25）、错误签名（signature）、状态感知检索与同组排除（retrieval）、离线构建与洞见提炼（build） |
| `src/prism_thesis/policies/` | 各实验条件：无记忆、完整记忆注入、人工阶段切换、自建静态基线、随机省略、LLMLingua-2、分层留一消融、PRISM |
| `src/prism_thesis/prism/` | PRISM 组成部分：风险表与估计、0-1 背包、显式错误补回 |
| `src/prism_thesis/stats.py` | 组 bootstrap、配对非劣检验、固定顺序检验、token 节省 |
| `src/prism_thesis/runner.py` | 批量运行、结果日志、续跑（按实验指纹核对配置）、累计费用上限、退出码 |
| `src/prism_thesis/checks.py` | 正式校准与最终比较前的输入检查（划分、题目集合、重复次数、模型、实验指纹） |
| `src/prism_thesis/cli.py` | 命令行入口 |
| `configs/default.json` | 实验配置（模型、解码参数、MySQL 镜像、ε 候选、δ 等） |
| `configs/extended_prompt.json` | 指令长度受控实验用的扩展版提示 |
| `tests/` | 测试（判分与官方实现逐例核对、解析、背包与穷举对照、策略、统计、数据库端到端） |
| `scripts/` | 环境安装脚本与不花钱的全流程演练脚本 |

## 安装

前提：Python 3.11 及以上；Docker Desktop（运行中）。

Windows（PowerShell，在本目录下）：

```
powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1
```

Linux / macOS / WSL：`bash scripts/setup_unix.sh`

LLMLingua-2 基线另需 `pip install -r requirements-llmlingua.txt`。
已有 MySQL 8 服务时，可在配置中把 `mysql.docker` 设为 false 并填写地址、账号与密码（此时只用一个工作线程）。

## 测试（不调用模型、不产生费用）

以下命令中的 `python` 指虚拟环境中的解释器（Windows 为 `.venv\Scripts\python`），需先设置 `PYTHONPATH=src`
（PowerShell：`$env:PYTHONPATH='src'`）。

```
python -m pytest -q tests                          # 离线测试；设置 PRISM_TEST_MYSQL=1 时包含数据库测试
python -m prism_thesis check-data                  # 题目划分与数量
python -m prism_thesis verify-env --split all      # 500 题参考 SQL 在 MySQL 中执行后全部判为成功
python -m prism_thesis show-prompt --task-id 5 --condition full
python scripts/dry_run_pipeline.py --limit 30      # 脚本化智能体跑通 无记忆运行→构建记忆库→消融→风险表→各方法→校准→比较（--rehearsal 演练模式，结果不得用于冻结）
```

## 零调用分析脚本（analysis/）

均可在项目根目录或任意目录运行；需要 numpy 与 tokenizers。第一步与第四步脚本只读地使用旧仓库中的系统提示、函数列表与
`first_action`，用 `--legacy-repo E:\硕士毕业论文` 或环境变量 `PRISM_LEGACY_REPO` 指定；不会在旧仓库写入任何内容。

```
python analysis/power_analysis/power_v2.py --bootstrap-check 300        # 功效模拟（研究计划表 7-1）
python analysis/step1_problem_analysis/tok_analysis.py --legacy-repo E:\硕士毕业论文
python analysis/step4_feasibility/units.py --legacy-repo E:\硕士毕业论文
python analysis/extract_n_runs.py --runs-dir E:\硕士毕业论文\runs\development_20261004_curve_v1 --out analysis/n_runs_check.jsonl
```

各目录 README 给出输入哈希与对应结果文件。

## 正式运行（调用付费接口）

需要环境变量 `OPENROUTER_API_KEY`，并显式加 `--allow-paid`。费用上限 `run.budget_usd` 针对整个结果文件
（包括以前各次运行已记录的费用，以及中途失败的运行已返回的模型响应），达到后停止派发。
结果按行写入 JSONL，中断后用相同命令续跑；每条记录带实验指纹（模型与解码参数、提示版本、数据、
与实验行为有关的代码、运行方式（付费 / 演练）、风险表内容等）；其中任何一项改变（包括修改代码）后续跑与正式分析都会拒绝，
须换新的输出路径重新运行。模型未返回 token 数时按本地计数估计费用。冻结后不要再改 src 中除 cli/checks/stats 以外的代码。数据库连接故障记为基础设施错误，续跑时重做。写入中断留下的不完整末行在续跑时移到 `<结果文件>.truncated-<时间>`（不覆盖已有备份）并重跑该次（其费用未计入）；正式分析遇到不完整末行或损坏的中间行会拒绝。
退出码：0 全部完成；1 有基础设施错误、工作线程失败或未完成；3 因费用上限停止。例如构建记忆库用的无记忆运行：

```
python -m prism_thesis run --split memory --conditions nomem --reps 2 --out results/memory/nomem.jsonl --allow-paid
```

## 经验记忆与实验条件

- **经验条目**：成功示例（任务指令、执行成功的 SQL、最终答案）、修复经验（错误签名、出错 SQL、改正后的 SQL）、
  洞见（模型离线从“失败—成功”对中提炼的规则，参照 ExpeL）。只由构建集与开发集上 `nomem` 条件的运行整理，
  测试集记录在构建与加载记忆库时都会被拒绝；每条记录来源题目与题目组。
- **检索**（不调用模型）：任务开始时按任务指令的 BM25 相似度取 `k_examples` 条示例（第 1 条为组 `exp_top1`，
  其余为 `exp_rest`）与 `k_insights` 条适用于该任务类型的洞见（组 `insight`）；完整记忆注入另按任务相似度取
  `k_repairs` 条修复经验（组 `repair`）。出错后按错误签名（错误号 + SQL 结构特征，逐级放宽）检索修复经验。
  与当前题目同组的条目一律排除。
- **条件**：`nomem` 无记忆；`full` 完整记忆注入（ExpeL 式，非劣检验与风险估计的参照）；`phase_switch` 与 PRISM
  使用状态感知检索（初始只取示例与洞见，出错后按签名取修复经验，单调保留）；`static`、`random_drop`、
  `llmlingua2` 与消融使用任务开始时的检索。经验在官方布局下放在第一条用户消息的任务指令之前，在 PRISM 中放在末尾临时块。
- **风险表**的组包括指令组与经验组（`exp_top1`、`exp_rest`、`insight`、`repair`）；`ablate:exp_top1@initial` 等
  照常估计。检索为空的经验组不算暴露。
- **k 的敏感性**：任何条件后加 `#k=N` 覆盖 `memory.k_examples`，例如 `full#k=1,prism@4#k=1`，
  比较时用 `compare --control full#k=1`。
- `show-memory --task-id N` 查看某题检索到的经验；`memory-stats --runs ...` 统计各条目被发送的次数与所在运行的
  成功率（记忆演化的扩展分析）。

开题报告各阶段与命令的对应：

| 阶段 | 命令 |
|---|---|
| 记忆库构建用的运行（构建集 + 开发集，每题 2 次） | `run --split memory --conditions nomem --reps 2 --out results/memory/nomem.jsonl` |
| 洞见提炼（离线，调用模型，费用上限 `memory.insights.budget_usd`） | `distill-insights --runs results/memory/nomem.jsonl --out results/memory/insights.json --allow-paid` |
| 构建记忆库（不调用模型；检查覆盖构建集与开发集全部题目、每题 `reps.memory_build` 次） | `build-memory --runs results/memory/nomem.jsonl --insights results/memory/insights.json` |
| 问题分析重新验证（开发集，无记忆与完整记忆） | `run --split dev --conditions nomem,full --reps 3` |
| 经验风险估计（开发集） | `run --split dev --conditions full,ablate:<组>@<状态>,...`，再 `risk-estimate`（检查开发集全部题目与 `reps.risk_estimate` 预定次数：full 3 次、消融 2 次，次数冻结前确定） |
| 覆盖要求 | 风险表只写入暴露题目组数不少于 `prism.min_exposed_groups`（默认 10）的条目，其余回退完整上下文；`risk-estimate` 打印覆盖诊断 |
| 互补组 | `complementary_pairs`（含 `exp_top1`+`insight`）中的两组在某状态下都做了单独消融时，必须同时运行联合消融 `ablate:<组1>+<组2>@<状态>`，否则 `risk-estimate` 拒绝；`--single-only`（供 `prism-singleonly` 消融）不使用联合消融记录 |
| 冻结 | 冻结 `configs/default.json`、`results/memory/memory_store.json`、`results/risk/risk_table.json` 与本目录代码版本 |
| 独立校准（构建集，10 次/题） | `run --split construction --conditions full,prism@1,prism@2,prism@4,prism@8 --reps 10`，再 `calibrate` |
| 最终对比（测试集，各条件 10 次） | `run --split test --conditions full,nomem,phase_switch,llmlingua2,static@4,random_drop,prism@<ε*> --reps 10`，再 `compare --split test` |
| 消融（开发集） | `prism-stateless@ε`、`prism-nosig@ε`、`prism-norecovery@ε`、`prism-singleonly@ε`、`prism-tailsys@ε`、`prism-greedy@ε` |
| 检索条数 k 的敏感性（开发集，3 次/题） | `run --split dev --conditions full#k=1,prism@<ε>#k=1,full#k=3,prism@<ε>#k=3,full#k=5,prism@<ε>#k=5 --reps 3`，再对每个 k 分别 `compare --split dev --reps-key k_sensitivity --control full#k=<k>`（输入只含该 k 的两个条件） |
| 计数核对 | `token-audit` |
| 动作格式离线复查（不调用模型，用当前解析规则重新解析已有记录的原始回复） | `reparse --runs <结果文件>` |
| 解析与判分一致性检查（把已有轨迹重放到本环境重新判分，需 Docker） | `python scripts/replay_runs.py <轨迹文件>` |
| 离线查看每局每轮的回复、数据库返回与诊断计数 | `show-runs --runs <结果文件> [--brief] [--instructions]` |
| 提示措辞对照（调试用，不用于正式实验） | 配置项 `prompt_overrides`（单元 id → 文本）与 `schema_style`（official / typed / legacy），示例见 `configs/probe/`；两者都计入实验指纹 |

使用记忆的条件，其实验指纹包含记忆库内容与检索设置：重建记忆库后须换新的输出路径。`nomem` 不依赖记忆库，
修改记忆库构建代码（`memory/build.py`）也不会使已完成的 `nomem` 运行失效。

## 与官方 db_bench 的差别（所有条件相同）

1. 系统指令为本研究编写（官方提示文件未能取得），采用预实验验证过的措辞，动作格式与官方解析规则一致；表信息写作
   `Schema: {列名: 类型}` 换行 `Table: 表名`（配置 `schema_style: legacy`，比官方的表信息句多了列类型）；唯一差别是 SQL 代码块内部的换行原样保留
   （官方替换为空格，会让 `-- ` 与 `#` 行注释吞掉下一行，例如 WHERE 条件）；
2. 每次操作只允许一条 SQL 语句；智能体账号只有本题数据库的增删改查权限；单条语句执行上限 10 秒；
3. 只保留模型回复中的第一个完整动作（Llama-3.1-8B 常在操作后自行编造结果）；官方解析判为无效、但第一个动作标签写成 SQL 动词
   （如 `Action: Query`、`Action: Update`）且后跟 ```sql 代码块的回复，统一改写为 `Action: Operation` 加该代码块
   （预实验中约四分之三的首轮回复属于这种情况）；没有代码块、动作行后直接写 SQL 的回复，取这条裸 SQL
   （到第一个以分号结尾的行、空行或模型自行编造的结果表格为止）；这两项只对官方解析判为无效的回复生效，
   既无可识别 SQL 也无 Final Answer 行的回复仍判为无效动作；
4. 修改题以结束时的表哈希判分（官方在作答与无效动作两种结束方式下也如此），查询题须给出 Final Answer。

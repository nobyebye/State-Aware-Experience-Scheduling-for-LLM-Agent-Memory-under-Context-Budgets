# 第一步：问题分析（零调用）

输入：`analysis/n_runs_extract.jsonl`（由 `analysis/extract_n_runs.py` 从旧仓库 runs/development_20261004_curve_v1 抽取的 330 次无经验基础运行）、
`data/tokenizer/llama31_8b_tokenizer.json`，以及旧仓库中运行器的系统提示、SQL 函数列表与 `actions.first_action`（只读）。
共用的加载与计数代码在 `analysis/common.py`，脚本不依赖当前工作目录。

依赖：Python ≥ 3.10、tokenizers（本次核对用 0.23.3）。在项目根目录运行，例如：

```
python analysis/step1_problem_analysis/tok_analysis.py --legacy-repo E:\硕士毕业论文 > analysis/step1_problem_analysis/results_tok_analysis.txt
```

也可先设置环境变量 `PRISM_LEGACY_REPO`（PowerShell：`$env:PRISM_LEGACY_REPO='E:\硕士毕业论文'`），再省略 `--legacy-repo`。
`--data` 与 `--tokenizer` 可改用其他输入。输入文件的 SHA-256 打印到标准错误，不进入结果文件。

- tok_analysis.py：重建每轮输入并计数，与服务商 prompt_tokens 核对（709/709 一致），给出构成与各轮长度。
- more.py：输出 token 浪费、任务描述长度、按操作类型的成功率、系统提示逐行 token。
- rep.py：跨轮重复发送占比、失败类别。
- fn.py：SQL 函数列表的实际使用情况。
- phase.py：各轮动作类型与动作序列分布。
- results_*.txt：对应输出。2026-10-22 改为共享模块后重跑，与原结果逐字一致
  （输入 SHA-256 前 16 位：n_runs_extract.jsonl `15ced1a5007a597a`，tokenizer `6b9e4e7fb171f92f`）。

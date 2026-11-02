# 第四步可行性估算

- `units.py`：把系统指令拆成单元，用第一步的 330 次基础运行（零调用）估算 PRISM 手工策略下的输入 token。
- 运行（项目根目录）：`python analysis/step4_feasibility/units.py --legacy-repo E:\硕士毕业论文`；输入与依赖同 `step1_problem_analysis/README.md`。
- 结果（`results_units.txt`）：每题输入 954.3 → 671.1 token（−29.7%）；函数预测召回 240/265（90.6%）；查询意图任务 228。
- 该数字沿用完整提示下的历史轨迹计算，是固定轨迹估算，不是实验结果；压缩后轨迹也可能改变（变长或变短），
  因此不能把它当作节省量的上界或下界。

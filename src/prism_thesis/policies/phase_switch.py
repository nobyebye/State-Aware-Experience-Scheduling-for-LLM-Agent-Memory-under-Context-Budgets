from .base import Plan, Policy, TurnState

ANSWER_FORMAT = ('answer_query', 'empty_scalar')


class PhaseSwitchPolicy(Policy):
    """人工阶段切换：按状态切换人工确定的指令与经验，不估计风险、不做优化。
    规则（冻结前写定）：
      初始：只保留预测会用到的函数说明；经验发送相似示例与洞见；
      出错后：发送全部指令单元；经验发送相似示例、洞见与按错误签名检索到的修复经验；
      成功执行后：省略 SQL 规则、全部函数说明与表信息；经验只发送洞见；
      非查询任务在初始与成功执行后省略查询答案格式说明。
    经验检索与 PRISM 相同（状态感知检索），二者只在“是否估计风险并优化选择”上不同。
    提示整体替换（官方布局），不使用临时块。"""
    name = 'phase_switch'

    def plan(self, state: TurnState) -> Plan:
        f = self.features
        repairs = self.collect_repairs(state)
        ex = self.exp_initial
        if state.phase == 'after_error':
            drop = ()
            exp = set(ex.get('exp_top1', ())) | set(ex.get('exp_rest', ())) | set(ex.get('insight', ())) | repairs
        elif state.phase == 'initial':
            drop = ('fn_other',)
            exp = set(ex.get('exp_top1', ())) | set(ex.get('exp_rest', ())) | set(ex.get('insight', ()))
        else:
            drop = ('one_stmt', 'scope', 'fn_predicted', 'fn_other', 'schema')
            exp = set(ex.get('insight', ()))
        if state.phase != 'after_error' and f.task_type == 'modify':
            drop += ANSWER_FORMAT
        instr = frozenset(u for u in self.drop_groups(drop) if not u.startswith('exp:'))
        return Plan(prefix=instr | frozenset(exp), info={'dropped': list(drop), 'n_exp': len(exp)})

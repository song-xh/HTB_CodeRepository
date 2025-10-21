# Todo List #

- [ ] 补全“作业库”为全流程 + 依赖/互斥/资源
    - 文件：utils/job.py
    - 动作：用技术资料构建 Job 集：进场(ZY_Z/ZY_M)、固定(ZY01)、保障(ZY02~)、解固(ZY_L)、转运(ZY_T)、出场(ZY_S/ZY_F)；
      每个 Job 增加字段：required_resources、predecessors、mutex、group、repeat_after_transfer、time_span(常量或-1表示按距离/设备到位计算)。
    - 目的：让后续“可行动作判断”和“时间计算”有真实语义支撑。

- [ ] 将任务编排从“固定串行”升级为“前置-后置DAG”
    - 文件：utils/task.py
    - 动作：实现 TaskGraph.enabled(finished_codes, ongoing_mutex) 返回候选作业；保留 simple_task_object 供兼容，但调度以 DAG 为准。
    - 目的：表达“前置满足即可进入候选、互斥不可并行”，为动作掩码提供第一道过滤。

- [ ] 飞机实体改为“状态机 + ETA（移动/加工分离）”
    - 文件：utils/plane.py
    - 动作：新增 status ∈ {IDLE, MOVING, PROCESSING, DONE}，current_job_code/current_site_id，
      eta_move_end/eta_proc_end，finished_codes/ongoing_mutex；提供 start_move/start_job/advance/finish_job。
    - 目的：支持“最小非零ETA推进”与后续“动态设备、路径占道”。

- [ ] 站位能力与资源配置规范化 + 不可用时窗
    - 文件：utils/site.py
    - 动作：以 Job.code→id 生成每个站位可执行作业集与资源数；预留字段：unavailable_windows、degraded_capacity_t。
    - 目的：避免随机能力造成训练不可控，并为扰动与重调度留接口。

- [ ] 动作掩码扩展为“DAG候选 × 能力/产能/时窗/空间”
    - 文件：environment.py（get_avail_agent_actions / 相关内部函数）
    - 动作：先取 TaskGraph.enabled(...) 作为候选作业；对每个站位动作做4类检查：
      (1) 能力包含该Job；(2) 资源数>0；(3) 不在不可用时窗；(4) 空间/路径不冲突(开关形式)。
      仍保留 WAIT/BUSY/DONE 三类基础动作；维持“同一步预占” has_chosen_action。
    - 目的：用掩码实现规则，保证训练样本全可行。

- [ ] 时间推进从“站位剩余时间最小值”改为“全体飞机ETA最小非零值”
    - 文件：environment.py（step / 推进逻辑）
    - 动作：从所有飞机的 {eta_move_end, eta_proc_end} 取最小非零值推进；ETA=0 触发状态流转与 finish_job/资源释放。
    - 目的：时间轴与真实过程一致，奖励对齐 makespan。

- [ ] 奖励函数保留“makespan为主 + 轻量塑形”
    - 文件：environment.py（reward）
    - 动作：终局奖励与 makespan 负相关；步时轻惩罚（移动/等待/冲突尝试），幅度小于终局目标；非法动作通过掩码禁止，不用重罚。
    - 目的：稳定学习、聚焦全局完工时间。

- [ ] 观测/全局状态追加“先验通道”并固定维度
    - 文件：environment.py（get_obs/get_state）
    - 动作：在 obs/state 末尾预留固定 padding 维度（如 32/64）；后续将 GNN 先验（φ_pred 和站位统计）写入此通道。
    - 目的：后续接入先验或新增特征无需改网络形状，可直接微调。

- [ ] 动态/移动资源（课程开关，逐步启用）
    - 文件：utils/site.py（设备池定义）、environment.py（设备到位时间/占用释放）
    - 动作：定义设备：位置/速度/占用至/维修复位；动作掩码检查“设备可达”；ETA 包含“设备到位时间”。
    - 目的：贴合“供给车/牵引车”等移动约束，作为高级难度开关。

- [ ] 空间/路径干涉（课程开关，逐步启用）
    - 文件：util.py（简单路径/占道模型）、environment.py（冲突检测/掩码）
    - 动作：用直线段+时间窗判定与他机/设备的占道冲突，冲突则屏蔽该站位动作；必要时在 step 中做防御性惩罚。
    - 目的：覆盖“空间干涉约束”，保证可行性。

- [ ] 扰动事件与重调度接口
    - 文件：environment.py（apply_disturbance）
    - 动作：实现通用事件：站位降级/不可用、设备故障、工时缩放等；刷新资源/时窗；支持“从当前时刻继续调度”评测。
    - 目的：满足任务书“特情识别与再调度”。

- [ ] 输出格式标准化（供任务三消费）
    - 文件：runner.py 或 evaluate 阶段
    - 动作：将 info["episodes_situation"] 序列化为 JSON：{plane_id, job_code, site_id, start_min, end_min, move_min, proc_min} 列表。
    - 目的：形成“当前时刻之后的完整后续计划”。

- [ ] 主程序开关与初始化顺序
    - 文件：MARL/common/arguments.py、main.py
    - 动作：增加 env_mode/enable_deps/enable_mutex/enable_dynres/enable_space 等开关；
      注意在 attach_prior(...) 之后再调用 env.get_env_info() 以正确设置 obs/state 维度。
    - 目的：支持“渐进式解锁约束”的课程训练与先验挂接。

- [ ] 兼容性与稳定性小补丁
    - 文件：MARL/policy/qmix.py 与 vdn.py（可选）
    - 动作：自动加载“最新编号”checkpoint；（可选）Double-QMIX 目标降低过估计。
    - 目的：训练迭代便捷、收敛更稳。

- [ ] 构建“测试图谱”（任务一未完成时的替代输入）
    - 文件：scripts/build_test_kg.py（新增）
        - 动作：从训练CSV + 当前环境采样生成 Graph/Temporal Graph：节点=Plane/Site/Job，关系=P→S、S→J、P→J；
        可按时间切 K 片形成时序快照；导出 demo_graph(_seq).json。
        - 目的：为 GNN 产生 φ_pred 和站位风险/可用性先验。

- [ ] 实现先验适配器（TKGPrior）
    - 文件：task2/prior_adapter.py（新增）、task2/tkg/dtrgnn.py（新增，可先用轻量GNN/启发式兜底）
    - 动作：从图谱前向得到 plane_prior(φ_pred=total,remain,eta_min) 与 site_prior(拥塞/可用性…)，提供 attach_prior 用的统一接口；
      若图不全，用“就近站位ETA + 剩余工序和”启发式兜底。
    - 目的：将知识图谱/预测转化为可注入的数值先验。

- [ ] 在环境中挂接先验并注入 obs/state
    - 文件：environment.py、main.py
    - 动作：env.attach_prior(prior, dims)；在 get_obs()/get_state() 的 padding 槽写入
      [φ_pred_i ⊕ site_prior(mean/min/max) ⊕ plane_prior全局统计]；attach 后再 get_env_info()。
    - 目的：让个体RNN与QMix超网络都“看见”先验（知识增强混合）。

- [ ] 校核动作掩码与“同一步预占”
    - 文件：environment.py（get_avail_agent_actions/has_chosen_action）、MARL/common/rollout.py
    - 动作：rollout 里先取掩码再 ε-greedy；若选站位，当步 has_chosen_action 预占；step 时按 ETA 推进并释放资源。
    - 目的：保证环境约束在训练采样中严格生效。

- [ ] 训练（在线采样 + 经验回放）
    - 命令：python main.py --n_agents=8 --alg qmix --learn True --load_model False \
            --env_mode techdoc_min --enable_deps --enable_mutex [--enable_dynres] [--enable_space]
        - 细节：开启新约束（dynres/space）时，清空 ReplayBuffer 但保留模型权重小步微调；奖励以 makespan 为主，步时轻惩。

- [ ] （可选）预热回放
    - 文件：runner.py/rollout.py（小开关）
    - 动作：用规则/GA/DE 先采 N 幕，批量写入 ReplayBuffer 作为 warm-start，再启动 RL 学习。
    - 目的：减少早期探索抖动、加速收敛。

- [ ] 评测与导出“完整后续计划”
    - 命令：python main.py --n_agents=8 --alg qmix --learn False --load_model True
    - 动作：Runner.evaluate 导出 episodes_situation 至标准 JSON（plane_id, job_code, site_id, start/end/move/proc）。
    - 目的：形成任务二的可用调度输出，也可直接作为任务三的输入。

- [ ] 课程式逐步解锁约束并微调
    - 顺序：DAG+互斥 → +动态/移动资源 → +空间/路径干涉 → +TKG/GNN先验强化
    - 动作：每解锁一项特性：保留模型、清空回放、短轮数微调；obs/state 形状不变（先验填入 padding）。

- [ ] 模型保存/加载与复现
    - 文件：MARL/policy/qmix.py（自动找最新ckpt）、MARL/common/arguments.py（记录随机种子、超参）
    - 动作：保存评测最优与最近；在 result_name 中包含环境开关，用于区分实验。
    - 目的：可复现、可比对、可回滚。

- [ ] 消融与对照（报告需要）
    - 动作：对比 “无先验 / 启发式先验 / GNN先验” 与 “VDN/QTRAN”等算法，记录 makespan/冲突率/移动时长；
      固定随机种子重复多次取均值方差。
    - 目的：证明“知识增强 + QMIX”的有效性。

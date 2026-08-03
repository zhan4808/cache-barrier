# WeChat update draft for Dr. Xiao — 2026-08-01

肖老师好，本周进展（数字为主）：

1. 您的 Qwen3.6-27B profile 里 aten::mm 占 GPU 时间 86.2%（25.8s/29.9s）——mm 提速 2× = 端到端 1.76×，其余全部无限加速也只有 1.16×，所以我们只做 GEMM。
2. 核心图已做出（clock-locked，CUDA graph 计时）：权重 8–128MB × T=1–512 × {bf16/W4A16/W8A8} 全平面。T≤32 时量化收益的正负号在 32→40MB 之间翻转，正好卡在实测 C_eff=36MB（非标称 50MB）：W8A8 从 0.7× 跳到 1.2–1.46×。
3. 这把我们和 MARLIN 的"矛盾"变成同一模型的两个区间：L2 驻留区量化无收益（我们的null结果），HBM流式区按字节比收益（MARLIN 的 4×）；W4A16 因 in-core 反量化上限（0.496 TB/s）远场只到平手——模型对翻转位置和上限都预测对了。
4. 模型精度（分区间报告）：cuBLAS 基线全域 12–21% MAPE；量化 Triton kernel 在 decode 区（T≤32）8–39%；T≥64 时 Triton kernel 偏离自身 roofline（诚实报告，属 kernel 限制，调度上也不会选它）。
5. 边界条件不变：serving 竞争下 L2 驻留区是阶跃式消失（7月已测），所以此图是隔离 microbenchmark 的上界；论文已按"容量门控+per-kernel谓词"重写 title/abstract，MLA 降为 case study。
6. 下一步：① served A/B+精度（harness已就绪）② A100 拟合→H100 预测的迁移 MAPE（可移植性关键数字）。另：tritonBLAS (2512.04226) 是最近的相邻工作——他们用 cache 模型选 tile 配置，我们选精度，差异已写进 related work。

---
## Evening addendum (served results, 2026-08-01)

7. 实测 serving A/B(真实 Qwen3.6-27B,vLLM,64 并发):decode fp8 W8A8 = **1.45×**(2931 vs 2026 tok/s),prefill 1.20×;wikitext-2 ΔPPL 仅 +0.032(7.528→7.560)。7月的 e2e 投影现在是实测值。
8. 诚实负结果:按 wave-band 边缘调 max_num_batched_tokens 反而降低 decode 吞吐(2931→2718/2818)——机制B属于 prefill batch shaping,decode 引擎 A/B 中不显现。
9. P5/P6 已完成:可移植参数 harness(H100 自校验通过;A100 迁移 20–29% MAPE,证明 kernel 参数必须实测不能外推)+ 调度成本模型(结论:quantized-primary 存储是唯一零成本策略)。论文正文已重写(18页编译通过);讲稿18页含每页数据来源。下一步只差:真实 A100 跑一次 harness(几小时)。

---
## A100 addendum (2026-08-01, session 5)

10. A100 实测参数替换估计值:迁移 MAPE 28.8/22.8% → **18.3/13.0%**(2026-06 eager 数据集,below/above gate);同卡 graph-timed 自洽目标 above-gate **19.0%**,below-gate 44%(模型形式残差,非常数问题——两个数据集一致)。实测 C_eff=31.2MB=0.78×标称(与 H100 同比率);r_dequant 实测 0.394 vs 全扫描拟合 0.406 TB/s(3%)。注意:到手的是 40GB 卡(HBM2 1.51 TB/s,非 80GB 的 1.94)——因此把 graph-timed 同卡重测提为主目标。附带发现:eager 数据里 58.7µs 的"kernel 固定成本"在 graph 计时下只有 3.5µs,是计时伪影。Gate 翻转 T∈{1,16,32} 减量扫描:**容量结构可迁移**(w8a8 收益峰值正好落在 31–62MB 非对称驻留带,2.07× @ 40MB),但 below-gate 的负号不迁移——A100 的 bf16 基线较慢,act-quant 开销压不过它,w8a8 在 T=1 全域获胜。结论收紧为:门的容量结构跨架构成立,below-gate 分支的符号取决于基线 kernel 质量(正是模型 per-kernel 项的职责)。

---
## H100 session 6 addendum (2026-08-01 night)

11. Below-gate 44% 残差分解完毕(两个实测原因,都不是常数问题):① **驻留由总 footprint 门控**(权重+激活+输出),非权重 operand——W=16MB 到 T=128(fp=36MB)仍 L2 驻留,W=32MB 在 T=24 就崩(fp=39MB);但塌陷是软的(40–60MB 过渡带 2–3TB/s),二值门只挽回 ~2 个点;② A100 小 kernel 带宽下限 ~1.2–1.4TB/s(门两侧同样,饱和问题非容量问题)。诚实更正:昨天"A100 bf16 延迟曲线在 31MB 可见拐点"的说法细读数据不成立,已在论文撤回。论文 case study 压缩完成:**18→15 页**,0 错误。机制 B prefill 侧:**相邻 chunk cap 896→1024 有可复现 +24% 台阶**(两端重复 ±0.4%,波段特征在 engine 级出现)——但任何 cap 都不赢 uncapped(首轮"1024 赢 17%"在重复实验下消解,uncapped 方差 ±10%);统一结论:engine 级 batch shaping 只能避免损失(896/512/460 亏 20–28%),不能创造收益。另:mb=2048 在此栈上直接挂死(vLLM 0.20.2+GDN+chunked prefill,已记录)。

12. A100 below-gate 44% 残差已定位(关机前在 A100 上补了探针):sm_80 cuBLAS 在 M=1 派发 wmma_16x16/gemvx 小 kernel(无 TMA 路径)——HBM 区达规格 98%(1.53/1.555 TB/s,无缺口),L2 区被卡在 ~0.9–1.3 TB/s 且与 CTA 数无关(H100 nvjet 同点 ~8 TB/s)。用与 W4A16 相同的两点校准法给 baseline 定价(1.28 TB/s):below-gate zero-shot MAPE **44.0%→12.8%**(14 个留出点)。结论闭环:不是模型形式、不是常数,是未测的 baseline kernel 项——**baseline kernel 也是 kernel 项**。论文已更新(15 页,0 错误)。

---
## H100 session 7 addendum (2026-08-02)

13. 两个 below-gate 残差全部闭环:① 软过渡带两参数拟合(C_hi=56MB,floor=2.10TB/s,只在 footprint 数据上拟合),留出验证 H100 gate sweep below-gate **20.9→14.3%**;归一化后零样本迁移到 A100(叠加 sm_80 baseline 校准项):**below 12.4% / above 13.2%**——迁移故事现在是"常数(harness)+ 每 kernel 两点项 + 一个归一化带形状"。② 对称性检验:H100 baseline 两点校准 = 6.51 ≈ harness 6.3,nvjet 无隐藏项。③ NCU 计数器直接证实 footprint 门控:固定 W=24MB,DRAM 重读 0.2→18.1→29.3→31.5MB 随 footprint 跨过 C_eff 软性爬升。④ 机制 B 桥接完成:27B 真实 prefill GEMM 形状在 M=896 处 per-token 代价比 1024 高 **+17%**(fp8/bf16 同形)——engine 级 +24% 台阶 = kernel 级锯齿 + chunk 数开销。论文 16 页 0 错误。

---
## B200 session addendum (2026-08-02)

14. B200 双目标全部落地。**目标一(第三架构点)**:实测 C_eff = **98.8 MB = 0.78× 标称 126.5**——A100/H100/B200 的有效/标称比在 harness 网格分辨率(±0.04)内一致为 **≈0.78**,跨 3.2× LLC 容量、三代架构一致,slide 12 的 LLC 增长故事拿到最干净的数据点(比率精度问题见第 15 条更正)。below-gate 区按论文预测大幅扩张:原 8–128MB 网格在 B200 上全部 ≤1.3× gate。零样本迁移阶梯:仅常数 17.6%/1.8%(below/above);A100 配方的两个修正项在 B200 上**双双失效且失效本身有信息量**——两点 baseline 校准被 sm_100 kernel 选择锯齿破坏(56MB 格点比邻居贵 2×),H100 拟合的持久 floor 被证伪(B200 远场跑满 HBM 速率 6.2–7.6 TB/s),回答了 session 7 的悬留问题:floor 是 H100 特有产物,不可迁移。保留带宽塌陷带的*宽度比*(C_hi/C_eff=1.56)、去掉 floor 和 baseline 项 → **零参数带,16.6%/3.9% 全零样本**。**目标二(native-FP4 预测)**:**确认**。同一张卡、同一 Mixtral 形状、同一 graph 计时:Marlin W4A16(dequant→bf16)在 T*≈159 交叉、T=2048 时输 2.4×;native W4A4 NVFP4(cutlass,FP4 tensor core,无 in-core dequant)**全程不交叉,2.1–3.1×**。T=2048 两种 FP4 表示相差 **6.3×**(785.6 vs 4944.8 µs)——这个差距就是 in-core dequant ceiling,先测到、再被硬件移除;carm_cuda_fit 预注册的 `b200 w4a4: none(wins)` 在真机上兑现。r_dequant → ∞。诚实声明:①本实例无法锁频(-lgc 被拒),实测持续负载频带 1237–1320 MHz ±3%,已记入每个结果文件;②vs-bf16 的倍数用的是 vLLM 默认 triton 配置 baseline(B200 无调优配置文件,guardrail 3),但 native-vs-Marlin 的 6.3× 不共享任何 bf16 baseline,是干净的主张;③W4A4 在随机数据上代价 ~22% 相对误差(软件仿真同为 0.2221,是精度本身的代价、非 kernel 错误)——这个速度不是免费的。fitted native FP4 峰值 3068.6 TF vs dequant ceiling 488.6 TF;native_peak_mult 实测 2.6×(投影 4.0× 偏乐观)。另:goal-1 gate sweep 里 triton w8a8 above-gate 优势消失(0.78–0.82× vs H100 1.19–1.46×)已由 CUDA 复现定性为 triton 3.4/sm_100 kernel 不成熟(Marlin 路径在同卡上小 T 赢 1.85×),非架构物理。

---
## H100 session 9 addendum (2026-08-02)

15. 三项收尾,一项更正。**① 比率精度更正(诚实声明)**:此前报告的 0.780/0.780/0.781"三位小数不变"是网格量化伪影——portable harness 的 cliff 扫描网格按标称 L2 等比取点(0.4–1.5×,14 点),候选比率在所有卡上是同一组有理数,三张卡落在同一格就必然逐位一致,真实分辨率是半步长 = 标称的 ±4%。H100 上以 0.5 MB 步长重扫(3 次重复):塌陷起点 **39.8±0.5 MB = 0.795±0.010× 标称**(塌陷在 ~3.5 MB 内完成,非瞬时悬崖)。诚实的跨架构表述:**三代架构共同的 ≈0.8× 分数**,不是三位小数常数。论文、deck、本文第 14 条已同步修正。**② floor-free 带在 H100 上复算**(session 7 悬留问题的另一半):去掉拟合 floor、ramp 到 bw_hbm(B200 的零参数形式),H100 留出 gate sweep below-gate 14.3%→17.3%,above-gate 持平(13.3→13.4%);floor-free 重拟合带宽度得 C_hi=1.22×C_eff(16.8%/12.1%)。结论:**H100 的远场确实低于 HBM 速率流(floor 在本卡是真实的 kernel 项,值 ~3 个点),但不可迁移**——可迁移形式是零参数带 + 各架构自己的局部 kernel 项,与 B200 的证伪完全自洽。**③ sm_clock 元数据修复**:五个 bench 脚本此前在启动时(空载)查询 nvidia-smi,记录的是 DVFS 地板(B200 文件里的 705/750 MHz 即此伪影,真实负载频带 1237–1320);现改为在饱和 GEMM 循环中采样,B200 旧结果文件已加注释(未改动实测数据)。顺带发现:H100 即使 -lgc 1755 锁频,饱和算力负载下也被功耗墙压到 ~1380–1395 MHz——"锁频"只封顶,不保底,采样器输出区间如实呈现。论文重编译 16 页 0 错误。

---
## H100 session 10 addendum (2026-08-03)

16. 容量门第一次作为 **kernel 设计工具** 闭环(两条线均先预注册后测量)。**① 现状测量**:fla 0.5.2 的 `fused_recurrent_gated_delta_rule`(Qwen3-Next/Kimi-Linear 类混合模型的 decode 内核)在 H100 上 8–160MB 全程平坦 ~2.3–2.4 TB/s——低于 HBM 流式速率 3.15,warm/rotated 优势 ≤1.13 且 24MB 后归零:**今天的生产 GDN decode kernel 完全看不见 L2 层**,门预言的 ~2.7× 余量(bw_l2 6.3 vs 实测 2.35)整段留在 below-gate 区没人拿。**② 动手拿**:~100 行 Triton 单程 kernel(每 (batch,head) 一个 program,流量恰为 2× state,正确性 1e-8),单算子对比 24MB 处 **2.20×**;再补齐完整 decode 步(short-conv+silu 滚动缓存、qk l2norm、delta rule、gated RMSNorm 全融合)对 fla 真实 3+ kernel 链:**below-gate 2.00–2.34×,above-gate 1.21–1.30×**——分解为融合红利 ~1.25×(全域)× 驻留 ~1.9×(窗口内),交叉点 40–48MB 正落在 fine-grid C_eff=39.8。门先说余量在哪、到哪消失,kernel 第一次配置搜索就拿到 ~80%。chunked prefill 变体分析后**有意不做**(matmul 分块吃 tensor core,驻留设计在 decode 才有付费点)——主张范围收在 decode。**③ 两容量问题 NCU 反证闭环**:warm-state DRAM 读计数显示 GEMM 语境的驻留转变在 **~34±2 MB**,纯重读在 **~40±2**——~6MB 的 GEMM-context 差是真的(cuBLAS tiling 持有驻留更差,但 44MB 处仍有 37% 命中尾巴)。结论:**C_eff 依赖 operand 语境**,模型今后携带 C_eff(re-read)=39.8 与 C_eff(GEMM)≈34 两个实测常数;6 月的 36 恰在两者之间,不是错。论文新增 §"The Gate as a Design Tool"(17 页 0 错误);fla upstream issue 草稿已备好待审发出。诚实范围:单卡、fp32 state、decode-only;fla 链 conv/norm 用其默认 bf16 而我们全 fp32(state 流量主导,二阶)。

---
## B300 session 11 addendum (2026-08-03)

17. **B300 第四架构点**(SXM6 风冷版,当天预注册当天测,严格按 PREREG 评分)。**① 标称容量**:device 报告 L2 = **126.5 MB,与 B200 完全相同**(SM 数也同为 148)——二手渠道的 192MB 传闻在此卡上被否定(caveat:这是 275GB 风冷版,288GB 水冷版无证据)。**② P1 预注册区间 0.77–0.82 被证伪(诚实记录)**:fine-grid C_eff = **91.6±1.0 MB = 0.724±0.008× 标称**(5 次重复,4 次聚簇);粗网格 harness 读 0.781——正是 session 9 网格量化批评的活例:粗网格分不清 0.72 和 0.80。跨架构表述最终收紧为:**0.72–0.80 的大而架构相关的分数**(fine-grid H100 0.795 / B300 0.724,A100/B200 粗网格 ±0.04),不是常数。**③ 方向性结论(P5 早期数据点)**:bw_l2 = 16.49 TB/s(+24% vs B200)而 bw_hbm 持平 6.72 → **L2:HBM = 2.46**(B200 1.96)——屏障的时延价值在这张卡上是增长的,不是被抹平。**④** t0 = 1.52 µs(torch 2.13):地板随 host 栈一路下降(2.30→1.80→1.52)横跨四代硅片,host-software 属性再次确认。**⑤ GDN 窗口跨架构兑现**:fla 在 B300 上跑 4.6–5.2 TB/s(新 triton + sm_103,绝对值比 H100 好得多)但依旧 residency-blind(warm 优势 ≤1.11);我们的 kernel warm 7.0–9.4 TB/s(峰值 9.38 @ 56MB),head-to-head **56MB 处 2.05×**,窗口在 96MB 塌陷——**speedup 窗口从 H100 的 ~40MB 移到 ~92MB,与实测 C_eff 完全一致**(B* ≈ 92 @ H=16):设计工具主张第一次在未见过的硅片上先预测后兑现。**⑥** w8a8 triton leg 被 triton 3.7 的 int8 tl.dot API 变更卡断(acc/out_dtype 断言语义变化,已诊断并以显式 out_dtype=tl.int32 修补)——kernel 移植问题,非架构物理。诚实声明:此卡锁频被拒(风冷功耗受限,peak fp16 1456 TF < B200 的 1547,持续频率已采样记录);P3(迁移)、P4(gate 扩张表需按 91.6 而非 150 重算)、P6(FP4 mult 预测 4.2)待跑。

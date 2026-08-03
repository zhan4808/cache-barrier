# WeChat draft for Dr. Xiao — 2026-07-02 (send after your own read-through)

## 中文版（建议发这版）

肖老师好，汇报一下这几天按您三项任务做的实验结果（都在 H100 上完成，关键结论均锁频复核过，slides 附后）：

1️⃣ 稠密模型（Qwen3.6-27B，逐层真实形状）：混合精度对主导运行时的 GEMM（您 profile 里的 86.2%）确实有收益，且确实受缓存约束——我们直接测出了"操作数感知"的 L2 容量门控：每种精度的带宽在**它自己的**权重大小越过 ~36MB 有效 L2 时跳崖（fp8 的悬崖在 bf16 的 2 倍处）。但更重要的另一半是：用多层驱逐做对照，这些缓存效应呈**阶跃式消失**——整模型服务的工作集总是超过 L2，所以微基准里的 2.4–3.0× 缓存红利在真实服务中压缩到约 1.7×（≈字节比）。两半合起来才是诚实的结论。

2️⃣ KV 缓存量化：**不受 L2 容量限制**（任何工作集都没出现 L2 带宽层级；小工作集受占用率限制，大的走 HBM）。fp8-KV 最多 +7%，小批量反而亏 30%（fp8 解码核自身带宽上限 ~1.6 对 bf16 ~3.0 TB/s）；按您 profile 全注意力只占 2.67%，端到端上限 ≤0.2%。结论：fp8-KV 在这类模型上是省显存的特性，不是提速的特性。

3️⃣ W8A8：锁频、对调优后的 bf16 基线，MoE 上**所有 T 均赢 1.8–1.9×、无反量化悬崖**（W8A16 在 M≈64–128/T≈300 处跳崖）。需要同时说明：逐张量 W8A8 相对误差 0.037（Marlin 0.0025），真实模型下游精度还没测，是当前的开放缺口。由逐 GEMM 测量推算（非整模型实测）：解码端到端 W8A16 约 1.50×，W8A8+融合激活量化约 1.6×。W4A4 在 H100 上无法原生运行（Hopper 无 FP4/INT4 张量核心），只能模拟。

想请您定的一件事：模型最锐利的可证伪预测——**原生 W4A4 突破反量化上限、在模拟 FP4 亏损的区间获胜**——只能在 Blackwell 上验证。我在 Prime Intellect 上有 B300 可用，测试框架已备好（需重新构建 Blackwell 版 vLLM）。您看是否值得现在跑？还是优先做真实模型精度评测或论文？

## English version (reference)

Dr. Xiao — results on your three items (all on H100, key claims re-verified clock-locked; slides attached):

1) Dense (Qwen3.6-27B, real per-layer shapes): mixed precision does help the dominant GEMMs (86.2% of runtime in your profile) and IS cache-constrained — we directly measured an operand-aware L2 capacity gate (each precision's bandwidth cliffs at its OWN weight size vs the ~36MB effective L2; fp8's cliff sits at 2× bf16's). Equally important: an eviction control shows these effects vanish as a step function once total working set exceeds L2 — which full-model serving always does — so the 2.4–3.0× microbenchmark cache bonus compresses to ~1.7× (≈byte ratio) under serving. Both halves are the finding.

2) KV cache quant: NOT L2-limited (no L2 bandwidth tier at any working set). fp8-KV: ≤+7% streamed, −30% at small batch (the fp8 decode kernel's own ~1.6 TB/s ceiling vs bf16's ~3.0); with full attention at 2.67% of runtime, end-to-end ceiling ≤0.2%. It's a memory-capacity feature, not a speed feature.

3) W8A8: vs a TUNED bf16, clock-locked — wins 1.8–1.9× at every T on MoE, no dequant cliff (W8A16 cliffs at M≈64–128 / T≈300). Caveat stated together: per-tensor W8A8 rel-err 0.037 vs Marlin 0.0025; real-model downstream accuracy unmeasured (open gap). Projected (not served) decode speedup: W8A16 ~1.50×, W8A8+fused act-quant ~1.6×. W4A4 cannot run natively on H100.

Decision requested: the model's sharpest untested prediction — native W4A4 breaking the dequant ceiling — needs Blackwell. I have B300 access on Prime Intellect and the harness is staged (needs a fresh Blackwell vLLM build). Worth running now, or should I prioritize accuracy eval / paper writing?

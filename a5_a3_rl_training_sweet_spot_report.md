# A5 相对 A3 的 RL 训练甜点配置分析报告

生成日期：2026-06-16

## 0. 结论摘要

当前 `EP8PP1TP2` 下，A5 已知有两个事实：

- 计算单元理论性能约为 A3 的 `2x`。
- 通信速度和 A3 基本持平。
- 当前 A5 profiling 中，未掩盖通信:计算约为 `1:2`，`free` 基本没有。
- 当前 A5 训练性能只有 A3 的 `0.7x`。

这四条放在一起，说明问题不只是“通信占比偏高”。如果只按“通信相同、A5 计算快 2 倍”建模，A5 本应比 A3 快：

```text
A5: comm + compute = 1 + 2 = 3
A3: comm + compute = 1 + 4 = 5
理想 A5/A3 性能比 = 5 / 3 = 1.67x
```

但实测是 `0.7x`。因此当前还存在一个很大的 A5 侧惩罚项，可能来自未掩盖通信、collective 等待、MoE straggler、路由负载不均衡、通信域映射或 profiler 口径外的同步等待。用上面的理想模型校准，当前 A5 只有理想相对表现的：

```text
0.7 / 1.67 ~= 0.42
```

因此，本报告后面的估算都分成两种口径：

- 保守口径：这个 `0.42` 的 A5 额外惩罚不变，只估算调参带来的相对变化。
- 改善口径：并行策略真的消除了 A5 侧等待/同步/尾部惩罚，则 A5/A3 有机会回到 `1.2x` 到 `1.7x`，极限接近 `2x` 计算优势。

核心判断：

1. TP all2all 稳定在每层反向约 `2.2ms`，主要因为它是固定 shape 的 dense collective。MoE 波动到 `5ms-26ms`，主要因为 MoE 是动态路由和 all2allv，wall time 由最慢 rank、最热 expert 和最慢 peer 决定。
2. 如果把 A3/A5 的计算都优化到一半，A5/A3 比值大概率不会变好，保守估计会从 `0.70x` 降到约 `0.63x`。原因是 A3 更计算受限，计算优化对 A3 的加速更大。
3. 如果调整并行策略主要减少未掩盖通信和等待，A5/A3 比值才会明显改善。若只把两边通信都减半，保守估计约 `0.76x`；若只让 A5 的有效未掩盖通信减半，约 `0.84x`；若还能消除 A5 侧额外惩罚，则可向 `1.3x+` 甚至 `1.67x` 靠近。
4. 调整序列长度会影响 TP/EP 的激活通信量。TP 激活通信和 EP dispatch/combine 通信通常都随 token 数线性增长；FSDP/参数类通信基本不随序列长度变化。若总 token 不变，只把 TND 从多短序列变成长序列，TP/EP 通信基本不变，但 full attention 计算按 `sum(seq_i^2)` 增加，这会提高计算/通信比，更偏向 A5。

## 1. 现有仓库证据和口径

仓库中已有报告 `qwen3_5_397b_layer_comm_compute_analysis.md` 给出过 Qwen3.5-397B-A17B 的单层通信与计算推导。这里引用其中与本问题直接相关的事实。

### 1.1 MoE 通信量公式

MoE EP dispatch/combine 搬运的是 token hidden state，不是 expert 中间维。因此它主要受以下因素影响：

```text
S_local: 本 rank token 数
top_k: 每 token 激活专家数
H: hidden size
EP: expert parallel size
```

单层 MoE dispatch + combine 的 remote-send 下界近似为：

```text
Bytes_EP_all2all
  ~= 2 * S_local * top_k * H * bytes_per_elem * (EP - 1) / EP
```

重要点：

- `moe_intermediate_size=1024` 会影响 expert 参数量和 expert GEMM。
- 但 EP dispatch/combine 搬运的是 hidden state，主要用 `hidden_size=4096`，不会因为 `moe_intermediate_size=1024` 直接小 4 倍。

### 1.2 event_timer 汇总中的稳定与波动

仓库已有 CSV 汇总支持“固定 dense 事件更稳定、MoE 事件尾部更重”的方向：

- `analysis_outputs/event_timer_report/mha_op_time_summary.csv` 中 31 个 MHA event 的均值范围为 `76.18ms-78.61ms`，均值约 `77.46ms`，相对波动约 `3.1%`。
- `analysis_outputs/event_timer_report/moe_decoder_layer_time_summary.csv` 中 60 层 MoE decoder layer 均值范围为 `140.74ms-341.86ms`，最大/最小约 `2.43x`。
- `analysis_outputs/event_timer_mha_tnd_report/selected_step_aux_heatmap_summary.csv` 中非 warmup 的 MoE op，p50 通常在 `49.8ms-50.9ms`，但 p99 可到 `237ms-394ms`，max 可到 `600ms-1223ms`。
- 同一份汇总中 MoE GMM 的 p50/p95 非常稳定，约 `13.7ms/14.8ms`，但 max 偶发到 `539ms`，说明常态计算核稳定，尾部更像等待、同步、异常 rank 或排队问题。

这些数据不是你当前 `TP all2all=2.2ms` 和 `MoE=5ms-26ms` 的逐事件原始文件，但能支撑同一个解释框架：固定 shape collective 稳定，动态 MoE 路由和 straggler 造成长尾。

## 2. 为什么 TP all2all 稳，而 MoE 波动大

### 2.1 TP all2all 每层反向约 2.2ms 的原因

TP 通信通常是 dense tensor collective。只要 micro-batch、sequence length、hidden size、TP size 不变，每层反向中参与 all2all 或 reduce/scatter/gather 的 tensor shape 基本固定：

```text
payload_TP ~= S * H * bytes * f(TP, op)
```

它的几个特点会让耗时稳定：

1. 每个 rank 的输入 shape 相同。
2. 每个 peer 的 chunk 大小近似相同。
3. collective 算法路径固定，通常不是 all2allv。
4. 不依赖 router 的 token-expert 分配。
5. 如果上游计算到达时间稳定，通信事件本身就会稳定。

所以你看到反向每层基本 `2.2ms` 是合理的。它更像固定税，每层都交一次，但方差不大。

### 2.2 MoE 5ms 到 26ms 波动的原因

MoE 的通信不是固定 dense all2all，而是 token routing 后的 dispatch/combine，常见实现是 all2allv 或 all2all 加 pack/unpack。它的 wall time 不由平均 token 数决定，而由最慢的 rank、最热的 expert、最慢的 peer 和最晚到达 collective 的 rank 决定。

主要波动来源有五类。

第一，router 负载不均衡。

每个 token 选 `top_k` 个 expert。即使全局 token 数固定，每层、每 step、每个 rank 路由到各 expert 的 token 数也会变化：

```text
load_r = rank r 实际接收到的 token-expert 数
rho = max(load_r) / mean(load_r)
```

MoE wall time 近似会被 `rho` 放大：

```text
T_moe_wall ~= rho * T_moe_avg + sync_wait
```

如果某个 rank 拿到热 expert 或更多 token，其他 rank 需要等它完成 dispatch、GMM 或 combine。

第二，all2allv 的 peer 消息大小不均。

TP all2all 通常每个 peer chunk 接近固定。MoE all2allv 的每个 peer 发送/接收大小由路由决定：

```text
send_bytes[r -> p] ~= tokens_routed_to_peer_p * H * bytes
```

如果某个 peer 收到大量 token，通信最长路径变长。HCCL/NCCL 对 all2allv 的调度也更容易被 small chunk、large chunk 混合和 peer skew 影响。

第三，MoE event 往往包含 pack/unpack、permute/unpermute、GMM 和同步等待。

profiling 里名为 MoE 的片段不一定只包含纯网络 memcpy。它可能包含：

- token sort、prefix sum、index build。
- hidden state pack/unpack。
- dispatch all2allv。
- expert grouped GEMM。
- combine all2allv。
- 等最慢 rank 到达下一个 collective。

因此 `5ms-26ms` 的波动，很可能不是“网络带宽忽快忽慢”，而是“动态路由 + 最慢 rank + 等待”共同进入了 wall time。

第四，反向比前向更容易放大尾部。

MoE 反向通常要走与 forward dispatch/combine 对偶的通信路径，还要处理 expert GEMM backward、梯度聚合和 activation 重排。前向的一点负载不均衡，到了反向可能叠加到更多依赖边上。

第五，当前 `free` 基本没有，说明没有 slack 吸收抖动。

如果 timeline 中有足够 free 或 overlap，MoE 某层多出来几毫秒可能被其他计算掩盖。现在 free 基本没有，意味着任何一个 rank 的 MoE 长尾都会直接推迟 step 关键路径。

### 2.3 建议优先检查的 profiler 指标

要确认 MoE 波动根因，建议下一轮 profiling 增加或提取以下指标：

- 每层每 rank 的 routed token 数。
- 每层每 expert 的 token 数。
- 每个 EP rank 的 received token-expert 数，统计 `max/avg`。
- EP all2allv 每个 peer 的 send/recv bytes。
- 每个 rank 进入 MoE dispatch collective 的时间戳差。
- MoE GMM 单独耗时与 MoE 总 wall time 的差。
- MoE combine 后到下一个 collective 前的等待时间。

如果 `MoE wall` 高但 `MoE GMM` 和 `memcpy` 不高，优先怀疑同步等待、路由 skew 或 collective 到达时间差。

## 3. 性能估算模型

为了把估算说清楚，先定义一个归一化模型。

以 A5 当前 profiling 为基准：

```text
A5 未掩盖通信 M = 1
A5 计算 C5 = 2
A5 当前 step 关键路径 ~= M + C5 = 3
```

因为 A5 计算单元约为 A3 的 `2x`，同样工作量下 A3 计算时间近似为：

```text
C3 = 2 * C5 = 4
```

如果通信耗时相同，则理想 A3 时间：

```text
A3 step ~= M + C3 = 1 + 4 = 5
```

理想 A5/A3 性能比：

```text
R_ideal = T_A3 / T_A5 = 5 / 3 = 1.67
```

但当前实测：

```text
R_observed = 0.70
```

因此定义一个 A5 当前额外惩罚校准系数：

```text
gamma = R_observed / R_ideal = 0.70 / 1.67 ~= 0.42
```

后面保守估计默认 `gamma` 不变，只看改动本身的相对影响。

## 4. 问题 2：计算都减半后的性能比

假设 A3 和 A5 的计算都优化到一半，通信不变。

```text
A5 new = M + C5/2 = 1 + 1 = 2
A3 new = M + C3/2 = 1 + 2 = 3
```

理想模型下：

```text
R_ideal_new = 3 / 2 = 1.50
```

带入当前 `gamma ~= 0.42`：

```text
R_observed_new ~= 1.50 * 0.42 = 0.63
```

也可以用当前 `0.70x` 做相对 speedup：

```text
A5 speedup = 3 / 2 = 1.50x
A3 speedup = 5 / 3 = 1.67x
new_ratio = 0.70 * 1.50 / 1.67 ~= 0.63
```

结论：如果 A3 和 A5 的计算都减半，A5/A3 比值大概率从 `0.70x` 降到约 `0.63x`，因为当前 A5 通信占比更高，计算优化反而更利好 A3。

补充：如果只优化 A5 的计算，不优化 A3，那么：

```text
A5 speedup = 3 / 2 = 1.50x
new_ratio ~= 0.70 * 1.50 = 1.05x
```

但这不是“双方计算都优化”的场景。

## 5. 问题 2：并行策略优化通信后的性能比

这里分三层估算。

### 5.1 只把双方未掩盖通信都减半

```text
A5 new = 2 + 0.5 = 2.5
A3 new = 4 + 0.5 = 4.5
```

相对当前：

```text
A5 speedup = 3 / 2.5 = 1.20x
A3 speedup = 5 / 4.5 = 1.11x
new_ratio = 0.70 * 1.20 / 1.11 ~= 0.76
```

保守估计：`0.76x`。

### 5.2 只降低 A5 的有效未掩盖通信

如果并行策略主要修的是 A5 的通信域映射、MoE straggler、collective 等待，而 A3 基本不变，则收益更大：

| A5 未掩盖通信缩放 | A5 时间 | A5 speedup | A5/A3 新比值 |
|---:|---:|---:|---:|
| 降低 25% | `2 + 0.75 = 2.75` | `1.09x` | `0.76x` |
| 降低 50% | `2 + 0.50 = 2.50` | `1.20x` | `0.84x` |
| 降低 75% | `2 + 0.25 = 2.25` | `1.33x` | `0.93x` |
| 全部消掉 | `2 + 0 = 2.00` | `1.50x` | `1.05x` |

这说明如果只是减少当前 A5 已暴露的 `M=1` 通信，最多把 `0.70x` 拉到约 `1.05x`。要超过这个，还必须解决 `gamma=0.42` 背后的额外惩罚。

### 5.3 并行策略同时消除 A5 额外惩罚

如果当前 `0.70x` 的主要原因就是并行策略导致 A5 出现大量等待、同步、通信域映射不佳或 MoE 长尾，那么优化并行策略可能不只是减少 `M`，还会把 `gamma` 从 `0.42` 往 `1.0` 拉回。

此时上界由物理模型决定：

```text
当前通信:计算 = 1:2
理想 A5/A3 = (A3 comm + A3 compute) / (A5 comm + A5 compute)
            = (1 + 4) / (1 + 2)
            = 1.67x
```

如果通信进一步下降，极限接近纯计算优势：

```text
通信趋近 0 时，A5/A3 -> 4 / 2 = 2.0x
```

因此真正值得追的目标不是“把 compute 再优化一半”，而是：

- 把 TP/EP 未掩盖通信从 `1:2` 推到至少 `1:4` 或更低。
- 降低 MoE p95/p99 长尾，让 `gamma` 从 `0.42` 恢复。
- 尽量让 A5 的强计算单元处在大 GEMM、长计算段、少同步等待的区间。

### 5.4 并行策略建议方向

当前 `EP8PP1TP2` 下，建议优先试这些方向。

第一，优先试 `TP1`，如果显存和单 rank 参数/激活允许。

TP2 的好处是拆 dense compute 和内存，坏处是每层增加 TP 通信。既然 TP all2all 每层反向稳定约 `2.2ms`，它就是固定关键路径税。A5 相对 A3 的优势在计算，不在通信，所以减少 TP 通信通常更利于 A5。

可试组合：

```text
EP8PP1TP1
EP4PP1TP1
EP4PP1TP2
```

第二，谨慎增加 EP，优先考虑降低 EP 或保持 EP8。

增大 EP 会让每 rank expert 参数和 expert compute 下降，但 MoE dispatch/combine 的 remote fraction 会略增，通信 peer 更多，all2allv 调度更复杂。对 A5/A3 比值来说，减少计算、增加通信复杂度通常不是好方向。

从通信 remote fraction 看：

```text
EP4:  (EP-1)/EP = 0.75
EP8:  0.875
EP16: 0.9375
```

EP 越大，MoE dispatch/combine 的远端比例越高。虽然差别不是线性翻倍，但 all2allv 的 peer 数、同步等待和负载不均衡风险会增加。

第三，PP1 本身没有 pipeline bubble，除非显存不够，不建议为了切层而引入 PP。

PP 能降低单 stage 层数和内存压力，但会引入 pipeline bubble、stage 间 activation 通信和调度复杂度。当前问题是 A5 通信/等待太高，PP 不应作为第一优先级。

第四，甜点配置的方向应是“少 TP、适中或更小 EP、更大本地计算块”。

定性目标：

```text
让 A5 的关键路径变成：长计算段 + 少量可预测通信
避免：短计算段 + 每层固定 TP 通信 + 动态 MoE all2allv 长尾
```

## 6. 问题 3：调整激活值/序列长度是否影响 TP 和 EP 通信量

答案是：会影响激活通信，但影响方式要分类型。

### 6.1 TP 激活通信

TP 里的 activation all2all、allgather、reduce-scatter 等，通常随 token 数线性变化：

```text
Bytes_TP_activation ~= S_local * H * bytes * f(TP, op_count)
```

所以如果你把序列长度或每 step token 数放大，TP 激活通信量通常线性增加。TP all2all 的 `2.2ms` 稳定，是因为当前 shape 稳定，不代表它和序列长度无关。

### 6.2 EP MoE dispatch/combine 通信

EP all2all 同样随 routed token 数线性变化：

```text
Bytes_EP_all2all
  ~= 2 * S_local * top_k * H * bytes * (EP - 1) / EP
```

影响项：

- `S_local` 增大，EP 通信量线性增大。
- `top_k` 增大，EP 通信量线性增大。
- `H` 增大，EP 通信量线性增大。
- `EP` 增大，remote fraction 从 `(EP-1)/EP` 增大并趋近 1。

不直接影响项：

- `moe_intermediate_size` 不直接决定 dispatch/combine hidden state 大小。
- expert 内部中间维影响 GMM compute 和参数，不直接让 EP all2all hidden state 变小。

### 6.3 参数通信

FSDP 参数 allgather、optimizer state 相关通信、部分参数梯度通信主要由参数量和并行切分决定，基本不随序列长度变化。

因此调整序列长度时，要区分：

| 通信类型 | 是否随序列/token 数变化 | 备注 |
|---|---:|---|
| TP activation collective | 是，通常线性 | shape 固定时耗时稳定，S 变了会变 |
| EP dispatch/combine | 是，通常线性 | 还受 router 负载不均衡影响 |
| FSDP 参数 allgather | 否 | 主要由参数量和 sharding 决定 |
| optimizer/参数梯度类通信 | 基本否 | 主要由参数量决定 |

## 7. 调整序列长度对 A5/A3 性能比的影响

序列长度有两种完全不同的调法。

### 7.1 总 token 数增加，序列结构近似不变

如果只是把每 step token 数整体放大，dense/MoE compute 和 TP/EP activation 通信大多一起线性增长。此时计算/通信比不一定改善：

```text
compute scale = q
communication scale = q
```

在这个场景下，A5/A3 比值大概率接近不变，仍围绕当前 `0.70x`，除非更大的 GEMM 显著提升 A5 MFU 或减少调度开销。

风险是 MoE all2all 的绝对字节数也变大。如果 router 负载不均衡没有改善，`5ms-26ms` 的尾部可能变成更大的尾部。

### 7.2 总 token 数不变，但序列从多短变少长

如果总 token 数 `S_total` 不变，只是样本 packing/TND 让序列更长，那么 TP/EP activation 通信主要看总 token 数，基本不变：

```text
Bytes_TP ~= S_total * ...
Bytes_EP ~= S_total * top_k * ...
```

但 full attention 计算看：

```text
F_attn ~= sum_i(seq_i^2)
```

多短序列变少长序列时，`sum_i(seq_i^2)` 会增大，计算增加，而通信基本不变。这会提高计算/通信比，理论上更偏向 A5。

这个方向是“让 A5 吃计算”的有效方法，但代价是每 token 计算量上升。它适合你想提高 A5 相对 A3 的硬件利用率，不一定适合想最大化 tokens/s。

### 7.3 保守估算公式

设：

```text
p = compute scale
q = communication scale
当前 A5: C5=2, M=1
当前 A3: C3=4, M=1
当前观测比 R0=0.70
```

若 `gamma` 不变，新的 A5/A3 比值可用：

```text
R_new ~= R0 * [3 / (2p + q)] / [5 / (4p + q)]
      = R0 * 3 * (4p + q) / [5 * (2p + q)]
```

几个例子：

| 场景 | compute scale p | comm scale q | A5/A3 保守估计 |
|---|---:|---:|---:|
| 计算和通信都线性增加 | `q` | `q` | 约 `0.70x` 不变 |
| 计算 +25%，通信不变 | `1.25` | `1.00` | `0.72x` |
| 计算 +50%，通信不变 | `1.50` | `1.00` | `0.735x` |
| 计算 +100%，通信不变 | `2.00` | `1.00` | `0.756x` |
| 通信减半，计算不变 | `1.00` | `0.50` | `0.756x` |
| 通信消除，计算不变 | `1.00` | `0.00` | `0.84x` |

这张表看起来收益不大，是因为它保守假设当前 A5 的 `gamma=0.42` 惩罚不变。真正的大收益来自让 `gamma` 恢复，也就是减少 MoE 长尾、通信域等待和 A5 上的同步惩罚。

## 8. 推荐实验矩阵

建议用小矩阵先找方向，不要一次改太多。

### 8.1 并行策略矩阵

优先级从高到低：

| 实验 | 目的 | 预期 |
|---|---|---|
| `EP8PP1TP1` | 去掉 TP2 固定通信税 | 若显存能扛，A5/A3 应改善 |
| `EP4PP1TP2` | 降低 EP all2all peer 数和 remote fraction | MoE 波动可能下降，计算占比上升 |
| `EP4PP1TP1` | 同时减少 TP/EP 通信 | 最符合 A5 甜点方向，但显存压力最大 |
| `EP16PP1TP1` | 验证更大 EP 是否只是在救显存 | 若 MoE 长尾变差，则不适合 A5 性能比 |

每个实验至少记录：

```text
A5/A3 tokens/s
A5 timeline 中 unmasked comm:compute
TP all2all p50/p95/p99
EP/MoE all2all p50/p95/p99
MoE GMM p50/p95/p99
每层每 rank routed token max/avg
free/overlap 比例
```

### 8.2 序列长度矩阵

建议分两组测试。

第一组，总 token 数变，观察线性扩展：

```text
0.5x tokens
1.0x tokens
1.5x tokens
2.0x tokens
```

目的：确认 TP/EP 通信是否按 token 数线性增长，确认 A5 MFU 是否随大 GEMM 提升。

第二组，总 token 数固定，改变 TND/packing：

```text
更多短序列
当前混合
更少长序列
```

目的：在 EP/TP 通信近似不变时，单独观察 full attention 计算增加是否改善 A5/A3 比。

## 9. 最终建议

当前最佳方向不是先做“计算都减半”的 kernel 优化。因为双方计算都优化时，A3 受益更大，A5/A3 比值可能从 `0.70x` 降到约 `0.63x`。

更应该先做并行策略和 MoE 长尾治理：

1. 尝试减少 TP，优先验证 `TP1` 是否可跑。
2. 尝试降低 EP 或保持 EP8，避免更大的 all2allv group 把 A5 拖进通信等待。
3. 对 MoE profiling 加 routed token 和 all2allv per-peer bytes，确认 `5ms-26ms` 的波动是负载不均、到达时间差还是 collective 调度。
4. 用固定总 token 的 TND/packing 实验提高 full attention 计算占比，观察 A5/A3 是否改善。

甜点配置的目标形态：

```text
更少 TP 通信
适中或更小 EP 通信域
更大的本地 GEMM/attention 计算块
MoE routed token max/avg 接近 1
A5 unmasked communication:compute <= 1:4
free/等待不再被 MoE p99 长尾吃掉
```

如果这个方向成立，A5/A3 不应停留在 `0.7x`。保守修通信可先看 `0.8x-1.0x`；如果能把 A5 额外惩罚从 `gamma=0.42` 拉回，才有机会进入 `1.3x-1.7x` 区间。

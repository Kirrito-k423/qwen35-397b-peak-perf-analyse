# A5 相对 A3 单卡的 RL 训练甜点配置分析报告

生成日期：2026-06-16

## 0. 修正后的结论摘要

本版修正了一个关键口径：A5 要和 A3 单卡比较，而 A3 单卡包含两颗 die。也就是说，若先估出 A3 单 die 的耗时，需要再除以 2 才能得到 A3 单卡吞吐等价耗时。

当前已知事实：

- 当前并行策略：`EP8PP1TP2`。
- A5 profiling 中未掩盖通信:计算约为 `1:2`，`free` 基本没有。
- A5 训练性能是 A3 单卡的 `0.7x`。
- “A5 计算单元约为 A3 单 die 的 2x”是预估，不应直接当成实测。

用 A5 profiling 归一化：

```text
A5 未掩盖通信 M = 1
A5 计算 C5 = 2
A5 当前耗时 T5 = 1 + 2 = 3
```

设 A3 单 die 计算耗时为 `C3_die`，通信速度与 A5 持平，则 A3 单 die 耗时为：

```text
T3_die = 1 + C3_die
```

A3 单卡有两颗 die，因此单卡吞吐等价耗时为：

```text
T3_card = T3_die / 2 = (1 + C3_die) / 2
```

由实测性能比反推：

```text
A5/A3_card 性能比 = T3_card / T5 = 0.7
(1 + C3_die) / 2 / 3 = 0.7
1 + C3_die = 4.2
C3_die = 3.2
```

所以当前配置下，反推得到的实际计算耗时比是：

```text
C3_die / C5 = 3.2 / 2 = 1.6x
```

这比“理论 2x”更保守。换句话说，在当前配置和 profiler 口径下，A5 相对 A3 单 die 的有效计算优势约为 `1.6x`，不是 `2x`。

核心修正结论：

1. 若 A3/A5 的计算都减半，A5/A3 单卡性能比从 `0.70x` 变为约 `0.65x`，方向仍然是变差。
2. 若双方未掩盖通信都减半，A5/A3 单卡性能比约为 `0.74x`；若双方通信都消掉，在当前反推的 `1.6x` 实际计算比下，上限约为 `0.80x`。
3. 若主要只降低 A5 的有效未掩盖通信，A5 通信减半时约为 `0.84x`，A5 通信完全消掉时约为 `1.05x`。
4. 如果后续优化让 A5 实际计算优势恢复到预估的 `2x`，并且通信不变，A5/A3 单卡也只是约 `0.83x`；双方通信都消掉时上限约 `1.0x`。因为对手是 A3 单卡两 die，A5 相对 A3 单 die 的 `2x` 计算优势会被 A3 单卡的两 die 吞吐抵消。
5. 因此当前甜点配置的目标不是追求“接近 2x A3 单卡”，而是先把 `0.70x` 拉到 `0.8x-1.0x`；要稳定超过 A3 单卡，需要 A5 侧通信/等待显著低于 A3，或 A5 实际计算优势超过 `2x` 单 die 预估。

## 1. 为什么 TP all2all 稳，而 MoE 波动大

### 1.1 TP all2all 稳定的原因

你看到 TP all2all 基本每层反向都是 `2.2ms`，这是合理的。TP 通信通常是固定 shape 的 dense collective。只要 micro-batch、序列长度、hidden size、TP size 不变，每层反向中参与 all2all 或 reduce-scatter/all-gather 的 tensor shape 基本固定：

```text
payload_TP ~= tokens * hidden_size * bytes * f(TP, op)
```

它稳定的原因是：

1. 每个 rank 输入 shape 相同。
2. peer 间 chunk 大小近似固定。
3. collective 算法路径固定。
4. 不依赖 router 的 token-expert 分配。
5. 如果上游计算到达时间稳定，通信事件本身方差就小。

所以 TP all2all 更像每层固定税。它会影响性能，但不太会解释 `5ms-26ms` 这种大幅波动。

### 1.2 MoE 波动大的原因

MoE 的通信和计算不是固定 dense pattern，而是由动态路由决定。常见路径包括：

- token sort / prefix sum / dispatch metadata。
- hidden state pack/unpack。
- EP dispatch all2allv。
- expert grouped GEMM。
- EP combine all2allv。
- 等最慢 rank 进入下一个 collective。

MoE wall time 不由平均 token 数决定，而由最慢 rank、最热 expert、最慢 peer 和最晚到达 collective 的 rank 决定。

定义：

```text
load_r = rank r 实际收到的 token-expert 数
load_avg = mean(load_r)
rho = max(load_r) / load_avg
```

MoE 的关键路径可以粗略写成：

```text
T_moe_wall ~= rho * T_moe_compute_avg + T_all2allv_skew + T_sync_wait
```

这解释了为什么 TP all2all 可以稳定在 `2.2ms`，而 MoE 会在 `5ms-26ms` 波动。

### 1.3 仓库中可参考的佐证

仓库已有 event_timer 汇总也支持“固定 dense 事件稳定、MoE 长尾重”的判断：

- `analysis_outputs/event_timer_report/mha_op_time_summary.csv` 中 31 个 MHA event 均值范围为 `76.18ms-78.61ms`，相对波动约 `3.1%`。
- `analysis_outputs/event_timer_report/moe_decoder_layer_time_summary.csv` 中 60 层 MoE decoder layer 均值范围为 `140.74ms-341.86ms`，最大/最小约 `2.43x`。
- `analysis_outputs/event_timer_mha_tnd_report/selected_step_aux_heatmap_summary.csv` 中非 warmup 的 MoE op，p50 通常在 `49.8ms-50.9ms`，但 p99 可到 `237ms-394ms`，max 可到 `600ms-1223ms`。
- 同一份汇总中 MoE GMM 的 p50/p95 很稳定，约 `13.7ms/14.8ms`，但 max 偶发到 `539ms`，更像等待、同步、异常 rank 或排队问题，而不是常态算子性能问题。

这些不是你当前 `EP8PP1TP2` 的原始逐事件数据，但与当前现象的解释方向一致。

### 1.4 建议下一轮 profiler 增加的指标

为了确认 MoE `5ms-26ms` 的来源，建议记录：

- 每层每 rank routed token 数。
- 每层每 expert token 数。
- 每个 EP rank received token-expert 的 `max/avg`。
- EP all2allv 每个 peer 的 send/recv bytes。
- 每个 rank 进入 MoE dispatch collective 的时间戳差。
- MoE GMM 单独耗时与 MoE 总 wall time 的差。
- MoE combine 后到下一个 collective 前的等待时间。

如果 `MoE wall` 高但 `MoE GMM` 和实际 memcpy 不高，优先怀疑同步等待、路由 skew 或 collective 到达时间差。

## 2. 修正后的性能模型

本报告后续统一使用以下归一化变量：

```text
A5:
  M5 = 1
  C5 = 2
  T5 = M5 + C5 = 3

A3 单 die:
  M3_die = 1
  C3_die = 3.2   # 由 0.7x 实测反推
  T3_die = 1 + 3.2 = 4.2

A3 单卡两 die:
  T3_card = T3_die / 2 = 2.1

当前性能比:
  A5/A3_card = T3_card / T5 = 2.1 / 3 = 0.7
```

注意这个 `3.2` 是当前配置下的反推值。它不等于理论硬件峰值，只表示在当前训练、并行、profiling 口径下，A3 单 die 的计算耗时相对 A5 计算耗时约为 `1.6x`。

如果坚持用预估的 `2x` 计算优势，则应有：

```text
C3_die_expected = 2 * C5 = 4
T3_card_expected = (1 + 4) / 2 = 2.5
A5/A3_card_expected = 2.5 / 3 = 0.83
```

但实测是 `0.70`，所以当前实际有效计算优势低于预估，或者 A5 的未掩盖等待/同步被统计到了计算/通信关键路径里。

## 3. 计算都减半时，A5/A3 单卡性能比是多少

假设 A3 和 A5 的计算都优化到一半，通信不变。

```text
A5:
  T5_new = 1 + 2/2 = 2

A3 单 die:
  T3_die_new = 1 + 3.2/2 = 2.6

A3 单卡:
  T3_card_new = 2.6 / 2 = 1.3
```

新的性能比：

```text
A5/A3_card_new = 1.3 / 2 = 0.65
```

也可以从 speedup 看：

```text
A5 speedup = 3 / 2 = 1.50x
A3_card speedup = 2.1 / 1.3 = 1.615x
new_ratio = 0.70 * 1.50 / 1.615 ~= 0.65
```

结论：双方计算都减半时，A5/A3 单卡性能比会从 `0.70x` 降到约 `0.65x`。原因是 A3 单卡两 die 后仍然很吃计算优化，计算优化对 A3 单卡也很有利。

补充：如果只优化 A5 计算，不优化 A3，则：

```text
A5 new = 1 + 1 = 2
A3_card = 2.1
A5/A3_card = 2.1 / 2 = 1.05
```

但这不是“双边计算都优化”的场景。

## 4. 并行策略优化通信时，A5/A3 单卡性能比是多少

### 4.1 双方未掩盖通信等比例下降

设双方通信缩放为 `q`，计算不变。

```text
A5_new = q + 2
A3_card_new = (q + 3.2) / 2
ratio_new = (q + 3.2) / [2 * (q + 2)]
```

| 双方通信缩放 q | A5 耗时 | A3 单卡等价耗时 | A5/A3 单卡 |
|---:|---:|---:|---:|
| `1.00` | `3.00` | `2.10` | `0.70x` |
| `0.75` | `2.75` | `1.975` | `0.718x` |
| `0.50` | `2.50` | `1.85` | `0.74x` |
| `0.25` | `2.25` | `1.725` | `0.767x` |
| `0.00` | `2.00` | `1.60` | `0.80x` |

结论：如果通信优化对 A3/A5 等比例生效，在当前反推的 `1.6x` 实际计算比下，即使双方通信都消掉，A5/A3 单卡也只有约 `0.80x`。

### 4.2 只降低 A5 的有效未掩盖通信

如果并行策略主要解决 A5 的通信域映射、MoE straggler、collective 等待，而 A3 单卡结果基本不变，则 A3 单卡等价耗时仍为 `2.1`：

```text
A5_new = q + 2
A3_card = 2.1
ratio_new = 2.1 / (q + 2)
```

| A5 通信缩放 q | A5 耗时 | A5 speedup | A5/A3 单卡 |
|---:|---:|---:|---:|
| `1.00` | `3.00` | `1.00x` | `0.70x` |
| `0.75` | `2.75` | `1.09x` | `0.764x` |
| `0.50` | `2.50` | `1.20x` | `0.84x` |
| `0.25` | `2.25` | `1.33x` | `0.933x` |
| `0.00` | `2.00` | `1.50x` | `1.05x` |

结论：如果能只减少 A5 的未掩盖通信/等待，通信减半可到约 `0.84x`，通信全消可到约 `1.05x`。

### 4.3 如果 A5 实际计算优势恢复到理论 2x

如果后续并行策略或 kernel 调整让 A5 有效计算优势恢复到预估的 `2x`，则使用：

```text
C3_die_expected = 4
```

双方通信仍相同且不变时：

```text
A5 = 1 + 2 = 3
A3_card = (1 + 4) / 2 = 2.5
A5/A3_card = 2.5 / 3 = 0.83x
```

双方通信都等比例下降时：

```text
ratio_expected = (q + 4) / [2 * (q + 2)]
```

| 双方通信缩放 q | A5/A3 单卡，按理论 2x 计算 |
|---:|---:|
| `1.00` | `0.833x` |
| `0.50` | `0.90x` |
| `0.00` | `1.00x` |

如果只降低 A5 通信，而 A3 仍按理论 2x 计算保持 `2.5` 等价耗时：

| A5 通信缩放 q | A5/A3 单卡，按理论 2x 计算 |
|---:|---:|
| `1.00` | `0.833x` |
| `0.50` | `1.00x` |
| `0.00` | `1.25x` |

这给出了更合理的上界：A5 要明显超过 A3 单卡，必须让 A5 的有效通信/等待显著低于 A3，或者让 A5 实际计算优势超过“单 die 2x”这个预估。

## 5. 并行策略建议

当前结论比上一版更保守：因为 A3 对手是两 die 单卡，A5 的 `2x` 单 die 计算预估并不会自动变成 `2x` 单卡性能。

仍然建议优先减少未掩盖通信和 MoE 长尾，而不是优先做“双边计算都减半”的优化。

### 5.1 TP

`TP2` 的固定通信税每层反向约 `2.2ms`。它稳定，但仍然是每层关键路径固定成本。A5 相对 A3 单卡的优势已经不大，固定通信税会继续压低 A5/A3 比值。

建议优先验证：

```text
EP8PP1TP1
```

如果显存能承受，`TP1` 有机会减少固定 TP 通信，使 A5 更接近计算受限。

### 5.2 EP

EP 增大会减少每 rank expert 参数和 expert compute，但会让 MoE dispatch/combine 的远端比例和 all2allv peer 数上升：

```text
EP4:  (EP - 1) / EP = 0.75
EP8:  0.875
EP16: 0.9375
```

对 A5/A3 单卡比值来说，减少计算、增加通信复杂度通常不是好方向。建议测试：

```text
EP4PP1TP2
EP4PP1TP1
```

判断标准不是平均 MoE 时间，而是 MoE p95/p99、每 rank routed token `max/avg`、all2allv per-peer bytes skew。

### 5.3 PP

`PP1` 没有 pipeline bubble。除非显存压力逼迫，不建议优先引入 PP，因为 PP 会增加 stage 间 activation 通信、调度复杂度和 pipeline bubble。

### 5.4 推荐实验矩阵

| 实验 | 目的 | 预期观察 |
|---|---|---|
| `EP8PP1TP1` | 去掉 TP2 固定通信税 | TP 通信下降，显存压力上升 |
| `EP4PP1TP2` | 降低 EP all2allv peer 数 | MoE p95/p99 可能下降 |
| `EP4PP1TP1` | 同时降低 TP/EP 通信 | 最符合 A5 甜点方向，但显存压力最大 |
| `EP16PP1TP1` | 验证更大 EP 是否只是救显存 | 若 MoE 长尾变大，不适合追 A5/A3 比 |

每个实验记录：

```text
A5/A3_card tokens/s
A5 unmasked comm:compute
TP all2all p50/p95/p99
EP/MoE all2all p50/p95/p99
MoE GMM p50/p95/p99
每层每 rank routed token max/avg
free/overlap 比例
```

## 6. 调整激活值/序列长度是否影响 TP 和 EP 通信量

会影响激活通信，但要分通信类型。

| 通信类型 | 是否随 token/序列长度变化 | 说明 |
|---|---:|---|
| TP activation collective | 是，通常线性 | `tokens * hidden_size` 量级 |
| EP dispatch/combine | 是，通常线性 | `tokens * top_k * hidden_size` 量级 |
| FSDP 参数 allgather | 基本否 | 主要由参数量和 sharding 决定 |
| optimizer/参数梯度类通信 | 基本否 | 主要由参数量决定 |

TP 激活通信近似：

```text
Bytes_TP_activation ~= tokens * hidden_size * bytes * f(TP, op)
```

EP MoE dispatch/combine 近似：

```text
Bytes_EP_all2all
  ~= 2 * tokens * top_k * hidden_size * bytes * (EP - 1) / EP
```

注意：`moe_intermediate_size` 会影响 expert 参数和 GMM 计算，但 EP dispatch/combine 搬运的是 hidden state，所以不直接让 EP all2all hidden state 变小。

## 7. 调整序列长度对 A5/A3 单卡比值的影响

设计算缩放为 `p`，通信缩放为 `q`。若 A3/A5 的计算和通信按相同规则缩放，则：

```text
A5_new = q + 2p
A3_card_new = (q + 3.2p) / 2
ratio_new = (q + 3.2p) / [2 * (q + 2p)]
```

### 7.1 总 token 数增加，序列结构近似不变

如果只是每 step token 数整体放大，dense/MoE compute 和 TP/EP activation 通信大多一起线性增长：

```text
p = q
ratio_new = (q + 3.2q) / [2 * (q + 2q)] = 4.2 / 6 = 0.70
```

因此，单纯增加总 token 数，若没有提升 A5 MFU 或减少等待，A5/A3 单卡比值大概率基本不变。

风险是 MoE all2all 的绝对字节数变大。如果 router 负载不均衡没有改善，MoE `5ms-26ms` 长尾可能进一步变大。

### 7.2 总 token 数不变，但序列从多短变少长

如果总 token 数不变，只改变 packing/TND，让序列更长：

```text
TP/EP activation 通信 ~= 看总 token 数，基本不变
full attention 计算 ~= sum_i(seq_i^2)，会增加
```

这等价于 `p > 1, q ~= 1`。在当前反推的 `1.6x` 计算比下：

| 计算缩放 p | 通信缩放 q | A5/A3 单卡 |
|---:|---:|---:|
| `1.00` | `1.00` | `0.700x` |
| `1.25` | `1.00` | `0.714x` |
| `1.50` | `1.00` | `0.725x` |
| `2.00` | `1.00` | `0.740x` |
| `infinity` | `1.00` | `0.800x` |

因此，增加计算占比会改善 A5/A3 单卡比值，但在当前反推的有效计算优势 `1.6x` 下，上限仍约 `0.8x`。若 A5 实际计算优势能恢复到理论 `2x`，纯计算上限才约 `1.0x`。

### 7.3 总 token 数不变且 A5 MFU 随长序列提升

更长序列可能让 A5 的 GEMM/attention 更饱满，使实际 `C3_die/C5` 从 `1.6x` 往 `2x` 靠近。这种情况下序列长度调整有双重收益：

1. 通信量基本不变，计算占比提高。
2. A5 计算效率可能提升，使实际计算比从 `1.6x` 接近 `2x`。

这才是“调激活值/序列长度”对 A5 有价值的场景。建议实验时同时记录 A5 MFU、MoE p95/p99 和 TP/EP all2all 字节量，避免把 token 数增加导致的通信增长误判成计算占比优化。

## 8. 最终建议

修正 A3 单卡两 die 口径后，结论更保守：

- 当前 `0.70x` 反推得到 A5 实际计算优势约为 A3 单 die 的 `1.6x`。
- 双方计算都减半，A5/A3 单卡约 `0.65x`。
- 双方通信都减半，A5/A3 单卡约 `0.74x`。
- 只降低 A5 通信，通信减半约 `0.84x`，通信全消约 `1.05x`。
- 若 A5 有效计算优势恢复到理论 `2x`，通信不变约 `0.83x`；双方通信都消掉约 `1.0x`；只消 A5 通信可到约 `1.25x`。

所以甜点配置的目标应改成：

```text
先把 A5/A3_card 从 0.70x 拉到 0.8x-1.0x
优先减少 A5-only 的未掩盖通信和 MoE 长尾
验证 A5 实际计算优势能否从 1.6x 恢复到 2x
不要指望双边计算优化提升 A5/A3_card 比值
```

推荐优先级：

1. 先试 `TP1`，减少每层固定 TP 通信税。
2. 试 `EP4`，观察 MoE all2allv p95/p99 和 routed token `max/avg` 是否下降。
3. 保持 `PP1`，除非显存压力迫使引入 PP。
4. 做两组序列长度实验：一组改总 token 数，一组固定总 token 改 packing/TND。
5. 用 `0.7x` 继续反推实际计算比，而不是直接套用理论 `2x`。

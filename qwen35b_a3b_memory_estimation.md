# Qwen3.5-35B-A3B 显存估计与 EP/TP 调参建议

生成日期：2026-06-16

## 0. 结论摘要

当前实验条件：

- 模型：`Qwen/Qwen3.5-35B-A3B`
- 配置来源：https://huggingface.co/Qwen/Qwen3.5-35B-A3B/blob/main/config.json
- 当前裁剪：ViT 和 LLM 都减层到 `1/4`
- 估算口径：LLM 从 40 层减到约 10 层
- 当前并行：`EP8PP1TP8`
- 当前序列长度：`16K`
- 当前观测：`npu-smi info` 长时间约 `48GB`

结论：

1. 当前 `48GB` 是合理的，不明显异常。即便 LLM 只保留 1/4 层，MoE expert 参数、训练态 optimizer/grad、16K activation、MoE dispatch/combine buffer、通信 workspace 和显存碎片叠加后，48GB 是可信量级。
2. `EP` 变小主要增加 expert 参数/梯度/优化器状态显存。若从 `EP8` 降到 `EP4`，且 expert 不被 TP 切分，持久训练态可能增加约 `16GB`，这通常需要继续减 LLM 层数。
3. `TP` 变小主要增加 TP shard 后的非 expert 参数、activation、attention workspace、MoE buffer 和通信 workspace。若从 `TP8` 降到 `TP4/TP2`，优先通过降低序列长度或激活保存量来试探。
4. 如果想同时把 `EP` 和 `TP` 开小，通常要同时减层和减激活。仅靠减序列长度不能抵消 EP 变小带来的 expert 参数态增长；仅靠减层也不一定能解决 TP 变小带来的 16K activation/workspace 压力。

## 1. Config 关键参数

从 Qwen3.5-35B-A3B config 采用以下关键参数：

```text
hidden_size H = 2048
num_hidden_layers = 40
num_experts E = 256
num_experts_per_tok top_k = 8
moe_intermediate_size I = 512
vocab_size = 248320
num_attention_heads = 16
num_key_value_heads = 2
head_dim = 256
```

当前 LLM 减层到 `1/4`，按：

```text
L_eff = 40 / 4 = 10 layers
```

进行估算。

## 2. 参数量估算

### 2.1 单个 expert 参数

MoE expert 是 SwiGLU 结构，包含 gate/up/down 三组投影：

```text
P_expert = 3 * H * I
         = 3 * 2048 * 512
         = 3,145,728 params
```

每层 routed experts：

```text
P_routed_layer = E * P_expert
               = 256 * 3,145,728
               = 805,306,368 params
               ~= 0.805B params / layer
```

10 层 routed experts：

```text
P_routed_10 = 10 * 0.805B
            ~= 8.05B params
```

### 2.2 非 expert 与 embedding/lm_head

按 config 和 Qwen3.5 attention 结构粗算：

```text
10 层非 expert 参数 ~= 0.36B params
embedding + lm_head ~= 1.02B params
```

所以 1/4 LLM 主模型参数量约：

```text
P_total_10 ~= 8.05B + 0.36B + 1.02B
           ~= 9.43B params
```

这说明即使只保留 1/4 层，MoE expert 仍然是显存主体。

## 3. 训练态显存口径

训练态每个本地参数大致可能包含：

```text
bf16 weight      2 bytes
bf16 grad        2 bytes
fp32 master      4 bytes
Adam m           4 bytes
Adam v           4 bytes
------------------------
合计             16 bytes / local param
```

如果没有 fp32 master，可能接近 `12 bytes/param`；如果使用 ZeRO/FSDP/分片 optimizer，实际会下降。下面主表使用保守的 `16 bytes/param`。

非常关键的一点：expert 参数是否被 TP 切分。如果 expert 不被 TP 切，`EP` 对显存的影响会非常大；如果 expert 也被 TP 切，`TP` 会显著分摊 expert 参数态。

## 4. EP/TP 下本地参数态估算

下面给出两种边界口径：

- `expertTP`：routed expert 参数同时按 `EP * TP` 切分。
- `expertNoTP`：routed expert 参数只按 `EP` 切分，非 expert 和 embedding/lm_head 按 TP 切分。

| EP | TP | expertTP 本地参数 | expertTP 训练态 | expertNoTP 本地参数 | expertNoTP 训练态 |
|---:|---:|---:|---:|---:|---:|
| 8 | 8 | `0.298B` | `4.8GB` | `1.178B` | `18.9GB` |
| 4 | 8 | `0.424B` | `6.8GB` | `2.185B` | `35.0GB` |
| 8 | 4 | `0.595B` | `9.5GB` | `1.350B` | `21.6GB` |
| 4 | 4 | `0.847B` | `13.6GB` | `2.357B` | `37.7GB` |
| 8 | 2 | `1.191B` | `19.1GB` | `1.694B` | `27.1GB` |
| 2 | 8 | `0.675B` | `10.8GB` | `4.198B` | `67.2GB` |
| 2 | 4 | `1.350B` | `21.6GB` | `4.370B` | `69.9GB` |

解释：

- 如果实际实现接近 `expertNoTP`，当前 `EP8TP8` 光持久训练态就可能约 `19GB`。再加 16K 激活、MoE buffer、通信 workspace、runtime cache 和碎片，`48GB` 很合理。
- 如果实际实现接近 `expertTP`，当前持久参数态只有约 `5GB`，那么 `48GB` 主要来自 activation、MoE/attention workspace、通信 buffer、FSDP/optimizer 实现细节和碎片。
- 因此下一步应先确认 expert weight shape 是否真的被 TP 切分，而不是只看 EP/TP 配置名。

## 5. EP 变小的显存影响

从 `EP8 -> EP4`，如果 expert 不被 TP 切：

```text
EP8TP8 expertNoTP 训练态 ~= 18.9GB
EP4TP8 expertNoTP 训练态 ~= 35.0GB
增量 ~= 16.1GB
```

这部分主要是 routed expert 参数、梯度和 optimizer 状态增长。它和序列长度关系不大，因此不能主要靠降低激活值解决。

每层 routed expert 训练态粗略为：

```text
EP8: 0.805B / 8 * 16 bytes ~= 1.61GB / layer
EP4: 0.805B / 4 * 16 bytes ~= 3.22GB / layer
EP2: 0.805B / 2 * 16 bytes ~= 6.44GB / layer
```

所以如果从 `EP8` 降到 `EP4`，想保持接近当前 48GB，最直接的方式是继续减 LLM 层数。例如：

```text
10 层 EP8 routed state ~= 16.1GB
5 层 EP4 routed state ~= 16.1GB
```

这只是 routed expert 参数态对齐，实际还要加非 expert、activation 和 workspace。因此 `EP4` 可能要把 LLM 从 10 层降到约 5-6 层，再配合序列长度探测。

## 6. TP 变小的显存影响

从 `TP8 -> TP4/TP2`，持久参数态会增加，但更大的风险通常是 activation、attention workspace、MoE dispatch/combine buffer 和通信 workspace。

16K 下一个 hidden tensor：

```text
16K * 2048 * 2 bytes ~= 67MB
```

MoE dispatch 单次逻辑 payload：

```text
16K * top_k(8) * hidden_size(2048) * 2 bytes
~= 0.537GB
```

dispatch + combine：

```text
~= 1.074GB / layer
```

实际训练中还会有：

- pack/unpack buffer
- backward grad buffer
- all2allv staging buffer
- attention workspace
- checkpoint/recompute 保留的 activation
- 通信库 workspace
- runtime allocator 碎片

因此 `TP8 -> TP4` 可能不只是参数态增加几 GB，而是 workspace 和 activation 峰值也上升。降低序列长度是更直接的探针：

```text
16K -> 12K: activation / MoE buffer 约降 25%
16K -> 8K:  activation / MoE buffer 约降 50%
```

## 7. 当前 48GB 是否合理

合理。

用 `expertNoTP` 口径，`EP8TP8` 训练态参数约 `18.9GB`。剩余：

```text
48GB - 18.9GB ~= 29GB
```

这 29GB 要容纳：

- 16K activation
- MoE dispatch/combine buffer
- attention/GDN workspace
- communication workspace
- optimizer/FSDP 临时 allgather buffer
- recompute/checkpoint 的保留张量
- CANN/HCCL/runtime cache
- allocator fragmentation
- ViT 和其他模块

所以长时间 `48GB` 不异常。

如果实际接近 `expertTP`，持久参数态较小，则说明 activation/workspace/临时 buffer 占比更高，后续开小 TP 时更需要先调序列长度。

## 8. 调参建议

### 8.1 想把 TP 开小

建议先保持 `EP8`，尝试：

```text
EP8PP1TP4, seq=16K
EP8PP1TP4, seq=12K
EP8PP1TP4, seq=8K
```

如果 `TP4 + 16K` OOM，而 `TP4 + 8K/12K` 能跑，说明主要瓶颈是 activation/workspace。

### 8.2 想把 EP 开小

建议先假设需要减层，尝试：

```text
EP4PP1TP8, LLM 5-6 layers, seq=16K
EP4PP1TP8, LLM 5-6 layers, seq=12K
EP4PP1TP8, LLM 5-6 layers, seq=8K
```

如果 `EP4` 在 10 层直接 OOM，原因大概率是 expert 参数/optimizer 状态，而不是激活。

### 8.3 想同时 EP/TP 都开小

建议从保守组合开始：

```text
EP4PP1TP4, LLM 5 layers, seq=8K
EP4PP1TP4, LLM 6 layers, seq=8K
EP4PP1TP4, LLM 5 layers, seq=12K
```

若稳定后再逐步加层或加序列，不要同时加两个维度。

## 9. 建议补充的观测

为了把显存模型校准到真实实现，建议在每组实验记录：

1. model init 后显存。
2. optimizer init 后显存。
3. 首次 forward 峰值。
4. backward 峰值。
5. optimizer step 峰值。
6. HCCL/CANN workspace 是否随 TP/EP 改变。
7. expert weight 在本地的 shape，确认是否被 TP 切。
8. activation checkpoint/recompute 是否打开。

最关键的是第 7 点：如果 expert 不被 TP 切，`EP` 是控制 expert 参数态的主旋钮；如果 expert 被 TP 切，`TP` 变小会显著增加 expert 参数态。

## 10. 一句话建议

当前 `EP8PP1TP8 + 16K + 1/4 layers` 的 `48GB` 是合理的。如果想减 TP，优先调序列长度；如果想减 EP，优先继续减 LLM 层数；如果 EP/TP 都减，先从 `EP4TP4 + 5-6 层 + 8K/12K` 这样的保守点开始。

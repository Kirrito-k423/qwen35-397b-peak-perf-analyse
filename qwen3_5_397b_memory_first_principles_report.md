# Qwen3.5-397B-A17B FSDP/EP 显存第一性原理推导

生成日期：2026-06-29

## 0. 结论摘要

本文按用户指定的 FSDP 峰值口径估算：

```text
FSDP peak ~= 常驻模型权重 shard
          + 当前 FSDP unit all-gather 出来的完整权重
          + 预取的下一 FSDP unit all-gather 权重
          + all-gather copy/staging buffer
          + 当前前反向所需 activation / communication / workspace buffer
```

如果统计完整训练显存，还需要在“常驻模型权重 shard”之外加入梯度、optimizer state、master weight 或 Muon/Adam 临时状态。本文先给第一性原理公式，再用 `/Users/Zhuanz/Downloads/rank0_memory_snapshot.pickle` 做一次实测校准。

核心结论：

1. 在 NPU `FSDP512 + SP16` 下，`EP` 对当前 MoE 层权重峰值影响非常大。平均单层 all-gather 权重从 `EP1=13.144GB` 降到 `EP8=1.869GB`，再到 `EP16=1.064GB`。
2. 由于 FSDP 开启 prefetch，峰值不能只算一层 all-gather。理论下限应至少按“两层 all-gather 权重”估算；`EP8` 从 `1.869GB` 修正为 `2 * 1.869 = 3.739GB`。snapshot 中实测 FSDP 相关 live allocation 峰值约 `5.894GB`，说明除了两层权重，还存在 copy-in/copy-out/staging buffer。
3. snapshot 的 trace 峰值为 `34.058GB active allocations`，其中 MoE forward 约 `18.723GB`，FSDP all-gather/prefetch 约 `5.894GB`，无 frame 的 workspace 约 `4.123GB`。snapshot 结束态 allocator reserved 为 `54.379GB`，但 active 只有 `7.816GB`，inactive/free blocks 高达 `46.563GB`，说明 reserved/碎片/可扩展 segment 复用也是 OOM 判断的关键。
4. `EP` 并不会从第一性原理上把 MoE dispatch 的平均激活总量 `S_local * top_k * hidden` 降下来。它只改变 token-expert 在 EP ranks 间的去向：本 rank 自己留下的比例约 `1/EP` 下降，但从整个 EP group 收到的平均 token-expert 总量仍约为 `S_local * top_k`。
5. 所以“FSDP+EP 是为了防止 MoE 层激活值过大”这句话只说对了一半：`EP` 主要是降低每 rank 需要 materialize 的 expert 权重和 optimizer/grad 状态；对 MoE hidden-state 激活 buffer，平均量级不随 EP 下降，负载不均衡只会在这个平均量上乘以 `rho=max/avg`。
6. NPU OOM 若发生在 `256k / SP16 = 16k local tokens`，更像是：`训练态常驻 shard + 两层 FSDP all-gather/prefetch + 多份 MoE/USP activation buffer + HCCL/CANN workspace + allocator reserved/fragmentation` 的合计逼近单 die HBM。EP 负载不均衡可以把边缘配置推 OOM，但不是平均激活显存下降的手段。

本文所有 GB 使用十进制 `1GB = 10^9 bytes`。换算 GiB 时乘以 `0.9313`。

## 1. 已知配置与符号

来自本仓库已有推导和 `/Users/Zhuanz/Downloads/node_0.txt` 日志：

```text
hidden_size H = 4096
num_hidden_layers L = 60
num_experts E = 512
num_experts_per_tok K = 10
moe_intermediate_size I = 1024
shared_expert_intermediate_size I_shared = 1024
vocab_size = 251392
bf16 bytes = 2
```

NPU 目标环境：

```text
32 机 * 16 die = 512 die
FSDP = 512
SP = 16
global sequence = 256k = 262144 tokens
local sequence S_local = 262144 / 16 = 16384 tokens
```

H200 对照环境：

```text
32 机 FSDP256
EP = 1
SP = 4
global sequence = 256k
local sequence S_local = 262144 / 4 = 65536 tokens
```

日志中实际总参数：

```text
Total trainable parameters = 403423.09M
```

按纯 text backbone 公式估算，不含日志中的额外 MTP/vision/projector，约为：

```text
P_text ~= 396.372B params
```

后文层内权重峰值使用 decoder 层公式；常驻 FSDP shard 同时给出 `396.372B` 和日志 `403.423B` 两个口径。

## 2. 单 expert 参数量

每个 routed expert 是 SwiGLU MLP，包含三组矩阵：

```text
gate_proj: [I, H]
up_proj:   [I, H]
down_proj: [H, I]
```

所以单 expert 参数量：

```text
P_expert = I * H + I * H + H * I
         = 3 * H * I
         = 3 * 4096 * 1024
         = 12,582,912 params
         = 12.583M params
```

512 个 routed experts 的全层 expert 参数：

```text
P_routed_all = E * P_expert
             = 512 * 12,582,912
             = 6,442,450,944 params
             = 6.442B params
```

bf16 权重大小：

```text
Bytes_routed_all = 6,442,450,944 * 2
                 = 12,884,901,888 bytes
                 = 12.885GB
```

这说明 Qwen3.5-397B-A17B 的单层权重主体几乎全在 routed experts。

## 3. 单层非 EP 参数量

非 routed expert 部分包括：

```text
shared expert
router
shared expert gate
norm
attention
```

### 3.1 Shared expert

```text
P_shared = 3 * H * I_shared
         = 3 * 4096 * 1024
         = 12,582,912 params
```

### 3.2 Router

```text
P_router = E * H
         = 512 * 4096
         = 2,097,152 params
```

### 3.3 Shared expert gate 与 norm

```text
P_shared_gate = H = 4096 params
P_norm = 2 * H = 8192 params
```

### 3.4 Attention 平均参数

已有本地推导给出：

```text
P_full_attn   = 104,858,112 params
P_linear_attn = 118,014,208 params
```

模型每 4 层 1 层 full attention、3 层 linear attention：

```text
P_attn_avg = (P_full_attn + 3 * P_linear_attn) / 4
           = (104,858,112 + 3 * 118,014,208) / 4
           = 114,725,184 params
```

### 3.5 单层非 EP 平均参数

```text
P_non_ep_avg = P_shared
             + P_router
             + P_shared_gate
             + P_norm
             + P_attn_avg

             = 12,582,912
             + 2,097,152
             + 4,096
             + 8,192
             + 114,725,184

             = 129,417,536 params
             = 129.418M params
```

bf16 大小：

```text
Bytes_non_ep_avg = 129,417,536 * 2
                 = 258,835,072 bytes
                 = 0.259GB
```

## 4. 单层权重随 EP 的变化

在 EP 下，每个 rank 只 materialize `E / EP` 个 routed experts 的参数；非 EP 参数仍按一层完整保留。

公式：

```text
E_local(EP) = E / EP

P_layer_local(EP)
  = P_routed_all / EP + P_non_ep_avg

Bytes_layer_local(EP)
  = P_layer_local(EP) * 2 bytes
```

NPU 上 `EP1/2/4/8/16` 对应的平均单层权重如下：

| EP | local experts | routed params / layer | non-EP params / layer | total params / layer | bf16 all-gather weight / layer |
|---:|---:|---:|---:|---:|---:|
| 1 | 512 | 6.442B | 0.129B | 6.572B | 13.144GB |
| 2 | 256 | 3.221B | 0.129B | 3.351B | 6.701GB |
| 4 | 128 | 1.611B | 0.129B | 1.740B | 3.480GB |
| 8 | 64 | 0.805B | 0.129B | 0.935B | 1.869GB |
| 16 | 32 | 0.403B | 0.129B | 0.532B | 1.064GB |

这就是 EP 对显存最直接的收益：当前 MoE decoder layer 的 all-gather 权重峰值近似按 `1/EP` 降低，但会被不可切的 `0.259GB` 非 EP 权重部分托底。

如果不考虑 FSDP shard，仅把“本 rank 需要拥有的全部 60 层 local-EP text 参数”materialize 出来，bf16 大小是：

| EP | 60 层 local-EP text 参数 bf16 大小 |
|---:|---:|
| 1 | 792.743GB |
| 2 | 406.196GB |
| 4 | 212.922GB |
| 8 | 116.286GB |
| 16 | 67.967GB |

这张表不是 FSDP 峰值公式中的常驻 shard，而是帮助理解：没有 FSDP 时，即便 EP16，单 rank 的 local experts + 非 EP 全层权重仍接近 68GB，几乎不可训练。

## 5. FSDP 常驻模型权重 shard

FSDP 的常驻模型权重按全模型参数在 FSDP group 内分片。权重-only 公式：

```text
M_weight_shard = P_total * 2 / FSDP
```

NPU `FSDP512`：

| 参数口径 | P_total | bf16 weight shard |
|---|---:|---:|
| text 公式估算 | 396.372B | 1.548GB |
| 日志 total trainable | 403.423B | 1.576GB |

H200 `FSDP256`：

| 参数口径 | P_total | bf16 weight shard |
|---|---:|---:|
| text 公式估算 | 396.372B | 3.097GB |
| 日志 total trainable | 403.423B | 3.152GB |

如果统计训练态常驻 shard，需要加入 grad 和 optimizer state。粗略区间：

```text
12 bytes/param: bf16 weight 2 + bf16 grad 2 + optimizer states 8
16 bytes/param: 再加 fp32 master weight 4
```

使用日志总参 `403.423B`：

| 环境 | FSDP | 12B/param training shard | 16B/param training shard |
|---|---:|---:|---:|
| NPU | 512 | 9.455GB | 12.607GB |
| H200 | 256 | 18.910GB | 25.214GB |

由于日志里有 Muon regular/MoE params 和 AdamW params，真实 optimizer state 不一定精确等于 Adam 的 12/16B 口径；这里是训练态显存的一阶范围，不是权重-only 峰值公式。

## 6. NPU FSDP 峰值按 EP 展开

按修正后的 FSDP prefetch-aware 口径：

```text
M_peak_weight_only(EP)
  ~= M_weight_shard
   + M_current_layer_allgather(EP)
   + M_prefetch_layer_allgather(EP)
   + M_allgather_staging(EP)
   + M_activation_peak
   + M_runtime_workspace
```

如果暂时不把 staging/copy buffer 展开，只先算“两层 all-gather 权重”下限：

```text
M_2layer_AG(EP) = 2 * M_layer_allgather(EP)
```

先只合并前两项，使用日志总参、NPU `FSDP512`：

```text
M_weight_shard = 1.576GB
```

| EP | one layer AG | two-layer prefetch AG | weight shard + two-layer AG |
|---:|---:|---:|---:|
| 1 | 13.144GB | 26.287GB | 27.863GB |
| 2 | 6.701GB | 13.403GB | 14.978GB |
| 4 | 3.480GB | 6.960GB | 8.536GB |
| 8 | 1.869GB | 3.739GB | 5.315GB |
| 16 | 1.064GB | 2.128GB | 3.704GB |

如果用训练态常驻 shard 替代 weight-only shard，取 `12.607GB` 上界：

| EP | training shard upper | two-layer prefetch AG | training shard + two-layer AG |
|---:|---:|---:|---:|
| 1 | 12.607GB | 26.287GB | 38.894GB |
| 2 | 12.607GB | 13.403GB | 26.010GB |
| 4 | 12.607GB | 6.960GB | 19.567GB |
| 8 | 12.607GB | 3.739GB | 16.346GB |
| 16 | 12.607GB | 2.128GB | 14.735GB |

这还没有加入 all-gather copy/staging buffer、activation、communication workspace、CANN/HCCL workspace 和 allocator fragmentation。

`rank0_memory_snapshot.pickle` 里 `EP8` 的 FSDP all-gather allocation 比理论值更大：

```text
theory one-layer AG EP8 = 1.869GB
observed large FSDP allocation ~= 2.059GB
observed two such allocations ~= 4.119GB

weight shard + observed two allocations
  = 1.576GB + 4.119GB
  = 5.695GB

training shard upper + observed two allocations
  = 12.607GB + 4.119GB
  = 16.726GB
```

此外，trace 峰值处 FSDP 相关 live allocation 合计约 `5.894GB`，比“两个 2.059GB unshard buffer”还多，说明 copy-in/copy-out/staging/prefetch buffer 也在峰值中重叠。

## 7. NPU 256k/SP16 的激活与 MoE buffer

NPU `SP16` 下：

```text
S_local = 262144 / 16 = 16384 tokens
```

### 7.1 一个 hidden state tensor

```text
Bytes_hidden = S_local * H * 2
             = 16384 * 4096 * 2
             = 134,217,728 bytes
             = 0.134GB
```

一个残差 hidden `[S_local, H]` 就是 `0.134GB`。训练中不会只存在一份 hidden，还会有 layernorm 输入、attention 输入输出、MoE 输入输出、grad buffer、通信 staging buffer 等。

### 7.2 MoE top-k dispatch hidden buffer

每个 token 路由到 `K=10` 个 experts。MoE dispatch 搬运的是 hidden state，不是 expert intermediate size。

单份 expanded token-expert hidden buffer：

```text
Bytes_moe_expanded = S_local * K * H * 2
                   = 16384 * 10 * 4096 * 2
                   = 1,342,177,280 bytes
                   = 1.342GB
```

这个 `1.342GB` 是 MoE hidden-state 激活显存的第一性原理核心量。

注意：`moe_intermediate_size=1024` 会让 expert 权重和 expert MLP compute 比 `I=4096` 小 4 倍；但 dispatch/combine 搬运的是 `hidden_size=4096`，所以 MoE 通信/激活 hidden buffer 不会因为 `I=1024` 降 4 倍。

### 7.3 EP 是否降低 MoE 激活 buffer

先看本 rank 自己发出的 token-expert hidden：

```text
send_total_per_rank = S_local * K * H * 2
                    = 1.342GB
```

在 EP 下，平均有 `1/EP` 留给本 rank local experts，`(EP-1)/EP` 发给其他 EP ranks：

| EP | 本 rank 自己留下 | remote send one-way | dispatch+combine remote bytes |
|---:|---:|---:|---:|
| 1 | 1.342GB | 0.000GB | 0.000GB |
| 2 | 0.671GB | 0.671GB | 1.342GB |
| 4 | 0.336GB | 1.007GB | 2.013GB |
| 8 | 0.168GB | 1.174GB | 2.349GB |
| 16 | 0.084GB | 1.258GB | 2.517GB |

但这张表只描述“本 rank 自己发出的 token-expert 分布”。从整个 EP group 的角度看：

```text
EP group total token-expert assignments = EP * S_local * K
每个目标 rank 平均接收 = (EP * S_local * K) / EP
                    = S_local * K
```

所以每个 EP rank 平均接收的 expert input hidden 仍然是：

```text
recv_total_avg_per_rank
  = S_local * K * H * 2
  = 1.342GB
```

这意味着：

```text
EP 不降低平均 MoE expert input activation buffer。
EP 降低的是本 rank 自己留在本地的比例，但其他 rank 发来的 token-expert 会补回来。
```

如果出现负载不均衡：

```text
rho = max_received_token_expert / avg_received_token_expert
```

则最热 rank 的单份 expert input buffer 约为：

```text
Bytes_hot_rank = rho * 1.342GB
```

| rho | 单份 expert input buffer |
|---:|---:|
| 1.00 | 1.342GB |
| 1.20 | 1.611GB |
| 1.50 | 2.013GB |
| 2.00 | 2.684GB |

如果实现中峰值同时存在 send buffer、recv buffer、pack/unpack buffer、combine buffer、backward grad buffer 等 `m` 份同量级 buffer，不均衡导致的额外显存约为：

```text
extra ~= m * (rho - 1) * 1.342GB
```

例如 `m=4`：

| rho | 额外显存 |
|---:|---:|
| 1.20 | 1.074GB |
| 1.50 | 2.684GB |
| 2.00 | 5.369GB |

例如 `m=8`：

| rho | 额外显存 |
|---:|---:|
| 1.20 | 2.147GB |
| 1.50 | 5.369GB |
| 2.00 | 10.737GB |

所以 EP 负载不均衡可以把本来余量很薄的配置推到 OOM；但它不是“EP 平均降低激活显存”的证明。

### 7.4 Router logits

Router logits 如果按 fp32 临时存：

```text
Bytes_router_logits = S_local * E * 4
                    = 16384 * 512 * 4
                    = 33,554,432 bytes
                    = 0.034GB
```

即使保留若干份，也远小于 top-k hidden dispatch buffer。

### 7.5 Attention / USP activation all-to-all

本仓库已有源码 shape 推导，NPU `SP16` 下平均每层 USP attention all-to-all 逻辑 payload 约：

```text
Bytes_USP_avg = 0.708GB / layer
```

full attention 层约 `0.805GB`，linear attention 层约 `0.675GB`。这部分随 `S_local` 近似线性变化，和 EP 没有直接关系。

## 8. 把 NPU EP8 的峰值拼起来

以用户当前关心的 NPU `EP8 + SP16 + FSDP512` 为例：

理论权重-only 基础项，如果只算两层 all-gather 权重、暂不算 staging：

```text
FSDP weight shard      ~= 1.576GB
current layer AG EP8   ~= 1.869GB
prefetch layer AG EP8  ~= 1.869GB
----------------------------------
subtotal               ~= 5.315GB
```

训练态常驻上界基础项：

```text
FSDP training shard upper ~= 12.607GB
current layer AG EP8      ~= 1.869GB
prefetch layer AG EP8     ~= 1.869GB
-------------------------------------
subtotal                  ~= 16.346GB
```

snapshot 实测的 FSDP large allocation 是 `2.059GB`，比理论 `1.869GB` 高约 `10.2%`。按实测值替换后：

```text
observed two FSDP large allocations ~= 2 * 2.059GB = 4.119GB

weight-only subtotal
  ~= 1.576GB + 4.119GB
  ~= 5.695GB

training-state upper subtotal
  ~= 12.607GB + 4.119GB
  ~= 16.726GB
```

但这仍然不是完整峰值。`rank0_memory_snapshot.pickle` 的 trace 重建显示：

```text
peak active allocation = 34.058GB
peak index = 30979
```

峰值处按调用栈粗分：

| 类别 | peak live allocation |
|---|---:|
| MoE forward | 18.723GB |
| FSDP all-gather / unshard / prefetch | 5.894GB |
| 无 frame 的 workspace / snapshot allocation | 4.123GB |
| other | 5.319GB |

峰值最大的几个 allocation 直接对应理论量级：

| allocation | 大小 | 证据/调用栈 |
|---|---:|---|
| workspace snapshot | 4.121GB | trace 起始 `workspace_snapshot` / `segment_alloc` |
| FSDP unshard output | 2.059GB | `alloc_all_gather_outputs` / `param_group_wait_for_unshard` |
| FSDP all-gather copy-in | 1.917GB-2.059GB | `all_gather_copy_in_npu` / `_prefetch_unshard` |
| MoE buffer | 1.342GB 多份 | `moe_decoder_layer.py` / `model/moe/moe.py` |
| MoE larger temporary | 2.147GB、2.059GB 多份 | `model/moe/moe.py` |

所以更合理的 EP8 峰值结构是：

```text
M_peak_EP8
  ~= training_state_shard
   + FSDP current/prefetch/copy buffers
   + MoE multi-buffer peak
   + attention/USP and other activations
   + workspace/reserved/fragmentation

snapshot calibrated active peak:
  ~= 34.058GB active allocations

snapshot ending allocator state:
  reserved total ~= 54.379GB
  active         ~= 7.816GB
  inactive/free  ~= 46.563GB
```

这说明 OOM 判断不能只看 active tensor，也不能只看两层权重；在 NPU allocator 里，reserved/inactive segment 可能已经把可用 HBM 吃得很薄。

激活/runtime 的第一性原理基础量仍然是：

```text
one hidden tensor             ~= 0.134GB
one MoE top-k hidden buffer   ~= 1.342GB
USP avg communication payload ~= 0.708GB
router logits fp32            ~= 0.034GB
```

如果某个 MoE 层峰值同时存在：

```text
4-8 份 MoE top-k hidden / staging / grad / combine 同量级 buffer
```

则仅 MoE hidden buffer 就可能是：

```text
4 * 1.342GB = 5.369GB
8 * 1.342GB = 10.737GB
```

再叠加 attention/USP、activation checkpoint/recompute 临时张量、CANN/HCCL workspace、FSDP prefetch/all-gather 临时 buffer、allocator reserved/fragmentation，实际峰值会显著高于公式中最干净的三项。

这也解释了为什么 NPU OOM 更可能是“FSDP 预取两层 + all-gather staging + MoE 多份大 buffer + workspace + reserved/碎片 + 少量 EP skew”组合，而不是单看 EP8 的单层权重。

## 9. H200 对照口径

H200 配置：

```text
FSDP256
EP1
SP4
local sequence = 262144 / 4 = 65536 tokens
```

### 9.1 FSDP shard 与单层 all-gather

使用日志总参 `403.423B`：

```text
FSDP256 weight shard = 403.423B * 2 / 256
                     = 3.152GB
```

训练态常驻 shard：

```text
12B/param ~= 18.910GB
16B/param ~= 25.214GB
```

EP1 下当前 decoder layer all-gather 权重：

```text
P_layer_EP1 = 6.572B params
Bytes_layer_EP1 = 13.144GB
```

如果 H200 侧同样开启 FSDP prefetch，则也应至少按两层权重估算：

```text
Bytes_two_layer_prefetch_EP1 = 2 * 13.144GB = 26.287GB
```

实际峰值还要继续加 all-gather copy/staging buffer。NPU snapshot 证明 staging/copy buffer 不能忽略；H200 是否同样重叠，需要用 H200 memory snapshot 校准。

### 9.2 H200 local activation 基础量

一个 hidden tensor：

```text
Bytes_hidden_H200 = 65536 * 4096 * 2
                  = 536,870,912 bytes
                  = 0.537GB
```

MoE top-k hidden buffer：

```text
Bytes_moe_expanded_H200 = 65536 * 10 * 4096 * 2
                         = 5,368,709,120 bytes
                         = 5.369GB
```

由于 H200 是 EP1，没有 EP all-to-all；但如果 grouped expert MLP 仍 materialize top-k expanded hidden，`5.369GB` 这个激活量仍然存在。H200 能承受，主要因为 H200 单卡 HBM 容量远大于 A3 单 die，且其 `FSDP256` 的训练态 shard 虽更大，但总体还有较大空间。

## 10. 对“FSDP+EP 是为了防 MoE 激活过大”的判断

这句话需要拆成三层：

### 10.1 对 expert 权重显存：是对的

EP 直接降低当前 MoE 层 all-gather expert 权重：

```text
EP1  -> 13.144GB / layer
EP8  ->  1.869GB / layer
EP16 ->  1.064GB / layer
```

如果开启 FSDP prefetch，权重项至少再乘以 2；如果实现还有 copy-in/copy-out staging，则峰值会继续增加。`rank0_memory_snapshot.pickle` 中 `EP8` 的 FSDP live allocation 峰值约 `5.894GB`，高于理论两层权重 `3.739GB`，就是这个现象。

它也降低每 rank 需要负责的 expert 参数、梯度、optimizer state 规模。没有 EP 时，MoE 397B 的单层 expert 权重太大，当前层 all-gather/prefetch 峰值和训练态参数态都很危险。

### 10.2 对 MoE dispatch hidden activation 平均量：不对

MoE dispatch/combine 的 hidden-state 激活基本量是：

```text
S_local * top_k * hidden_size * bytes
```

在 NPU `SP16` 下就是：

```text
16384 * 10 * 4096 * 2 = 1.342GB
```

这个平均量不因为 EP 从 1 到 16 而下降。EP 只是把这 `1.342GB` 的 token-expert assignments 分发给不同 expert owners；从目标 rank 看，平均接收量仍然是 `1.342GB`。

### 10.3 对最热 rank 激活峰值：EP 可能让问题更敏感

EP 越大：

```text
remote fraction = (EP - 1) / EP
```

从 EP8 到 EP16，remote fraction 从 `87.5%` 到 `93.75%`，通信 peer 更多、chunk 更碎、等待长尾概率更高。若 router 负载不均衡，最热 rank 的 recv buffer 变成 `rho * 1.342GB`，并且多份 staging/backward buffer 会放大这个差异。

所以更准确的说法是：

```text
FSDP+EP 主要是防 expert 权重/参数态过大；
SP 主要是降低每 rank sequence activation；
FSDP prefetch 会让至少两层权重同时存在，还可能叠加 copy/staging buffer；
EP 负载均衡差会放大 MoE activation/workspace 峰值，但 EP 本身不是降低平均 MoE activation 的工具。
```

## 11. 建议下一步校准项

为了确认 NPU OOM 是否真由 EP imbalance 触发，需要记录：

1. 每层每 rank `received token-expert count`，计算 `rho=max/avg`。
2. 每层每 expert token count，确认是否有 hot expert。
3. EP all2allv per-peer send/recv bytes。
4. OOM 前最后一个 MoE 层的 send/recv/pack/combine buffer shape。
5. FSDP all-gather 当前 unit、prefetch unit、copy-in/copy-out staging 的实际字节数，确认同时 live 的 FSDP allocation 数量。
6. 分阶段显存：model init、optimizer init、first forward peak、backward peak、optimizer step peak。
7. `allocated`、`reserved`、`inactive/free blocks` 同时记录。当前 snapshot 结束态为 `reserved 54.379GB / active 7.816GB / inactive 46.563GB`，碎片和可复用 segment 不能忽略。

判断标准：

```text
如果 rho <= 1.2，EP imbalance 很难单独解释 OOM；
如果 rho >= 1.5，并且 OOM 点恰在 MoE dispatch/combine/backward，则 EP imbalance 很可能是触发器；
如果 FSDP live allocation 接近或超过 2 * layer_AG，则优先查 FSDP prefetch、copy/staging buffer 和 materialize 范围；
如果 reserved 远大于 active，则优先查 allocator fragmentation、expandable segment、CANN/HCCL workspace 与释放时机。
```

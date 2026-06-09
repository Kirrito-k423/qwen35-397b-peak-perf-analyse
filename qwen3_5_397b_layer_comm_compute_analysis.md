# Qwen3.5-397B-A17B 单层通信与计算耗时推导

## 1. 目标

分析 Qwen3.5-397B-A17B 在 32 机 Ascend A3、单机 16 die、总 512 die 环境下，采用 `256k_ep16_sp16` 并行时，一层模型的理论通信负载、理论计算量，以及在 H800 和 A3 超节点上的耗时预测。

这里的 `256k_ep16_sp16` 按如下方式理解：

- 全局序列长度：`256k tokens`
- Sequence Parallel：`SP = 16`
- 单 die local sequence length：`256k / 16 = 16k tokens`
- Expert Parallel：`EP = 16`
- 数据类型：`bf16`，即 `2 bytes/parameter`

## 2. 事实来源

### 2.1 HuggingFace config

模型配置来自：

`https://huggingface.co/Qwen/Qwen3.5-397B-A17B/blob/main/config.json`

关键字段如下：

```json
{
  "text_config": {
    "hidden_size": 4096,
    "num_hidden_layers": 60,
    "num_attention_heads": 32,
    "num_key_value_heads": 2,
    "head_dim": 256,
    "num_experts": 512,
    "num_experts_per_tok": 10,
    "moe_intermediate_size": 1024,
    "shared_expert_intermediate_size": 1024,
    "full_attention_interval": 4,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 64,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4
  }
}
```

因此：

- `H = hidden_size = 4096`
- `L = num_hidden_layers = 60`
- `E = num_experts = 512`
- `K = num_experts_per_tok = 10`
- `I = moe_intermediate_size = 1024`
- `I_shared = shared_expert_intermediate_size = 1024`
- full attention 层比例为 `1/4`
- linear attention 层比例为 `3/4`

### 2.2 本地 node_0.txt

本地文件：

`/Users/Zhuanz/Downloads/node_0.txt`

日志中关键事实：

```text
WORLD_SIZE=32
torchrun --nproc_per_node 16 --nnodes 32 ... --config hyf_test/sft_397b_gq.py
Qwen3_5_VLMoE397BA17SplitConfig
```

参数形状示例：

```text
language_model.layers.47.self_attn.q_proj.weight: shape = [16384, 4096]
language_model.layers.47.self_attn.k_proj.weight: shape = [512, 4096]
language_model.layers.47.self_attn.v_proj.weight: shape = [512, 4096]
language_model.layers.47.self_attn.o_proj.weight: shape = [4096, 8192]

language_model.layers.48.self_attn.in_proj_qkv.weight: shape = [12288, 4096]
language_model.layers.48.self_attn.in_proj_z.weight: shape = [8192, 4096]
language_model.layers.48.self_attn.in_proj_b.weight: shape = [64, 4096]
language_model.layers.48.self_attn.in_proj_a.weight: shape = [64, 4096]
language_model.layers.48.self_attn.out_proj.weight: shape = [4096, 8192]

language_model.layers.48.shared_experts.gate_proj.weight: shape = [1024, 4096]
language_model.layers.48.shared_experts.up_proj.weight: shape = [1024, 4096]
language_model.layers.48.shared_experts.down_proj.weight: shape = [4096, 1024]
language_model.layers.48.gate.weight: shape = [512, 4096]
language_model.layers.48.experts.fused_w1w3.weight: shape = [1048576, 4096]
language_model.layers.48.experts.fused_w2.weight: shape = [2097152, 1024]
```

这些形状与 HF config 对齐：

- `1048576 = 512 experts * 2 * 1024`
- `2097152 = 512 experts * 4096`
- `q_proj [16384, 4096] = 32 heads * 256 head_dim * 2 gate factor`
- `k_proj/v_proj [512, 4096] = 2 kv heads * 256 head_dim`

注意：日志中实际配置片段显示 `tp_size=1, ep_size=8, sp_size=16`。本文按用户指定的目标配置 `ep16_sp16` 进行理论预测；如果按日志实际 `ep8_sp16`，通信域和专家分片假设需要单独重算。

## 3. 单层参数量推导

下面先推导一层全局全量参数量，再推导 `EP=16` 后单个 EP rank 本地持有的专家参数量。通信 AllGather 负载应使用“EP 切分后的本地专家参数 + 非 EP 参数”，而不是 512 个 routed experts 的全局总参数。

### 3.1 Routed experts 参数量

每个 routed expert 是 SwiGLU 结构，包含：

- `gate_proj`: `[I, H]`
- `up_proj`: `[I, H]`
- `down_proj`: `[H, I]`

单个 expert 参数量：

```text
P_expert = I * H + I * H + H * I
         = 3 * H * I
         = 3 * 4096 * 1024
         = 12,582,912 params
```

512 个 routed experts 全量参数量：

```text
P_routed_all = E * P_expert
             = 512 * 12,582,912
             = 6,442,450,944 params
```

这里需要特别注意 `moe_intermediate_size = 1024` 的影响。单个 expert 的 FFN 中间维不是 `hidden_size = 4096`，而是 `I = 1024`。因此 expert 参数量和 expert 内部 GEMM 计算量都按 `3 * H * I` 计算：

```text
P_expert = 3 * 4096 * 1024
         = 12,582,912 params
```

如果误按 `I = 4096` 估算，则会得到：

```text
P_expert_wrong = 3 * 4096 * 4096
               = 50,331,648 params
```

两者相差正好 `4x`：

```text
P_expert_wrong / P_expert
= 50,331,648 / 12,582,912
= 4
```

所以 MoE expert 参数量、expert 内部 `gate/up/down` GEMM，以及与 expert 中间维相关的估算，都必须使用 `moe_intermediate_size = 1024`。这也是 profiling 中 MoE 相关计算/搬运规模可能比按 `4096` 中间维粗估小 `4x` 的主要原因。

但这不等价于所有 MoE 通信都小 `4x`。EP dispatch / combine 通常搬运的是 token hidden state，shape 主要由：

```text
tokens * top_k * hidden_size
```

决定，其中使用的是 `hidden_size = 4096`，不是 `moe_intermediate_size = 1024`。因此需要区分两类口径：

| 项目 | 使用维度 | 是否受 `moe_intermediate_size=1024` 影响 |
|---|---:|---:|
| expert 参数量 | `3 * hidden_size * moe_intermediate_size` | 是，较 `I=4096` 小 `4x` |
| expert 内部 GEMM | `2 * tokens * active_expert_params` | 是，较 `I=4096` 小 `4x` |
| EP token dispatch/combine hidden state | `tokens * top_k * hidden_size` | 否，主要由 `hidden_size=4096` 决定 |
| router logits | `tokens * num_experts` | 否，主要由 `num_experts=512` 决定 |

### 3.2 Shared expert 参数量

shared expert 也是 SwiGLU 结构，中间维为 `1024`：

```text
P_shared = 3 * H * I_shared
         = 3 * 4096 * 1024
         = 12,582,912 params
```

### 3.3 Router 和 norm 参数量

Router：

```text
P_router = E * H
         = 512 * 4096
         = 2,097,152 params
```

Shared expert gate：

```text
P_shared_gate = 1 * H
              = 4,096 params
```

两个 RMSNorm：

```text
P_norm = 2 * H
       = 8,192 params
```

### 3.4 Full attention 层参数量

full attention 层日志形状为：

```text
q_proj: [16384, 4096]
k_proj: [512, 4096]
v_proj: [512, 4096]
o_proj: [4096, 8192]
q_norm: [256]
k_norm: [256]
```

参数量：

```text
P_full_attn = 16384 * 4096
            + 512 * 4096
            + 512 * 4096
            + 4096 * 8192
            + 256
            + 256
            = 104,858,112 params
```

### 3.5 Linear attention 层参数量

linear attention 层日志形状为：

```text
conv1d.weight: [12288, 1, 4]
norm.weight: [128]
out_proj.weight: [4096, 8192]
in_proj_qkv.weight: [12288, 4096]
in_proj_z.weight: [8192, 4096]
in_proj_b.weight: [64, 4096]
in_proj_a.weight: [64, 4096]
dt_bias: [64]
A_log: [64]
```

参数量：

```text
P_linear_attn = 12288 * 1 * 4
              + 128
              + 4096 * 8192
              + 12288 * 4096
              + 8192 * 4096
              + 64 * 4096
              + 64 * 4096
              + 64
              + 64
              = 118,014,208 params
```

### 3.6 单层全量参数量

full attention 层：

```text
P_layer_full = P_routed_all
             + P_shared
             + P_router
             + P_shared_gate
             + P_norm
             + P_full_attn

             = 6,442,450,944
             + 12,582,912
             + 2,097,152
             + 4,096
             + 8,192
             + 104,858,112

             = 6,562,001,408 params
```

linear attention 层：

```text
P_layer_linear = P_routed_all
               + P_shared
               + P_router
               + P_shared_gate
               + P_norm
               + P_linear_attn

               = 6,442,450,944
               + 12,582,912
               + 2,097,152
               + 4,096
               + 8,192
               + 118,014,208

               = 6,575,157,504 params
```

因为 60 层中每 4 层 1 层 full attention、3 层 linear attention，平均单层参数量为：

```text
P_layer_avg = (1 * P_layer_full + 3 * P_layer_linear) / 4
            = (6,562,001,408 + 3 * 6,575,157,504) / 4
            = 6,571,868,480 params
```

换成十进制 G 参数：

```text
P_layer_avg = 6.572G params
```

## 4. EP16 后单层 AllGather 通信负载

上面的 `P_layer_avg = 6.572G params` 是全局单层参数量，包含 512 个 routed experts。采用 `EP=16` 时，每个 EP rank 只负责：

```text
E_local = E / EP
        = 512 / 16
        = 32 experts
```

因此每个 EP rank 的本地 routed expert 参数量为：

```text
P_routed_ep16 = E_local * P_expert
               = 32 * 12,582,912
               = 402,653,184 params
```

非 EP 参数包括 shared expert、router、shared expert gate、norm 和 attention。平均每层非 EP 参数量为：

```text
P_non_ep_avg = P_shared
             + P_router
             + P_shared_gate
             + P_norm
             + (P_full_attn + 3 * P_linear_attn) / 4

             = 12,582,912
             + 2,097,152
             + 4,096
             + 8,192
             + (104,858,112 + 3 * 118,014,208) / 4

             = 129,417,536 params
```

所以 `EP16` 后，单个 EP rank 每层需要通过 FSDP AllGather 恢复的参数量约为：

```text
P_fsdp_ep16 = P_routed_ep16 + P_non_ep_avg
             = 402,653,184 + 129,417,536
             = 532,070,720 params
```

bf16 每个参数 2 bytes，因此每层 AllGather 的通信负载约为：

```text
Bytes_layer_ep16 = P_fsdp_ep16 * 2
                 = 532,070,720 * 2
                 = 1,064,141,440 bytes
                 = 1.064 GB
```

结论：

```text
Qwen3.5-397B-A17B 全局平均每层参数量约 6.57G；但在 EP16 下，单个 EP rank 的 FSDP AllGather 参数量约 0.532G，bf16 通信负载约 1.064GB。
```

## 5. SP16 / USP all-to-all 激活通信

除了 FSDP 参数 AllGather，代码中还存在 USP 的序列并行 all-to-all 激活通信。

### 5.1 代码依据

`xtuner/v1/module/attention/mha.py` 中，full attention 路径在 `seq_ctx.sequence_parallel_mesh.size() > 1` 时：

```python
query_states = ulysses_all_to_all(query_states, scatter_dim=1, gather_dim=2, ...)
key_states = ulysses_all_to_all(key_states, scatter_dim=1, gather_dim=2, ...)
value_states = ulysses_all_to_all(value_states, scatter_dim=1, gather_dim=2, ...)
...
raw_output = ulysses_all_to_all(raw_output, scatter_dim=1, gather_dim=2, ...)
```

也就是 full attention 每层有 4 次 USP all-to-all：`Q`、`K`、`V`、`O`。

`xtuner/v1/module/attention/gated_deltanet.py` 中，linear attention / GatedDeltaNet 路径在 SP 下：

```python
query = _all_to_all_conv_pre_qk(...)
key = _all_to_all_conv_pre_qk(...)
value = _all_to_all_conv_pre_v(...)
g = _all_to_all_gb(...)
beta = _all_to_all_gb(...)
core_attn_out = _all_to_all_out(...)
```

也就是 linear attention 每层有 6 次 USP all-to-all：`Q`、`K`、`V`、`g`、`beta`、`O`。

### 5.2 all-to-all 通信口径

`ulysses_all_to_all` 的实现对输入 tensor 做 all-to-all single。本文采用与前面通信估算一致的“逻辑 payload”口径：每次 all-to-all 的通信负载按输入 tensor 的 bf16 字节数估算。

严格按网络发送字节算，all-to-all 中每个 rank 只发送 `(SP-1)/SP` 的本地 tensor 给其他 rank；`SP=16` 时发送字节约为逻辑 payload 的 `15/16`。由于这里还要和 FSDP AllGather 示例口径对齐，主表使用逻辑 payload，并在备注中说明发送字节修正。

### 5.3 Full attention USP 通信量

full attention 路径中：

- `Q` 进入 all-to-all 前 shape 约为 `[1, 32, 16k, 256]`
- `K/V` 原始 KV heads 为 2，但因为 `SP=16 > num_kv_heads=2`，代码先 `repeat_kv` 到 16 heads，再做 all-to-all，因此 `K/V` shape 约为 `[1, 16, 16k, 256]`
- `O/raw_output` shape 约为 `[1, 32, 16k, 256]`

bf16 每元素 2 bytes，因此：

```text
Bytes_Q_full = 16,384 * 32 * 256 * 2
             = 268,435,456 bytes
             = 0.268 GB

Bytes_K_full = 16,384 * 16 * 256 * 2
             = 134,217,728 bytes
             = 0.134 GB

Bytes_V_full = 16,384 * 16 * 256 * 2
             = 134,217,728 bytes
             = 0.134 GB

Bytes_O_full = 16,384 * 32 * 256 * 2
             = 268,435,456 bytes
             = 0.268 GB
```

full attention 单层 USP all-to-all 逻辑 payload：

```text
Bytes_USP_full = Bytes_Q_full + Bytes_K_full + Bytes_V_full + Bytes_O_full
               = 805,306,368 bytes
               = 0.805 GB
```

### 5.4 Linear attention / GatedDeltaNet USP 通信量

linear attention 配置：

- `linear_num_key_heads = 16`
- `linear_key_head_dim = 128`
- `linear_num_value_heads = 64`
- `linear_value_head_dim = 128`

因此：

```text
Q dim = 16 * 128 = 2,048
K dim = 16 * 128 = 2,048
V dim = 64 * 128 = 8,192
g dim = 64
beta dim = 64
O dim = 64 * 128 = 8,192
```

bf16 通信量：

```text
Bytes_Q_linear = 16,384 * 2,048 * 2
               = 67,108,864 bytes
               = 0.067 GB

Bytes_K_linear = 16,384 * 2,048 * 2
               = 67,108,864 bytes
               = 0.067 GB

Bytes_V_linear = 16,384 * 8,192 * 2
               = 268,435,456 bytes
               = 0.268 GB

Bytes_g_linear = 16,384 * 64 * 2
               = 2,097,152 bytes
               = 0.002 GB

Bytes_beta_linear = 16,384 * 64 * 2
                  = 2,097,152 bytes
                  = 0.002 GB

Bytes_O_linear = 16,384 * 8,192 * 2
               = 268,435,456 bytes
               = 0.268 GB
```

linear attention 单层 USP all-to-all 逻辑 payload：

```text
Bytes_USP_linear = Bytes_Q_linear
                 + Bytes_K_linear
                 + Bytes_V_linear
                 + Bytes_g_linear
                 + Bytes_beta_linear
                 + Bytes_O_linear

                 = 675,282,944 bytes
                 = 0.675 GB
```

### 5.5 平均单层 USP 通信量

Qwen3.5-397B-A17B 每 4 层中 1 层 full attention、3 层 linear attention，因此平均单层 USP all-to-all 逻辑 payload：

```text
Bytes_USP_avg = (Bytes_USP_full + 3 * Bytes_USP_linear) / 4
              = (805,306,368 + 3 * 675,282,944) / 4
              = 707,788,800 bytes
              = 0.708 GB
```

如果按 all-to-all 实际发送给其他 rank 的字节数估算：

```text
Bytes_USP_avg_send = Bytes_USP_avg * (SP - 1) / SP
                   = 707,788,800 * 15 / 16
                   = 663,552,000 bytes
                   = 0.664 GB
```

### 5.6 参数通信 + USP 激活通信合计

主表使用逻辑 payload 口径：

```text
Bytes_comm_total = Bytes_layer_ep16 + Bytes_USP_avg
                 = 1,064,141,440 + 707,788,800
                 = 1,771,930,240 bytes
                 = 1.772 GB
```

结论：

```text
按源码 tensor shape 估算，在 EP16 + SP16 下，平均单层通信逻辑 payload 约为 1.772GB，其中 FSDP 参数 AllGather 约 1.064GB，USP all-to-all 激活通信约 0.708GB。
```

## 6. 单层 active 计算量推导

计算量需要使用 active 参数，而不是全量参数。MoE 中每个 token 只激活 `top_k = 10` 个 routed experts，同时始终计算 shared expert、router 和 attention。

### 6.1 Active routed experts 参数量

```text
P_routed_active = K * P_expert
                = 10 * 12,582,912
                = 125,829,120 params
```

### 6.2 Full attention 层 active 参数量

```text
P_active_full = P_routed_active
              + P_shared
              + P_router
              + P_shared_gate
              + P_norm
              + P_full_attn

              = 125,829,120
              + 12,582,912
              + 2,097,152
              + 4,096
              + 8,192
              + 104,858,112

              = 245,379,584 params
```

### 6.3 Linear attention 层 active 参数量

```text
P_active_linear = P_routed_active
                + P_shared
                + P_router
                + P_shared_gate
                + P_norm
                + P_linear_attn

                = 125,829,120
                + 12,582,912
                + 2,097,152
                + 4,096
                + 8,192
                + 118,014,208

                = 258,535,680 params
```

平均 active 参数量：

```text
P_active_avg = (1 * P_active_full + 3 * P_active_linear) / 4
             = (245,379,584 + 3 * 258,535,680) / 4
             = 255,246,656 params
```

### 6.4 GEMM 计算量

对线性层，矩阵乘通常按 `2 FLOPs / parameter / token` 估算。

单 die local sequence length：

```text
S_local = 256k / SP
        = 262,144 / 16
        = 16,384 tokens
```

平均每层 GEMM FLOPs：

```text
F_gemm_avg = 2 * S_local * P_active_avg
           = 2 * 16,384 * 255,246,656
           = 8,363,922,423,808 FLOPs
           = 8.36 TFlops
```

### 6.5 Full attention 二次项

full attention 每 4 层出现一次。对 full attention 的 `QK^T` 和 `AV` 二次项，粗略按：

```text
F_attn_quad_full = 4 * S_local^2 * num_attention_heads * head_dim
                 = 4 * 16,384^2 * 32 * 256
                 = 8,796,093,022,208 FLOPs
                 = 8.80 TFlops / full-attention layer
```

平均到每层：

```text
F_attn_quad_avg = F_attn_quad_full / 4
                = 2.20 TFlops / layer
```

因此，计入 full attention 二次项后的平均单层计算量为：

```text
F_total_avg = F_gemm_avg + F_attn_quad_avg
            = 8.36 + 2.20
            = 10.56 TFlops / layer
```

结论：

```text
以 16k local sequence length 为 base，Qwen3.5-397B-A17B 平均每层 active GEMM 计算量约 8.36TFlops；如果计入 full attention 二次项，平均每层约 10.56TFlops。
```

## 7. 耗时预测公式

### 7.1 通信耗时

通信耗时使用：

```text
T_comm = Bytes_layer / BW_effective
```

为与用户给出的 Qwen3 235B 示例保持同一假设，使用示例反推的有效通信带宽：

```text
H800 effective BW = 4.6GB / 0.115s = 40.0 GB/s
A3 effective BW   = 4.6GB / 0.029s = 158.6 GB/s，近似取 159 GB/s
```

Qwen3.5 在 `EP16` 后单 rank FSDP AllGather 通信负载为 `1.064GB`，USP all-to-all 平均激活通信逻辑 payload 为 `0.708GB`，合计通信逻辑 payload 为 `1.772GB`。对应理论通信耗时为：

```text
T_fsdp_H800 = 1.064GB / 40.0GB/s
            = 27ms

T_usp_H800 = 0.708GB / 40.0GB/s
           = 18ms

T_comm_total_H800 = 1.772GB / 40.0GB/s
                  = 44ms

T_fsdp_A3 = 1.064GB / 159GB/s
          = 7ms

T_usp_A3 = 0.708GB / 159GB/s
         = 4ms

T_comm_total_A3 = 1.772GB / 159GB/s
                = 11ms
```

### 7.2 计算耗时

计算耗时使用：

```text
T_compute = FLOPs / (Peak_BF16 * MFU)
```

沿用用户示例中的 MFU 假设：

- H800 MFU：`0.3`
- A3 MFU：`0.4`

芯片 BF16 理论峰值按：

- H800：`989 TFLOPS`
- A3 单 die：`380 TFLOPS`

有效计算吞吐：

```text
H800 effective compute = 989 * 0.3 = 296.7 TFLOPS
A3 effective compute   = 380 * 0.4 = 152.0 TFLOPS
```

只计 GEMM 的计算耗时：

```text
T_compute_H800_gemm = 8.36T / 296.7T/s
                    = 0.0282s
                    = 28ms

T_compute_A3_gemm = 8.36T / 152.0T/s
                  = 0.0550s
                  = 55ms
```

计入 full attention 二次项后的计算耗时：

```text
T_compute_H800_total = 10.56T / 296.7T/s
                     = 0.0356s
                     = 36ms

T_compute_A3_total = 10.56T / 152.0T/s
                   = 0.0695s
                   = 69ms
```

## 8. 最终预测表

| 项目 | H800 | A3 超节点 |
|---|---:|---:|
| 1 层 FSDP 参数 AllGather，EP16 后单 rank 参数 | 27 ms | 7 ms |
| 1 层 USP all-to-all 激活通信，SP16 平均层 | 18 ms | 4 ms |
| 1 层总通信，FSDP + USP | 44 ms | 11 ms |
| 1 层 16k 序列计算，只计 active GEMM | 28 ms | 55 ms |
| 1 层 16k 序列计算，计入 full-attn 平均二次项 | 36 ms | 69 ms |

## 9. Profiling 校准与解释建议

用户提供的 profiling 汇总窗口为 12 层：

```text
12 layers = 3 * (3 linear + 1 full)
```

实测结果：

```text
计算总耗时：1102 ms
单次 full attention：约 82 ms
FSDP 通信 wall time：2503 ms，其中实际 memcpy：305 ms
all2all 通信 wall time：1297 ms，其中实际 memcpy：231 ms
  - EP all2all wall time：300 ms，其中实际 memcpy：207 ms
  - SP all2all wall time：990 ms，其中实际 memcpy：30 ms
```

### 9.1 12 层理论值

按本文 A3 理论模型：

```text
FSDP 参数通信：1.064GB / layer
USP all-to-all 源码 shape 粗估：0.708GB / layer
A3 有效通信带宽：159GB/s
```

12 层理论通信耗时，按源码 shape 粗估：

```text
T_fsdp_theory_12 = 12 * 1.064GB / 159GB/s
                 = 80.3 ms

T_usp_theory_12 = 12 * 0.708GB / 159GB/s
                = 53.4 ms

T_comm_theory_12 = 80.3 + 53.4
                 = 133.7 ms
```

注意：上述 `T_usp_theory_12` 是源码 tensor shape 的粗估，用来给出 SP attention all-to-all 的量级；它没有把 EP 前后 `permute / unpermute` 的 all2allv profiling 事件直接折算成理论通信量。EP all2allv 的 profiling size 需要先确认单位、per-peer 还是聚合、是否包含 pack/unpack 和 send/recv 两侧 memcpy，再单独建模。

EP all2all 的理论期望按 MoE dispatch/combine 的 remote send 下界估算：

```text
T_ep_all2all_theory_12
  = 12 * 2 * S_local * top_k * hidden_size * 2 bytes * (EP - 1) / EP / 159GB/s
  = 12 * 2 * 16,384 * 10 * 4096 * 2 * 15 / 16 / 159GB/s
  = 189.9 ms
```

12 层理论计算耗时：

```text
只计 active GEMM：12 * 55.0ms = 660.3 ms
计入 full-attn 二次项：12 * 69.5ms = 833.9 ms
```

full attention 的二次项单次理论耗时：

```text
T_full_attn_quad_A3 = 8.796TFlops / 152TFlops/s
                    = 57.9 ms
```

### 9.2 实测与理论对比

| 项目 | 理论耗时 | 实际 memcpy | 实际 wall time | memcpy / 理论 | wall / 理论 | 等待/调度耗时 | 等待占 wall |
|---|---:|---:|---:|---:|---:|---:|---:|
| FSDP 参数通信，12 层 | 80.3 ms | 305 ms | 2503 ms | 3.80x | 31.17x | 2198 ms | 87.8% |

all2all 拆成 EP 与 SP 分别看：

| 项目 | 理论耗时 | 实际 memcpy | 实际 wall time | memcpy / 理论 | wall / 理论 | wall / memcpy | 等待/调度耗时 | 等待占 wall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| EP all2all，12 层 | 189.9 ms | 207 ms | 300 ms | 1.09x | 1.58x | 1.45x | 93 ms | 31.0% |
| SP all2all，12 层 | 53.4 ms | 30 ms | 990 ms | 0.56x | 18.53x | 33.00x | 960 ms | 97.0% |
| EP + SP all2all，12 层 | 243.3 ms | 237 ms | 1290 ms | 0.97x | 5.30x | 5.44x | 1053 ms | 81.6% |

注：用户给出的 all2all 总数为 `1297ms(memcpy 231ms)`，EP/SP 拆分相加为 `1290ms(memcpy 237ms)`，存在少量 profiling 分类或四舍五入口径差异。下面的归因分析按 EP/SP 拆分项判断瓶颈。

通信合计按 FSDP + EP all2all + SP all2all 拆分：

| 项目 | 理论耗时 | 实际 memcpy | 实际 wall time | memcpy / 理论 | wall / 理论 | wall / memcpy | 等待/调度耗时 | 等待占 wall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FSDP + EP all2all + SP all2all，12 层 | 323.6 ms | 542 ms | 3793 ms | 1.68x | 11.72x | 7.00x | 3251 ms | 85.7% |

计算部分：

| 项目 | 理论耗时 | 实测耗时 | 实测 / 理论 |
|---|---:|---:|---:|
| 12 层计算，只计 active GEMM | 660.3 ms | 856 ms | 1.30x |
| 12 层计算，计入 full-attn 二次项 | 833.9 ms | 1102 ms | 1.32x |
| 单次 full attention 二次项 | 57.9 ms | 82 ms | 1.42x |

### 9.3 建议解释口径

这组 profiling 可以支持如下判断：

1. 计算侧相对理论值的差距在 `1.3x` 左右，属于 MFU、kernel 实现、linear attention kernel、重计算和调度开销共同造成的范围。只计 active GEMM 时，应从 12 层总计算中扣除 3 次 full attention，即 `1102 - 82 * 3 = 856ms`，约为理论的 `1.30x`；若采用“计入 full-attn 二次项”的总计算口径，12 层实测为理论的 `1.32x`。
2. FSDP 参数通信单独看，实际 memcpy 是理论的 `3.80x`，wall time 是理论的 `31.17x`，说明 FSDP 侧既有 memcpy/带宽差距，也有严重等待。
3. EP all2all 的 memcpy `207ms` 接近理论 `190ms`，wall time `300ms`，说明 EP all2all 主要差距不是 memcpy 带宽，而是约 `93ms` 的等待/调度。
4. SP all2all 的 memcpy 只有 `30ms`，但 wall time 达到 `990ms`，wall/memcpy 为 `33x`，等待占比约 `97%`。因此 SP all2all 的主要问题是快慢卡等待、collective 同步、SP 组内 straggler 或 overlap 不充分，而不是纯 memcpy。
5. 通信合计 wall time 相比 memcpy 是 `7.00x`，额外 `3251ms` 主要来自等待/同步，而不是纯 memcpy。

可以在汇报中写成：

```text
在 12 层 profiling 中，FSDP 参数通信理论约 80ms，实际 memcpy 为 305ms，约为理论的 3.8x；FSDP wall time 为 2503ms，约为理论的 31.2x。all2all 需要拆开看：EP all2all 理论约 190ms，memcpy 207ms，wall 300ms，基本接近理论；SP all2all 理论约 53ms，memcpy 30ms，但 wall 990ms，wall/memcpy 达到 33x，等待占比约 97%。因此通信 wall time 的主要膨胀来自 FSDP 等待和 SP all2all 等待，而不是 EP all2all memcpy 本身。

计算侧只计 active GEMM 时，实测应扣除 3 次 full attention，即 1102 - 82 * 3 = 856ms，对比理论 660ms，约为 1.30x。若计入 full attention 二次项，总计算理论为 834ms，12 层实测 1102ms，约为 1.32x。单次 full attention 实测约 82ms，对比二次项理论 58ms，约为 1.42x。
```

## 10. 可引用结论

Qwen3.5-397B-A17B 全局平均每层参数量约为 `6.57G params`。但在 `EP16` 下，512 个 routed experts 被切成每个 EP rank 只持有 `32` 个 experts，因此单 rank 每层 FSDP AllGather 参数量约为 `0.532G params`，bf16 参数通信负载约为 `1.064GB`。此外，代码中的 USP 会在 `SP16` 下引入 attention 激活 all-to-all，按源码 tensor shape 粗估平均每层逻辑 payload 约为 `0.708GB`。因此平均每层 `FSDP + USP` 通信逻辑 payload 约为 `1.772GB`。

profiling 中还观测到 EP 前后 `permute / unpermute` 的 all2allv，以及 linear attention SP 的细粒度 memcpy size。这些数据说明 all2all 实测中存在额外的 pack/unpack、per-peer chunk、send/recv 和等待同步开销；但在确认 profiler size 的单位和聚合口径前，不应把这些 memcpy size 直接当作网络理论通信量。

以 `256k_ep16_sp16` 并行、单 die `16k` 序列长度为 base，平均每层 active GEMM 计算量约为 `8.36TFlops`；如果计入每 4 层一次 full attention 的二次项，平均每层计算量约为 `10.56TFlops`。

假设通信带宽与 Qwen3 235B 示例一致，即 H800 有效通信带宽约 `40GB/s`、A3 超节点有效通信带宽约 `159GB/s`，并假设 H800 BF16 MFU 为 `0.3`、A3 BF16 MFU 为 `0.4`，则预测：

| 芯片 | 1 层 FSDP 参数通信 | 1 层 USP 激活通信 | 1 层 FSDP+USP 通信 | 1 层 16k 序列计算 |
|---|---:|---:|---:|---:|
| H800 | 27 ms | 18 ms | 44 ms | 28 ms GEMM / 36 ms 含 full-attn 二次项 |
| A3 超节点 | 7 ms | 4 ms | 11 ms | 55 ms GEMM / 69 ms 含 full-attn 二次项 |

## 11. 备注和不确定项

1. 本文通信负载按 `EP16` 后单 EP rank 的 FSDP 参数 AllGather 估算，即 routed experts 只计 `512 / 16 = 32` 个本地 experts。全局 `6.57G params/layer` 只用于说明模型层规模，不用于通信耗时。
2. 本文没有再扣除 FSDP 分片局部已持有参数，也没有加入 ring allgather 链路放大系数；这是为了与用户给出的 Qwen3 235B 示例口径尽量保持一致。
3. 日志中实际 `ep_size=8`，用户目标为 `ep16`。本文最终表按用户目标 `ep16_sp16` 的 local sequence 和参数通信口径计算；如果要精确模拟日志配置，需要按 `ep8_sp16` 重新分析专家切分和通信域。
4. profiling 中的 EP all2allv 和 linear attention SP memcpy size 暂作为实测解释材料，不直接作为最终理论通信量；需要确认 profiler size 是 bytes 还是 bf16 elements、是 per-peer chunk 还是聚合量、是否包含 pack/unpack 和 send/recv 两侧 memcpy。
5. full attention 二次项按标准 `QK^T + AV` 粗略估算，实际 FA kernel、mask、MTP、重计算、overlap、EP dispatch/combine 通信都会影响真实耗时。
6. 本文未计 optimizer、activation checkpoint、MoE token dispatch/combine 的计算开销、load balance、MTP 层和视觉塔计算，仅聚焦语言模型单 decoder layer 的参数 AllGather、USP attention 激活 all-to-all 和主计算。

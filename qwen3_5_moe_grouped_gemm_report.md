# Qwen3.5 MoE Routed Expert Grouped GEMM 使用报告

生成日期：2026-06-30

## 0. 结论摘要

`grouped_matmul` / NPU fused grouped GEMM 在 Qwen3.5 MoE 中的定位可以概括为：

```text
不是替换 attention；
不是替换 dense Qwen3.5 的普通 MLP；
而是在 Qwen3.5 MoE 的 MLP/FFN 子层中，
把 routed experts 的两次矩阵乘批量化为 grouped GEMM。
```

更精确地说：

1. Dense 版 `qwen3_5` 的 `Qwen3_5MLP` 仍然是普通 `gate_proj / up_proj / down_proj`。
2. MoE 版 `qwen3_5_moe` 的 decoder layer 中，`self.mlp` 是 `Qwen3_5MoeSparseMoeBlock`，这相当于 FFN/MLP 子层被 MoE block 承担。
3. `grouped_matmul` 只作用在 `SparseMoeBlock.experts` 这一路 routed experts 的两次线性层：
   - `hidden -> gate/up`
   - `intermediate -> hidden`
4. `shared_expert` 仍然是普通 MLP；router 的线性打分也不是 grouped GEMM。
5. 在 Qwen3.5-397B 配置中，`expert_parallel_size: 16` 且 `ep_plan.apply_modules` 指向 `model.language_model.layers.{*}.mlp.experts`，因此 production EP 路径会走 `mindspeed_mm.fsdp.ops.moe_ops.gemm.grouped_matmul`。

一句话判断：

```text
“MLP 部分替换”这个说法只对了一半。
架构层面，Qwen3.5 MoE 的 MLP 子层是 SparseMoeBlock；
算子层面，GMM 替换的是 routed expert 内部两次 Linear 的执行方式，
不是把整个 MLP 子层、shared expert 或 dense MLP 全部替换成 GMM。
```

## 1. 事实来源

本报告基于 MindSpeed-MM PR 2664 和 PR head 源码核验：

- PR diff：`https://gitcode.com/Ascend/MindSpeed-MM/pull/2664/diffs`
- PR head：`b55b02d1b389494a846ff8752b0c4556d86a2a95`
- 本地核验目录：`/tmp/mindspeed-mm-pr2664`

PR 2664 本身只新增了 `grouped_matmul` 单测，并放宽了 `unpermute` 单测容差；它没有直接改 Qwen3.5 模型代码。模型如何使用 GMM 需要沿主干已有源码追调用链。

关键源码位置：

| 模块 | 作用 |
| --- | --- |
| `mindspeed_mm/fsdp/ops/moe_ops/gemm.py` | 定义 `grouped_matmul`，NPU 上调用 `torch_npu.npu_grouped_matmul`，否则走 eager per-group matmul |
| `mindspeed_mm/fsdp/distributed/expert_parallel/ep_dispatcher.py` | EP dispatcher 中直接 `from mindspeed_mm.fsdp.ops.moe_ops.gemm import grouped_matmul`，并在 experts 的两次 GEMM 上调用 |
| `mindspeed_mm/fsdp/models/qwen3_5_moe/modeling_qwen3_5_moe.py` | 定义 Qwen3.5 MoE 的 router、experts、SparseMoeBlock 和 decoder layer |
| `mindspeed_mm/fsdp/distributed/expert_parallel/expert_parallel.py` | 将配置中匹配到的 experts 模块替换为 EP forward |
| `examples/qwen3_5/qwen3_5_397B_config.yaml` | 397B 配置中启用 `qwen3_5_moe`、EP16、experts EP plan 和 `use_grouped_expert_matmul` |

## 2. `grouped_matmul` 的语义

`grouped_matmul(x, weight, group_list)` 的抽象语义是：

```text
x:          [sum(group_list), input_dim]
weight:     [num_groups, input_dim, output_dim]
group_list: [m_0, m_1, ..., m_{num_groups-1}]

for expert i:
    y[start_i:end_i] = x[start_i:end_i] @ weight[i]
```

其中 `start_i/end_i` 由 `group_list` 前缀和决定。也就是说，所有 token 已经按 expert 连续排好，`group_list` 告诉算子每个 expert 对应多少行。

NPU fused 路径中，`GmmFunction.forward` 调用：

```python
torch_npu.npu_grouped_matmul(
    [x],
    [weight],
    bias=None,
    group_list=group_list,
    split_item=2,
    group_type=0,
    group_list_type=1,
)[0]
```

PR 2664 新增的单测验证了 fused 和 eager 两种实现的一致性：输入 `x` 是 `[batch_size, input_dim]`，权重是 `[num_experts, input_dim, output_dim]`，`group_list` 的和等于 batch size。

## 3. Qwen3.5 dense 版没有使用这条 GMM 路径

Dense 版 `mindspeed_mm/fsdp/models/qwen3_5/modeling_qwen3_5.py` 中，MLP 是普通三线性层：

```python
class Qwen3_5MLP(nn.Module):
    self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
    self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
    self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
```

`Qwen3_5DecoderLayer` 直接构造：

```python
self.mlp = Qwen3_5MLP(config, config.intermediate_size)
```

因此如果模型配置是 `model_id: qwen3_5`，不能说 `grouped_matmul` 替换了它的普通 MLP。

## 4. Qwen3.5 MoE 的 MLP 子层结构

MoE 版 `mindspeed_mm/fsdp/models/qwen3_5_moe/modeling_qwen3_5_moe.py` 中，decoder layer 的 FFN/MLP 子层是：

```python
self.mlp = Qwen3_5MoeSparseMoeBlock(config)
```

`Qwen3_5MoeSparseMoeBlock` 内部包含三块：

```text
gate                -> Qwen3_5MoeTopKRouter
experts             -> Qwen3_5MoeExperts
shared_expert       -> Qwen3_5MoeMLP
shared_expert_gate  -> Linear(hidden_size, 1)
```

forward 的主逻辑是：

```text
hidden_states
  -> flatten 为 [tokens, hidden]
  -> shared_expert(hidden_states)                      # 普通 MLP
  -> router 得到 selected_experts / routing_weights     # top-k 路由
  -> experts(hidden_states, selected_experts, weights)  # routed experts
  -> sigmoid(shared_expert_gate) * shared_expert_output
  -> routed expert 输出 + shared expert 输出
  -> reshape 回 [batch, seq, hidden]
```

所以，MoE 版里“MLP 子层”不是单个 dense MLP，而是 `SparseMoeBlock = routed experts + shared expert + router/gate`。

## 5. Routed experts 内部的两次矩阵乘

Qwen3.5 MoE experts 的参数以 3D tensor 保存：

```python
self.gate_up_proj = nn.Parameter(
    torch.empty(num_experts, hidden_dim, 2 * intermediate_dim)
)
self.down_proj = nn.Parameter(
    torch.empty(num_experts, intermediate_dim, hidden_dim)
)
```

对一个 token-expert assignment 来说，expert FFN 等价于：

```text
gate_up = hidden @ W_gate_up[expert]       # [H] -> [2I]
act     = SwiGLU(gate_up)                  # [2I] -> [I]
out     = act @ W_down[expert]             # [I] -> [H]
```

`grouped_matmul` 做的事情就是把很多 expert 的这些小批量矩阵乘合到一个 grouped GEMM 调用中：

```text
permuted_hidden_states
  -> grouped_matmul(..., gate_up_proj, tokens_per_expert)
  -> swiglu
  -> grouped_matmul(..., down_proj, tokens_per_expert)
  -> unpermute(..., probs=routing_weights)
```

这不改变数学计算量，只改变执行组织方式：减少 Python per-expert loop 和 kernel launch，提升 NPU 上 grouped expert GEMM 的利用率。

## 6. EP16 下的实际调用链

Qwen3.5-397B 配置文件 `examples/qwen3_5/qwen3_5_397B_config.yaml` 的关键配置是：

```yaml
parallel:
  expert_parallel_size: 16
  ep_plan:
    apply_modules:
      - model.language_model.layers.{*}.mlp.experts
    dispatcher: alltoall

model:
  model_id: qwen3_5_moe
  use_grouped_expert_matmul: true
```

此外，`EPPlanConfig.use_npu_fused_ops` 默认是 `true`。该 yaml 没有覆盖它，因此 EP dispatcher 默认以 fused=True 运行。

实际 production EP 调用链是：

```text
TorchParallelize.apply_ep_modules
  -> expert_parallelize_modules
    -> 匹配 model.language_model.layers.{*}.mlp.experts
    -> distribute_experts_module
    -> module.forward = partial(module.ep_forward, ep_group, ep_plan)

Qwen3_5MoeExperts.ep_forward
  -> ep_dispatcher.ep_forward / ep_mc2_forward / ep_allgather_forward

ep_dispatcher.ep_forward
  -> dispatch_preprocess 统计 token-expert 分布
  -> permute + all_to_all dispatch
  -> grouped_matmul(hidden, fc1_weight, group_list)
  -> swiglu / clamp_swiglu
  -> grouped_matmul(activation, fc2_weight, group_list)
  -> unpermute + all_to_all combine
```

这里的 `ep_dispatcher.py` 顶部正是：

```python
from mindspeed_mm.fsdp.ops.moe_ops.gemm import grouped_matmul
```

因此对 Qwen3.5-397B 的 EP16 配置来说，用户提到的这个 `grouped_matmul` 确实在 production expert forward 中使用。

## 7. 以 Qwen3.5-397B-A17B 的形状理解

按本仓库既有分析口径，Qwen3.5-397B-A17B text backbone 的关键 MoE 参数是：

```text
H = hidden_size = 4096
E = num_experts = 512
K = num_experts_per_tok = 10
I = moe_intermediate_size = 1024
EP = 16
```

EP16 下每个 rank 持有：

```text
E_local = E / EP = 512 / 16 = 32 experts
```

因此 local routed expert GEMM 的典型权重形状是：

```text
fc1 / gate_up_proj: [32, 4096, 2048]
fc2 / down_proj:    [32, 1024, 4096]
```

若当前 rank 在 all-to-all dispatch 后收到的本地 token-expert assignments 总数为 `T_local`，则两次 grouped GEMM 的形状是：

```text
第一次:
  input  [T_local, 4096]
  weight [32, 4096, 2048]
  output [T_local, 2048]

SwiGLU:
  [T_local, 2048] -> [T_local, 1024]

第二次:
  input  [T_local, 1024]
  weight [32, 1024, 4096]
  output [T_local, 4096]
```

`group_list` 长度是 local expert 数 `32`，每个元素是当前 rank 上该 local expert 收到的 token-expert 行数。由于 router top-k 和负载不均衡，每个 expert 的 group size 通常不相等。

## 8. 非 EP 路径的补充说明

`Qwen3_5MoeExperts.forward` 还有一个非 EP 或直接 forward 的 NPU fused 分支：

```text
if self.use_grouped_expert_matmul and IS_NPU_AVAILABLE:
    torch_npu.npu_moe_token_permute
    npu_group_gemm(..., gate_up_proj, tokens_per_expert)
    torch_npu.npu_swiglu
    npu_group_gemm(..., down_proj, tokens_per_expert)
    torch_npu.npu_moe_token_unpermute
```

这里调用的是 `mindspeed_mm.models.common.gmm.npu_group_gemm`，不是 `mindspeed_mm.fsdp.ops.moe_ops.gemm.grouped_matmul` 这个 wrapper 名字，但底层同样是 `torch_npu.npu_grouped_matmul`。

也就是说：

```text
EP 路径：    使用 fsdp.ops.moe_ops.gemm.grouped_matmul
非 EP fused：使用 models.common.gmm.npu_group_gemm
底层语义：   都是按 expert 分组的 NPU grouped matmul
```

对 397B 的 EP16 配置，重点是前者。

## 9. 对性能分析的含义

Grouped GEMM 的收益主要来自执行效率，而不是减少理论 FLOPs。

对每个 routed token-expert assignment，expert FFN 的主 FLOPs 近似为：

```text
FC1: H * 2I MAC
FC2: I * H  MAC
合计: 3 * H * I MAC
按 2 FLOPs / MAC 计：6 * H * I FLOPs
```

Qwen3.5-397B 中每个 token 选择 `K = 10` 个 routed experts，因此 routed experts 的理论计算量仍然与 `tokens * K` 成正比。GMM 不会把 `K=10` 的算术量变少，它只是把很多不同 expert 的小/中型 GEMM 批量交给 NPU grouped GEMM。

对 EP16 而言，GMM 位于：

```text
all-to-all dispatch 之后
all-to-all combine 之前
```

因此性能瓶颈通常由三部分共同决定：

1. EP all-to-all / all2allv 通信耗时。
2. token-expert 负载不均衡导致的最大 group/rank 拖尾。
3. grouped GEMM 本身在不同 group size 下的 NPU kernel 效率。

这也解释了为什么评估 MoE 层时不能只看总 FLOPs，还需要看 `tokens_per_expert` 的分布和 EP group 内最慢 rank。

## 10. 边界结论

可以这样表述：

```text
Qwen3.5-397B 的 MoE FFN/MLP 子层中，routed expert 计算会先按 router top-k
把 token 分组到 expert，再用 grouped matmul / NPU grouped GEMM 批量完成
expert 的 gate_up 和 down 两次矩阵乘。它是 expert 内部 Linear 执行方式的
融合/批量化，不是改变 MoE 数学结构，也不是替换 dense Qwen3.5 的普通 MLP。
```

推荐在后续性能报告中使用更精确的术语：

```text
“routed expert FFN 的两次 Linear 使用 grouped GEMM 批量执行”
```

不推荐笼统写成：

```text
“MLP 被 GMM 替换”
```

因为这会混淆 dense MLP、shared expert、router 和 routed experts 四个不同概念。

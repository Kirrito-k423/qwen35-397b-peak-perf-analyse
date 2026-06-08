# Qwen3.5 397B Peak Performance Analysis

This repository contains a theoretical communication and compute bound analysis for Qwen3.5-397B-A17B under large-scale Ascend A3 / H800 training assumptions.

Main document:

- [Qwen3.5-397B-A17B 单层通信与计算耗时推导](./qwen3_5_397b_layer_comm_compute_analysis.md)

The analysis covers:

- per-layer parameter decomposition
- EP16 FSDP parameter communication
- SP16 / USP all-to-all activation communication
- EP permute / unpermute all2allv communication
- active MoE compute FLOPs
- full attention quadratic FLOPs
- profiling-based calibration against measured compute and communication timings

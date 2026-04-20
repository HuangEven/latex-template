# 多 GPU 实验设置与结果汇总

## 表 5-8 多 GPU 实验设置

| GPU数 | 方案 | 数据规模 | 索引组织 | 批大小 | 候选数量 | k值 | 归并路径 | I/O 绑定 | 备注 |
| --- | --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |
| 1 | Replicated | 10M | Full replica | 4096 | 128 | 50 | None | Topology-aware | 单卡吞吐扩展基线 |
| 2 | Replicated | 10M | Full replica | 4096 | 128 | 50 | None | Topology-aware | 双卡 replicated 扩展基线 |
| 4 | Replicated | 10M | Full replica | 4096 | 128 | 50 | None | Topology-aware | 四卡 replicated 扩展基线 |
| 4 | Local scoring + GPU merge | 100M | Sharded index | 3072 | 128 | 50 | NCCL all-gather + GPU merge | Topology-aware | 显存受限时的 GPU 侧归并兜底方案 |
| 4 | Local scoring + CPU aggregation | 100M | Sharded index | 3072 | 128 | 50 | CPU aggregation | Topology-aware | CPU 聚合对比基线 |
| 2 | Topology-aware binding | 100M | Full replica | 4096 | 128 | 50 | None | Topology-aware | I/O lane 绑定对比实验 |
| 4 | Topology-aware binding | 100M | Full replica | 4096 | 128 | 50 | None | Topology-aware | I/O lane 绑定对比实验 |
| 2 | Shared-path conflict | 100M | Full replica | 4096 | 128 | 50 | None | Shared PCIe/SSD path | 共享路径冲突对比基线 |
| 4 | Shared-path conflict | 100M | Full replica | 4096 | 128 | 50 | None | Shared PCIe/SSD path | 共享路径冲突对比基线 |

## 表 5-9 多 GPU 实验结果汇总

| GPU数 | 方案 | 数据规模 | QPS | avg_latency_ms | p99_latency_ms | CPU利用率 (%) | GPU利用率 (%) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | Replicated | 10M | 4550.00 | 22.80 | 34.80 | 47.00 | 63.50 |
| 2 | Replicated | 10M | 7780.00 | 21.00 | 31.90 | 42.00 | 68.80 |
| 4 | Replicated | 10M | 13620.00 | 19.80 | 30.20 | 37.80 | 73.50 |
| 4 | Local scoring + GPU merge | 100M | 11380.00 | 21.50 | 33.80 | 41.50 | 69.80 |
| 4 | Local scoring + CPU aggregation | 100M | 9580.00 | 24.60 | 39.00 | 48.50 | 62.50 |
| 2 | Topology-aware binding | 100M | 7520.00 | 23.40 | 37.20 | 43.80 | 67.50 |
| 4 | Topology-aware binding | 100M | 12650.00 | 22.10 | 45.10 | 39.80 | 72.20 |
| 2 | Shared-path conflict | 100M | 6830.00 | 25.10 | 41.40 | 45.80 | 61.80 |
| 4 | Shared-path conflict | 100M | 10420.00 | 28.90 | 56.60 | 44.50 | 56.50 |

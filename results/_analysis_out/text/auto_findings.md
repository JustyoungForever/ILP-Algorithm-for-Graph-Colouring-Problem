# Auto findings draft

- 覆盖实例数（summary层面）: **10**
- 覆盖算法数: **3**

## 解质量胜场（按 UB_mean 最小计）
- algo=dsatur, ablation=baseline, TL=300: win_count=5
- algo=slo, ablation=baseline, TL=300: win_count=5
- algo=iterlp2_full, ablation=full, TL=300: win_count=5
- algo=dsatur, ablation=baseline, TL=3: win_count=3
- algo=iterlp2_full, ablation=full, TL=5: win_count=3
- algo=slo, ablation=baseline, TL=5: win_count=3
- algo=dsatur, ablation=baseline, TL=10: win_count=3
- algo=slo, ablation=baseline, TL=10: win_count=3
- algo=dsatur, ablation=baseline, TL=5: win_count=3
- algo=iterlp2_full, ablation=full, TL=10: win_count=3

## stop_reason 主要分布（每 algo-ablation Top2）
- dsatur | baseline: baseline(1.00)
- iterlp2_full | full: time_limit(0.90), stalled_at_K_eq_ceil_zLP(0.10)
- slo | baseline: baseline(1.00)

## IterLP 改进发生时机（粗略）
- best_time_mean_sec≈0 的比例（表示最好 UB 多半来自初始化）: **0.94**
- 注：该指标需结合具体实例族进一步解读（高密度图更可能出现后期改进）。

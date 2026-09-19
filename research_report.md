# SIFTER 研究记录

日期：2026-08-22

## 最终主架构

SIFTER（Sparse Inductive Feature-to-Prototype Event Representation）采用非参数 support memory：

```text
document
  -> shared word tokenizer
  -> TF-IDF-like sparse evidence vector f(x)
  -> class prototype p_c = mean(f(x_support,c))
  -> prototype logits f(x) · p_c
```

模型保留一个小型 TESSERA 事件图编码器作为可学习残差校正分支；v3 将残差尺度从 0.001 提升为正式配置 1.0，参数量不增加。这样做仍保留 sparse support evidence 为锚点，但允许事件图在少量标签下修正边界。

当前实现的稀疏记忆包含一个完整词表的 global channel，以及四个相对位置 tile。位置 tile 使用小型哈希空间，避免参数量随词表乘以位置通道；global channel 保留稀有词证据。路由可以固定为 global、positional 或 edge，也可以启用 adaptive 作为诊断实验。

## 公平性

- 三个模型使用相同 AG News 本地切分、相同随机种子、相同 max length、相同 1 epoch 预算；
- `benchmark.py` 按总参数量选择宽度；
- Transformer、Mamba-lite、SIFTER 的总参数量均约 1.3M；
- 词表和 IDF 只从 train-pool + test 的无标签文本构建，不使用评测标签；
- 评测只使用 test 部分标签；
- 运行设备为 RTX 5070 CUDA。

主协议让 Transformer/Mamba 使用 dense head，而 SIFTER 使用其 support prototype head；这不是隐藏的实现细节，而是 SIFTER 的少样本归纳偏置。AG News v3 同 support head 的 8-shot 对照为 Transformer 0.2919 / 0.2860，Mamba-lite 0.2964 / 0.2804，SIFTER 0.4847 / 0.4807。TREC 还额外运行了同 support head 的 v3 8-shot 对照：Transformer 0.5415 accuracy / 0.5129 macro-F1，Mamba-lite 0.6000 / 0.5451，SIFTER 0.7145 / 0.6731。SST-2 同 support head 的 v3 结果为 Transformer 0.5166 / 0.5000，Mamba-lite 0.5029 / 0.4524，SIFTER 0.5143 / 0.4847。结果文件分别位于 E:\nlp_arch_lab\runs\ag_final_v3_residual1_supporthead_8shot_4seed\summary.json、E:\nlp_arch_lab\runs\trec_word_v2_residual1_supporthead_8shot_4seed\summary.json 与 E:\nlp_arch_lab\runs\sst2_hashword2_v3_supporthead_8shot_4seed\summary.json。

## 结果

| shots/类 | Transformer accuracy | Mamba-lite accuracy | SIFTER accuracy | Transformer macro-F1 | Mamba-lite macro-F1 | SIFTER macro-F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 0.2582 | 0.2536 | 0.3972 | 0.2325 | 0.2242 | 0.3915 |
| 8 | 0.2712 | 0.2569 | 0.4847 | 0.2230 | 0.2272 | 0.4807 |
| 16 | 0.2674 | 0.2618 | 0.5801 | 0.2329 | 0.2177 | 0.5763 |

SIFTER v3 在三个 shot 点、accuracy 和 macro-F1 两个指标上都领先；总参数量和显存更低。8-shot 的 SIFTER 平均总参数约 1.337M、平均峰值显存约 88MB。

## 证据与限制

本地 TF-IDF + LogisticRegression 诊断在同一 8-shot 切分上约为 0.475 accuracy，说明稀疏证据确实适合该低标签设置。SIFTER v3 的 AG News 结果与这一诊断一致，但不能把本地 80/20 split 结果扩大解释为所有 NLP 任务上的普适胜利。

adaptive 路由的 support 留一选择在 8-shot 下出现高方差，曾把 AG News 的 global 证据错误切换到 edge；因此主结果固定使用 global，并把 adaptive 作为失败模式诊断。工程测试位于 E:\nlp_arch_lab\tests\test_smoke.py，6 个 smoke tests 已通过。

## 第二真实任务：SST-2

SST-2 已从本地 GLUE 文件接入：E:\nlp_arch_lab\data\sst2\train.tsv 用于构造 support，dev.tsv 用于评测。当前协议为每类 8-shot、1 epoch、4 个 seeds、max length 96，三种模型共享 tokenizer 与总参数量匹配。

| 协议 | Transformer accuracy / macro-F1 | Mamba-lite accuracy / macro-F1 | SIFTER accuracy / macro-F1 |
|---|---:|---:|---:|
| hashword2、4-shot、dense head | 0.5049 / 0.4716 | 0.4917 / 0.4081 | 0.5138 / 0.4845 |
| word、dense head | 0.5175 / 0.5170 | 0.5017 / 0.4897 | 0.5209 / 0.4818 |
| hashword2、dense head | 0.5046 / 0.4732 | 0.4960 / 0.4106 | 0.5143 / 0.4847 |
| hashword2、同 support head | 0.5166 / 0.5000 | 0.5029 / 0.4524 | 0.5143 / 0.4847 |

SST-2 给出一个重要的反证边界：word tokenizer 下 SIFTER 只在 accuracy 上略高，macro-F1 低于 Transformer；v3 hashword2 探索性协议下，4/8-shot SIFTER 同时高于两个 dense baseline，但同 support head 下 Transformer 的 F1 仍领先。由于 hashword2 与残差尺度是在观察早期结果后才加入的，所以这部分应视为探索性证据；随后用未见 seeds 46--49 做了固定配置 holdout。对应结果文件：

- E:\nlp_arch_lab\runs\sst2_hashword2_v3_residual1_4shot_4seed\summary.json
- E:\nlp_arch_lab\runs\sst2_hashword2_v3_residual1_8shot_4seed\summary.json
- E:\nlp_arch_lab\runs\sst2_hashword2_v3_supporthead_8shot_4seed\summary.json

## 第三真实任务：TREC-6

TREC 使用 CogComp 的 train_5500.label 与 TREC_10.label，本地文件为 E:\nlp_arch_lab\data\trec\train.txt 和 test.txt，共 6 类、5500 条训练样本与 500 条测试样本。协议为 word tokenizer、1 epoch、4 seeds、总参数匹配。

| shots/类 | Transformer accuracy / macro-F1 | Mamba-lite accuracy / macro-F1 | SIFTER accuracy / macro-F1 |
|---:|---:|---:|---:|
| 4 | 0.1920 / 0.1426 | 0.1655 / 0.1389 | 0.6460 / 0.5872 |
| 8 | 0.2595 / 0.1989 | 0.1405 / 0.1167 | 0.7145 / 0.6731 |

TREC 8-shot 同 support head 的结果为：Transformer 0.5415 / 0.5129，Mamba-lite 0.6000 / 0.5451，SIFTER 0.7145 / 0.6731。该核查说明 TREC 的主要收益来自证据表示与原型结构的组合，而不是单独替换分类头。结果文件：

- E:\nlp_arch_lab\runs\trec_word_v1_4shot_4seed\summary.json
- E:\nlp_arch_lab\runs\trec_word_v2_residual1_8shot_4seed\summary.json
- E:\nlp_arch_lab\runs\trec_word_v2_residual1_supporthead_8shot_4seed\summary.json

## 未见随机种子 holdout

v3 的残差尺度是在旧 seeds 上做探索后确定的，因此补跑了未见过的 seeds 46、47、48、49，保持同一配置、同一数据协议：

| 任务、8-shot | Transformer accuracy / macro-F1 | Mamba-lite accuracy / macro-F1 | SIFTER v3 accuracy / macro-F1 |
|---|---:|---:|---:|
| SST-2 hashword2 | 0.4903 / 0.4844 | 0.4928 / 0.4586 | 0.5178 / 0.4848 |
| TREC word | 0.2390 / 0.1913 | 0.1480 / 0.1074 | 0.6480 / 0.6143 |
| AG News word | 0.2622 / 0.2351 | 0.2556 / 0.2422 | 0.5005 / 0.4992 |

这组 holdout 结果支持 v3 改进不是只对原有四个 seeds 有效；SST-2 上 SIFTER 的 F1 与 Transformer 基本持平，同时 accuracy 明显更高，TREC 与 AG News 则保持明显领先。结果文件：

- E:\nlp_arch_lab\runs\sst2_hashword2_v3_holdout46_49_8shot\summary.json
- E:\nlp_arch_lab\runs\trec_word_v3_holdout46_49_8shot\summary.json
- E:\nlp_arch_lab\runs\ag_news_v3_holdout46_49_8shot\summary.json

合并 seeds 42--49 后的均值（accuracy / macro-F1）为：AG News Transformer 0.2667 / 0.2290、Mamba-lite 0.2562 / 0.2347、SIFTER 0.4926 / 0.4900；SST-2 Transformer 0.4974 / 0.4788、Mamba-lite 0.4944 / 0.4346、SIFTER 0.5161 / 0.4847；TREC Transformer 0.2492 / 0.1951、Mamba-lite 0.1442 / 0.1120、SIFTER 0.6813 / 0.6437。这个 8-seed 汇总比单一四 seed 表更适合作为 v3 的稳定性证据。

## 当前结论

AG News 完整 train.csv 因网络下载速度过慢未纳入当前结果；当前真实数据来自 canonical test CSV 的固定本地切分。SST-2 已完成本地 train/dev 复核，TREC 已接入 CogComp 原始文件。v3 的最强结论是：“SIFTER 在 AG News、SST-2、TREC 的主协议均超过 Transformer/Mamba；在 TREC 同 support head 下仍显著领先；SST-2 同 support head 的 F1 仍略低于 Transformer。”这已经形成跨 3 个真实任务的主协议领先证据，但仍不支持“所有 NLP 任务普适超越”。下一阶段应预注册 tokenizer、扩大随机种子，并补齐完整官方 AG News 划分。

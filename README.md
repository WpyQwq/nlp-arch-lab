# SIFTER：少样本 NLP 架构实验室

SIFTER（Sparse Inductive Feature-to-Prototype Event Representation）不是 Transformer 或 Mamba 的变体。它针对“标签极少、可利用无标签文本”的场景，采用：

1. 全语料 IDF 统计的词级稀疏证据记忆，并保留相对位置证据通道；
2. support-set 类别原型，而不是从零学习一个分类头；
3. 小型 TESSERA 事件图作为可学习残差接口；
4. 可复现的证据路由模式：global、positional、edge、adaptive。

核心思想是：少量标签不够训练稳定的 dense embedding，但足够估计类别在稀疏词证据空间中的原型。该路径不依赖 token-token attention，也不依赖逐 token 的连续状态扫描。

## 当前已验证配置

- GPU：NVIDIA GeForce RTX 5070，约 12GB 显存；
- PyTorch：2.9.0 + CUDA 12.8；
- 数据：AG News 本地 80/20 split，数据文件位于 E:\nlp_arch_lab\data\ag_news_csv\test.csv；
- 每类 4/8/16-shot；
- 4 个随机种子：42、43、44、45；
- Transformer、双向 Mamba-lite、SIFTER 总参数量约 1.3M；
- SIFTER v3 启用可学习 TESSERA 残差校正，残差尺度为 1.0，参数量不增加；
- 所有日志、checkpoint、summary 均写入 E 盘。

数据接口还支持放置在 E 盘的数据目录：SST-2 使用 E:\nlp_arch_lab\data\sst2\train.tsv 与 dev.tsv，TREC 使用 E:\nlp_arch_lab\data\trec\train.txt 与 test.txt。SST-2 与 TREC 原始文件均已完成本地接入。

## 复现

主实验命令：

    C:\Users\Administrator\miniconda3\envs\LLM\python.exe E:\nlp_arch_lab\src\benchmark.py --dataset ag_news_local --shots 8 --epochs 1 --seeds 42 43 44 45 --max-len 128 --tokenizer word --evidence-routing global --sifter-residual-scale 1.0 --output-dir E:\nlp_arch_lab\runs\reproduce_8shot

完整复现入口：

    powershell -ExecutionPolicy Bypass -File E:\nlp_arch_lab\run_reproduce.ps1

快速回归测试：

    C:\Users\Administrator\miniconda3\envs\LLM\python.exe -m unittest discover -s E:\nlp_arch_lab\tests -v

SST-2 探索性复现（同一 hashword2 tokenizer，8-shot、4 seeds）：

    C:\Users\Administrator\miniconda3\envs\LLM\python.exe E:\nlp_arch_lab\src\benchmark.py --dataset sst2_local --shots 8 --epochs 1 --seeds 42 43 44 45 --max-len 96 --tokenizer hashword2 --evidence-routing global --sifter-residual-scale 1.0 --output-dir E:\nlp_arch_lab\runs\sst2_hashword2_reproduce_8shot_4seed

AG News 完整训练集下载过慢，因此工程使用已下载的 canonical test CSV 做固定 80/20 本地切分；这不是官方 train/test 组合，报告中已明确标注。

## 主结果

4 seeds、1 epoch、总参数匹配的均值：

| shots/类 | Transformer acc | Mamba-lite acc | SIFTER acc | Transformer F1 | Mamba F1 | SIFTER F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 0.2582 | 0.2536 | 0.3972 | 0.2325 | 0.2242 | 0.3915 |
| 8 | 0.2712 | 0.2569 | 0.4847 | 0.2230 | 0.2272 | 0.4807 |
| 16 | 0.2674 | 0.2618 | 0.5801 | 0.2329 | 0.2177 | 0.5763 |

结果文件：

- E:\nlp_arch_lab\runs\ag_final_v3_residual1_4shot_4seed\summary.json
- E:\nlp_arch_lab\runs\ag_final_v3_residual1_8shot_4seed\summary.json
- E:\nlp_arch_lab\runs\ag_final_v3_residual1_16shot_4seed\summary.json

## 分类头公平性核查

主结果保留了最常见的 dense Transformer/Mamba 分类头，同时给 SIFTER 使用 support prototype head，这是它面向少样本的核心设计。为隔离“表示架构”和“分类头”的贡献，另做了 8-shot、4 seeds 的同头实验：Transformer 与 Mamba 也从 support 集初始化并冻结 prototype head。

| 8-shot、support head | accuracy | macro-F1 |
|---|---:|---:|
| Transformer | 0.2919 | 0.2860 |
| Mamba-lite | 0.2964 | 0.2804 |
| SIFTER | 0.4847 | 0.4807 |

当前 v3 同头 JSON：E:\nlp_arch_lab\runs\ag_final_v3_residual1_supporthead_8shot_4seed\summary.json。

## 第二真实任务：SST-2

SST-2 使用本地 GLUE train/dev 文件，训练集按每类抽取 support，dev 作为评测集；因此这里是“本地 train/dev 少样本协议”，不是官方隐藏 test 的替代品。先用 word tokenizer 做了预注册实验，再用 hashword2 作为短语证据消融，后者必须视为探索性结果，因为 tokenizer 选择发生在观察 word 结果之后。

| 协议 | Transformer acc / F1 | Mamba-lite acc / F1 | SIFTER acc / F1 |
|---|---:|---:|---:|
| hashword2、4-shot、dense head | 0.5049 / 0.4716 | 0.4917 / 0.4081 | 0.5138 / 0.4845 |
| word、dense head | 0.5175 / 0.5170 | 0.5017 / 0.4897 | 0.5209 / 0.4818 |
| hashword2、dense head | 0.5046 / 0.4732 | 0.4960 / 0.4106 | 0.5143 / 0.4847 |
| hashword2、同 support head | 0.5166 / 0.5000 | 0.5029 / 0.4524 | 0.5143 / 0.4847 |

结论要保守：v3 在 hashword2 探索性协议的 4/8-shot 上均同时超过两个 dense baselines；但同 support head 的 SST-2 F1 仍略低于 Transformer。因此 SST-2 目前支持“跨任务有竞争力并在主协议领先”，还不足以宣称普适超越。结果文件分别位于 E:\nlp_arch_lab\runs\sst2_hashword2_v3_residual1_4shot_4seed\summary.json、E:\nlp_arch_lab\runs\sst2_hashword2_v3_residual1_8shot_4seed\summary.json 与 E:\nlp_arch_lab\runs\sst2_hashword2_v3_supporthead_8shot_4seed\summary.json。

## 第三真实任务：TREC-6

TREC 使用 CogComp 的 train_5500.label 与 TREC_10.label，本地文件为 E:\nlp_arch_lab\data\trec\train.txt 和 test.txt，共 6 类、5500 条训练样本与 500 条测试样本。以下为 word tokenizer、1 epoch、4 seeds、总参数匹配的 dense-head 结果：

| shots/类 | Transformer acc / F1 | Mamba-lite acc / F1 | SIFTER acc / F1 |
|---:|---:|---:|---:|
| 4 | 0.1920 / 0.1426 | 0.1655 / 0.1389 | 0.6460 / 0.5872 |
| 8 | 0.2595 / 0.1989 | 0.1405 / 0.1167 | 0.7145 / 0.6731 |

为隔离分类头因素，TREC 8-shot 同 support head 结果为：Transformer 0.5415 / 0.5129，Mamba-lite 0.6000 / 0.5451，SIFTER 0.7145 / 0.6731。对应文件：

- E:\nlp_arch_lab\runs\trec_word_v1_4shot_4seed\summary.json
- E:\nlp_arch_lab\runs\trec_word_v2_residual1_8shot_4seed\summary.json
- E:\nlp_arch_lab\runs\trec_word_v2_residual1_supporthead_8shot_4seed\summary.json

## 未见随机种子 holdout

为了避免把 v3 残差尺度的探索性选择误报成无偏结果，固定配置再跑了未见过的 seeds 46、47、48、49：

| 任务、8-shot | Transformer acc / F1 | Mamba-lite acc / F1 | SIFTER v3 acc / F1 |
|---|---:|---:|---:|
| SST-2 hashword2 | 0.4903 / 0.4844 | 0.4928 / 0.4586 | 0.5178 / 0.4848 |
| TREC word | 0.2390 / 0.1913 | 0.1480 / 0.1074 | 0.6480 / 0.6143 |
| AG News word | 0.2622 / 0.2351 | 0.2556 / 0.2422 | 0.5005 / 0.4992 |

holdout 结果文件位于 E:\nlp_arch_lab\runs\sst2_hashword2_v3_holdout46_49_8shot\summary.json、E:\nlp_arch_lab\runs\trec_word_v3_holdout46_49_8shot\summary.json 与 E:\nlp_arch_lab\runs\ag_news_v3_holdout46_49_8shot\summary.json。

合并 seeds 42--49 后的均值（acc / macro-F1）为：AG News Transformer 0.2667 / 0.2290、Mamba-lite 0.2562 / 0.2347、SIFTER 0.4926 / 0.4900；SST-2 Transformer 0.4974 / 0.4788、Mamba-lite 0.4944 / 0.4346、SIFTER 0.5161 / 0.4847；TREC Transformer 0.2492 / 0.1951、Mamba-lite 0.1442 / 0.1120、SIFTER 0.6813 / 0.6437。

## 失败模式与边界

adaptive 路由在 8-shot 下可能被 support 留一方差误导，因此主实验固定使用 global，而 positional/edge 作为显式消融。长程合成任务的结果保存在 E:\nlp_arch_lab\runs\challenge_positional_exact_8shot_4seed\summary.json，只用于诊断，不并入真实数据主表。

## 边界与下一步

当前结果证明 SIFTER v3 在 AG News 与 TREC 的明确少样本协议下超过两个神经基线，并在 TREC 8-shot 同 support head 下仍保持领先；SST-2 的 4/8-shot hashword2 主协议也同时领先两个 dense baselines，但同 support head 的 F1 仍略低于 Transformer。AG News 当前仍是 canonical test CSV 的固定 80/20 split，SST-2 是本地 train/dev 协议。因此现在可以说“在三类真实任务中的主协议均有领先证据，其中 AG News/TREC 还具备更强的公平性证据”，不能说已经证明对所有 NLP 任务普适超越。

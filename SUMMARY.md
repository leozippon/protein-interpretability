# InterpretabilityTransfer 研究总结

**状态：** 官方进度总览。对科学结论而言仍是支持性汇总，不是权威来源；结论以权威审计为准，本文负责让当前进度一眼可读，并在新结果产出时同步更新。

**更新：** 2026-08-07

**权威来源：** [`docs/INTERPRETABILITY_TRANSFER_AUDIT.md`](docs/INTERPRETABILITY_TRANSFER_AUDIT.md)

本文从最终结论和核心实验两个层次概括本研究。若本文与权威审计或后续实验记录不一致，以权威审计和 [`docs/EXPERIMENT_LOG.md`](docs/EXPERIMENT_LOG.md) 为准。实验日志和项目日志中带章节号的旧 `check.md` 引用属于已经被权威审计取代的历史内容，不指向本文。

“测量成功”表示实验能够稳定地区分所定义的量；“方法成功”还要求解释对象通过因果检查；“完全成功”还要求跨模型、数据和蛋白家族泛化，并排除架构、数据、tokenization 和评价接口等混淆。目前没有方法达到最后一级。“冒烟”只证明接口能够端到端运行，不是科学结果。

## 研究结论总览

| 结论编号 | Research Objective | 核心主题 | 本研究取得的最终结论 | 结论状态 | 尚未解决的问题 |
|---|---|---|---|---|---|
| R1.1 | 1. 比较模型家族 | 整体可解释性难度 | 目前不能笼统声称“蛋白模型比语言模型更难解释”。差异主要出现在特定 selector、评价接口和数据构造上，而不是所有方法普遍失效。 | **已确认** | 需要更多匹配模型验证适用范围。 |
| R1.2 | 1. 比较模型家族 | 计算路径组织 | 文本模型整体更偏向 MLP 路径；已测蛋白模型相对更依赖 attention 或更加均衡。这是目前最有价值的候选计算差异。 | **较强候选** | tokenization 和独立蛋白家族控制尚未完成。 |
| R1.3 | 1. 比较模型家族 | Attention 与 induction | 蛋白模型中存在具有真实因果作用的 induction heads；“蛋白模型 induction heads 更少或更弱”不是可靠结论。 | **已确认其存在；模态差异未确认** | 需要自然重复和 family-disjoint 验证。 |
| R1.4 | 1. 比较模型家族 | 输出语义接口 | 逐残基蛋白模型的小词表形成低秩输出 aperture，限制 Logit Lens、DLA 等方法；这是词表和接口性质，不是蛋白模型的必要属性。 | **已确认** | 需要输出秩归一化的跨模型比较方法。 |
| R1.5 | 1. 比较模型家族 | MoE routing | ProGen3 第一层 routing 接近当前氨基酸查表（相对多数类的 skill 为 0.870）；该 skill 非单调下降，到第 7 层已降为零，即深层 routing 不再携带残基身份信息。**注意：**“深层转为上下文决定”是过度解读——本实验只证明残基不再有预测力，没有测量上下文是否有。 | **ProGen3 描述性发现；单折、无区间** | 没有文本 MoE 和第二个蛋白 MoE，不能形成模态结论。 |
| R1.6 | 1. 比较模型家族 | 蛋白知识与新颖性 | ProGen3 的 fitness 预测优于简单替换矩阵，但尚不能确认它超过训练语料同源检索，也没有证明模型能生成真正新颖且有效的蛋白质。 | **部分成立** | Retrieval bound、独立 fitness 和新颖性验证尚未完成。 |
| R2.1 | 2. 评估方法迁移 | 能够迁移的方法 | Tuned Lens、Activation Patching、部分 Probe 和 Concept Erasure 能够应用于蛋白模型并产生稳定结果。 | **已确认** | 这些方法多数只能回答局部问题，不能单独形成完整机制解释。 |
| R2.2 | 2. 评估方法迁移 | Attention selector 迁移 | 本项目 PAA 在文本模型上能够找回因果重要 heads，但在多数蛋白模型上失败；原有 induction census 的全层统计又受到深度混淆。Attention 模式不能直接当成蛋白模型的因果机制。 | **核心迁移结果** | PAA 的蛋白失败尚未定位到具体 selector 因素。 |
| R2.3 | 2. 评估方法迁移 | SAE、Transcoder 与 CLT | 稀疏 replacement 确实学到了廉价线性映射学不到的非线性结构（自由基线被明确超越），但良好重建既不保证行为忠实也不保证因果忠实：重建排序在忠实度门控下完全消失。CLT 的等宽优势是容量效应而非跨层连接效应——且该结论只在参数预算或墙钟预算下成立，在推理 FLOP 预算下反转。**方法本身已被排除：** 同一协议在 GPT-2 上行为恢复 0.909，首次越过 0.80 门控（EXP-R2-141）。 | **核心负面结果；方法侧已排除** | 失败属于模态还是 MoE 架构，取决于 ProtGPT2 的重训结果——原 ProtGPT2 字典 75% 死亡，已撤出该比较。 |
| R2.4 | 2. 评估方法迁移 | 统一因果审计 | 重建率、probe accuracy、attention pattern 和 performance recovery 都不足以单独支持机制解释；必须同时检查 attainability、fidelity、causal retrieval、分母有效性和独立数据稳定性。 | **主要方法学贡献** | 需要把同一套门控扩展到更多新一代方法。 |
| R2.5 | 2. 评估方法迁移 | 蛋白测量基底 | 蛋白实验对输入渲染、序列区间、同源性、文件顺序、tokenization 和小字母表更敏感；很多表面模态差异实际来自这些接口。 | **已确认** | 需要把这些控制固化为统一蛋白评价协议。 |
| R3.1 | 3. 开发适配方法 | PAA | PAA 是本研究提出并完成验证的方法：它在文本上有效，并揭示了蛋白 attention selector 的迁移失败；但它尚不是成功的蛋白特异方法。 | **方法创新已完成；蛋白适配失败** | 没有发现足够具体、可复现的失败因子，因此暂不应继续堆叠启发式特征。 |
| R3.2 | 3. 开发适配方法 | 因果忠实 replacement | 现有结果支持设计直接优化 intervention consistency 或 causal-rank consistency 的 replacement，而不是继续只优化重建误差。 | **方向已获得；方法未完成** | 需要先完成 Dense 对照，再决定是否加入 residue interface 或 routing path。 |
| R3.3 | 3. 开发适配方法 | 新一代方法 | DAS、causal abstraction、完整 circuit tracing 和 routing-path interventions 尚未形成完成实验，不能写成已有成果。 | **未完成** | 必须通过负控制和统一因果门控后才能进入结论。 |
| S.1 | 整体结论 | 研究贡献 | 当前最稳固的贡献是建立了文本可解释性方法向蛋白生成模型迁移的因果审计框架，并区分了可迁移方法、方法固有限制、蛋白评价接口问题和真实迁移失败。 | **论文主线基本形成** | 尚缺正式 Dense replacement 对照、retrieval bound，以及一个经过验证的蛋白适配方法或生物学发现。 |

## 核心实验与结果索引

本表一行对应一个能够支撑研究结论的实验系列，而不是单个 seed、窗口或运行任务。它只记录直接测量事实；研究解释和后续缺口只在上表陈述。详细配置、数值和撤回过程保留在权威审计与实验日志中。

| 核心实验系列 | 模型与关键对照 | 直接测量对象 | 直接实验结果 | 证据状态 | 支持结论 |
|---|---|---|---|---|---|
| 蛋白测量基底审计 | Dense 文本/蛋白面板 | 输入渲染、cohort、长度、距离单位和 tokenization 对结果的影响 | 多个早期效应在修正输入渲染、抽样或距离单位后发生实质变化，其中部分被撤回。 | **已完成并冻结为限制目录** | R1.1、R2.4、R2.5 |
| Pathway Budget | 多个 Dense 文本模型 vs 多个 Dense 蛋白模型 | MLP/attention 全路径消融比 | 文本模型为 1.50–2.10，蛋白模型为 0.55–1.12，当前范围不重叠。 | **带 tokenization caveat** | R1.2 |
| Induction Causal Audit | 7 个文本模型 vs 5 个蛋白模型 | Census 与 exhaustive path-patching 因果排序的一致性 | 蛋白模型的高排名 heads 具有因果作用；全层相关和固定阈值计数受深度与 bulk 噪声影响。 | **已完成** | R1.3、R2.2 |
| PAA Copy-suppression | 多个文本正控制 vs 多个 Dense 蛋白模型 | PAA selector 对因果 top heads 的 retrieval | 文本模型均高于自身随机水平；逐残基蛋白模型处于或低于随机水平，ProtGPT2 的结果主要由层深携带。 | **核心结果已冻结** | R2.2、R3.1 |
| Lens Transfer | GPT-2 系列 vs Dense 蛋白模型 | Tuned Lens 相对 Logit Lens 的 KL 改善 | 所有正式计分模型的改善为正；扩大蛋白序列区间会使改善下降 54–82%，文本控制不下降。 | **已完成** | R1.4、R2.1、R2.5 |
| Probe and Erasure | GPT-2-large 控制 vs 多个 Dense 蛋白模型 | 属性 decodability 与线性 erasure 后的行为效应 | 蛋白属性的 probe skill 与 erasure 后行为依赖可以方向相反。 | **部分 cohort 结论受限** | R2.1、R2.4 |
| Output Aperture / J-Lens | Dense 文本模型 vs 不同词表的 Dense 蛋白模型 | 输出 Jacobian 秩及性质方向的定义 | ProGen2 的 aperture 被小词表限制，ProtGPT2 接近文本模型；非输出函数的生物属性没有可靠 J-Space 定义。 | **已完成，J-Space 关闭** | R1.4、R2.4 |
| Dictionary Fidelity | GPT-2-large 与多个 Dense 蛋白字典 | FVU、重建与 behavioural recovery 的对应关系 | FVU 与重建排序不能认证行为或因果忠实（Spearman 约为 0）。 | **已完成** | R2.3、R2.4 |
| Replacement 自由基线 | ProGen3-112M：逐层仿射最小二乘映射 vs 全部 transcoder | 自由基线（rule 28）的行为恢复率 | 线性映射恢复 −1.61 至 −1.64，比均值消融地板还差，且在 16 倍拟合数据范围内饱和（变化 0.003），说明这是块非线性的性质而非欠拟合。全部 transcoder 明确胜出。 | **已完成；本项目字典方向的首个正面结果** | R2.3、R2.4 |
| ProGen3 Loader Audit | ProGen3-112M MegaBlocks 权重 vs eager 路径 | checkpoint 加载后的真实计算 | 直接 eager 加载的 MoE experts 保持随机；严格转换后的 scored self-check 恢复合理 NLL。 | **已完成** | R2.4、R2.5 |
| CLT/PLT Capacity Match | ProGen3 上的 CLT、等宽 PLT、参数匹配 PLT，各 2 个语料 seed | 重建误差在不同资源匹配下的排序，以及该排序是否进入忠实度 | 等宽 CLT 比 PLT 好 11.7%；匹配参数后宽 PLT 反而好 0.126（按 seed 配对差 +0.1294/+0.1228，95% CI [0.084, 0.168]，并在独立 Swiss-Prot cohort 上复现）。但该排序**不进入忠实度**：行为恢复 0.1450 vs 0.1464，两个配对差符号相反，本设计的最小可检测差 0.0546 大于全部七个 arm 的总极差 0.0436。 | **已完成；等宽比较为嵌套模型类** | R2.3 |
| Replacement Faithfulness | 同一冻结快照、同一协议（12× 扩展、k=64、20000 步）下的 gpt2、ProtGPT2、ProGen3，各自对照自身自由线性基线 | sequential replacement 的行为恢复和 causal retrieval | 行为恢复：gpt2 **+0.9091** [+0.9037, +0.9136]（越过 0.80 门控），ProGen3 +0.1337 [+0.1180, +0.1503]，ProtGPT2 +0.0542（**已撤出**，见下）。三者都胜过自身自由基线（+0.263 / −1.611 / −0.716），attainability 全部 ATTAINABLE，区间互不重叠。 | **文本控制完成并通过；ProGen3 完成并失败；ProtGPT2 因字典欠训练撤回** | R2.3、R2.4、R3.2 |
| MoE Routing Audit | ProGen3-112M；随机分组（等基数自由基线）与残基身份（混淆项） | selected-set 与 boundary-margin 分组对 replacement residual 的留出解释量 | 在 replacement 真正失败的第 4–8 层，selected-set 分组**不减少任何留出误差**（−0.0007 至 −0.00002），只是比等基数随机分组少亏 +0.0002–0.0008，算术上限 0.0022–0.0034；boundary-margin 分组在每一层都更弱。 | **已完成；预注册 null 成立** | R1.5、R2.3、R3.2 |
| Frozen-attention Check | GPT-2-large vs ZymCTRL | 冻结 attention 后保留的上下文信息 | 两个模型均保留约 77%，没有出现蛋白侧额外下降。 | **已完成的负结果** | R1.1、R2.4 |
| Fitness Baseline | ProGen3-112M vs BLOSUM62，ProteinGym | Zero-shot fitness Spearman 优势 | 完整 benchmark 上模型优势为 +0.0647，置信区间高于零；论文的小面板无法单独解析该优势。 | **已完成** | R1.6 |
| Homology Control | Dense 蛋白模型与 UniRef50 同源层 | induction 指标随训练集同源性的变化 | 没有真实低同源层；head count 未随同源层分离，峰值强度与记忆解释一致。 | **受数据覆盖限制** | R1.3、R1.6 |
| Retrieval Bound | 217 个 ProteinGym assay；LOOKUP（UniRef50 位点独立 profile）vs BLOSUM62 vs 三个 arm | 模型 fitness 是否超过对自身训练语料的同源检索 | LOOKUP 通道本身已测定：全 benchmark 平均 Spearman **+0.3537**（中位 +0.3560），明显高于本仓库冻结的 BLOSUM62 基线 +0.2098；六个通道的打乱标签对照全部落在 ±0.006 内，说明统计量已校准。187 个野生型聚为 174 个 50% 同源簇，其中 73 个是 UniRef50 的逐字节相同记录。模型侧 Spearman 与 MODEL − LOOKUP 判定仍在计算。 | **LOOKUP 已完成；模型对比进行中** | R1.6 |

## 当前进行中

本节是本文作为进度总览的操作性部分：只记录已经启动、结果尚未产生的实验，以及它们各自将要判定的问题。结果一旦产出，对应行移入上表并从本节删除。

| 进行中的实验 | 判定什么 | 判定规则（已预先声明） |
|---|---|---|
| **最小归因对照（收尾中）**：gpt2 与 ProGen3 已完成，**ProtGPT2 按匹配 token 预算重训**（80000 步 ≈ 80M token，对齐 gpt2 的 73M），反向对照 gpt2 在 ProtGPT2 原预算（20M token）下重训；gpt2-large 的 PLT 仍在训练 | replacement 的失败属于模态、架构、方法还是评价接口 | 方法一支已排除（gpt2 恢复 0.909）。**剩余判定完全落在重训后的 ProtGPT2 上**：重训后仍失败 → 模态（蛋白 Dense 与 MoE 都失败）→ residue/interface 或扩展 PAA 方向；重训后通过 → 架构（只有 MoE 失败）→ expert/router/path 方向。gpt2-large 只用于排除深度混淆（36 层文本对照 ProtGPT2 的 36 层）。 |
| **训练语料检索上界（收尾中）**：LOOKUP 通道已完成（+0.3537），三个 arm 的 zero-shot fitness 打分正在 H200 上运行，随后是 MODEL − LOOKUP 的等价性判定。**重采样单位是 50% 同源簇而非 assay**：187 个野生型聚为 174 簇，其中 73 个是 UniRef50 的逐字节相同记录（已实测）。 | 各 arm 的 fitness 优势是否超过对自身语料的同源检索 | 若不超过，则任何以该 fitness 为分母的 recovery ratio（包括 ProGenMech 的与本项目的）都不能支撑机制主张。正控制下限取本仓库自身冻结的 BLOSUM62 全benchmark均值 +0.2098（本地独立复现为 +0.2124），而非无法在本机核实的已发表数值。**LOOKUP 已经高于该下限 0.14，因此这条上界是有约束力的，不是形式检查。** |

**已知的不可消除限制，先记录再测量：** ProGen3-112M 的模型卡未声明训练语料，ProGen2-medium 的 BFD30 未落盘，因此这两个 arm 的 LOOKUP 只能以 UniRef50 作代理，**系统性低估其语料支持度，偏向于让模型通过**。ProtGPT2 是唯一语料被精确识别的 arm（其声明语料就是已落盘的 UniRef50），因此它的结果权重最高。

**分支状态更新（EXP-R2-141）：** “全部失败”一支已被排除——GPT-2 在同一协议下恢复 0.909，是本项目第一个越过 0.80 行为门控的 arm，因此失败不属于方法本身。“只有 ProGen3 失败”一支仍然存活，但不再是最可能的一支：它现在与“蛋白模态失败”并列，二者只能由重训后的 ProtGPT2 分开。

**本对照自身暴露的设计缺陷，记录而非掩盖：** 三个 arm 匹配的是*序列数*而不是 *token 数*。由于每个 arm 的渲染方式不同（每序列 125 / 456 / 932 token），ProtGPT2 实际只看到 20M token 却要拟合 8.3 倍于 gpt2 的字典参数，训练结束时 **75.1% 的 latent 已死亡**（gpt2 为 8.3%，ProGen3 为 0.6%）。这样的字典没有测量它的 arm，因此 ProtGPT2 的 0.054 被撤出模态判定，而不是当作模态证据报告。修正方式是把 token 预算而非序列数声明为匹配量。

当前最有根据的建设方向依次是：因果忠实的 replacement 目标、层内因果校准的 selector，以及 tokenization 匹配的 pathway 分析。DAS/causal abstraction、完整新一代 circuit tracing、完整蛋白 attribution graph 和文本 MoE 对照尚无完成实验，不写成已有成果。

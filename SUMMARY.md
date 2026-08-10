# InterpretabilityTransfer 研究总结

**状态：** 官方进度总览。对科学结论而言仍是支持性汇总，不是权威来源；结论以权威审计为准，本文负责让当前进度一眼可读，并在新结果产出时同步更新。

**更新：** 2026-08-09

**权威来源：** [`docs/INTERPRETABILITY_TRANSFER_AUDIT.md`](docs/INTERPRETABILITY_TRANSFER_AUDIT.md)

本文从最终结论和核心实验两个层次概括本研究。若本文与权威审计或后续实验记录不一致，以权威审计和 [`docs/EXPERIMENT_LOG.md`](docs/EXPERIMENT_LOG.md) 为准。实验日志和项目日志中带章节号的旧 `check.md` 引用属于已经被权威审计取代的历史内容，不指向本文。

“测量成功”表示实验能够稳定地区分所定义的量；“方法成功”还要求解释对象通过因果检查；“完全成功”还要求跨模型、数据和蛋白家族泛化，并排除架构、数据、tokenization 和评价接口等混淆。目前没有方法达到最后一级。“冒烟”只证明接口能够端到端运行，不是科学结果。

## 研究结论总览

| 结论编号 | Research Objective | 核心主题 | 本研究取得的最终结论 | 结论状态 | 尚未解决的问题 |
|---|---|---|---|---|---|
| R1.1 | 1. 比较模型家族 | 整体可解释性难度 | 目前不能笼统声称“蛋白模型比语言模型更难解释”。差异主要出现在特定 selector、评价接口和数据构造上，而不是所有方法普遍失效。 | **已确认** | 需要更多匹配模型验证适用范围。 |
| R1.2 | 1. 比较模型家族 | 计算路径组织 | 文本模型整体更偏向 MLP 路径；已测蛋白模型相对更依赖 attention 或更加均衡。这是目前最有价值的候选计算差异。 | **较强候选** | tokenization 和独立蛋白家族控制尚未完成。 |
| R1.3 | 1. 比较模型家族 | Attention 与 induction | 蛋白模型中存在具有真实因果作用的 induction heads；“蛋白模型 induction heads 更少或更弱”不是可靠结论。 | **已确认其存在；模态差异未确认** | 需要自然重复和 family-disjoint 验证。 |
| R1.4 | 1. 比较模型家族 | 输出语义接口 | 逐残基蛋白模型的小词表形成低秩输出 aperture，限制 Logit Lens、DLA 等方法；这是词表和接口性质，不是蛋白模型的必要属性。 | **已确认** | 需要输出秩归一化或概念对齐的比较方法。 |
| R1.5 | 1. 比较模型家族 | MoE routing | ProGen3 第一层 routing 接近当前氨基酸查表（相对多数类的 skill 为 0.870）；该 skill 非单调下降，到第 7 层已降为零，即深层 routing 不再携带残基身份信息。**注意：**“深层转为上下文决定”是过度解读——本实验只证明残基不再有预测力，没有测量上下文是否有。 | **ProGen3 描述性发现；单折、无区间** | 没有文本 MoE 和第二个蛋白 MoE，不能形成模态结论。 |
| R1.6 | 1. 比较模型家族 | 蛋白知识与新颖性 | **三个蛋白解码器无一超过对其自身训练语料的位点独立 profile 检索，最小的那个还明显低于它。** ProtGPT2 −0.0143 [−0.0381, +0.0101]、ProGen2-medium −0.0043 [−0.0303, +0.0216]（均为 **retrieval_bounded**，retrieval share 1.107 / 1.029），ProGen3-112M **−0.0808 [−0.1139, −0.0481]**，share **2.186**，判定为 **retrieval_dominated**。跨 112M–774M 参数、三种 tokenization。因此 ProteinGym 替换 benchmark 上的 fitness Spearman 不足以支撑“模型习得生物学知识”的主张，本项目与 ProGenMech 中以该 fitness 为分母的 recovery ratio 都继承这一限制。 | **核心负面结果；ProtGPT2 一支语料精确识别** | 这不等于“模型在记忆”——只说明该评价接口无法区分二者。生成新颖蛋白的验证尚未开始。 |
| R2.1 | 2. 评估方法迁移 | 能够迁移的方法 | Tuned Lens、Activation Patching、部分 Probe 和 Concept Erasure 能够应用于蛋白模型并产生稳定结果。 | **已确认** | 这些方法多数只能回答局部问题，不能单独形成完整机制解释。 |
| R2.2 | 2. 评估方法迁移 | Attention selector 迁移 | 本项目 PAA 在文本模型上能够找回因果重要 heads，但在多数蛋白模型上失败；原有 induction census 的全层统计又受到深度混淆。Attention 模式不能直接当成蛋白模型的因果机制。 | **核心迁移结果** | PAA 的蛋白失败尚未定位到具体 selector 因素。 |
| R2.3 | 2. 评估方法迁移 | SAE、Transcoder 与 CLT | **sequential replacement 的行为忠实失败属于蛋白模态，不属于方法、字典预算、深度或 MoE 路由（EXP-R2-147）。** 匹配对 gpt2-large 与 ProtGPT2 同为 36×1280、字典同为 552,960 个 latent、estimand 逐位相同：文本恢复 **0.9322**，蛋白恢复 **0.1641**，而蛋白字典的重建反而更好（每层 NMSE 0.2376 对 0.2750）。dense 蛋白 arm 与 sparse-MoE 蛋白 arm 同处 0.13–0.16，与同构文本 arm 的 0.93 相距甚远，**MoE 由此被排除**。**EXP-R2-149 进一步排除了分词**：ZymCTRL（与 gpt2-large 同为 36×1280、但逐残基分词）恢复 **+0.0916 / +0.0923**（两 seed 差 0.0007）。蛋白侧现有三种分词、两种架构共三个 arm，全部落在 **0.09–0.16**，文本两个 arm 为 **0.91–0.93**，相差 6–10 倍且不重叠。 | **核心负面结果；归因已完成，属于迁移限制而非方法限制** | 尚未解释「为什么」。**ZymCTRL 自身有一项未排除的反对意见：它的字典重建反而差约一倍（每层 0.529 对 0.275 / 0.238），因此「字典更差」对该 arm 成立**；更高预算（~154 token/latent，高于 gpt2-large 的 132）的直接对照正在训练。自由基线不再支持模态二分（见下）。 |
| R2.4 | 2. 评估方法迁移 | 统一因果审计 | 重建率、probe accuracy、attention pattern 和 performance recovery 都不足以单独支持机制解释；必须同时检查 attainability、fidelity、causal retrieval、分母有效性和独立数据稳定性。**该门控中的 causal-rank 一项已被本项目自己撤出模态用途**（EXP-R2-148）：在同为 720 head 的网格上，行为恢复 0.93 的文本对照 gpt2-large 自身也失败（ρ +0.4119），因此它不能证明蛋白 replacement 在因果上不忠实。 | **主要方法学贡献；其中一项估计量已自我限缩** | 需要把同一套门控扩展到更多新一代方法；causal-rank 需要一个能在文本对照上达标的替代统计量。 |
| R2.5 | 2. 评估方法迁移 | 蛋白测量基底 | 蛋白实验对输入渲染、序列区间、同源性、文件顺序、tokenization 和小字母表更敏感；很多表面模态差异实际来自这些接口。 | **已确认** | 需要把这些控制固化为统一蛋白评价协议。 |
| R3.1 | 3. 开发适配方法 | PAA | PAA 是本研究提出并完成验证的方法：它在文本上有效，并揭示了蛋白 attention selector 的迁移失败；但它尚不是成功的蛋白特异方法。 | **方法创新已完成；蛋白适配失败** | 没有发现足够具体、可复现的失败因子，因此暂不应继续堆叠启发式特征。 |
| R3.2 | 3. 开发适配方法 | 因果忠实 replacement | 现有结果支持设计直接优化 intervention consistency 或 causal-rank consistency 的 replacement，而不是继续只优化重建误差。 | **方向已获得；方法未完成** | 需要先完成 Dense 对照，再决定是否加入 residue interface 或 routing path。 |
| R3.3 | 3. 开发适配方法 | 新一代方法 | DAS、causal abstraction、完整 circuit tracing 和 routing-path interventions 尚未形成完成实验，不能写成已有成果。 | **未完成** | 必须通过负控制和统一因果门控后才能进入结论。 |
| R3.4 | 3. 开发适配方法 | Concept-aligned Lens | 候选方法将不同 token 输出投影到预先声明的概念空间：蛋白模型内部可使用疏水性、电荷、体积和实测结构标签；文本–蛋白比较只能使用复制、距离依赖和约束满足等共同抽象。 | **候选；尚未实现** | 需要通过随机属性、输出秩匹配、因果干预和独立蛋白家族验证；通过前不能写成方法创新。 |
| S.1 | 整体结论 | 研究贡献 | 当前最稳固的贡献是建立了文本可解释性方法向蛋白生成模型迁移的因果审计框架，并区分了可迁移方法、方法固有限制、蛋白评价接口问题和真实迁移失败。**两项此前缺失的关键对照现已完成**：最小归因对照把 replacement 的失败定位到蛋白模态（排除方法、字典预算、深度与 MoE 路由），训练语料检索上界表明三个蛋白解码器的 fitness 优势都不超过对自身语料的检索。 | **论文主线基本形成，两项核心对照完成** | 尚缺一个经过验证的蛋白适配方法或生物学发现；每个模态仅两个 arm。 |

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
| Replacement 自由基线 | 四个 arm 的逐层仿射最小二乘映射；其中 gpt2-large 与 ProtGPT2 为 36×1280 完全同构、基线参数量同为 59,028,480 | 自由基线（rule 28）的行为恢复率 | **文本 +0.2630 / +0.2941，蛋白 −0.7158 / −1.6112，两组区间完全不重叠**；同构匹配对相差 **1.010**。线性映射在蛋白上比均值消融地板还差，且在 16 倍拟合数据范围内饱和（变化 0.003），说明是块非线性的性质而非欠拟合。全部 transcoder 明确胜出自身基线。 | **已完成；但模态二分已被 ZymCTRL 推翻**：三次抽样分别为 −0.0096 / +0.0107 / +0.0242（含一次与前两次不相交的抽样），符号随抽样改变，即该 arm 的自由基线与均值消融地板无法区分。蛋白侧因此跨越 −1.78 至 +0.02，**「两组不重叠」只对本实验的四个 arm 成立，不是模态结论**。行为恢复率那条腿不受影响。 | R1.2、R2.3、R2.4 |
| ProGen3 Loader Audit | ProGen3-112M MegaBlocks 权重 vs eager 路径 | checkpoint 加载后的真实计算 | 直接 eager 加载的 MoE experts 保持随机；严格转换后的 scored self-check 恢复合理 NLL。 | **已完成** | R2.4、R2.5 |
| CLT/PLT Capacity Match | ProGen3 上的 CLT、等宽 PLT、参数匹配 PLT，各 2 个语料 seed | 重建误差在不同资源匹配下的排序，以及该排序是否进入忠实度 | 等宽 CLT 比 PLT 好 11.7%；匹配参数后宽 PLT 反而好 0.126（按 seed 配对差 +0.1294/+0.1228，95% CI [0.084, 0.168]，并在独立 Swiss-Prot cohort 上复现）。但该排序**不进入忠实度**：行为恢复 0.1450 vs 0.1464，两个配对差符号相反，本设计的最小可检测差 0.0546 大于全部七个 arm 的总极差 0.0436。 | **已完成；等宽比较为嵌套模型类** | R2.3 |
| Replacement Faithfulness（**归因已完成**） | 同一协议（12× 扩展、k=64）下的 gpt2、gpt2-large、ProtGPT2、ProGen3；**gpt2-large 与 ProtGPT2 构成 36×1280 同构匹配对**；gpt2 与 ProGen3 各有第二个语料 seed | sequential replacement 的行为恢复和 causal retrieval | **文本 gpt2-large +0.9322 [+0.9286, +0.9352]、gpt2 +0.9091（seed 07 +0.9084，差 0.0007）；蛋白 ProtGPT2 重训后 +0.1641、ProGen3 +0.1337（seed 07 +0.1262，差 0.0075）。** dense 蛋白与 sparse-MoE 蛋白同处 0.13–0.16，与同构文本 arm 的 0.93 相距甚远。关键在于**蛋白字典的重建反而更好**（每层 NMSE 0.2376 对 0.2750），且其 cohort 污染率实测 **89.8%**（gpt2-large 为 40.4%）——所有残余偏置都指向让蛋白 arm 通过，它仍然失败。深度亦已排除（36 层反而略高于 12 层）。 | **全部完成；预注册判定规则于结果产出前写定** | R2.3、R2.4、R3.2 |
| MoE Routing Audit | ProGen3-112M；随机分组（等基数自由基线）与残基身份（混淆项） | selected-set 与 boundary-margin 分组对 replacement residual 的留出解释量 | 在 replacement 真正失败的第 4–8 层，selected-set 分组**不减少任何留出误差**（−0.0007 至 −0.00002），只是比等基数随机分组少亏 +0.0002–0.0008，算术上限 0.0022–0.0034；boundary-margin 分组在每一层都更弱。 | **已完成；预注册 null 成立** | R1.5、R2.3、R3.2 |
| Causal-rank 门控（**自我限缩**） | 同一快照下的 gpt2（144 head）、gpt2-large（720）、ProtGPT2（720）、ProGen3（60）；ProtGPT2 的匹配对照是 gpt2-large | 原模型与 replacement 的逐部件消融效应秩相关，门槛 ρ ≥ 0.50 | **只有网格最小的 gpt2 通过（+0.7096）。在与 ProtGPT2 完全同网格的 720 head 上，gpt2-large 失败（+0.4119，n=512 为 +0.4161），ProtGPT2 也失败（+0.2024）。** 全部 arm 的 attainability ceiling 均高于门槛（0.695–0.984），所以这些失败是实测失败而非规格缺陷。深度（12 对 36）与网格规模（144 对 720）在本设计中未分离。 | **已完成；结论是把该估计量撤出模态用途** | R2.2、R2.4 |
| Tokenization 对照（EXP-R2-149） | gpt2-large（文本/BPE）、ProtGPT2（蛋白/multi-residue BPE）、**ZymCTRL（蛋白/逐残基、EC 条件）** 三者同为 36×1280；ProGen3 为第四个 arm；ZymCTRL 两个语料 seed | sequential replacement 的行为恢复率 | **ZymCTRL +0.0916 / +0.0923（seed 差 0.0007），低于另外两个蛋白 arm。** 蛋白侧三种分词、两种架构全部落在 0.09–0.16，文本 0.91–0.93。三项偏置（EC 泄漏 2.4487 nats、实测 76.6% cohort 污染、clean NLL 仅 0.8364）全部指向让它通过，它仍失败。 | **已完成；判定规则于启动前写定。保留意见：该 arm 字典重建差约一倍，「字典更差」未排除** | R1.1、R2.3 |
| Frozen-attention Check | GPT-2-large vs ZymCTRL | 冻结 attention 后保留的上下文信息 | 两个模型均保留约 77%，没有出现蛋白侧额外下降。 | **已完成的负结果** | R1.1、R2.4 |
| Fitness Baseline | ProGen3-112M vs BLOSUM62，ProteinGym | Zero-shot fitness Spearman 优势 | 完整 benchmark 上模型优势为 +0.0647，置信区间高于零；论文的小面板无法单独解析该优势。 | **已完成** | R1.6 |
| Homology Control | Dense 蛋白模型与 UniRef50 同源层 | induction 指标随训练集同源性的变化 | 没有真实低同源层；head count 未随同源层分离，峰值强度与记忆解释一致。 | **受数据覆盖限制** | R1.3、R1.6 |
| Retrieval Bound | 217 个 ProteinGym assay，174 个 50% 同源簇为重采样单位；LOOKUP（自身声明语料的位点独立 profile）vs BLOSUM62 vs 三个 arm | 模型 fitness 是否超过对自身训练语料的同源检索 | LOOKUP **+0.3537** vs BLOSUM62 **+0.2098**。ProtGPT2 −0.0143 [−0.0381, +0.0101]，share 1.107；ProGen2-medium −0.0043 [−0.0303, +0.0216]，share 1.029；**ProGen3-112M −0.0808 [−0.1139, −0.0481]，share 2.186，retrieval_dominated**。三个对照全部通过；BLOSUM62 复现为 +0.20982（冻结值 +0.2098），ProGen3 的 MODEL − BLOSUM62 复现为 +0.0681（冻结值 +0.0647），是对 EXP-R2-134 的两次独立复现。同源分层无梯度。 | **三个 arm 全部完成** | R1.6 |

## 当前进行中

本节是本文作为进度总览的操作性部分：只记录已经启动、结果尚未产生的实验，以及它们各自将要判定的问题。结果一旦产出，对应行移入上表并从本节删除。

| 进行中的实验 | 判定什么 | 判定规则（已预先声明） |
|---|---|---|
| **ZymCTRL 更高预算字典**：28,000 步 / 约 154 token per latent（高于 gpt2-large 的 132，且不重复任何一条记录）；以及 s06 的 n=512 cohort 稳健性检查 | ZymCTRL 的失败是否部分来自它较差的字典重建（每层 0.529 对 0.275 / 0.238） | 若在高于文本对照的 token 预算下恢复率仍停在 0.09–0.16，则「字典更差」被排除，该 arm 的贡献与 ProtGPT2 等强；若显著上升，则该 arm 的失败部分归于字典预算，三 arm 陈述改以另外两个 arm 为准。 |

**已知的不可消除限制，先记录再测量：** ProGen3-112M 的模型卡未声明训练语料，ProGen2-medium 的 BFD30 未落盘，因此这两个 arm 的 LOOKUP 只能以 UniRef50 作代理，**系统性低估其语料支持度，偏向于让模型通过**。ProtGPT2 是唯一语料被精确识别的 arm（其声明语料就是已落盘的 UniRef50），因此它的结果权重最高。

**分支状态更新（EXP-R2-141 至 144）：** “全部失败”一支已被排除——GPT-2 在同一协议下恢复 **0.909**，是本项目第一个越过 0.80 行为门控的 arm；把它饿到 ProtGPT2 的 20M token 预算后仍恢复 **0.832**，依然通过。因此失败既不属于方法，也不属于字典的数据预算。

**ProGen3 的失败已排除数据预算解释。** 按 token/latent 排序：ProGen3 为 **6,470**、死亡 latent 仅 **0.6%**，是全面板喂得最饱、最健康的字典（gpt2 为 660 / 8.3%），恢复率却只有 0.134——恢复率与 token/latent 在四次训练中甚至呈反序。当时剩下的唯一开放轴是**蛋白模态 vs 稀疏 MoE 架构**，**该轴已由 EXP-R2-147 关闭**：与 ProGen3 无任何 routing 的 dense 蛋白 arm（ProtGPT2 重训后 +0.1641）失败方式和幅度都与之相同，因此 MoE 被排除，归因落在蛋白模态。以下两段是该分支关闭前写下的推理过程，保留为记录。

**该轴当时已有一份不依赖字典的独立证据（EXP-R2-144），它先于 ProtGPT2 重训指向了同一答案。** 自由线性基线不需要训练任何字典，因此完全不受 token/latent、死亡 latent 与语料规模的影响。它在文本上恢复 +0.2630 / +0.2941，在蛋白上恢复 −0.7158 / −1.6112，两组不重叠；而 gpt2-large 与 ProtGPT2 是同深度同宽度、基线参数量完全相同（59,028,480）的匹配对，相差 1.010。**它把 dense 蛋白 arm 归到 MoE 蛋白 arm 一侧，而不是归到与它同构的文本 arm 一侧**，因此支持模态解释，且早于本该给出该判定的 ProtGPT2 重训。需要注意的保留：recovery 以各 arm 自身的 clean→ablated 间隔归一，而两者的分母并不相同（6.91 对 4.30）。

**本对照自身暴露的设计缺陷，记录而非掩盖：** 三个 arm 匹配的是*序列数*而不是 *token 数*，更不是 *token/latent*。ProtGPT2 只看到 20M token 却要拟合 552,960 个 latent（gpt2 为 110,592，因为 36×1280 对 12×768 而扩展比固定为 12×），即每 latent 仅 36 个 token，训练结束时 **75.1% 的 latent 已死亡**。这样的字典没有测量它的 arm，因此 ProtGPT2 的 0.054 被撤出模态判定。**由此也重新指认了对照组：gpt2 从来不是 ProtGPT2 的对照，gpt2-large 才是**——同为 36×1280、字典同为 552,960 个 latent，token/latent 为 132 对 118，相差 12%。

**另一处已记录的自我更正：** 本日志曾断言 gpt2 的 cohort “约 98% 未见过”，那是假设而非测量，且是错的。已落盘的 OpenWebText 只有 80 个分片中的 4 个（396,133 条合格记录），gpt2 的 160,000 条训练序列占其自身评分池的 **40.4%**。这一点两面都要说：EXP-R2-141 的主对比反而更稳（gpt2 是污染更重的一方，40.4% 对 ProtGPT2 的 29.2%），但其中每个 recovery 数字都因此被向上偏置了一个未测量的量。

当前最有根据的建设方向依次是：因果忠实的 replacement 目标、层内因果校准的 selector，以及 tokenization 匹配的 pathway 分析。DAS/causal abstraction、完整新一代 circuit tracing、完整蛋白 attribution graph 和文本 MoE 对照尚无完成实验，不写成已有成果。

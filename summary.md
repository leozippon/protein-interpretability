# InterpretabilityTransfer 研究方向与当前结论

**更新日期：** 2026-08-25

本项目比较纯文本、纯蛋白质和语言–蛋白质联合生成模型，研究它们具有什么能力、这些能力如何形成，以及模型究竟学到了什么。

研究分三步：**先说明模型具有什么能力，再用现有方法解释能力如何形成、表示和计算；如果现有方法不足，再开发新方法判断模型学到了什么。**

## 主要模型

| 模型谱系 | 模态与比较作用 | Tokenizer / 基本单位 | 关键架构 | 训练数据或阶段 |
| --- | --- | --- | --- | --- |
| GPT-2 | 纯文本基础对照和规模阶梯 | 50,257 词表 BPE | 12–48 层，宽 768–1,600，约 1.24 亿–15.6 亿参数 | WebText |
| DialoGPT-small | 文本语料对照 | 与 GPT-2 相同的 50,257 词表 BPE | 与 GPT-2 small 相同的 12 层、宽 768 | Reddit 对话 |
| Qwen2.5-0.5B / Llama-3.2-3B | 非 GPT-2 文本架构对照 | 151,936 / 128,256 词表 | 24 层、宽 896 / 28 层、宽 3,072 | 多语言、代码和数学混合 / 网络文本与蒸馏数据 |
| ByGPT5 small/base/medium | 字节级文本对照和规模阶梯 | 384 符号字节词表，英文约一字符一 token | T5 式仅解码器；4–12 层，宽 1,472–1,536 | 仓库未确认训练语料 |
| ProtGPT2 | 多残基蛋白对照 | 50,257 词表，多残基 BPE；原生 FASTA 输入 | GPT-2 36 层、宽 1,280、20 个头，约 7.74 亿参数 | UniRef50 |
| ZymCTRL | 带 EC 条件的逐残基蛋白模型 | 458 词表，基本单位为单残基 | GPT-2 36 层、宽 1,280 | EC 标注的 UniProt 蛋白 |
| ProGen2 | 逐残基蛋白谱系和规模对照 | 32 词表，单残基 token，带生成方向标记 | 12–32 层，宽 1,024–4,096，约 1.51 亿–64.4 亿参数 | 多数规模使用 UniRef90 与 BFD30；base 的具体训练混合未确认 |
| ProGen3-112M | 稀疏 MoE 蛋白对照 | 134 词表；更精确的基本单位未确认 | 10 层、宽 384；每层 8 个专家，每 token 选择 2 个 | 模型卡未声明预训练语料 |
| Galactica / InstructProtein | 语言–蛋白联合模型对照 | 文本词表与原生蛋白表示 | Galactica-1.3B 与 InstructProtein 同为 OPT 24 层、宽 2,048 | 科学语料，其中蛋白低于 1% / UniRef100 继续预训练与指令微调 |
| Llama-2 → ProLLaMA | 同一谱系的文本、蛋白和训练阶段对照 | 共享 32k SentencePiece；蛋白约 1.53 residues/token | 32 层、宽 4,096、32 个头 | Llama-2 → UniRef50 继续预训练 → 蛋白指令微调 |

本文把围绕同一科学问题的重复抽样、参数扫描和修复运行合并为一个实验族，只列能够产生独立结论、使旧结论撤回、改变归因，或决定路线是否继续的实验。表中的“数据”是评测集或实验队列，不一定是模型的完整训练语料；训练语料或样本量无法准确确认时不作猜测。示例只解释记录格式，不代表真实数据内容。详细证据、限制和撤回以 [`docs/INTERPRETABILITY_TRANSFER_AUDIT.md`](docs/INTERPRETABILITY_TRANSFER_AUDIT.md) 为准，最新但尚未晋升的记录见 [`docs/EXPERIMENT_LOG.md`](docs/EXPERIMENT_LOG.md)。

| 状态 | 含义 |
| --- | --- |
| 已完成 | 已有可引用结果，但只能在表中写明的模型、数据和设计范围内解释 |
| 已关闭或停止 | 已回答当前问题、触发停止条件，或证明现有设计无法区分目标假设 |
| 已收窄或撤回 | 测量仍保留，但原来的宽泛解释不再成立 |
| 正在运行 | 已冻结并启动正式队列，但终态产物尚未齐全，不能从部分输出形成科学结论 |
| 开放或未运行 | 仪器验证、小规模试运行或预注册已经完成，但正式模型实验尚未启动 |

## 方向一：比较模型具有什么能力

先在模型的原生接口上测量行为，并用相似文本任务和其他基线说明能力边界。

| 实验 | 模型 | 数据(格式、示例、数量) | 怎么做 | 结果 | 状态与局限 |
| --- | --- | --- | --- | --- | --- |
| 上下文信息资格门 | 当前基础面板的 15 个自回归解码器：10 个文本模型（GPT-2 small/medium/large/xl、DialoGPT-small、Qwen2.5-0.5B、Llama-3.2-3B、ByGPT5 small/base/medium）和 5 个蛋白模型（ProtGPT2、ZymCTRL、ProGen2 small/base/medium） | 文本使用 OpenWebText；蛋白使用 Swiss-Prot 或带 EC 标签的 UniProt，每个区块评分 200 条记录。记录按原生格式分词，最多保留开头 384 token；每个区块另请求 4,000 条一元频率参考记录 | ①在全部合格记录的固定随机排列上构造 8 个区块，每个区块从 1,000 条候选记录中评分 200 条，区块起点相隔 7,000 条。②删除评分集与参考集的完全重复内容。③用各模型 tokenizer 和 α=1 平滑拟合一元基线。④按蛋白残基 5-mer 或文本五词 shingle 合并近重复组。⑤进行 2,000 次组级配对 bootstrap，并校正参考集重采样位移。⑥以校正后 95% 区间下界严格大于零识别上下文信号，以 `Î>8.7652×SE(Î)` 判断能否将 `I` 用作分母 | 15 个模型在 8 个区块上均得到区间；DialoGPT-small 的上下文信息为负，8 个区块均未通过，其余 14 个模型均识别出上下文信号并通过安全分母条件。区块选择可移动约 0.10–0.17 nats；平滑扫描对 `I` 的最大影响为 0.1767 nats，删除共享 30-mer 参考记录后最多移动 +0.0235 nats | 这是方向一后续能力比较的统一样本框和可测性入口，不产生模型排名。区块内区间不包含区块选择波动；蛋白参考集存在跨区块重叠；nats/token 依赖 tokenizer。通过只表示当前接口、长度带和数据队列可测，不说明模型机制或生物知识 |
| Galactica / InstructProtein 双模式能力 | Galactica-125M/1.3B、InstructProtein；Galactica-1.3B 与 InstructProtein 都是 24 层、宽 2048 的 OPT 形状 | 文本为 OpenWebText/WikiText-103 记录；蛋白为 Swiss-Prot 64–246 aa，如 `MKT...`。每个 checkpoint/mode 评分 128 条，另用 400 条独立记录拟合 unigram | ①确定两种模式的原生包装。②分别测相对 unigram 的上下文信息。③反转蛋白序列并检查方向代价。④检查文本输出概率是否仍落在文本 token。⑤从逐记录统计做 2,000 次近重复组 bootstrap，并扣除参考集重采样造成的区间位移 | Galactica-1.3B 的蛋白模式信号很小：+0.0477 nats/token、SE 0.0046，位移校正后的区间下界为 +0.0387；InstructProtein 的蛋白模式可测，但文本位置约 96.3% 概率质量落在残基 token | Galactica-1.3B 蛋白模式可识别，但低幅度信号不自动形成可靠的双模式机制桥；InstructProtein 文本模式仍关闭。相同架构和规模不保证双模式可比较，这两个 checkpoint 也不能代表所有联合模型 |
| ProLLaMA 训练阶段与双模式能力 | Llama-2-7B → ProLLaMA Stage 1 → Stage 2；共享 32 层、宽 4096、32k tokenizer | OpenWebText/WikiText 文本和 Swiss-Prot 64–246 aa，如文本段落与 `Seq=<MKT...>`。每个 stage/mode 评分 128 条，unigram 参考 400 条，并使用独立抽样复核 | ①按各阶段原生格式渲染文本和蛋白。②测上下文信息与反转代价。③沿同一权重谱系定位蛋白能力出现阶段。④把文本代价分到蛋白继续预训练和后续指令微调 | Stage 1 获得可测蛋白模式，同时损失约 4.69 nats/token，即基础文本上下文信息的约 85%；Stage 2 只再改变约 0.10，未恢复文本能力 | 已完成。Stage 1/2 是现有合格双模式桥；蛋白约为 1.53 residues/token，不能与逐残基模型按 token 横比 |
| ProLLaMA 文本与跨模态检索能力变化 | Llama-2-7B、ProLLaMA Stage 1、Stage 2 | 20,000 条 Swiss-Prot 记录，每条含蛋白序列、推荐名称、人工功能描述、非 IEA GO 概念及 EC 概念；遮蔽描述中的概念名称。组外 eval 为 4,499 条、3,752 组；family holdout 为 4,999 条 | ①按序列近重复组和蛋白家族切分。②在预注册的第 22 层拟合 Procrustes 映射。③以遮蔽描述为 query，在同一 1,000-candidate gallery 中分别检索映射后的蛋白名称（文本模式）或蛋白序列（蛋白模式）。④比较三个训练阶段的 top-1，并在家族外复核 | 文本组外 top-1 为 0.2387 → 0.0954、0.1065，家族外为 0.1722 → 0.0546、0.0580；蛋白组外为 0.0104 → 0.0107、0.0164，家族外为 0.0042 → 0.0042、0.0024。蛋白适配没有形成家族外检索增益，指令微调也没有恢复文本检索 | 已完成能力测量。Llama-2 的蛋白模式只作表示参考；这些 top-1 不说明概念对齐超过表面统计，更不说明因果使用，相关方法门和停止结果见方向三“跨模态概念对齐”。只覆盖一个谱系、一个队列和一次抽样 |
| 蛋白突变适应度预测 | ProtGPT2、ProGen2-medium、ProGen3-112M | ProteinGym DMS 突变行，例如 `mutant=A42G, DMS_score=...`。217 个 substitution assays、187 个不同野生型，按 50% identity 合为 174 families；每 assay 最多抽 1,000 行，实际 63–1,000，未冻结一个总突变行数 | ①在同一 mutant digest 上计算模型序列似然排序。②从可用训练语料建立逐位 profile lookup。③以 family 为 bootstrap 单位比较 MODEL−LOOKUP。④用错误 profile、标签打乱和 `BLOSUM62` 作对照 | MODEL−LOOKUP：ProtGPT2 −0.0143，ProGen2-medium −0.0043，ProGen3 −0.0808；置信区间都不支持模型超过 lookup | 已测且当前接口受检索约束。ProtGPT2 语料识别最完整；ProGen2/3 是偏向模型的下界代理。不能推出记忆、无知识或无生成新颖性 |
| 设计蛋白稳定性预测 | 主结论为 ProtGPT2；ProGen2 base/medium 只作语料未完全识别的边界，ProGen2-small 因控制失败不读取结论 | MegaScale `dataset2` 的设计变体行，例如 `aa_seq, mut_type=A42G, WT_name, WT_cluster, ddG_ML`。130 个无 UniRef50 整序列命中的设计 WT、40 个 design series、110,730 variants；天然对照为 266 WT、124 clusters、404,114 variants | ①排除有整序列命中的设计。②按 WT 计算模型 likelihood 与实测 `ddG_ML` 的 Spearman。③比较 composition、`BLOSUM62`、Kyte–Doolittle 和 3–7-mer。④再用天然域、长度 placebo、组成平衡和 fragment-order sweep 检查 | ProtGPT2 相对 3-mer 为 +0.1439，到 7-mer 缩为 +0.0516；相对 hydropathy 为 −0.3548。模型从天然域到设计序列时整体排名能力下降 | 已测负结果。证书只排除整序列检索；≤6-mer 全覆盖，7-mer 有 99.57% 可见；设计与组成不可分。不能声称已经发现知识或模型“只靠疏水性” |
| 更大 checkpoint 的描述性能力门 | 第一轮为 ProGen2 medium→large→xlarge；独立第二阶段计划 Galactica 1.3→6.7→30B、ProGen3 112M→3B、Qwen2.5-base 0.5→7→32B，并把 ProteinGLM-7B-CLM、RITA_xl 作为新谱系存在性点 | 复用上下文资格队列、ProteinGym 217 assays/174 families，以及 MegaScale 130/40 设计单位和 266/124 天然单位；组 bootstrap 固定 2,000 次、seed 20260825 | ①严格加载、原生渲染、固定 NLL 与错误渲染或 shuffle 负控制。②不同评分方向分层。③只对同家族相邻 rung 判断预注册的复合门。④DMS 要求较大 rung 超过固定 LOOKUP 基线且 raw Spearman 配对增量下界大于零；MegaScale 要求设计和天然侧同时超过 hydropathy、BLOSUM62，并有设计 series 配对增量 | 尚无 large/xlarge 或新模型的能力结果，不能报告门翻转 | EXP-R2-224/225 已预注册但未运行。只允许描述 checkpoint 差异，不能归因为参数量；跨家族和单点模型不得标为门翻转；30B/32B 是否单卡可运行仍需实测资源门 |

## 方向二：用现有方法解释能力如何形成、表示和计算

再用现有方法分析能力如何形成、表示和计算，并比较这些方法在文本与蛋白质模型上是否同样有效。

| 实验 | 模型 | 数据(格式、示例、数量) | 怎么做 | 结果 | 状态与局限 |
| --- | --- | --- | --- | --- | --- |
| 方法功效与抽样敏感性 | 以 gpt2-large 为正对照，并覆盖合格文本和蛋白模型 | 文本为连续文档，蛋白为 `MKT...` 序列，ZymCTRL 另带 EC 前缀。解释通道复核 500 条 Swiss-Prot、122,671 个计分位置、每个 38 个 triplet 的 top-100 事件，并检查 23,586 个 AlphaFold 模型 | ①测目标自身的因果足迹。②计算门槛解析上限、功效和匹配置换 null。③改为从全语料中按固定随机种子抽样。④用 skip-offset 和独立队列检查选择敏感性 | 单层 MLP 目标的因果足迹约 0.02 nats/token，因此要求恢复 80% 的门在该目标上不可达；top-100/122,671 的互信息上限只有 0.0066 nats。七类统计都显示蛋白队列比匹配文本对照更敏感 | 已完成。结果给出方法功效和抽样敏感性的正式边界：不能把“测不到”写成模型没有机制 |
| 匹配架构的重复识别机制 | gpt2-large、ProtGPT2；均为 36 层、宽 1280、20 个头、词表 50,257、约 7.74 亿参数 | 合成重复 probe，结构如 `[前缀][片段 S][同一片段 S]`；主复现实验每模型 256 个 probe、复制长度 64。自然精确和近似重复另作稳健性检查 | ①固定架构、规模和 probe 几何。②统计每个头对前一个对应位置的注意力。③扫描阈值和 probe 类型。④加入 tokenizer identity ceiling 与更大样本复现 | 固定阈值下 gpt2-large 的高分头比例为 ProtGPT2 的 5.46 倍；按各自可达上限校正后约为 5.62 倍 | 已完成并收窄到这一匹配对。两者训练语料中的重复结构仍不同，不能推广成普遍文本—蛋白规律 |
| 分词粒度对照 | ByGPT5-medium；对照 BPE 文本、逐残基蛋白及 GPT-2/ProtGPT2 | 合成字符或 byte 重复，如 `abc...abc...`；每个模型使用 256 个 probe、复制长度 64，并为每个真 probe 构造无重复 collision null，运行 2,000 次 bootstrap | ①用近似“一字符一 token”的文本模型拆开模态与分词。②分别测量每个模型的同 token 竞争和 identity ceiling。③比较旧 induction 固定阈值。④再比较 PAA 能否找回真正因果头 | ByGPT5 在旧 induction 统计上落入蛋白区间，永久推翻面板级模态排序；在 PAA 因果检索中又表现得像文本模型，达到自身机会水平的 5.3 倍 | 已完成。分词能制造部分伪差异，但不能解释全部差异；ByGPT5 同时改变架构和训练语料，不能单独识别字母表的因果效应 |
| 长程影响与距离尺度 | GPT-2-large/xl、Qwen2.5、Llama-3.2；ProtGPT2、ProGen2 small/base/medium | 连续文本或 `MKT...` 蛋白序列；距离分别用 token、字符或残基表示。核心内容距离实验为每个模型设置 5 个独立窗口、每个距离带 256 个扰动案例 | ①在源位置替换 token 或内容片段。②测不同距离目标位置的输出变化。③把横轴从 token 距离改成字符或残基距离。④扫描严格与宽松效应阈值并比较衰减率 | ProtGPT2 与 ProGen2 只因距离单位改变就会交换次序，旧的远距离绝对水平结论撤回。内容距离下，严格大效应阈值显示蛋白模型组衰减较慢；宽松阈值下不构成普遍规律 | 已完成并收窄。字符与残基仍不是同一内容单位，结论只适用于指定模型、窗口和阈值 |
| 多残基分词的干预对齐 | ProtGPT2；ZymCTRL、ProGen2-medium 和 gpt2-large 作对照 | 单点突变记录，如 `WT=MKT...; mutation=A42V`：4 个 ProteinGym assay 各抽 200 个突变。另有 128 个合成 8-residue 重复 probe，如两处 `ACDEFGHI` | ①分别 tokenize 野生型和突变体。②检查序列是否等 token 长并且只有一个 token 改变。③把同一残基片段放在不同上下文相位。④检查两处是否仍得到相同 token 序列 | ProtGPT2 只有 47.0%–54.5% 单点突变保持唯一局部对齐；正式 128-probe 重复实验中只有 21.1% 保持 token 一致 | 已完成。其余样本的残基级 patch 没有唯一数学定义；幸存者又是非随机的 BPE 稳定子集。序列级 likelihood 不受此限 |
| 蛋白训练能力的组件来源 | Llama-2-7B、ProLLaMA Stage 1 | OpenWebText 文本与 Swiss-Prot 64–246 aa。每种模式每次抽样 128 条，共使用两组互不重叠且固定随机种子的样本 | ①逐张量验证两个 checkpoint 可交换。②双向替换输入 embedding、输出头、完整词汇接口或 Transformer 主体。③在相同上下文信息指标上比较混合模型。④用第二组独立样本复现 | 约 4.69 nats 文本损失主要随主体移动；完整接口只解释正向约 2%、反向约 14%–15%，接口内主要由输出头承担，输入 embedding 约 1% | 已完成并复现。效应不具可加性，且 next-token loss 由输出头直接参数化，不能当作一般组件贡献占比 |
| 蛋白训练后的激活结构 | Llama-2-7B、ProLLaMA Stage 1、Stage 2 | OpenWebText 与 Swiss-Prot。每个模型阶段与模式组合取 1,024 条留出记录、每条最多 64 个位置，共 65,536 个 token 位置；蛋白短记录明确拒绝 | ①读取每层 feed-forward 输出。②以 float64 计算协方差谱、参与率、有效秩和 `r99`。③与 isotropic、坐标独立 null 比较。④比较三阶段的逐层峰 | Stage 1 在 28/28 个内部文本层使谱更集中；蛋白高维区从基础模型的早层峰移到 Stage 1/2 的 17–22 层，Stage 2 不再移动蛋白峰 | 已完成。只覆盖一个谱系和一个模块输出位置；不同谱指标不能合成单一“坍缩倍数”，谱形也不等于行为能力 |
| 架构与路由的扰动响应 | Dense：GPT-2、ProtGPT2、ZymCTRL、ProGen2；MoE：ProGen3-112M；同权重 ProLLaMA Stage 1/2 | 原生文本或 Swiss-Prot 64–246 序列。主要实验条件通常各用 128 条记录、每个强度 3 个随机方向；ProGen3 路由诊断另用 256 条、41,155 个残基位置，因此没有统一 n | ①在每层 MLP 输出加入相对范数匹配的各向同性扰动。②比较 Dense/MoE 和同权重双模式。③用随机分组、残基分组和路由边界作控制。④检查结果能否由路由解释 | 蛋白模型内部差异可大于文本—蛋白差异；ProLLaMA 双模式排序在 Stage 1 与 Stage 2 间反转。Dense 与 MoE 都能出现方法失败 | 已完成。排除了“蛋白、MoE 或路由必然更脆弱”等简单解释，但没有识别统一近端机制 |
| 模型规模与扰动响应 | ProGen2 small/medium/large/xlarge，约 1.51 亿、7.65 亿、27.8 亿、64.4 亿参数 | Swiss-Prot 64–246 aa，一字母序列如 `MKT...`。每个规模、每个条件 128 条、3 个扰动方向、1,000 次 bootstrap；主样本与错开抽样起点的独立样本都运行 | ①固定 ProGen2 谱系、逐残基 tokenizer 和评测长度带。②扫描相同的相对 MLP 扰动强度。③比较四个规模。④检验排序是否单调且能否在第二组独立样本中重现 | 除最小扰动点外，各强度和条件均为 xlarge > medium > large > small；medium 始终高于参数更多的 large | 已完成。不存在简单规模律；深度、宽度和参数量共同变化，不能单独识别参数量效应 |
| 中间层预测读出 | gpt2-large、ProtGPT2、ZymCTRL、ProGen2 small/base/medium | OpenWebText 连续文档与 Swiss-Prot `MKT...` 序列。蛋白比较包含 64–120 和 64–246 两个长度带；canonical audit 未统一规定每个模型的独立抽样次数 | ①比较 Logit Lens 与 Tuned Lens。②用有限差分验证输出 Jacobian。③把 Jacobian 秩与输出词表孔径分开。④修复宽带截断。⑤对化学性质投影加入 shuffled/rank-matched null | Tuned Lens 在所有已测模型和两个长度带都更好；扩大到 64–246 后，蛋白侧优势缩小 54%–82%，文本对照增加约 4%。小残基词表的低秩主要是输出孔径 | 已完成，F6 冻结。Lens 结果必须同时写长度带；性质投影没有匹配随机分区就不可解释 |
| 可解码表示与因果使用 | ProtGPT2、ProGen2-medium；ZymCTRL 等无有效因果分母的实验条件被拒绝 | 记录如 `{sequence, Pfam_family}` 或 `{sequence, residue_index, SS3}`。相关流水线覆盖 820 条 Swiss-Prot、44,626 个标注残基 | ①按家族或近重复组切分。②训练线性/MLP Probe。③用 `LEACE` 擦除读出子空间。④重新测模型下一 token loss。⑤比较“可解码强度”和“行为损伤” | ProtGPT2 的 Pfam 可解码技能约 +0.705，但擦除后预测改善约 −0.179 nats；ProGen2-medium 的 SS3 行为依赖只有约 +0.031 nats，远小于可解码强度 | 已完成。“可解码”不等于“模型使用”。ProtGPT2 多残基 BPE 不允许启发式映射到 SS3、burial 等残基标签 |
| 接触关系的状态与注意力读出 | ZymCTRL、ProGen2 small/base/medium | 接触记录如 `{sequence, i, j, contact, sequence_distance, decoy}`；按 Pfam family 和 k-mer cluster 隔离。canonical audit 未统一记录正式接触对总数 | ①先修正结构筛选顺序。②训练单位置 hidden-state 控制。③读取原始 attention。④加入距离匹配 decoy 和只含序列间距的控制。⑤比较家族外接触预测 | 修正后，attention 相对简单控制的优势由约 0.10 缩到 0.03–0.05，相对纯间距控制约 0.03 | 已完成并关闭当前跨位置稀疏特征动机。只适用于 token 与残基明确对齐的模型；ProtGPT2 不可启发式对齐 |
| 生化属性的层间读出 | gpt2、gpt2-large、ProtGPT2、ProGen2 base/medium、ZymCTRL | 400 个 Swiss-Prot 蛋白，64–246 aa；Pfam 分成 188 个 seen 与 187 个 unseen families。每个残基带身份、疏水性、电荷和体积标签 | ①在每层比较残基身份和粗粒度化学属性的读出。②加入 shuffled-property 与 rank-matched 小字母表 null。③要求属性比残基身份更早稳定。④固定 ZymCTRL 的 EC 标签检查泄漏 | 三个无条件蛋白模型中，属性没有比残基身份更早稳定；charge/volume 在各蛋白模型中均为 0/11 层通过。ZymCTRL 的疏水性正结果在固定 EC 标签后从 7/11 降为 0/11 | 已关闭，后续因果阶段未获授权。只否定这一输出 Lens，不说明残差流没有生化信息 |
| 重复序列的注意力机制 | 核心匹配对为 gpt2-large/ProtGPT2；另含文本面板、ZymCTRL 和 ProGen2 | 合成两次重复片段；自然精确或近似重复文本/蛋白。203,063 个合格 EC 记录中找到 48 个精确重复和 817 个近似重复；同源分层使用 UniRef50 | ①统计 prefix-matching 头并扫描阈值。②加入 tokenizer collision ceiling。③穷举头级因果效应并做路径修补。④关闭 DIAMOND masking 后按训练语料相似度分层。⑤修正 bootstrap 单位和 collision null 特异性 | 面板级“蛋白 induction 更少”被 ByGPT5 和失效的 collision null 推翻；匹配对仍保留 5.46×/5.62× 差异。路径修补未证明蛋白通路更间接，同源分层也未定案记忆贡献 | 已完成并收窄。只可陈述匹配对；当前没有真正低同源蛋白层，也没有对称文本训练语料控制 |
| 注意力筛选的因果准确性 | 文本：GPT-2、Qwen、Llama、ByGPT5；蛋白：ProtGPT2、ZymCTRL、ProGen2 small/base/medium | 单条记录形如 `{native_record, query_position, antecedent_position, predicted_token, matched_decoy_positions}`。不同正式实验和失败审计对“模型 × 独立抽样”的统计口径不同，因此不合并成一个总 n | ①穷举消融，得到每个模型自己的因果 top-k。②用 PAA 分数找回 top-k。③与自身机会水平和只知道层号的基线比较。④排除格式 token 并固定随机种子。⑤在发现集和留出集上检验可能的失败因素 | 所有逐残基蛋白模型只达到或低于自身机会，所有文本模型至少达到 3.6 倍机会；ByGPT5 也通过。ProtGPT2 的表面通过主要由层深携带 | F1–F4 已冻结，既定面板关闭。结论是筛选器在这些蛋白接口上失真，不是模型没有 copy-suppression 机制 |
| MLP 与注意力的行为贡献 | 六个可测文本模型；ProtGPT2、ZymCTRL、ProGen2 small/base/medium。DialoGPT-small 因资格失败排除 | 连续原生文本或 64–246-residue 蛋白序列。使用从全语料中按固定随机种子抽取的样本；F7 未冻结统一记录数 | ①分别消融 MLP 和 attention。②以每个模型自己的上下文信息归一化。③从文件顺序队列改为全语料随机队列。④用第二组独立样本复核资格和结果 | MLP/attention 代价比在文本为 1.491–2.070，在蛋白为 0.530–1.129，范围不重叠；重新抽样后各模型的结果复现到约两位小数 | F7 已冻结，是当前较稳健的模型差异。该比值不是贡献分解且可以大于 1；尚未完成分词尺度校正 |
| 稀疏字典的表示质量 | ProtGPT2、ZymCTRL、ProGen2-medium；联合谱系另测 ProLLaMA | 训练流为 UniRef50、ZymCTRL EC FASTA 或模型原生文本；评测为连续文档或 `MKT...` 序列。不同字典实验的训练 token 数不同，没有统一 n | ①训练不同宽度和预算的字典。②检查死特征、FVU/NMSE 和 Probe 可恢复性。③人工或自动注释特征。④做 steering。⑤检查更宽字典是否改变结论 | 扩宽能减少死特征、改善部分 Probe，却没有带来稳定 steering；FVU、命名性和行为保持的排序不一致 | 已完成方法审计。死特征、低重建误差、可命名性和 Probe 都只能作诊断，不能单独构成机制证据 |
| 稀疏表示的行为保持 | gpt2、gpt2-large；ProtGPT2、ZymCTRL、ProGen3-112M | 原生连续文本或蛋白一字母序列。F11 主比较统一到 64–246 residues、固定 12× 字典宽度；蛋白另扫描约 64–1022 residues 的四个长度带 | ①逐层用字典重建替换 MLP。②以原模型和均值消融定义行为恢复。③加入闭式仿射、原始神经元和随机扰动基线。④控制深度、训练预算、路由、分词和长度带。⑤撤回连文本正对照也失败的 causal-rank gate | 固定 12× 下文本恢复 0.9084–0.9322，蛋白只有 0.0916–0.1641；四个长度带上蛋白仍为 0.0425–0.2724，与文本无重叠。更好重建不保证更多行为恢复 | F11 已冻结；差异对长度带稳健，但蛋白侧字典宽度稳健性和近端原因仍开放。不能推断蛋白表示必然更稠密 |
| 神经元与匹配扰动对照 | gpt2-large、ProtGPT2、ZymCTRL；另含 ProGen2 规模阶梯和 ProLLaMA 双模式 | 模型原生文本或蛋白队列。每个子实验的模型和 n 不同，没有统一样本数 | ①把学习字典换成原始 MLP 神经元。②构造与字典误差同范数、同夹角的随机扰动。③比较模型、模态和规模排序。④检查误差方向是否异常 | 原始神经元在部分蛋白模型上胜过学习字典，指向字典训练问题；误差方向并不异常，扰动耐受也不按模态或参数量单调 | 已测但未晋升为普遍规律。只能陈述具体模型，不能说蛋白表示必然更稠密或更脆弱 |
| 蛋白生成的特征控制 | ZymCTRL v2；ProGen2-medium 只用于确认干预 hook | 8 类 EC 生成队列：lysozyme、trypsin、ADH、catalase、DNA polymerase、lipase、kinase、carbonic anhydrase；另用 100 个真实 lysozyme 与 100 个长度匹配 UniRef50 蛋白校准指标 | ①按 CLT 直接效应选择特征。②在 L3/L12/L30 做 top-k 流形内 steering。③用 Pfam、CLEAN、ESMFold/Foldseek 评价。④先验证指标能区分真实与随机蛋白。⑤比较 steering 与未干预生成 | 8 类中显著改善为 0/8；真实/随机校准强分离，说明外部指标工作，但 steering 没有优于未干预生成 | 已关闭强 steering、药物设计和湿实验主张。hook 确实移动 logits，因此负结果不能归因于干预路径断开 |
| 蛋白训练前后的表示差异 | Llama-2-7B → ProLLaMA Stage 1；文本和蛋白模式分别运行 | 同输入、同位置、同层的 OpenWebText 与 Swiss-Prot 队列；使用近重复分组和逐层非退化 `r99` 审核字典基底 | ①先做 offset、正交和线性对齐。②训练独立字典与 Crosscoder。③扫描稀疏权重。④检查拟合充分和特征极化是否同时成立。⑤对单 latent 做匹配随机消融 | 线性对齐仍留残差；Crosscoder 没有同时满足充分拟合与极化。文本侧约 13%–16% live latents 显示超出控制的差异依赖 | 受限候选。差异依赖不等于新增或删除特征；基础模型蛋白行为不可测，蛋白侧因果差分无定义 |
| 同权重文本—蛋白计算子空间 | ProLLaMA Stage 1/2 双模式；Llama-2 作训练前参考 | OpenWebText 与 Swiss-Prot；匹配记录数、每条记录位置数和总计分位置。正式队列包含 3 个 checkpoint、2 个独立语料抽样，共 6 个实验单元 | ①逐层测 occupancy。②消融自身主方向测 necessity。③与随机同维子空间比较 overlap。④做 2×2 cross-mode driveability。⑤把损伤拆成 unigram 和残余上下文部分 | 合成已知答案的 8 项检查全部通过；正式运行中没有任何层得到许可的“分离子空间”或“共享子空间”判决。Llama-2 蛋白模式在多数层可测到非一元频率损伤，但有的层残余占比不足，有的层找不到文本侧必要秩，因此仍不能形成重叠结论 | 已完成，但没有得到共享或分离计算子空间的正式结论。此前按固定 0.30 门槛拒绝 Llama-2 蛋白模式的做法已撤回；重新测量没有把拒绝变成正结论。任何子空间重叠都不是生物知识证据 |

**相关工作边界。** ProteinGuide 在冻结的蛋白生成模型之外训练属性预测器，并在采样时用该预测器重加权离散转移率；论文展示了稳定性、金属结合和 TadA 等属性的生成控制。这证明外部预测器可以有效引导采样，但不等于生成模型内部已经表示或因果使用同一属性，也没有排除同源检索、组成或其他表面捷径。它与本项目的内部特征 steering 不是同一种干预，因此当前作为相关工作引用，不复现为正式实验。

## 方向三：开发新方法判断模型学到了什么

当现有方法不足时，设计新的对照和解释方法，区分语料规律与生物知识。

| 实验 | 模型 | 数据(格式、示例、数量) | 怎么做 | 结果 | 状态与局限 |
| --- | --- | --- | --- | --- | --- |
| 突变互作与语料耦合 | 真实生物阶段未读取模型；gpt2-large、ProtGPT2、ProGen2-medium 只用于合成仪器检查 | 双突变行，例如 `A42G:B57V`，且两个 single mutation 必须在同 assay 实测。51 assays、48 identity clusters、7,298 个至少含 10 个 doubles 的位置对；可构建语料耦合者为 23 assays、22 clusters、5,325 对 | ①用两个 singles 建 additive prediction。②用 cross-fitted isotonic regression 去掉全局非线性，得到 specific epistasis。③从语料比对建立 APC 校正耦合并与列置换 null 比较。④只有多数蛋白通过语料正对照才允许读取真实模型 | 语料耦合正对照 0/22 通过，因此模型 epistasis、内部中介和因果阶段都没有运行；合成文本任务可以找回目标 pair | Stage 0 仪器停止。只说明当前 DIAMOND/UniRef50 hit-list 接口不能建立二阶减项，不说明语料或模型没有二阶知识 |
| 跨模态概念对齐 | Llama-2-7B 参考；ProLLaMA Stage 1/2 的文本和蛋白模式 | sequence-description pair，例如 `MKT... ↔ “[MASK] catalyses ...”`。Swiss-Prot `fullName + CC FUNCTION`；池 20,000 对、19,385 个不同序列、17 个 concepts；主 eval 为 4,499 条、3,752 组，gallery 1,000 | ①按近重复组和家族切分并遮蔽概念名。②在预注册 layer 22 拟合 mean/Procrustes/affine 对齐。③比较 shuffled、rank、description-only、composition 和 3-mer。④只有蛋白模式清除全部上界才允许文本方向因果注入和外部 Pfam 验证 | Stage 1 蛋白 excess 为 +0.0097，低于 3-mer 的 +0.0272；Stage 2 同向。相同仪器在三个 checkpoint 的文本模式通过 | 正式归入重组并停止；Stage 36 因 STOP-35 从未运行。只覆盖一个谱系、一个层和一次独立抽样，且是序列级表示 |
| 催化活性与家族统计的矛盾检验 | ProGen2-small、ProGen2-medium；ProtGPT2 与 ZymCTRL 被仪器拒绝 | matched pseudokinase pair，例如 `(RYK_dead, bit-score-matched active kinase)`。原池 461 条；主队列 15 对；另有 8 个 `active_despite_degradation` 记录，与 dead 侧组成 23-group counter-stratum | ①用 20-bit caliper 中和 Pfam bit score。②用催化位点 motif reader 作生物参照。③对相同 anchor 替换计算模型 NLL contrast。④同时比较 1–7-mer、最近活性激酶检索和 composition。⑤检查 motif-degraded active stratum | ProGen2-small AUROC 0.9200、medium 0.8844，但低于 7-mer ceiling 0.9733；counter-stratum 约 0.73，区间覆盖机会 | 小规模预算试运行的读出已归入重组。ProtGPT2 单符号覆盖不足，ZymCTRL 的 EC 标签携带标签，均被拒绝；双 caliper 可达性仍待测 |
| 折叠与序列统计的矛盾检验 | 最终关闭不依赖模型；ProGen2-small/medium 只有小规模试运行，ProtGPT2/ZymCTRL 被拒绝 | fold triple：`(anchor, composition-near/different-fold C_seq, sequence-far/same-fold C_str)`；199 triples、367 members、346 near-duplicate groups | ①用 DIAMOND 排除可对齐关系。②验证组成与折叠给出反向排序。③计算 fragment、composition 和 Pfam/profile ceiling。④只用 anchor 前缀运行 jackhmmer。⑤若任一统计成员与结构预测同向，就在读取模型结论前关闭 | Profile 找回同折叠伙伴 56/199，找回组成伙伴 0/199，说明 alignment-clean 不等于 profile-clean | 模型无关地归入重组并关闭。F15 是队列构造方法结论，不是模型负结果；只覆盖人类 AlphaFold 子集、staged Swiss-Prot 和这一 triple 构造 |
| 氨基酸化学与语料统计的矛盾检验 | ProGen2-small/base/medium、ZymCTRL；两次确认各有一个独立 ByGPT5-medium 文本控制。GPT-2-large 与 ProtGPT2 被拒绝 | 20 种残基形成 190 个无序对，例如 `(A,V; 理化轴, 语料轴)`。早期单侧版本只用 48 条序列；当前双侧版本为每个蛋白模型冻结 4,096 条轴构造记录和两个各 4,096 条的独立确认队列，并排除队列间精确相同或 5-mer containment≥0.5 的记录 | ①替换输入 embedding，测每个残基对造成的 likelihood damage。②以疏水性、电荷、体积和极性定义理化轴。③用相同记录上的 order-7 fragment substitution damage 定义相反的语料轴和匹配上界。④按近重复序列组×被替换符号组做 2,000 次交叉 bootstrap。⑤首次双侧确认因队列内容重叠按预注册规则关闭；新版本确定性填充三个跨队列去重的队列，并要求两个独立确认分别通过 | 截至 2026-08-25，14 个单元中 9 个正常结束、3 个运行、2 个待运行，零失败。四个蛋白轴均构造成功，两个 ByGPT5 控制均通过；ProGen2-base、ProGen2-medium 和 ZymCTRL 的第一次确认标签为 `CHEMISTRY`，但它们的第二次确认尚未完成，ProGen2-small 两次确认均未完成。这些部分输出不能形成模型臂或整个实验的科学结论 | 正在运行，尚无可晋升结果。已完成单元的精确 JSON 和 SHA-256 sidecar 均通过核验；队列终态、5 个剩余单元和运行后资源快照仍缺。任一模型只有两次确认都满足全部预注册条件并返回 `CHEMISTRY`，才构成候选正结果；即使如此，也只说明该 checkpoint 的 likelihood 对输入行替换呈现与所声明化学描述一致的敏感性，不证明下游因果使用、机制或生物知识。GPT-2-large 与 ProtGPT2 的单残基覆盖约 0.5%，属于仪器拒绝 |

## 补充实验与测量前提

以下项目作为正式实验之后的附录性补充。它们不单独回答模型具有什么能力、机制如何工作或模型学到了什么，而是确认后续实验使用了有效的接口和独立评价单位。

| 补充检查 | 对象与数据 | 怎么做 | 关键结果 | 对后续实验的作用 |
| --- | --- | --- | --- | --- |
| 原生接口与计分校验 | ProtGPT2、ZymCTRL、ProGen2、Galactica、InstructProtein、ProLLaMA 等正式候选；使用文本记录或蛋白一字母序列。安全格式示例：ProtGPT2 为 `<EOT>` 后接每 60 residues 换行的 FASTA；ZymCTRL 为 `3.2.1.17<sep><start>MKT...<end>`；ProGen2 为方向标记加序列，如 `1MKT...`。这些检查跨多个资格流程，没有统一样本量 | ①按模型训练格式渲染。②核对提示、边界和实际计入 next-token loss 的位置。③检查输出概率是否落在正确词表。④比较正确与错误格式的损失和序列反转代价。⑤拒绝输出语义或上下文信息不合格的模式 | ProtGPT2 的错误 FASTA 格式产生约 1.42 nats/token 差异；ZymCTRL 的 EC 标签会携带条件信息；InstructProtein 的残基化输出词表会破坏文本读出 | 已成为后续实验的接口前提。不同模式和输出空间的数值不能直接横比；通过校验只说明接口可用，不表示模型具有生物能力 |
| 近重复切分与留出泄漏检查 | 方法不依赖单一模型；在 ProLLaMA 字典训练和序列—描述实验中复核。10,240-record 池的按记录留出集有 2,048 条，其中 871 条仍有 ≥95% 近亲、370 条完全相同 | ①比较按记录、精确字符串和近重复组切分。②用 DIAMOND 检查留出集中的近亲序列。③把训练流和验证流一起分组。④在 OpenWebText 上运行同样检查作对照 | 普通记录划分后，留出集中仍有 42.5% 的蛋白带有 ≥95% 同一性的近亲，18.1% 完全相同；按组切分把 ≥90% 跨集近亲降为零 | 已成为评价前提。蛋白实验不能只做精确去重或更换随机种子；这项检查保证留出单位更接近独立，但不解释模型机制 |

## 当前结论与下一步

| 主题 | 当前可以下的结论 |
| --- | --- |
| 模型比较 | 分词、原生输入输出接口、训练阶段、语料、架构和抽样队列都会显著影响跨模态比较；当前没有简单、普遍的模态或规模规律 |
| 方法审计 | 多种文本可解释性方法在蛋白模型上存在已测得的失真或适用边界；可解码、可重建、可命名或注意力清晰都不等于因果忠实 |
| 生物知识 | 目前没有任何方法完成“超过重组上界—因果使用—独立生物验证”链条，也没有建立新的生物学知识 |
| 不能声称 | 不能断言模型已经学到、没有学到或只是在记忆；不能把无整序列同源、超过随机基线、合理生成物或结构预测写成知识证据 |
| 方向一、二下一步 | 同权重模式子空间实验已经完成但没有得到许可的共享或分离子空间结论；下一步优先补充第二个可靠的多残基蛋白模型、尽可能匹配的 BPE 文本对照，以及不依赖 token 对齐的序列级测量，而不是扩大未通过判定门的子空间扫描 |
| 方向三下一步 | 等待当前字母表—化学队列进入终态，并核对全部精确 JSON、SHA-256 sidecar 和运行后资源快照；不得从单次确认或部分输出推断结果。只有同一模型的两个独立确认都满足全部预注册条件并返回 `CHEMISTRY`，才进入另行预注册的内部因果使用、未见家族泛化和独立生物验证 |

# InterpretabilityTransfer 研究方向与当前结论

**更新日期：** 2026-09-17

本项目比较纯文本、纯蛋白质和语言–蛋白质联合生成模型。

研究分三步：先比较三者具有什么能力，再用现有方法解释能力如何形成、表示和计算；现有方法不足时开发新方法判断模型学到了什么。

## 主要模型

本表按谱系列出本方向实际使用或计划使用的 checkpoint。同一谱系的笼统覆盖不等于每个尺寸都已跑过同一实验；列入本表不等于接口、数值或实验已经获得资格或已有完成结果。

上下文长度是各 checkpoint 声明的服务上限，不是本项目的计分窗口：上下文信息资格测量默认在 384 token 窗口内计分。该值取自各 checkpoint 的 config，tokenizer 与 config 冲突时以 config 为准。

| 模型谱系 | 模态与比较作用 | Tokenizer / 基本单位 | 关键架构 | 训练数据或阶段 | 上下文长度 | 特殊限制 |
| --- | --- | --- | --- | --- | --- | --- |
| GPT-2 | 纯文本基础对照和规模阶梯 | 50,257 词表 BPE | gpt2：12 层、宽 768，约 1.24 亿；gpt2-medium：24 层、宽 1,024，约 3.55 亿；gpt2-large：36 层、宽 1,280，约 7.74 亿；gpt2-xl：48 层、宽 1,600，约 15.6 亿 | WebText | 1,024 tokens | config 未声明 pad token，填充复用 EOT（50,256） |
| DialoGPT-small | 文本语料对照 | 与 GPT-2 相同的 50,257 词表 BPE | 与 GPT-2 small 相同的 12 层、宽 768 | Reddit 对话 | 1,024 tokens | config 未声明 pad token，填充复用 EOT（50,256） |
| Qwen2.5 0.5B/7B/32B | 非 GPT-2 文本架构对照，兼纯文本规模阶梯 | 0.5B 词表 151,936；7B/32B 配置词表 152,064 | 稠密 Qwen2：0.5B 24 层宽 896；7B 28 层宽 3,584；32B 64 层宽 5,120 | 多语言、代码和数学混合 | 0.5B 32,768；7B/32B 131,072 tokens | 不添加 BOS；`<\|endoftext\|>` 兼作文档分隔与填充 |
| Llama-3.2-3B | 非 GPT-2 文本架构对照 | 128,256 词表 | 28 层、宽 3,072 | 网络文本与蒸馏数据 | 131,072 tokens（位置外推自 8,192） | tokenizer 自动添加 BOS（128,000） |
| Qwen3-8B-Base | 纯文本预训练基座候选，用于检验现有文本侧发现是否依赖 Qwen2.5 这一代 | 151,936 词表；具体基本单位尚待核验 | 稠密 Qwen3 自回归 Transformer：36 层、宽 4,096，GQA 32 查询/8 KV，QK-Norm、RoPE、SwiGLU；约 82 亿参数 | 预训练基座；具体语料混合未确认 | 32,768 tokens | 不添加 BOS；pad 与 EOS 同为 151,643 |
| ByGPT5 small/base/medium | 字节级文本对照和规模阶梯 | 384 符号字节词表，英文约一字符一 token | T5 式仅解码器：small 4 层宽 1,472；base 6 层宽 1,536；medium 12 层宽 1,536 | 仓库未确认训练语料 | 未声明（无绝对位置表） | 字节级，不添加 BOS/EOS；本地 config 的 auto_map 与 use_cache=false 是它在本项目所用 transformers 上离线前向成功的条件 |
| ProtGPT2 | 多残基蛋白对照 | 50,257 词表，多残基 BPE；原生 FASTA 输入 | GPT-2 36 层、宽 1,280、20 个头，约 7.74 亿参数 | UniRef50 | 1,024 tokens | `<EOT>` 前缀加每 60 残基换行，无尾随 `<EOT>` |
| ZymCTRL | 带 EC 条件的逐残基蛋白模型 | 458 词表，基本单位为单残基 | GPT-2 36 层、宽 1,280 | EC 标注的 UniProt 蛋白 | 1,024 tokens | EC 条件输入，缺 EC 标签即跳过；ProteinGym 217 个 assay 全部跳过 |
| ProGen2 | 逐残基蛋白谱系和规模对照 | 32 词表，单残基 token，带生成方向标记 | small：12 层宽 1,024，约 1.51 亿；base/medium：27 层宽 1,536，约 7.65 亿；large：32 层宽 2,560，约 27.8 亿；xlarge：32 层宽 4,096，约 64.4 亿 | small/medium/large/xlarge 使用 UniRef90 与 BFD30；base 的具体训练混合未确认 | small/medium/large/xlarge 1,024；base 2,048 tokens | 渲染以方向标记 `1` 开头；输出按完整列计分、不裁剪；large 的 51,200 列表头使截断曲线不可算；small/base 存 float32 但上下文信息阶段按 bfloat16 计分，medium 单独按 float32，ProteinGym 阶段一律 bfloat16 |
| RITA-xl | 纯蛋白自回归模型，ProteinGym 独立单点对照 | 单残基 token，26 维输出词表 | 24 层、宽 2,048、32 个头，约 12 亿参数 | UniRef100 | 1,024 tokens | 文档边界 `<EOS>` 前置，末尾 `<EOS>` 由 tokenizer 附加且与前者同为 id 2；两端 `<EOS>` 不计分、残基 1 计入；不发明 pad、不填充；按 float32 计分 |
| ProGen3 112M/3B | 稀疏 MoE 蛋白对照，兼蛋白规模阶梯 | 134 词表；更精确的基本单位未确认 | 112M：10 层、宽 384；3B：24 层、宽 1,280；两档均为每层 8 专家、每 token 选 2 | 112M 的模型卡未声明预训练语料；3B 声明 Profluent Protein Atlas v1 | 65,536 tokens | 渲染 `<bos>1…2<eos>`，标记 token 不计分；上下文信息只按 N→C，ProteinGym 按双向均值 |
| ProteinGLM-7B-CLM | 纯蛋白因果生成候选，补充稠密 GLM 谱系参照 | 128 词表；仓库已测 tokenizer 按单残基切分 | 稠密自定义 GLM 因果解码器：36 层、宽 4,096、32 头 MHA，一维 RoPE、GEGLU；非 MoE | 蛋白 CLM 预训练；不可与 100B 旗舰或 MLM 版本混同；这份 7B 权重对应的具体训练混合尚未独立核实 | 1,024 tokens | 需 float32；`<gmask><sop><eos>` 前缀后计分残基 2..L；1,024 token 窗口拒绝更长序列 |
| ProtGPT3-1.3B | 蛋白自回归 MoE 候选，作为 ProGen3 之外的另一谱系参照 | 31 词表；具体基本单位尚待核验 | Mixtral 式仅解码器 MoE：17 层、宽 1,024，8 专家、每 token 选 2，GQA 16 查询/4 KV，RoPE、SwiGLU；约 13.3 亿总参数，active 未声明 | 蛋白序列因果预训练基础版；不混用 DPO 或 MSA 版本；具体语料混合尚待核验 | 1,025 tokens | 需 `<\|bos\|>` 加方向 token（`1` N→C、`2` C→N），两个标记不计分 |
| Galactica 125M/1.3B/6.7B/30B | 语言–蛋白联合模型对照，兼联合规模阶梯 | 50,000 词表文本 BPE；蛋白为同一词表中的单字母残基，另加残基分隔标记 | 同为 OPT 形状解码器：125M 12 层宽 768，约 1.25 亿；1.3B 24 层宽 2,048；6.7B 32 层宽 4,096；30B 48 层宽 7,168 | 科学语料，其中蛋白低于 1% | 2,048 tokens | 蛋白需 `[START_AMINO]…[END_AMINO]` 渲染，计分用 FP32 协议（30B 单卡峰值 115,153 MiB）；125M 蛋白模式未识别 |
| InstructProtein | 语言–蛋白联合对照，但不是 Galactica 的规模阶梯 | 文本词表加专用残基 token（`<protein>`、`</protein>` 与 `ƤA..ƤY`） | OPT 形状：24 层、宽 2,048，与 Galactica 1.3B 同形状 | UniRef100 继续预训练与指令微调 | 2,048 tokens | `<protein>ƤA..ƤY</protein>` 外加 tokenizer BOS；残基化输出词表使文本模式不可读；2,048 窗口排除 4 个 assay |
| Llama-2 → ProLLaMA | 同一谱系的文本、蛋白和训练阶段对照 | 共享 32k SentencePiece；蛋白约 1.53 residues/token | Llama-2-7B、ProLLaMA Stage 1、Stage 2 三个 checkpoint 共享 32 层、宽 4,096、32 个头 | Llama-2 → UniRef50 继续预训练 → 蛋白指令微调 | 4,096 tokens | 逐 token 求和与逐残基求和不是同一泛函；Stage 2 的蛋白模式在其调优模板之外 |

本文把围绕同一科学问题的重复抽样、参数扫描和修复运行合并为一个实验族，只列能产生独立结论、撤回旧结论、改变归因或决定路线是否继续的实验。表中的“数据”是评测集或实验样本，不一定是模型的完整训练语料；训练语料或样本量无法准确确认时不作猜测。示例只解释记录格式，不代表真实数据内容。详细证据、限制和撤回以 [`docs/INTERPRETABILITY_TRANSFER_AUDIT.md`](docs/INTERPRETABILITY_TRANSFER_AUDIT.md) 为准，尚未纳入正式结论的最新记录见 [`docs/EXPERIMENT_LOG.md`](docs/EXPERIMENT_LOG.md)。

| 状态 | 含义 |
| --- | --- |
| 已完成 | 已有可引用结果，但只能在表中写明的模型、数据和设计范围内解释 |
| 已关闭或停止 | 已回答当前问题、触发停止条件，或证明现有设计无法区分目标假设 |
| 已收窄或撤回 | 测量仍保留，但原来的宽泛解释不再成立 |
| 正在运行 | 正式测量已启动，但终态结果尚未齐全，不能从部分输出形成科学结论 |
| 开放或未运行 | 方法校验、小规模试运行或预注册已经完成，但正式模型实验尚未启动 |

## 共同测量前提

以下是三个方向共用的测量入口，不是第四个研究方向，也不构成能力榜。先确认模型接口、计分和评价单位有效，再分别判断任务表现、机制解释与知识鉴别能支持什么结论。

| 检查 | 模型与数据 | 方法 | 结果 | 状态与边界 |
| --- | --- | --- | --- | --- |
| 上下文信息资格门 | 主表生成模型。文本用 OpenWebText，蛋白用 Swiss-Prot 或带 EC 标签的 UniProt。每个模型 8 个区块、每区块 200 条，4,000 条独立记录拟合一元基线。当前覆盖 GPT-2 四档、DialoGPT-small、Qwen2.5-base 0.5B/7B/32B、Llama-3.2-3B、ByGPT5 small/base/medium、Qwen3-8B-Base、ProtGPT2、ZymCTRL、ProGen2 small/base/medium/large/xlarge、ProGen3 112M/3B、ProtGPT3-1.3B、ProteinGLM-7B-CLM、RITA-xl 的各自模态，以及 Galactica 125M/1.3B/6.7B/30B、InstructProtein、Llama-2-7B、ProLLaMA Stage 1/2 的文本与蛋白模式 | 删除评分集与参考集的完全重复内容，按近重复组做 2,000 次配对 bootstrap，校正参考集重采样位移；识别要求校正区间下界大于零，比值另需分母资格。区间含零或失败也是完成的测量。ProteinGLM 计分位置为残基 2..L；RITA 逐条计分、不使用 pad | DialoGPT-small 在 8 个区块上均未识别出上下文信号；GPT-2 四档、Qwen2.5-0.5B、Llama-3.2-3B、ByGPT5 三档、ProtGPT2、ZymCTRL 与 ProGen2 small/base/medium 均识别。Qwen2.5 三个 checkpoint 的 8 个区块位移校正后 95% 置信区间上下端依次为 [+4.7151,+4.9499]、[+5.3046,+5.5489]、[+5.4725,+5.7533] nats/token，不是合并的总体 95% 区间。识别（8 个区块校正下界均大于零）的还有 Qwen3-8B-Base [+5.2716,+5.5215]，Llama-2-7B 文本 [+5.3131,+5.5085] 与蛋白 [+0.0452,+0.1178]，Galactica 125M 文本 [+3.7227,+3.9440]、1.3B 文本 [+4.3574,+4.5806] 与蛋白 [+0.0304,+0.0947]、6.7B 文本 [+4.6051,+4.8293] 与蛋白 [+0.1196,+0.2129]、30B 文本 [+4.7419,+4.9693] 与蛋白 [+0.2452,+0.4762]，ProLLaMA Stage 1 文本 [+0.4269,+0.8780] 与蛋白 [+0.4868,+0.6424]、Stage 2 文本 [+0.3986,+1.0487] 与蛋白 [+0.4598,+0.6040]，ProGen2-large [+1.1115,+1.4309] 与 xlarge [+1.4891,+1.8276]，ProGen3 112M [+0.8276,+1.1306] 与 3B [+1.5228,+1.9188]，ProteinGLM-7B-CLM [+1.5648,+1.8978]，RITA-xl [+1.1942,+1.5878]，InstructProtein 蛋白 [+1.8885,+2.2812]，ProtGPT3-1.3B [+0.8563,+1.1788]。未识别（8 个区块校正区间整体在零以下，仍是完成的测量）：Galactica-125M 蛋白 [−0.1873,−0.0660]，InstructProtein 文本 [−16.5512,−15.9196] | 覆盖的各 checkpoint 与模态的资格门均已完成；未识别的是 Galactica-125M 蛋白与 InstructProtein 文本。ProtGPT3-1.3B 按原生格式识别。联合 checkpoint 的两种模式分别过门，数字不能横比。识别与否不是能力排名，也不改写已有区间，也不阻止后续实验。区块选择可使同一模型移动约 0.10–0.17 nats；结果只适用于声明的接口、语料、长度带和 tokenizer。Qwen2.5 的上升没有任务级能力资格，也不能归因于参数量 |
| 原生接口与计分校验 | 主表里实际计分的生成模型。文本按各家 tokenizer 喂连续文档；蛋白用一字母序列，并加上该 checkpoint 的训练标记。已核对的格式包括：ProtGPT2 为 `<EOT>` 后接每 60 residues 换行的 FASTA；ZymCTRL 为 `3.2.1.17<sep><start>MKT...<end>`；ProGen2 各档为方向标记加序列，如 `1MKT...`；Galactica 为氨基酸起止标记；ProLLaMA 为共享 SentencePiece；RITA 为前置文档边界 `<EOS>` 加序列；ProteinGLM 为 `<gmask><sop><eos>` 前缀、残基 2..L；ProGen3 为专用 MoE 加载后的 N→C、标记 token 不计分；ProtGPT3 为 `<\|bos\|>` 加方向 token 后接序列。没有统一样本量 | ①按模型训练格式渲染。②核对提示、边界和实际计入 next-token loss 的位置。③检查输出概率是否落在正确词表。④比较正确与错误格式的损失和序列反转代价。⑤拒绝输出语义或上下文信息不合格的模式 | ProtGPT2 的错误 FASTA 格式产生约 1.42 nats/token 差异；ZymCTRL 的 EC 标签会携带条件信息；InstructProtein 文本位置约 92.8% 的概率质量落在残基 token，不能与蛋白模式组成同权重对照。Galactica 1.3B/6.7B/30B 的反向蛋白条件未识别。计分跨度按各臂自己的声明核对：渲染声明的标记位置不计分（ProtGPT3 的 `<\|bos\|>` 与方向 token、RITA-xl 前置的文档边界 `<EOS>`、ProGen3 的 `<bos>…<eos>`、ProteinGLM 的 `<gmask><sop><eos>`），按残基计分处一个残基对应一个 token，Qwen3 不添加 BOS；声明与跨度不一致的格式直接拒绝，不另做一套格式损失表 | 已成为后续实验的接口前提，不是单独的覆盖实验。输入须被钉住才可引用：analyse 拒绝与 probe 指纹不一致的 LOOKUP，产物带 CSV/FASTA 指纹与冻结 run-id（尾段为代码 hash）。不同模式和输出空间的数值不能直接横比；通过校验只说明接口可用，不表示模型具有生物能力 |
| 近重复切分与留出泄漏检查 | 方法不依赖单一模型；在 ProLLaMA 字典训练和序列—描述实验中复核。10,240-record 池的按记录留出集有 2,048 条，其中 871 条仍有 ≥95% 近亲、350 条整串相同 | ①比较按记录、精确字符串和近重复组切分。②用 DIAMOND 检查留出集中的近亲序列。③跨训练/验证分组或筛查近亲。④在 OpenWebText 上运行同样检查作对照 | 普通记录划分后，留出集中仍有 42.5% 的蛋白带有 ≥95% 同一性的近亲，17.1% 整串相同；同一性按较短序列归一，100% 不等于整串相同。按组切分把本次检测条件下的 ≥90% 跨集近亲降为零 | 已成为评价前提。蛋白实验不能只做精确去重或更换随机种子；这项检查保证留出单位更接近独立，但不解释模型机制，也不表示没有更远同源或预训练未见过这些序列 |
| 计分跨度规则 | 所有计分臂；十三档纯文本氨基酸串对照（`text-aa-fp32-v1`） | 渲染声明的标记位置不计分；按残基计分处一个残基必须对应一个 token，合并跨度按不同泛函拒绝 | 合并跨度相对逐残基计分的差异在 galactica-1.3b 上实测约 2.9 nats/token（EXP-R2-151）；十三档文本串对照中只有 ByGPT5 三档为字节级（约一字一 token），其余十档因合并不能作逐残基对照 | 硬缺陷而不是约定：`joint_modes.verify_one_token_per_residue` 在计分 token 数不等于残基数时抛出，`text_aa_fitness` 直接拒绝合并编码；ProtGPT3 的 `<\|bos\|>` 加方向 token、RITA-xl 前置的文档边界 `<EOS>`、ProGen3 的 `<bos>…<eos>` 都按此排除标记位置 |
| 精度 / 数值约定 | 所有计分臂；ProGen2 各档、ProteinGLM-7B-CLM、RITA-xl 与 Galactica 蛋白模式的精度已在主表声明 | 存储 dtype 与计分 dtype 分别声明、允许不同，并逐端点固定。计分默认 N→C 求和；一次比较内方向必须相同 | ProGen2-medium 的截断曲线统计在 bfloat16 下移动约 16%（0.6266 → 0.7293 nats），由此单独按 float32 计分；ProteinGym 阶段一律 bfloat16 | 精度和方向都是可比性条件：同一谱系的梯子若各档精度不同，比较的是算术而不是 checkpoint。不同方向的数字不混合，也不要求采用某一 checkpoint 的官方双向计分 |
| 抽样种子纪律 | 所有抽取语料的阶段；文本 OpenWebText，蛋白 Swiss-Prot 与带 EC 标签的 UniProt | 语料抽取是带种子的置换：`protein_cohort` 与 `text_cohort`（及其 repeat 版本）在声明的 draw seed 下置换后再取 `--cohort-skip` 窗口，种子写入充分统计量 `seeds.cohort_draw` | 文件顺序前缀是语料的一个区域而不是样本：生物语料按 cluster 排序，网页语料按 shard 排序；`--cohort-draw-seed 0` 的文件顺序变体只在对照中记录 | 前提而非逐实验选择：`tests/test_cohort_draw_contract.py` 静态检查每个抽取语料的阶段的调用点，新增未声明 draw seed 的抽取即失败 |

计分产物写明方向（默认 N→C）、跨度、精度、字母表和 tokenizer 策略。输入格式声明须引用模型卡、论文或本仓库 probe。

## 方向一：比较模型具有什么能力

本节关注模型在外部任务、原生生成和条件／上下文利用中实际做得怎样，包括未通过对照的结果。序列评分、生成质量与产出、条件选择性和泛化是不同的能力维度，不能合成一个排名。已有蛋白的实验标签不能转移给新生成物；生成评测须用天然、阴性与简单参照校准，计算证据也不等于实测折叠或功能。

### 外部蛋白任务上的序列评分

| 实验 | 模型与数据 | 方法 | 结果 | 状态与边界 |
| --- | --- | --- | --- | --- |
| ProteinGym 突变适应度 | ProtGPT2；ProGen2 small/base/medium/large/xlarge；ProGen3 112M/3B；ProteinGLM-7B-CLM；ProtGPT3-1.3B；InstructProtein 蛋白；RITA-xl；Llama-2-7B 与 ProLLaMA Stage 1/2；Galactica 125M/1.3B/6.7B/30B 蛋白按 FP32 v2。基础评测集为 217 个 substitution assays / 174 个家族；1024-token 臂共同支持 201 assays / 163 families，2048-token 臂为 213 assays / 171 families。ZymCTRL 因缺 EC 标签，217 个 assay 全部跳过 | 在固定 mutant digest 上用模型似然排序突变，以逐位 profile 和 BLOSUM62 作序列基线；按家族 bootstrap 比较 MODEL−LOOKUP 和 MODEL−BLOSUM62，同谱系 checkpoint 只在相同分析集和评分方向内形成配对增量 | ProtGPT2 的 MODEL−LOOKUP 为 −0.0143。ProGen2 三个 checkpoint 为 −0.0043、+0.0122、+0.0039，区间均覆盖零。ProGen3 从 112M 的 −0.0801 [−0.1135,−0.0460] 变为 3B 的 +0.0555 [+0.0345,+0.0774]，raw Spearman 增量为 +0.1356 [+0.1082,+0.1647]。ProLLaMA 的 raw Spearman 为 −0.0071→+0.1358→+0.1507，但最佳 checkpoint 仍低于 LOOKUP +0.3537 和 BLOSUM62 +0.2097。ProGen2-small/base 的 MODEL−LOOKUP 各自在自己的支持集上读取：201 assays/163 families 为 −0.0554 [−0.0816,−0.0291]，indeterminate；213 assays/171 families 为 +0.0037 [−0.0252,+0.0325]，retrieval_bounded；InstructProtein 的 MODEL−LOOKUP 为 −0.0754 [−0.1021,−0.0481]，retrieval_dominated，同样在自己的 213 assays/171 families 支持集上读取。ProteinGLM-7B-CLM 在 201 assays / 163 families 上 MODEL−LOOKUP 为 +0.0384 [+0.0175, +0.0592]，判决 acquired，LOOKUP 为外部 UniRef50 profile。Galactica FP32 v2 各档均为 213 assays / 171 families：raw Spearman 为 125M −0.0077 [−0.0245, +0.0084]、1.3B +0.0484 [+0.0307, +0.0672]、6.7B +0.0954 [+0.0744, +0.1167]、30B +0.1479 [+0.1249, +0.1710]；MODEL−LOOKUP 均为负，125M −0.3633 [−0.3909, −0.3352]、1.3B −0.3071 [−0.3366, −0.2769]、6.7B −0.2602 [−0.2930, −0.2270]、30B −0.2076 [−0.2393, −0.1782]。RITA-xl 在 201 assays / 163 families 上 MODEL−LOOKUP 为 +0.0075 [−0.0141, +0.0282]，区间覆盖零。ProtGPT3-1.3B 在同一支持集上 MODEL−LOOKUP 为 −0.0512 [−0.0791, −0.0246]，判决 indeterminate。MODEL−BLOSUM62 用同一套家族 bootstrap：ProtGPT2 为 +0.1343 [+0.1056, +0.1607]。ProGen2 medium/large/xlarge 为 +0.1522 [+0.1248, +0.1787]、+0.1688 [+0.1465, +0.1916]、+0.1604 [+0.1412, +0.1789]；small 在 201/163 上为 +0.1012 [+0.0725, +0.1273]，base 在 213/171 上为 +0.1531 [+0.1254, +0.1797]。ProGen3 从 112M 的 +0.0688 [+0.0398, +0.0975] 变为 3B 的 +0.2044 [+0.1835, +0.2259]。Llama-2 到 ProLLaMA Stage 1/2 为 −0.2133 [−0.2345, −0.1896]→−0.0704 [−0.0999, −0.0411]→−0.0555 [−0.0840, −0.0282]。InstructProtein 为 +0.0740 [+0.0532, +0.0947]。ProteinGLM-7B-CLM 为 +0.1949 [+0.1725, +0.2165]。Galactica 各档均为负：125M −0.2138 [−0.2360, −0.1910]、1.3B −0.1577 [−0.1789, −0.1357]、6.7B −0.1107 [−0.1336, −0.0861]、30B −0.0582 [−0.0805, −0.0339]。RITA-xl 为 +0.1640 [+0.1431, +0.1833]。ProtGPT3-1.3B 为 +0.1053 [+0.0770, +0.1323] | ProGen2 相邻规模 checkpoint 均未改变通过或不通过的结论；ProGen3 只支持两个具名 checkpoint 的描述性转变。各组的分析集、评分方向、语料和架构不同，不能跨组排名或形成参数量因果。ProGen2 的 profile 覆盖不完整；ProGen3、ProteinGLM、Galactica 与 InstructProtein 的 LOOKUP 是外部 UniRef50 基线，不排除训练语料检索；赢过 LOOKUP 不是获得生物学知识；赢过 BLOSUM62 只说明排序好过替换表。ZymCTRL 的全部 217 个 assay 因语料原因跳过，是完成的整体拒绝，不产生适应度数字。Galactica-125M 蛋白资格门未识别，ProteinGym 仍按正式结果读取，与 1.3B/6.7B/30B 写在同一组；该组分析不给出 acquired 或 retrieval_bounded 判决。RITA-xl 配对已从冻结清单副本重跑，数字与上次相同，现可作为正式结果引用。ProtGPT3-1.3B 按原生格式计分，LOOKUP 同为外部 UniRef50 |
| MegaScale 设计稳定性 | ProtGPT2 与 ProGen2 medium/large/xlarge；设计评测含 130 个无可检出 UniRef50 整序列同源的 WT、40 个 series、110,730 个变体，天然对照含 266 WT、124 clusters、404,114 个变体 | 计算模型 likelihood 与实测稳定性的 Spearman，并比较 hydropathy、BLOSUM62、composition 和 3–7-mer 基线；按设计 series 或天然 cluster 重采样，同谱系 checkpoint 在相同评测集上比较 | ProtGPT2 的模型相关为 +0.0837；相对 7-mer 为 +0.0516 [+0.0213,+0.0807]，但相对 hydropathy 为 −0.3548，相对 BLOSUM62 为 −0.0779。ProGen2 三个 checkpoint 在设计侧都显著低于 hydropathy 和 BLOSUM62，相邻规模均未改变通过或不通过的结论 | 已完成。证书只排除整序列检索，设计序列的 7-mer 仍有 99.57% 可在语料中找到；结果不证明功能、知识或机制，设计身份与组成也仍不可分。ProGen2 上两个 checkpoint 的高阶 k-mer margin 尚无结果；ProLLaMA 未通过天然对照资格，ProGen3 尚无可读结果 |

### 按条件生成与使用上下文

| 实验 | 模型与数据 | 方法 | 结果 | 状态与边界 |
| --- | --- | --- | --- | --- |
| 原生条件生成 | ZymCTRL 的 EC 标签和 ProLLaMA Stage 2 的 superfamily 指令；分别保留 14 个可测 EC 类别和 15 个可测 superfamily，每个类别与条件生成 200 条 | 同一模型比较请求标签、固定错配标签和无条件下限；先用真实蛋白与随机序列校准 HMMER/Pfam-A 判定器，再按类别和近重复组做 2,000 次 bootstrap | ZymCTRL 的请求减错配为 +0.8822 [+0.7202,+0.9872]，14/14 类别为正；ProLLaMA 为 +0.1267 [+0.0527,+0.2360]，11/15 类别为正。两个模型都满足“生成朝请求类别移动”的复合条件 | 已完成。结论只是策展 profile 对生成物的类别归属，不是功能验证。ZymCTRL 生成物对 UniRef50 的最大 identity 中位数为 69.2%，未排除训练分布内检索；Pfam 只能识别 13.5% 的 ProLLaMA 生成物，两个模型的比率也不直接比较 |
| 同源上下文利用 | gpt2-large、ProtGPT2、ProGen2-small、ProGen2-medium；每个模型 800 个目标单位，蛋白同源上下文按 70–90、50–70、30–50、<30% identity 分档 | 在固定 1,024-token 预算内匹配上下文条数、长度和填充，比较真同源、组成匹配无关序列、打乱同源和位置占位控制；要求效应在 `<30%` identity 的最低局部重叠层仍成立 | ProtGPT2 和 ProGen2-small 的合并 AUROC 为 0.786 和 0.653，但随局部重叠降低分别变为 0.915→0.935→0.770→0.525 和 0.860→0.670→0.550→0.530，低于 30% identity 后回到机会；ProGen2-medium 和 gpt2-large 的合并 AUROC 为 0.379 和 0.504 | 已完成。四个未再训练的 checkpoint 都没有显示超出局部复制的同源利用；结论只适用于当前 token 预算和实际 k，不约束专门为多序列条件训练的模型。文本 BM25 带与蛋白 identity 带不是同一因素，ProtGPT2 的 FASTA 多序列预训练也是独立混杂 |

### 生成物的顺序信息、产出与参考距离

EXP-R2-232/233 已完成结构与原生生成终态，补充回答“生成了多少、序列顺序是否提供信息、实际产出与参考支持怎样”，与上表的请求类别选择性是不同问题，各自保留原分母。原条件生成仍是 ZymCTRL 的 EC 标签任务和 ProLLaMA 的 superfamily 指令任务；本次没有把它们改成同任务模型排名，也没有开展两套 ProLLaMA 解码配方比较。

| 已完成测量 | 结果 | 能支持的结论与边界 |
| --- | --- | --- |
| 全尝试与非冗余识别 | 主要请求条件下的目标 profile 命中为 ZymCTRL 2,488/2,800、ProLLaMA 380/3,000，分别落入 1,688 和 380 个原条件单元内的近重复组 | 正确条件的选择性和实际非冗余产出是不同指标；profile 命中不是 EC 活性或生成物功能 |
| 天然校准及条件生成结构 | 116 个天然／自身扰乱预测全部完成并通过校准；main 的 2,320 个预测、1,160 对全部有效。ZymCTRL 的生成减自身扰乱 mean CA-pLDDT 为 +56.68，97.5% 类别 bootstrap 区间 [45.90, 62.57]；ProLLaMA 为 +5.92 [2.49, 10.57] | 两者都有超出组成和长度的、预测器可识别的序列顺序信息；自然／扰乱对照不是实测可折叠／不可折叠分类器 |
| 绝对置信度及全分母估计 | 请求条件 mean CA-pLDDT 为 86.00 和 44.39；固定 confidence event 的全尝试点估计约 86.36% 和 1.36%，后者一个短输入未知使支持上界为 1.40% | 正 margin 不等于高绝对产率。这些是固定预测器的样本加权估计，支持范围不是置信区间，更不是实测成功率 |
| ProGen3-3B 原生无条件生成 | 800 次全部保留：501 个原生接口接受的输出、299 个预算截断前缀；496 个获任意 Pfam 识别。128 原产物和 128 自身扰乱均预测成功，combined margin +38.39，95% 序列组 bootstrap 区间 [34.53, 42.27] | 支持这个具名原生任务的顺序信息；primary 包含 89 个完整原生输出和 39 个截断前缀，后者的置信度不能计作完整产物证据，也不构成跨家族泛化 |
| 完整参考检索核对 | fresh 16,400 条全部核验；原字段全部保留，7,043 个既有非空 identity 数值逐值不变。主要请求条件评测 ZymCTRL 2,658/2,800 有命中，identity／query coverage 中位 70.00%／98.82%；ProLLaMA 390/3,000，32.05%／79.85% | 身份与双覆盖来自同一命中；有无参考支持与置信度同时报告，描述性分层不能当成训练外泛化或记忆因果证明 |

全部测量保持固定样本和端点，未因结果扩张模型或解码矩阵。参考库距离不等于完整训练外；fresh 检索同时保留 identity 与同一命中的覆盖率。结构预测只是计算证据，实际功能仍需对应候选的实验数据；这些有界生成结果也不表示方向三的知识鉴别已经通过。设计与停止条件见[生成补充实验预注册](docs/D1_GENERATION_BIOLOGY_PREREGISTRATION.md)，详细终态、已完成的本地审阅包及作者侧待办见[科学审计](docs/INTERPRETABILITY_TRANSFER_AUDIT.md#current-assessment-and-next-decisions-2026-09-05)。

### 联合训练阶段的行为收益与文本代价

上面的序列评分衡量蛋白适配带来的任务收益；这里补充同一训练谱系的方向性读出和文本代价，组件归因留在方向二。

| 实验 | 模型与数据 | 方法 | 结果 | 状态与边界 |
| --- | --- | --- | --- | --- |
| ProLLaMA 蛋白适配与文本代价 | Llama-2-7B→ProLLaMA Stage 1→Stage 2；三个 checkpoint 共享 32 层、宽 4,096 和同一 tokenizer。评测含文本与蛋白上下文，以及 20,000 条 sequence–description 记录 | 在相同评测集上测上下文信息和蛋白序列反向代价；文本侧在近重复组外和家族外做 1,000-candidate 描述检索 | 基础模型的蛋白反向代价为 −0.0013、区间包含零，Stage 1 为约 +0.144 nats/token；Stage 1 同时损失约 4.69 nats/token 的文本上下文信息。文本检索 top-1 在组外为 0.2387→0.0954→0.1065，家族外为 0.1722→0.0546→0.0580 | 已完成。Stage 1/2 构成合格双模式对照，但指令阶段没有恢复文本检索；蛋白约为 1.53 residues/token，不能与逐残基模型按 token 横比，也不能把变化归因于某个单独组件 |

### 尚未完成的能力评测

EXP-R2-225 扩展的 7B 接口资格仍未作为该扩展运行；派生预算资格代码不是 7B 运行。ProteinGym 上 ProteinGLM、Galactica FP32 v2、RITA-xl 与 ProtGPT3-1.3B 的数字见上表。ProGen3 的 MegaScale 行仍缺稳定性评分、逐 checkpoint 的语料识别记录和评分分层标注。这些不替代方向二、三的下一步。

## 方向二：用现有方法解释能力如何形成、表示和计算

本节按研究决策顺序组织，而不是按工具名称罗列。先校准方法能否测得到，再从扰动响应和注意力模式寻找计算线索；观察性的解释须经得起分词和因果对照，而“能读出来”离“模型真的在用”还很远。稀疏重建与内部特征控制检验解释能否保持或改变行为，最后沿同一训练谱系看训练本身改变了什么。诊断结果与因果证据分别保留，不把前者写成已经解释完整机制。

### 先校准方法是否测得到

| 实验 | 模型 | 数据(格式、示例、数量) | 怎么做 | 结果 | 状态与局限 |
| --- | --- | --- | --- | --- | --- |
| 方法功效与抽样敏感性 | 以 gpt2-large 为正对照，并覆盖合格文本和蛋白模型 | 文本为连续文档，蛋白为 `MKT...` 序列，ZymCTRL 另带 EC 前缀。解释方法复核 500 条 Swiss-Prot、122,671 个计分位置、每个 38 个 triplet 的 top-100 事件，并检查 23,586 个 AlphaFold 模型 | ①测目标自身的因果足迹。②计算门槛解析上限、功效和匹配置换 null。③改为从全语料中按固定随机种子抽样。④用 skip-offset 和独立样本检查选择敏感性 | 单层 MLP 目标的因果足迹约 0.02 nats/token，因此要求恢复 80% 的标准在该目标上不可达；top-100/122,671 的互信息上限只有 0.0066 nats。七类统计都显示蛋白样本比匹配文本对照更敏感 | 已完成。结果给出方法功效和抽样敏感性的正式边界：不能把“测不到”写成模型没有机制 |
| 多残基分词的干预对齐 | ProtGPT2；ZymCTRL、ProGen2-medium 和 gpt2-large 作对照 | 单点突变记录，如 `WT=MKT...; mutation=A42V`：4 个 ProteinGym assay 各抽 200 个突变。另有 128 个合成 8-residue 重复 probe，如两处 `ACDEFGHI` | ①分别 tokenize 野生型和突变体。②检查序列是否等 token 长并且只有一个 token 改变。③把同一残基片段放在不同上下文相位。④检查两处是否仍得到相同 token 序列 | ProtGPT2 只有 47.0%–54.5% 单点突变保持唯一局部对齐；正式 128-probe 重复实验中只有 21.1% 保持 token 一致 | 已完成。其余样本的残基级 patch 没有唯一数学定义；幸存者又是非随机的 BPE 稳定子集。序列级 likelihood 不受此限 |

### 从扰动响应到计算环节

输入扰动的距离衰减先描述输入—输出敏感性；模块消融和内部扰动再定位可能参与的计算环节。响应持续得更远不等于正确利用了远处信息，也不表示传播路径已被识别。

| 实验 | 模型 | 数据(格式、示例、数量) | 怎么做 | 结果 | 状态与局限 |
| --- | --- | --- | --- | --- | --- |
| 输入扰动的距离衰减 | GPT-2、DialoGPT-small、Qwen、Llama 文本模型，以及 ProtGPT2、ProGen2 small/base/medium | 每个模型使用同一固定种子排列上的 5 个不重叠语料切片，每窗 24 条源记录、每个距离带抽 256 个扰动案例；案例不是独立序列，重采样以源记录为单位 | ①按经验一元词频替换源 token，而非定向改写或生物学突变。②用字符／残基声明扰动跨度（5 字符／3 残基）和 9–16、17–32、33–64 距离带，再按各评测集平均每 token 内容量折算，而非逐案例精确对齐。③固定原始预测的最高与次高候选，测扰动前后二者 logit 差的绝对变化。④比较远端／近端达阈比例的衰减比 | 在较严格的 0.50 阈值下，所测蛋白模型组的衰减比较高，即相对各自近端水平下降较慢；在宽松的 0.25 下，文本和蛋白范围重叠，不形成普遍排序或绝对水平结论。阈值是 logit 差变化量，不是概率百分比 | 已完成描述性诊断。字符和残基仍非等价单位，阈值会改变组间差异是否可见；仅适用于指定模型、窗口与扰动定义。不证明有用的长程信息利用、生物学相互作用或完整传播机制 |
| MLP 与注意力的行为贡献 | 六个可测文本模型；ProtGPT2、ZymCTRL、ProGen2 small/base/medium。DialoGPT-small 因资格失败排除 | 连续原生文本或 64–246-residue 蛋白序列。使用从全语料中按固定随机种子抽取的样本；F7 未固定统一记录数 | ①分别消融 MLP 和 attention。②以每个模型自己的上下文信息归一化。③从文件顺序样本改为全语料随机样本。④用第二组独立样本复核资格和结果 | MLP/attention 代价比在文本为 1.491–2.070，在蛋白为 0.530–1.129，范围不重叠；重新抽样后各模型的结果复现到约两位小数 | F7 结果已确定，是当前较稳健的模型差异。该比值不是贡献分解且可以大于 1；尚未完成分词尺度校正 |
| 架构与路由的扰动响应 | Dense：GPT-2、ProtGPT2、ZymCTRL、ProGen2；MoE：ProGen3-112M；同权重 ProLLaMA Stage 1/2 | 原生文本或 Swiss-Prot 64–246 序列。主要实验条件通常各用 128 条记录、每个强度 3 个随机方向；ProGen3 路由诊断另用 256 条、41,155 个残基位置，因此没有统一 n | ①在每层 MLP 输出加入相对范数匹配的各向同性扰动。②比较 Dense/MoE 和同权重双模式。③用随机分组、残基分组和路由边界作控制。④检查结果能否由路由解释 | 蛋白模型内部差异可大于文本—蛋白差异；ProLLaMA 双模式排序在 Stage 1 与 Stage 2 间反转。Dense 与 MoE 都能出现方法失败 | 已完成。排除了“蛋白、MoE 或路由必然更脆弱”等简单解释，但没有识别统一近端机制 |
| 模型规模与扰动响应 | ProGen2 small/medium/large/xlarge，约 1.51 亿、7.65 亿、27.8 亿、64.4 亿参数 | Swiss-Prot 64–246 aa，一字母序列如 `MKT...`。每个规模、每个条件 128 条、3 个扰动方向、1,000 次 bootstrap；主样本与错开抽样起点的独立样本都运行 | ①固定 ProGen2 谱系、逐残基 tokenizer 和评测长度带。②扫描相同的相对 MLP 扰动强度。③比较四个规模。④检验排序是否单调且能否在第二组独立样本中重现 | 除最小扰动点外，各强度和条件均为 xlarge > medium > large > small；medium 始终高于参数更多的 large | 已完成。不存在简单规模律；深度、宽度和参数量共同变化，不能单独识别参数量效应 |

### 精确重复前缀：从注意力模式到因果筛选

| 实验 | 模型 | 数据(格式、示例、数量) | 怎么做 | 结果 | 状态与局限 |
| --- | --- | --- | --- | --- | --- |
| 精确重复前缀的注意力模式普查 | gpt2-large 与 ProtGPT2；两者均为 36 层、宽 1,280、20 个头、50,257 词表和约 7.74 亿参数。另含其他文本模型、ZymCTRL 和 ProGen2 | 每个模型使用 256 个合成重复 probe，复制片段长度为 64 tokens（不是残基），如起始标记后接两份相同的 64-token 片段。自然精确或近似重复文本/蛋白。203,063 个合格 EC 记录中找到 48 个精确重复和 817 个近似重复；同源分层使用 UniRef50 | ①固定架构、规模和 probe 几何，统计 prefix-matching 头并扫描阈值。②加入 tokenizer 单 token 身份候选均分参照。③检查自然精确和近似重复。④穷举头级因果效应并做路径修补。⑤关闭 DIAMOND masking 后按训练语料相似度分层。⑥修正 bootstrap 单位和 collision null 特异性 | 固定 0.10 阈值下，gpt2-large 为 71/720 高分头、ProtGPT2 为 13/720，约 5.46 倍；按各自单 token 身份候选均分参照重读为 5.62 倍。路径修补未证明蛋白通路更间接，同源分层也未定案记忆贡献 | 已完成并收窄。这是匹配对上的描述性注意力模式普查，不是行为能力差异，也不表示完整因果机制已被解释。单头统计不能刻画完整多层计算。身份参照不是多层上下文头的绝对上限。只可陈述匹配对，不是普遍文本—蛋白规律；训练语料中的重复结构仍未匹配，当前没有真正低同源蛋白层，也没有对称文本训练语料控制 |
| 分词粒度对照 | ByGPT5-medium；对照 BPE 文本、逐残基蛋白及 GPT-2/ProtGPT2 | 合成字符或 byte 重复，如 `abc...abc...`；每个模型使用 256 个 probe、复制长度 64，并为每个真 probe 构造无重复 collision null，运行 2,000 次 bootstrap | ①用近似“一字符一 token”的文本模型拆开模态与分词。②分别测量每个模型的同 token 竞争和 identity ceiling。③比较固定阈值 induction 统计。④再比较 PAA 能否找回真正因果头 | ByGPT5 在固定阈值 induction 统计上落入蛋白区间，因此按模型组的模态排序不成立；在 PAA 因果检索中表现得像文本模型，达到自身机会水平的 5.3 倍 | 已完成。分词能制造部分伪差异，但不能解释全部差异；ByGPT5 同时改变架构和训练语料，不能单独识别字母表的因果效应 |
| 注意力筛选的因果准确性 | 文本：GPT-2、Qwen、Llama、ByGPT5；蛋白：ProtGPT2、ZymCTRL、ProGen2 small/base/medium | 单条记录形如 `{native_record, query_position, antecedent_position, predicted_token, matched_decoy_positions}`。不同正式实验和失败审计对“模型 × 独立抽样”的统计口径不同，因此不合并成一个总 n | ①穷举消融，得到每个模型自己的因果 top-k。②用 PAA 分数找回 top-k。③与自身机会水平和只知道层号的基线比较。④排除格式 token 并固定随机种子。⑤在发现集和留出集上检验可能的失败因素 | 所有逐残基蛋白模型只达到或低于自身机会，所有文本模型至少达到 3.6 倍机会；ByGPT5 也通过。ProtGPT2 的表面通过主要由层深携带 | F1–F4 结果已确定，既定模型组已关闭。结论是筛选器在这些蛋白接口上失真，不是模型没有 copy-suppression 机制 |

### 从可读出表示检验到因果使用

| 实验 | 模型 | 数据(格式、示例、数量) | 怎么做 | 结果 | 状态与局限 |
| --- | --- | --- | --- | --- | --- |
| 中间层预测读出 | gpt2-large、ProtGPT2、ZymCTRL、ProGen2 small/base/medium | OpenWebText 连续文档与 Swiss-Prot `MKT...` 序列。蛋白比较包含 64–120 和 64–246 两个长度带；正式审计未统一规定每个模型的独立抽样次数 | ①比较 Logit Lens 与 Tuned Lens。②用有限差分验证输出 Jacobian。③把 Jacobian 秩与输出词表孔径分开。④修复宽带截断。⑤对化学性质投影加入 shuffled/rank-matched null | Tuned Lens 在所有已测模型和两个长度带都更好；扩大到 64–246 后，蛋白侧优势缩小 54%–82%，文本对照增加约 4%。小残基词表的低秩主要是输出孔径 | 已完成，F6 结果已确定。Lens 结果必须同时写长度带；性质投影没有匹配随机分区就不可解释 |
| 同一权重下两种模式的读出深度 | Galactica 1.3B、6.7B 和 30B 的文本与蛋白模式；125M 只作方法参照 | 与各 checkpoint 自己的上下文资格样本使用同一批 128 条蛋白和匹配文本记录，共 20,866 个蛋白目标与 20,864 个文本目标，蛋白渲染为每个计分 token 一个残基 | 在 float32 下用未调优 logit lens 逐层读取模型自己的最终分布，要求输出头重建 KL 不超过 1e-2 nats；以 top-1 一致率的相对跨越深度和跨度归一 KL 深度作为两个统计量，按记录组做 2,000 次 bootstrap，并报告位置上限与朴素渲染控制 | 三个可识别 checkpoint 都显示蛋白模式更晚接近各自的最终预测：蛋白减文本的相对深度差为 1.3B +0.1380 [+0.1331,+0.1434]、6.7B +0.1352 [+0.1292,+0.1412]、30B +0.0263 [+0.0248,+0.0278]；KL 深度差依次为 +0.2808、+0.1947、+0.0946。30B 的两个模式都只在最后一段明显接近最终预测，最深内部网格点的 top-1 一致率为文本 0.3891、蛋白 0.1161 | 已完成，但不进入可引用结论。未调优 lens 不能区分“计算更晚形成”和“中间状态离最终输出基更远”；30B 的两个跨越点都位于最后一个五层宽网格区间，方向可读而幅度未解析。三个 checkpoint 的深度、宽度和参数量共同变化，因此曲线不是规模因果；还缺调优 lens 和独立数据复核 |
| 接触关系的状态与注意力读出 | ZymCTRL、ProGen2 small/base/medium | 接触记录如 `{sequence, i, j, contact, sequence_distance, decoy}`；按 Pfam family 和 k-mer cluster 隔离。正式审计未统一记录正式接触对总数 | ①先修正结构筛选顺序。②训练单位置 hidden-state 控制。③读取原始 attention。④加入距离匹配 decoy 和只含序列间距的控制。⑤比较家族外接触预测 | attention 相对简单控制的优势约 0.03–0.05，相对纯间距控制约 0.03 | 已完成并关闭当前跨位置稀疏特征动机。只适用于 token 与残基明确对齐的模型；ProtGPT2 不可启发式对齐 |
| 生化属性的层间读出 | gpt2、gpt2-large、ProtGPT2、ProGen2 base/medium、ZymCTRL | 400 个 Swiss-Prot 蛋白，64–246 aa；Pfam 分成 188 个 seen 与 187 个 unseen families。每个残基带身份、疏水性、电荷和体积标签 | ①在每层比较残基身份和粗粒度化学属性的读出。②加入 shuffled-property 与 rank-matched 小字母表 null。③要求属性比残基身份更早稳定。④固定 ZymCTRL 的 EC 标签检查泄漏 | 三个无条件蛋白模型中，属性没有比残基身份更早稳定；charge/volume 在各蛋白模型中均为 0/11 层通过。ZymCTRL 的疏水性正结果在固定 EC 标签后从 7/11 降为 0/11 | 已关闭，后续因果阶段未获授权。只否定这一输出 Lens，不说明残差流没有生化信息 |
| 可解码表示与因果使用 | ProtGPT2、ProGen2-medium；ZymCTRL 等无有效因果分母的实验条件被拒绝 | 记录如 `{sequence, Pfam_family}` 或 `{sequence, residue_index, SS3}`。相关流水线覆盖 820 条 Swiss-Prot、44,626 个标注残基 | ①按家族或近重复组切分。②训练线性/MLP Probe。③用 `LEACE` 擦除读出子空间。④重新测模型下一 token loss。⑤比较“可解码强度”和“行为损伤” | ProtGPT2 的 Pfam 可解码技能约 +0.705，但擦除后预测改善约 −0.179 nats；ProGen2-medium 的 SS3 行为依赖只有约 +0.031 nats，远小于可解码强度 | 已完成。“可解码”不等于“模型使用”。ProtGPT2 多残基 BPE 不允许启发式映射到 SS3、burial 等残基标签 |

### 从稀疏表示重建走向行为控制

| 实验 | 模型 | 数据(格式、示例、数量) | 怎么做 | 结果 | 状态与局限 |
| --- | --- | --- | --- | --- | --- |
| 稀疏字典的表示质量 | ProtGPT2、ZymCTRL、ProGen2-medium；联合谱系另测 ProLLaMA | 训练流为 UniRef50、ZymCTRL EC FASTA 或模型原生文本；评测为连续文档或 `MKT...` 序列。不同字典实验的训练 token 数不同，没有统一 n | ①训练不同宽度和预算的字典。②检查死特征、FVU/NMSE 和 Probe 可恢复性。③人工或自动注释特征。④做 steering。⑤检查更宽字典是否改变结论 | 扩宽能减少死特征、改善部分 Probe，却没有带来稳定 steering；FVU、命名性和行为保持的排序不一致 | 已完成方法审计。死特征、低重建误差、可命名性和 Probe 都只能作诊断，不能单独构成机制证据 |
| 稀疏表示的行为保持 | gpt2、gpt2-large；ProtGPT2、ZymCTRL、ProGen3-112M | 原生连续文本或蛋白一字母序列。F11 蛋白主比较使用 64–246 residues、固定 12× 字典宽度；蛋白另扫描约 64–1022 residues 的四个长度带 | ①逐层用字典重建替换 MLP，attention 在受扰残差上重新计算。②在固定序列目标上以原模型和均值消融定义行为恢复，不评估自由生成。③加入闭式仿射、原始神经元和随机扰动基线。④控制深度、训练预算、路由、分词和长度带。⑤不采用连文本正对照也失败的 causal-rank 资格 | 固定 12× 下文本恢复 0.9084–0.9322，蛋白只有 0.0916–0.1641；四个长度带上蛋白仍为 0.0425–0.2724，与文本无重叠。更好重建不保证更多行为恢复 | F11 结果已确定；差异对长度带稳健，但蛋白侧字典宽度稳健性和近端原因仍开放。不能推断蛋白表示必然更稠密 |
| 神经元与匹配扰动对照 | gpt2-large、ProtGPT2、ZymCTRL；另含 ProGen2 规模阶梯和 ProLLaMA 双模式 | 模型原生文本或蛋白评测集。每个子实验的模型和 n 不同，没有统一样本数 | ①把学习字典换成原始 MLP 神经元。②构造与字典误差同范数、同夹角的随机扰动。③比较模型、模态和规模排序。④检查误差方向是否异常 | 原始神经元在部分蛋白模型上胜过学习字典，指向字典训练问题；误差方向并不异常，扰动耐受也不按模态或参数量单调 | 已测但未推广为普遍规律。只能陈述具体模型，不能说蛋白表示必然更稠密或更脆弱 |
| 蛋白生成的特征控制 | ZymCTRL v2；ProGen2-medium 只用于确认干预 hook | 8 类 EC 生成评测：lysozyme、trypsin、ADH、catalase、DNA polymerase、lipase、kinase、carbonic anhydrase；另用 100 个真实 lysozyme 与 100 个长度匹配 UniRef50 蛋白校准指标 | ①按 CLT 直接效应选择特征。②在 L3/L12/L30 做 top-k 流形内 steering。③用 Pfam、CLEAN、ESMFold/Foldseek 评价。④先验证指标能区分真实与随机蛋白。⑤比较 steering 与未干预生成 | 8 类中显著改善为 0/8；真实/随机校准强分离，说明外部指标工作，但 steering 没有优于未干预生成 | 已关闭强 steering、药物设计和湿实验室实验主张。hook 确实移动 logits，因此负结果不能归因于干预路径断开 |

### 沿联合谱系追踪训练带来的变化

| 实验 | 模型 | 数据(格式、示例、数量) | 怎么做 | 结果 | 状态与局限 |
| --- | --- | --- | --- | --- | --- |
| 蛋白训练能力的组件来源 | Llama-2-7B、ProLLaMA Stage 1 | OpenWebText 文本与 Swiss-Prot 64–246 aa。每种模式每次抽样 128 条，共使用两组互不重叠且固定随机种子的样本 | ①逐张量验证两个 checkpoint 可交换。②双向替换输入 embedding、输出头、完整词汇接口或 Transformer 主体。③在相同上下文信息指标上比较混合模型。④用第二组独立样本复现 | 约 4.69 nats 文本损失主要随主体移动；完整接口只解释正向约 2%、反向约 14%–15%，接口内主要由输出头承担，输入 embedding 约 1% | 已完成并复现。效应不具可加性，且 next-token loss 由输出头直接参数化，不能当作一般组件贡献占比 |
| 蛋白训练后的激活结构 | Llama-2-7B、ProLLaMA Stage 1、Stage 2 | OpenWebText 与 Swiss-Prot。每个模型阶段与模式组合取 1,024 条留出记录、每条最多 64 个位置，共 65,536 个 token 位置；蛋白短记录明确拒绝 | ①读取每层 feed-forward 输出。②以 float64 计算协方差谱、参与率、有效秩和 `r99`。③与 isotropic、坐标独立 null 比较。④比较三阶段的逐层峰 | Stage 1 在 28/28 个内部文本层使谱更集中；蛋白高维区从基础模型的早层峰移到 Stage 1/2 的 17–22 层，Stage 2 不再移动蛋白峰 | 已完成。只覆盖一个谱系和一个模块输出位置；不同谱指标不能合成单一“坍缩倍数”，谱形也不等于行为能力 |
| 蛋白训练前后的表示差异 | Llama-2-7B → ProLLaMA Stage 1；文本和蛋白模式分别运行 | 同输入、同位置、同层的 OpenWebText 与 Swiss-Prot 评测集；使用近重复分组和逐层非退化 `r99` 审核字典基底 | ①先做 offset、正交和线性对齐。②训练独立字典与 Crosscoder。③扫描稀疏权重。④检查拟合充分和特征极化是否同时成立。⑤对单 latent 做匹配随机消融 | 线性对齐仍留残差；Crosscoder 没有同时满足充分拟合与极化。文本侧约 13%–16% live latents 显示超出控制的差异依赖 | 受限候选。差异依赖不等于新增或删除特征；基础模型蛋白行为不可测，蛋白侧因果差分无定义 |
| 同权重文本—蛋白计算子空间 | ProLLaMA Stage 1/2 双模式；Llama-2 作训练前参考 | OpenWebText 与 Swiss-Prot；匹配记录数、每条记录位置数和总计分位置。正式评测包含 3 个 checkpoint、2 个独立语料抽样，共 6 个实验单元 | ①逐层测 occupancy。②消融自身主方向测 necessity。③与随机同维子空间比较 overlap。④做 2×2 cross-mode driveability。⑤把损伤拆成 unigram 和残余上下文部分 | 合成已知答案的 8 项检查全部通过；正式运行中没有任何层得到许可的“分离子空间”或“共享子空间”判决。Llama-2 蛋白模式在多数层可测到非一元频率损伤，但有的层残余占比不足，有的层找不到文本侧必要秩，因此仍不能形成重叠结论 | 已完成，但没有得到共享或分离计算子空间的正式结论。任何子空间重叠都不是生物知识证据 |

**相关工作边界。** ProteinGuide 在固定权重的蛋白生成模型之外训练属性预测器，并在采样时用该预测器重加权离散转移率；论文展示了稳定性、金属结合和 TadA 等属性的生成控制。这证明外部预测器可以有效引导采样，但不等于生成模型内部已经表示或因果使用同一属性，也没有排除同源检索、组成或其他表面捷径。它与本项目的内部特征 steering 不是同一种干预，因此当前作为相关工作引用，不复现为正式实验。

## 方向三：开发新方法判断模型学到了什么

当现有方法不足时，设计新的对照和解释方法，区分语料规律与生物知识。本节按要鉴别的知识问题组织：突变互作、跨模态概念，以及催化、折叠和氨基酸化学与统计替代解释的矛盾。方法资格未通过、结果仍在重组解释内和受限阳性候选都保留在各自问题中；它们不等于已经识别了具体知识。

| 实验 | 模型 | 数据(格式、示例、数量) | 怎么做 | 结果 | 状态与局限 |
| --- | --- | --- | --- | --- | --- |
| 突变互作与语料耦合 | 真实生物阶段未读取模型；gpt2-large、ProtGPT2、ProGen2-medium 只用于合成方法检查 | 双突变行，例如 `A42G:B57V`，且两个 single mutation 必须在同 assay 实测。51 assays、48 identity clusters、7,298 个至少含 10 个 doubles 的位置对；可构建语料耦合者为 23 assays、22 clusters、5,325 对 | ①用两个 singles 建 additive prediction。②用 cross-fitted isotonic regression 去掉全局非线性，得到 specific epistasis。③从语料比对建立 APC 校正耦合并与列置换 null 比较。④只有多数蛋白通过语料正对照才允许读取真实模型 | 语料耦合正对照 0/22 通过，因此模型 epistasis、内部中介和因果阶段都没有运行；合成文本任务可以找回目标 pair | Stage 0 因方法资格未通过而停止。只说明当前 DIAMOND/UniRef50 hit-list 接口不能建立二阶减项，不说明语料或模型没有二阶知识 |
| 跨模态概念对齐 | Llama-2-7B 参考；ProLLaMA Stage 1/2 的文本和蛋白模式 | sequence-description pair，例如 `MKT... ↔ “[MASK] catalyses ...”`。Swiss-Prot `fullName + CC FUNCTION`；池 20,000 对、19,385 个不同序列、17 个 concepts；主 eval 为 4,499 条、3,752 组，gallery 1,000 | ①按近重复组和家族切分并遮蔽概念名。②在预注册 layer 22 拟合 mean/Procrustes/affine 对齐。③比较 shuffled、rank、description-only、composition 和 3-mer。④只有蛋白模式清除全部上界才允许文本方向因果注入和外部 Pfam 验证 | Stage 1 蛋白 excess 为 +0.0097，低于 3-mer 的 +0.0272；Stage 2 同向。相同测量在三个 checkpoint 的文本模式通过 | 正式归入重组并停止；Stage 36 因 STOP-35 从未运行。只覆盖一个谱系、一个层和一次独立抽样，且是序列级表示 |
| 催化活性与家族统计的矛盾检验 | ProGen2-small、ProGen2-medium；ProtGPT2 与 ZymCTRL 未通过方法资格 | matched pseudokinase pair，例如 `(RYK_dead, bit-score-matched active kinase)`。原池 461 条；主评测 15 对；另有 8 个 `active_despite_degradation` 记录，与 dead 侧组成 23-group counter-stratum | ①用 20-bit caliper 中和 Pfam bit score。②用催化位点 motif reader 作生物参照。③对相同 anchor 替换计算模型 NLL contrast。④同时比较 1–7-mer、最近活性激酶检索和 composition。⑤检查 motif-degraded active stratum | ProGen2-small AUROC 0.9200、medium 0.8844，但低于 7-mer ceiling 0.9733；counter-stratum 约 0.73，尚未建立超过 motif 参照的能力 | 小规模预算试运行的读出按所测基线归入重组。2026-09-05 的 CPU 核验发现，固定的 16 对双 caliper 候选全部已超过第一道 20-bit 限制，最小差距 21.1 bits，因此交集为零，低于 8 对下限，当前评测关闭。这不排除其他配对设计；ProtGPT2 与 ZymCTRL 未通过方法资格 |
| 折叠与序列统计的矛盾检验 | 最终关闭不依赖模型；ProGen2-small/medium 只有小规模试运行，ProtGPT2/ZymCTRL 被拒绝 | fold triple：`(anchor, composition-near/different-fold C_seq, sequence-far/same-fold C_str)`；199 triples、367 members、346 near-duplicate groups | ①用 DIAMOND 排除可对齐关系。②验证组成与折叠给出反向排序。③计算 fragment、composition 和 Pfam/profile ceiling。④只用 anchor 前缀运行 jackhmmer。⑤若任一统计成员与结构预测同向，就在读取模型结论前关闭 | Profile 找回同折叠伙伴 56/199，找回组成伙伴 0/199，说明 alignment-clean 不等于 profile-clean | 模型无关地归入重组并关闭。F15 是样本构造方法结论，不是模型负结果；只覆盖人类 AlphaFold 子集、staged Swiss-Prot 和这一 triple 构造 |
| 氨基酸化学与语料统计的矛盾检验 | ProGen2-small/base/medium、ZymCTRL；两次确认各有一个独立 ByGPT5-medium 文本控制。GPT-2-large 与 ProtGPT2 被拒绝 | 20 种残基形成 190 个无序对，例如 `(A,V; 理化轴, 语料轴)`。双侧版本为每个蛋白模型固定 4,096 条轴构造记录和两个各 4,096 条的独立确认样本，并排除样本间精确相同或 5-mer containment≥0.5 的记录 | ①替换输入 embedding，测每个残基对造成的 likelihood damage。②以疏水性、电荷、体积和极性定义理化轴。③用相同记录上的 order-7 fragment substitution damage 定义相反的语料轴和匹配上界。④按近重复序列组×被替换符号组做 2,000 次交叉 bootstrap。⑤确定性填充三个跨样本去重的集合，并要求两个独立确认分别通过 | 14/14 正常结束，零失败；四个轴构造和两个文本控制通过，四个蛋白模型的两次确认共八个主读出均为 `CHEMISTRY`。ProGen2 三个 checkpoint 在 tercile/quartile 通过，但 quintile 均回到上界内；ZymCTRL 的模型减上界区间为正，自身化学方向的区间却在两次确认都覆盖零 | 报告结果已完成，但完整重放受限。14 个原始 SHA-256 校验文件、16 个前置交叉引用及运行前后资源记录均已核验；预注册要求的逐记录充分统计未保存，现有产物只能复算 pair-mean 主效应，不能独立重建交叉 bootstrap 区间。结果是针对指定化学描述和 order-7 片段替代解释的 likelihood 敏感性候选，依赖切分和模型；尚不证明下游因果使用、机制或生物知识。GPT-2-large 与 ProtGPT2 的单残基覆盖约 0.5%，两者未通过方法资格 |

## 当前结论与下一步

现有能力结果没有形成统一的规模规律。ProGen2 到 64.4 亿参数仍未越过 profile 通道，ProGen3 的 3B checkpoint 越过了自己的外部 profile 基线；ProLLaMA 的蛋白适配明显改善突变排序，但仍低于 profile 与 BLOSUM62，同时付出显著文本代价。原生条件接口能够把生成推向请求类别，但没有排除训练分布内检索，也不证明功能；固定权重的解码器从上下文同源序列获得的正增益局限于局部重叠较高的层次，与复制解释一致，但没有因果定位复制机制。不同谱系使用不同语料、分析集和评分方向，不能据此比较“3B 与 6.44B 谁更强”，也不能把差异归因于参数量。方向二的同权重读出显示 Galactica 三个可识别 checkpoint 均为蛋白模式更晚接近最终预测，但差值在 30B 明显缩小；未调优 lens 和末端粗网格仍不允许把它解释为计算深度或规模规律。

隐状态桥接实验 EXP-R2-234–239 已关闭或停止：在固定 EC-7 任务上，预训练供体模型相对随机初始化且权重固定的对照，区间覆盖零，未确立预训练特异增益；普通显式分类器交接优于软桥；该轮未扩大规模。这不是能力排名，也不是新方法，不改变 Galactica tuned lens、F11 与 D3 的下一步顺序。

| 主题 | 当前可以下的结论 |
| --- | --- |
| 模型比较 | 分词、原生输入输出接口、训练阶段、语料、架构和抽样评测都会显著影响跨模态比较；至今没有一条简单的模态规律或规模规律能替代这些因素 |
| 方法审计 | 多种文本可解释性方法在蛋白模型上存在已测得的失真或适用边界；可解码、可重建、可命名或注意力清晰都不等于因果忠实 |
| 生物信息的行为证据 | 条件家族特征与经天然对照校准的序列顺序信号支持模型学到了可用于生成的生物信息；正信号与高有效产出仍须分开 |
| 具体规律与机制 | 方向二、三尚未完成因果使用与独立生物验证，也未建立新的生物学规律。其方法资格规则不是知识存在的普遍定义 |
| 不能声称 | 不能从有限阳性断言普遍理解，也不能从阴性断言没有知识或只在记忆。经过校准的生成表现可提供强度不同的间接证据，但 profile 命中、结构预测和低参考相似均不等于对应序列实测功能，也不识别具体知识或机制 |
| 方向一、二下一步 | EXP-R2-232/233 的论文与可复现审阅包已在本地汇齐，原始代码已公开；投稿仍需作者批准和实际数据访问安排。下一步选定一个已测能力解释其机制：Galactica-1.3B 先做等容量 tuned lens、独立分组留出和末端逐层测量，仍有差异才做因果 patch/消融及 6.7B 复核；F11 则直接检验干净激活拟合与顺序替换评分的分布差异。二者不由这次结构预测替代，进行中的 Galactica 原生 ProteinGym 计分也不替代。EXP-R2-234–239 已关闭，不插入第四个研究方向 |
| 方向三下一步 | 字母表—化学实验已有双确认候选，但须同时保留切分敏感性、ZymCTRL 自身效应不确定和逐记录统计缺失的限制。优先选择自身效应区间为正的 ProGen2 checkpoint 与一个已建立的下游能力，用现有干预验证因果选择性、未见家族泛化和独立实验标签；若只造成通用 likelihood 损伤，或已有基线仍能解释，就停止该设计。只有明确的现有方法失败才支持开发新方法；未来记录须保存可重放的充分统计。方向三的重组上界是针对已实现替代解释的资格规则，不是“模型没有生物知识”的证明 |

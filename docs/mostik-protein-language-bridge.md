# Mostik 与蛋白质–语言隐状态桥：公开评估与候选方案

**日期：** 2026-09-07

本文是公开资料评估与候选方案备忘录，不是新增执行授权，不改变方向一→二→三的既定顺序，也不承担试验运行状态记录。候选任务是带来源与不确定性的功能假设说明，不是已验证的生物功能，也不是 mechanistic explanation。当前科学计划、已完成结论、撤回与准入仍以 `docs/INTERPRETABILITY_TRANSFER_AUDIT.md` 为权威，用户侧概述见 `summary.md`，执行翻译见 `docs/RESEARCH_PLAN.md`。本文不替代上述文件，不重开已关闭的 D3.g / stage 35–36 路线，也不把旧阴性结果改写成“任何 learned adapter 都不可能”。M0/M1 只是方向一的能力增量验证：冻结生成式 pLM 与冻结语言模型之间的短通道，能否在受控任务上相对既有接口与检索/标签基线增加可核查信息。干预约束的概念通信是后续另需准入和授权的机制研究，标为 M2 候选，不是 M1 通过门，现在也不授权执行。若 M0 显示该任务上原生接口与检索基线已经饱和，则不必为“桥的额外价值”进入 M1；这不否定既有 F17 或自然语言说明能力本身。

2026-09-07 用户已授权 M0/M1 及必要模型/GPU扩展；实际预算、执行边界与运行状态见 [`docs/LATENT_BRIDGE_CAPABILITY_PILOT.md`](LATENT_BRIDGE_CAPABILITY_PILOT.md)。

## Mostik 公开查证

官方技术页是 [mostik.ai/read-more](https://mostik.ai/read-more)。可核实的核心设定是：发送端与接收端权重冻结，只训练一个小桥，在模型之间传递 hidden states，中间不传生成文本。公开演示把 GLM-5.2（官方写 753B）仅用于 prompt prefill，再把状态交给 Qwen3.5 4B 完成全部 decode。这是同模态文本模型之间的计算交接，不是蛋白质–语言跨模态论文的默认设定。

官方图 3（[原图](https://framerusercontent.com/images/l2ZWP8FlIqMnqMSsza9xLubEY.png?width=2946&height=828)）给出五个数学/代码基准的 AVG@4 均值：Receiver 62.35，桥接系统 76.79，Sender 92.66。按这组均值，绝对提升 14.44 个百分点，相对 Receiver 约 23.2%，相对 Sender–Receiver 差距补上约 47.6%。正文中的约 25% 自身提升、关上约 50% 差距只能称作近似口径，不断言与图 3 严格等同或已经过未展示的舍入规则。官方图 5（[原图](https://framerusercontent.com/images/vTiO0mTlPWw2RceAXLTalEHoxw.png?width=2368&height=1244)）是 AIME22–26 的 Pass@4 成本曲线，货币单位，与图 3 的准确率口径不同，二者不可混写成同一加速比。文中相对“打出同样分数的中型单模型”约 2.5 倍更省，是另一条比较，不能改写成实测 20 倍加速；也不把未在本次核验中钉死的“完整大模型成本的 1/20”写成已测墙钟加速。

本次检索未找到可复现论文或官方代码；这只说明当前可见公开材料不足，不能绝对断言不存在未索引仓库。活跃参数、服务配置、成本配方、桥所接的层与算子、训练数据与损失权重均未披露。官方用约 2 MB 内部状态对约 17 bit 输出 token 作表示尺寸比喻，它不是可传语义信息量的测量，更不是相对文本交接的优势证明。官方自承 “We have not conclusively shown that a bridge is an observability tool”：干预通道是值得借鉴的问题设定，不是已完成的可观测性结果。

竞赛公开榜见 [ARC Prize 2026 / ARC-AGI-3](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/leaderboard)。2026-09-07 经 jina 获取的页面上，队名为 mostik.ai、公开分 7.51、当时列第 4；动态页面与代理缓存都不是实时终榜。先前新闻把 CSTL 写成第一，队名变更只是推断，不当已证事实。竞赛完整 harness、所用模型以及桥是否进入提交均未披露；官方技术页不提竞赛，因此不能把 753B→4B 演示说成竞赛方案。[WIRED 报道](https://www.wired.com/story/russian-startup-mostik-ai-models-communication/) 只能作辅助叙述。

对本项目有用的不是复现 Mostik 未公开算法，而是可操作的双冻结 latent 通道：冻结两端、只训桥；发送端做一次并行读取（prefill），接收端写输出。Prefill 是读取，不是已完成的深度推理或规划；若还要额外推理 token，必须另计账。计算比较依赖实际输入/输出长度、模型、硬件与采样预算，不能把图 5 的货币曲线或未测的序列长短默认搬过来。用替换发送态观察接收行为属于 M2 候选，不是 M1 能力门。下面的方案是仓库可实现设计，明确不是 Mostik 公开算法。

## 仓库既有桥与公开先行工作

仓库已经做过同权联合模型上的序列级语言–蛋白对齐，而不是双模型隐状态交接。D3.g（EXP-R2-213）比较序列级池化、概念名遮蔽后的描述、以及 mean / Procrustes / affine 线性对齐；蛋白模式 Procrustes excess 为 +0.0097，低于 3-mer 上界 +0.0272（审计 F13）。该阴性只否定既定桥、既定数据与既定检验，不否定一切非线性 adapter。Stage 36 因果注入因 STOP-35 从未在真模型上运行，D3.i 经此路径不可达。旧设计是同一联合 checkpoint 的两种模式、序列级池化、线性对齐与最近邻检索；新假说是两个冻结生成模型之间的短通道，以及干预后接收行为是否按方向改变。对象、损失与成功标准都不同，因此不能把 STOP-35 写成对新桥的预判决，也不能把新桥写成对 STOP-35 的补丁。本文不自动重开该路线，也不改 stage 35 / 36 的门。

方向一已经建立的是受控接口，不是自由自然语言生成。F17 / EXP-R2-227 中，ZymCTRL 的原生 EC 标签与 ProLLaMA Stage 2 的 superfamily 指令相对错配类别的请求减错配分别为 +0.8822 与 +0.1267。这支持“生成朝请求类别移动”，不是任意自然语言驱动、不是内部 feature steering，也不是酶活或稳定性实验。结构补充表明两类产物都有超出长度与组成的顺序信号，但绝对产率与参考相似性不同，不能据此给模型排名，更不能把 profile 命中写成功能。旧内部 steering 为 0/8，已退役，不得借新桥复活。InstructProtein 论文声称双向生成，但仓库实测文本模式资格失败（文本位置约 96.3% 概率质量落在残基 token），不能当作已合格双向桥。

公开先行工作里，最接近“冻一侧、学小投影、让语言模型说话或让蛋白模型被条件化”的是跨模态适配，而不是 Mostik 那种同模态双冻结隐状态交接。ProtT3 冻结 ESM-2 150M，用 Q-Former（约 8 个 query token）接到 Galactica 1.3B 的 LoRA，做蛋白 caption / QA / 检索，代码见 [github.com/acharkq/ProtT3](https://github.com/acharkq/ProtT3)，论文 [arxiv.org/html/2405.12564](https://arxiv.org/html/2405.12564)。只冻结蛋白 encoder 再 LoRA 语言模型，并不等于双冻结；Q-Former 也不是 Mostik 已披露的桥。它仍可作为本方案新桥的候选实现：若池化 MLP 成为瓶颈，再考虑 query-resampler，而不是一开始就上更重模块。Prot2Chat 冻结改写后的 ProteinMPNN，用问题感知 adapter 压成 virtual tokens，再 LoRA LLaMA3 做问答，见 [arxiv.org/abs/2502.06846](https://arxiv.org/abs/2502.06846) 与 [github.com/wangzc1233/Prot2Chat](https://github.com/wangzc1233/Prot2Chat)。ProteinDT 用对比学习、facilitator 与独立 decoder 做文本引导序列生成，见 [arxiv.org/html/2302.04611](https://arxiv.org/html/2302.04611) 与 [github.com/chao1224/ProteinDT](https://github.com/chao1224/ProteinDT)；其“大于 90% 生成准确率”是同一 CLAP 上的检索多选，不能当独立生物学验证。BioT5+ 是约 252M 的统一 T5，预训练含蛋白–文本双向翻译，蛋白设计没有成熟基准，见 [arxiv.org/html/2402.17810v2](https://arxiv.org/html/2402.17810v2) 与 [github.com/QizhiPei/BioT5](https://github.com/QizhiPei/BioT5)。ProTrek 是序列–结构–文本三模态检索，不是生成，见 [Nature Biotechnology 论文](https://www.nature.com/articles/s41587-025-02836-0) 与 [github.com/westlake-repl/ProTrek](https://github.com/westlake-repl/ProTrek)。以上检索不是穷尽综述，不能声称全球首个，也不能声称不存在任何对应开源实现。

对本文方案，这些工作给出三条边界。第一，注释复述（BLEU、exact match）很容易看起来像“解释”，却不必增加序列本身没有写在 Swiss-Prot 里的信息。第二，对比检索可以把“语言驱动生成”做成最近邻查找；ProTrek 类系统应标成 baseline，而不是创新。第三，统一词表里硬学 FASTA 的 LLM / T5，与冻结的生成式 pLM 隐状态不是同一对象。ESM-2 与 ProtBERT 是 encoder；真正的序列生成需要自回归或扩散接口。DPLM-2 一类工作是序列–结构联合生成，原文不以自然语言为条件，不能算作语言桥的已完成对照。

## 两个方向的最小方案

A 是蛋白→语言，目标是带来源与不确定性的功能假设说明，不冒称 mechanistic explanation，也不把复述注释得分叫生物功能已经验证。发送端取仓库资格通过的生成式 pLM（例如 ProGen2；最终以现有 `load_arm` / `load_arm_spec` 可加载臂为准），对整条输入序列做 prefill，读取 hidden states。面板成员走 `load_arm`；更高 rung 若只作为 staged non-member 存在，则必须显式走 `load_arm_spec`，不得 silently 替换。ESM-2 只作为强工程对照，明确不能当自回归生成器：它没有 token-by-token 生成接口。Prefill 只提供该次前向的隐状态，不把它写成已经完成规划。接收端冻结一个可读隐状态的小型开源文本模型，从现成文本臂起，不加载 753B。桥的起步实现是池化 MLP，把发送态压成 8 或 16 个 soft tokens；只有确认该瓶颈后才加 query-resampler。桥的输出进入冻结语言模型，生成受限 JSON 字段或短自然语言：功能假设、可核查证据、不确定或拒答。残基级证据必须来自独立遮挡、替换或外部标签，不能把 attention 当解释，也不强行词与残基一一对应。训练只更新桥，用配对监督交叉熵起步。检索损失与对语言模型的 LoRA 是明确分开的实验臂，不能在主臂里偷偷解冻。没有 Q&A 或指令资格的基础语言模型先做接口正对照：若它连字段格式都跟不住，后续“说明质量”不可解释，应先换有指令资格的接收端，而不是把失败算成桥失败。评估时固定检索、工具与注释的可见性，比较有桥与无桥的增量；工具和 latent 可以互补。去掉工具只是拆来源消融，不是证明额外价值的必要条件。输出模式建议固定为少数字段：功能假设、支持该假设的可复核证据、与证据冲突时的拒答。自由散文只作为人工阅读附件，不进入主指标。最低成本部署替代是检索或工具调用加上有引用的语言模型说明，标签为 baseline，不是创新。优先做 A 的理由是：全序列前向直接、真实注释评估较便宜；B 的生成产率与独立功能验证更贵。

B 是语言→蛋白。第一步训练免费：把自然语言映射到受控 EC 或 superfamily 约束，再走现有 ZymCTRL / ProLLaMA 原生 prompt。越出接口词表的需求拒绝或标 `unsupported`，不把“高稳定”“高可溶”等自由形容词转成已验证控制。真正的候选桥是：开源文本模型只 prefill 需求；可选地把显式短规划文本作为独立对照，而不是默认混进潜变量。若显式短规划已经达到与桥相同的 requested-versus-mismatched，则 latent 没有额外任务价值。小 MLP 把语言 hidden 压成生成式 pLM 的 soft-prefix 或 KV-prefix，MVP 只选一种模型兼容方式，KV 是后备而不是必须同时实现。两端冻结，用 sequence 交叉熵做 teacher forcing。训练时模型只看见文本条件与 \(x_{<t}\)；任何用作生成条件的蛋白发送端不能读目标全序列或未来 token。全序列目标在 A 合法，在 B 作为额外输入会泄漏。蛋白生成器必须有真实自回归或扩散接口；ESM-2 与 ProTrek 检索不是生成。范围先限制在已有良性酶类别和足量配对数据，不承诺一次覆盖任意新功能。量化属性只有存在独立实验标签时才能进入。对封闭类，EC 标签本身往往已带足任务信息，因此只有同类细属性、未见约束组合、或独立反事实增益，才能论证 latent 相对原生标签的额外价值。B 的采样必须按全部 attempts 记账，选择后的高置信候选不得改写分母。

两个方向不强制共用同一个矩阵，也不要求互为逆映射。描述对序列是多对一，不把序列↔描述一一对应写成成功标准。A 允许同一功能描述对应许多序列；B 允许同一约束生成多样序列。成功看的是约束是否被满足、无关约束是否保持，而不是重建训练对。

## 待验证的研究创新：干预约束的概念通信

本节是 M2 候选：需单独准入与授权，不是现在执行，也不是 M1 通过门。若只把桥训到生成通顺，得到的是多模态适配能力，不是“模型已经把功能概念交给另一侧”。本文把真正待检验的对象称作干预约束的概念通信器：固定桥与接收模型，替换发送态或概念子通道，观察语言断言或生成的外部属性是否按预定方向改变，同时无关属性与生成有效性尽量保留。这是待验证假说，不是首创声明。

研究问题是通道到底在传同源或词典记忆，还是可组合的功能差异。构造应使用同家族异功能或近序列单突变、独立实验标签、以及预先写好的可用矛盾集。没有合格矛盾集时，不得事后从生成失败里反推“机制”。干预包括 patch、随机方向、剂量与负对照；必要时做 rescue，而不是强行循环一致性。最小必要测试只有三项。有方向：把发送态从功能 A 换到功能 B，接收端的断言或外部属性应朝 B 移动，而不是朝任意第三类。有特异性：无关字段、无关催化类别或生成有效性不应同步崩溃；随机方向与剂量为零时应回到基线附近。干预后移除或恢复：撤掉 patch 后效应应回落，或把原发送态写回后效应应恢复；只升不降的曲线不能当因果。不要堆一组方法名称来代替这三项。

桥训练成功且两端冻结，并不证明生物学本来写在发送端里。监督来自注释时，桥完全可能把注释统计学进通道。能力增量验证应在相同桥容量、训练数据、receiver 与评测下，比较预训练 pLM、保留同样输入的随机冻结编码器、以及 k-mer / 组成特征。其中预训练特异的 held-out 增益，再加上固定桥对发送态的干预，可以加强“通道用了 sender 信息”的证据，仍不能单凭性能证明原生生物学知识。尚不能证明先天通用几何，也不能声称新的生物学发现。若没有合格矛盾集，M0/M1 仍可做能力工程评估，但必须停止“学到了什么 / 机制”主张；M2 不得用新 adapter 绕过审计 §7.0 的准入，也尚未授权。

所有方案共用同一套验证纪律。按同源组或家族拆分；注释来源、模板与条件组合留出；时间切分与训练可见性在未知时如实记录。低 identity 不保证 profile-clean：F15 已说明 alignment-clean 的折叠对照仍可被 profile HMM 找回。训练、验证与最终 test 隔离。验证集可以调桥、选概念；真正禁止用 final-test 选概念或调参。若发生 final 泄漏，应撤回该测试结论，并使用新的独立测试集，不能在同一批已窥见数据上重新随机切分后号称独立。匹配桥参数、训练样本、候选采样数和计算账。至少包含：无桥；显式文本或原生标签交接；最近邻 / ProTrek 类检索；经同容量桥的 k-mer 或组成特征；错配 pair 或 shuffled latent。现成联合模型可附加，但必须声明资格，避免方法 × 模型的全笛卡尔积。Galactica 合格双模式 checkpoint 与 ProLLaMA Stage 1/2 可以作附加对照，InstructProtein 不能。现有渲染与泄漏纪律保留。新的能力 pilot 独立定义对象、主指标与成功标准，不把旧 stage 的统计门或同权词典约束当成所有新跨模型桥的强制接口，也不绕过已闭合的科学路线。

A 的指标按证据等级分开写。注释数据只报告留出注释或标签一致性及支持性，不能把模型复述注释的得分叫生物功能已经验证。只有独立实验标签才报告实验性质预测准确性。拒答校准看的是模型在证据不足时是否拒绝。跨家族增益与突变对上的断言差异，同样按标签来源分级。BLEU 只作辅助。B 按全部 attempts 报告 requested-versus-mismatched、有效率、唯一性、多样性、相对参考近邻、以及独立属性或结构指标；选择后的候选与总 yield 分开。共享训练的 CLAP 分数不当独立验证。结构置信度不等于功能；in silico 不替代湿实验。最强主张必须有独立酶活、荧光或稳定性实验与阴性对照。成功统计是预注册收益差 \(\Delta\) 的 family-bootstrap 置信区间下界大于 \(\varepsilon\)。\(\varepsilon\) 以及主指标、成功标准在独立 pilot 之后、最终 test 之前冻结；本文不虚构已定数值。若桥相对检索或标签 baseline 没有该增量，停止“桥有额外说明或生成价值”的主张，但不否定既有 F17，也不否定自然语言说明能力本身。若当前双冻结失败而 LoRA 成功，只说明该双冻结配置未证增量、适配仍可用，不把概念通信的全部可能性一并否定。固定预算后停止。

## 分阶段可行性与停机标准

M0 与 M1 只验证方向一的能力增量，不是机制研究授权。M0 不需要新训练：复用 F17 原生条件与检索基线，确认数据、接口和新问题的正对照，不碰既有关闭阶段。M0 还应确认接收端能否在无桥条件下跟随简单字段格式，以及发送端 hidden 是否能被现有 nnsight / 前向钩子读出；任一项失败则停在接口，不进入训练。M1 在两个方向中优先做一个小的 A 向双冻结桥，再做一个窄类 B 向条件桥。先用不超过 32 条记录做接口检查，再按实际显存 profile 定 batch。约 1–5 万条训练配对是建议预算，不是现有数据承诺，也不是已清点资源。桥可以缓存发送态，但存储按 \(N \times K \times d \times \mathrm{bytes}\) 记账。两端冻结仍须让梯度通过接收端到达桥，不能把整图包进 `no_grad`。显存必须实际 profile，不能断言 7B 级双模型一定单卡可跑。所有实质训练与生成走 H200 与 `scripts/transfer/run_transfer_h200.sh`；L20 只做接口检查。本节只提出候选计算方案，不报告实测速率；获授权试验的执行状态以上述合同为准。记录 prefill、bridge、decode、通信、候选数及独立评估成本，并与文本 handoff 对照；账按实际输入/输出长度、模型、硬件与采样预算计算，只陈述预期，不许诺复现 Mostik 的倍率。M2 候选是干预约束的概念通信，需单独准入与授权，不能作为 M1 通过条件，现在也不执行。

可复用而不改门的现有入口包括：`src/transfer/arms.py`、`joint_modes.py`、`joint_lineage.py`、`sequence_description.py`、`concept_alignment.py`、`concept_injection.py`、`conditioned_generation.py`，以及 `scripts/transfer/45_conditioned_generation.py` 与 `scripts/transfer/run_transfer_h200.sh`。`sequence_description.py` 与 stage 34 队列可用于配对数据的切分与遮蔽，但不得把 stage 35 的线性对齐结果当作新桥已经失败或已经成功。候选路线须独立立项获准后实现；已获授权的 EC 类别能力试验以其合同为执行依据。

停机标准预先写明，并与 M1 一致。A 在预注册 \(\Delta\) 上不优于固定可见性的检索加引用 baseline，或拒答校准崩溃，则停止“桥有额外说明价值”的主张；不因此否定注释数据上的功能假设说明任务，也不否定 F17。B 在原生 EC / superfamily 接口已经饱和、latent 相对该接口没有独立反事实增益时，停止“桥有额外生成价值”的主张，同样不否定原生条件接口本身。任一方向若当前双冻结失败而 LoRA 成功，只记录该双冻结配置未证增量、适配可用，不把概念通信一并否定。没有合格矛盾集时，不启动或升级 M2 机制叙事。计算账相对文本交接没有可辩护的节省，且质量不增，则不把双冻结桥当作部署方案。pilot 用尽建议样本与计算预算，仍不能在最终 test 前已冻结的 \(\varepsilon\) 上使 family-bootstrap 下界大于 \(\varepsilon\)，同样停止，不追加模型或损失项。

## 来源

Mostik 官方技术页：<https://mostik.ai/read-more>。ARC-AGI-3 公开榜：<https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/leaderboard>。辅助报道：<https://www.wired.com/story/russian-startup-mostik-ai-models-communication/>。ProtT3：<https://arxiv.org/html/2405.12564>，<https://github.com/acharkq/ProtT3>。Prot2Chat：<https://arxiv.org/abs/2502.06846>，<https://github.com/wangzc1233/Prot2Chat>。ProteinDT：<https://arxiv.org/html/2302.04611>，<https://github.com/chao1224/ProteinDT>。BioT5+：<https://arxiv.org/html/2402.17810v2>，<https://github.com/QizhiPei/BioT5>。ProTrek：<https://www.nature.com/articles/s41587-025-02836-0>，<https://github.com/westlake-repl/ProTrek>。InstructProtein 论文与仓库资格冲突见正文，不把它列为已合格双向资产。仓库内 D3.g、F17 与生成补充的权威叙述仍在审计与 `summary.md`，此处只引用已给定的数字，不重做审计。公开论文与仓库路径以各节正文为准；许可证与离线体积只在实现立项时再核，本文不堆未使用的条款。竞赛榜与 Mostik 演示数字会变，引用时必须标明获取日期，不得把快照写成终态。本文写作时竞赛页快照日期为 2026-09-07，仅用于标明来源时效，不更新科学计划或排行主张。

# 隐状态桥能力试验合同（M0/M1）

**日期：** 2026-09-07
**状态：** 已完成。接口验证通过，但尚未确立预训练带来的独有增益；主准确率还低于事后补充的训练集多数类参考。目前不扩大模型或进入机制研究。事实单源见 EXP-R2-234（本地 prepare/L20 M0）与 EXP-R2-235（H200 执行与锁定测试）。
**权威：** 科学结论、关闭路线与准入仍以 `docs/INTERPRETABILITY_TRANSFER_AUDIT.md` 为准。方案背景见 `docs/mostik-protein-language-bridge.md`。本文只规定本试验的执行合同，不更新 F/L，不改变方向一→二→三的顺序。

## 范围

用户授权 M0/M1 方向一系统能力试验，以及必要时更大模型与更多 H200。这不是旧 stage 35/36，不是 M2 机制或知识结论，不重开 D3.g，不进入 `STAGE_CONTRACTS`。冻结两端只是控制条件，不是原生语义对齐的证明。Mostik 公开的是同模态双冻结隐状态交接原则；本试验不复现其未公开结构、训练配方或竞赛系统，也不沿用其速度倍率。

问题：冻结生成式蛋白模型对内容位做 mean pooling 得到 hidden，经 K=8 的小 MLP 压成 soft-prefix，交给冻结文本模型，能否传递 EC 顶级类信息，并相对规定基线产生可核查增量。成功只说明该通道在该任务上有通信能力；注释监督可能进入桥，因此训练成功不能自动证明生物学知识原本就在蛋白模型里。

首轮任务仅 EC 顶级 7 类，规范英文 JSON `{class: 1..7, name: <canonical enzyme class>}`。不要 evidence/abstain 伪字段。不称自由功能解释，也不称生物功能验证。

| class | name |
|---|---|
| 1 | oxidoreductase |
| 2 | transferase |
| 3 | hydrolase |
| 4 | lyase |
| 5 | isomerase |
| 6 | ligase |
| 7 | translocase |

## 数据

复用 stage 34 `non_iea` 记录字段：`ec`（list）、`sequence`、`accession`、`dup_group`/`family_group`/`split`。过滤前 fit 10502 / eval 4499 / family_holdout 4999。这是历史队列复用，不是新前瞻生物证据，也不把 Swiss-Prot 注释一致性写成湿实验功能。

从 `ec` 提取唯一顶级数字。空、无法解析、跨顶级类、非法氨基酸、超过声明长度的记录分别显式计数后排除，不静默 truncate，也不用描述文本补标签。监督来自 `ec` 而非 `description_masked`。donor 只见 `'1'` + 原序列，不见 EC、类名或注释。receiver 在桥接臂只见桥给出的 soft-prefix 与要求输出该 JSON 的指令，不见序列与标签；raw-sequence 对照臂可见完整序列，仍不见标签。

split：fit=train，eval=val，family_holdout=锁定 test。验证 accession 与 dup 不交、held-out family-group 不交；任一泄漏则停止，不改 split 后继续。cap 按固定 seed 与 ID hash 选，不按标签难度或模型分数挑样本。prepare 须写出过滤前/后计数、按原因排除数、各类 support、实有 train/val/test 条数、长度分布与哈希。资源预查只 admit EC 1–6 概念；EC7 的实际样本 support 由 prepare 报告，不能承诺七类都可评价。有效数量先 prepare 统计；不足声明 cap 时记录实有数量，任何改预算在训练前写入，不得在看 test 后改。

## 模型与训练

两端独立加载、权重冻结，只训桥。无 nnsight、无 peft。冻结 receiver 但仍反传到 bridge，优化器只更新桥。缓存仅未训练 donor 统计，不得缓存已训桥或 receiver 梯度。

| 角色 | 首轮 | 备注 |
|---|---|---|
| donor | ProGen2-medium | H200 已有权重；small 可做接口。L20 M0 用 medium。 |
| receiver | Qwen2.5-0.5B-Instruct | Compute 已转存 GPFS 并做内容 SHA 校验；H200 M0 已从该路径加载并通过接口门。不得冒充 PANEL 同名 base。 |
| 对照文本 | 同一 0.5B-Instruct | 首轮 mandatory raw-sequence 用该 Instruct。7B/32B base 仅可选后续，不混入主臂。 |
| 扩容 | 本轮不扩容 | 现成 3B-Instruct 仅在后续另有容量证据时再议；不按当前 test 扩大 receiver 或 GPU。 |

ct 4.57.3 与 H200 标准环境版本不同须记录。Qwen3.5 不受现有环境支持，先不升级。Instruct 权重必须与 PANEL 同名 base 分目录存放，加载失败显式报错，不得回退到 base。

桥：mean-content hidden → K=8 小 MLP → 冻结文本 soft-prefix，trainable budget 3,000,000。首轮训练预算在未见任何 test 表现前确定：三桥（预训练 donor、随机冻结 donor、3-mer 同预算）train/val 同 steps；epochs 20，batch 32，AdamW lr 1e-3、weight_decay 0.01、默认 betas (0.9, 0.999)、eps 1e-8，grad_clip 1，seed 20260907，random_donor_seed 20260907，bf16/chat。caps train 4096 / val 512 / test 1024，max_residues 512，max-tokens 与 max-prompt-tokens 1024，max-new-tokens 64。最佳 checkpoint 仅按 val token NLL 选。先一组配置，不全模型网格。看 test 后调整必须承认探索性，不能再称同一 test 独立确认。

## M0 / M1

M0 不超过 32 条非 test 记录，检验：原生渲染与方向标记、声明长度拒绝而非截断、内容位 mask、桥参数有梯度、donor 与 receiver 权重不变、oracle-label 能跟住上述 JSON。任一项失败则停在接口，不进入 M1。L20 只空闲 GPU1 做接口；训练全 H200。接口过才训。M0 通过只说明仪器可跑，不是能力结果。

M1 首轮预算见上节。checkpoint 只在 train/val 上选，训练不扫 test。三桥 checkpoint 固定后，各臂对锁定 test 各评一次，不能按 test 调参。各臂共享同一锁定 ID 列表，不得为某一臂单独补样本。

## 基线与读出

同 ID 对齐的必需臂：

| 臂 | 作用 |
|---|---|
| receiver 直接读完整 raw sequence | 不能只盲猜 |
| fit-only 原 8000 维 3-mer 最近邻 | 片段统计对照 |
| 3-mer → 同训练参数预算桥 | 同容量非预训练对照 |
| 预训练 donor 桥 | 主臂 |
| 同架构随机初始化冻结 donor 桥 | 预训练特异对照 |
| oracle-label | 只格式仪器，不当能力结果 |

所有 attempt 生成 JSON。主读出是 class accuracy：JSON 有效且 class 正确才算对；无效 JSON 计为 class 错，不丢弃分母。valid JSON、name 一致性、macro-F1、support、confusion 分列，不把辅助列升格为主读出。name 一致性是辅助单列，不改写主指标；可另列 joint class+name 准确率。无 support 类的 macro-F1 政策须写入 run config 与 support 表，不臆造分数。主差值是全部 attempt 上的 class accuracy 差（记录加权比例，不是各族准确率再平均的族宏差）。共同 ID 必须严格对齐；重复 accession 或错配 ID/`family_group` 拒绝，不折叠。family paired-bootstrap 以 `family_group` 为有放回抽样单位，抽中一次即保留该族全部记录权重，重复抽中则重复计入，不把各族准确率等权平均。单位不足报 `unavailable`，不做记录级或 token 级回退。分析在未见任何真实 test 表现前固定：`n_resamples` 5000、`seed` 20260907、95% percentile 区间。四个预训练 donor 桥减去规定对照（raw-sequence、fit-only 3-mer 最近邻、3-mer 同预算桥、随机冻结 donor 桥）的区间是探索性边际估计，不作多重校正，不设 epsilon，不报 confirmatory PASS。3-mer 最近邻只在 fit 上建 gallery、拟合与调参，不得把 val/test 用于这三项；把 val/test 序列当查询合法。

首轮探索性 pilot 只报估计和区间，不事后挑 epsilon 宣称确认 PASS，也不把 val 选模结果写成 test 确认。若后续确认，epsilon、主要对照、预算在独立 pilot 后、final 前冻结，要求 CI 下界大于该 epsilon。注释一致性、CLAP、结构置信都不是湿实验功能。训练超参与既有 M0/M1 门不变。

## 决策

比较对象是预训练 donor 桥相对规定基线的差值，不是绝对准确率。无桥、文本直读或 3-mer 同样好，则不支持这套 bridge 增量，不否定 F17，也不把阴性写成一切跨模型桥不可能。随机冻结 donor 不差于预训练 donor，则不支持“用了该蛋白预训练表示”的能力主张。oracle-label 失败则先修格式，不解释为桥失败。本轮只做蛋白→语言的 EC 顶级类通信，不做语言→蛋白生成，也不做残基级功能解释。M2 另需准入与授权，不能用本试验绕过审计 §7.0。

## 执行

H200 用 `scripts/transfer/run_transfer_h200.sh` 冻结，再以现有 `scripts/transfer/run_external_baseline_h200.sh --stage 48_latent_bridge.py` 执行。代码需先提交明确 PIN 再 freeze，排除 AGENTS/CLAUDE 用户改动。训练/评估源码 PIN 为 `acb254abc74c4bb9f2fc5ac09ee963ea2a58192f`；独立分析 PIN `111c3d4076d5fe0c44d5e26dc61b83420ae80da4` 不同，不改变训练或生成。绑定 code、data、model、runtime receipts。receipts 至少含 commit、数据哈希、模型路径与版本、Python/PyTorch/Transformers 版本、GPU 计数与峰值显存。本用户先用现有 1 张空闲 H200，不抢其他进程，不持久化 pod 名。缓存、训练、推理的时间与峰值显存分别记录，三者不得混成一条墙钟。L20 M0 通过不单独构成 H200 验收。

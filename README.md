# DTSeek 决策模型引擎

> **基于双端解耦与自回归切片发射（Pointer + Action Tokens）的非自回归/语句切片分类决策引擎**。  
> 融合 **TypeSafe Jev / Laya** 的非自回归极速判决理念、**DETR / YOLO** 目标检测范式，以及 **nanoSeek** 紧凑字符级词表与测量纪律。

---

## 核心特性

1. **单次前向长文本缓存（Memory Cache）**：
   - 文本编码器（Doc Encoder）对长篇上下文仅做一次特征抽取；下游不同任务或同一任务的连续多切片查询共享此特征图，杜绝自回归逐字生成带来的平方级算力消耗。
2. **纯数字锚点输出与零文本幻觉**：
   - 模型在推理时**不生成任何原始字符串**，严格输出 1-based 闭区间位置指针 `[start, end]`、类别及置信度；
   - 终端渲染引擎基于位置指针自动在原文中进行彩色下划线高亮。
3. **自回归序列切片发射（`<slice>...<cont>/<eos>`）**：
   - 彻底摆脱 YOLO 固定数量锚点与 NMS 的死板过滤；
   - 每步吐出一个切片并预测动作 Token：`<cont>`（继续发射下一个）或 `<eos>`（终止输出），天然支持 0 到任意多个可变切片。
4. **长文档标点感知自适应分句**：
   - 内置 `segmenter` 引擎，自动将千字长篇大论拆解为自适应语义子句（$\le 55$ 字符），保证每个子任务天然位于小模型（$\sim 0.1\text{B}$）的最优感受野内，并自动无损换算回整篇长文的全局字符坐标。
5. **纯正 6000 字符级分词器**：
   - 依赖独立的 `nano-char-tokenizer` 模块，1 汉字 = 1 Token，彻底规避 BPE 乱码碎片。

---

## 项目目录结构

```text
DTSeek/
├── checkpoints/                 # 模型训练权重 (.pt)
│   ├── robust_ar_dtseek.pt      # 增强型自回归切片发射模型（支持长文本分句高亮）
│   ├── slot_multispan_dtseek.pt # DETR-Slot 多切片检测模型
│   ├── yolo_dtseek.pt           # YOLO 式单切片定位模型
│   └── pronoun_dtseek.pt        # 基础代词判决模型
├── dev-notes/                   # 历次架构探索与踩坑演进笔记
│   ├── 01-架构探索与初级训练踩坑笔记.md
│   └── 02-自回归语句切片分类与长文档自适应分句演进笔记.md
├── examples/                    # 推理与交互式演示入口
│   ├── example_robust_ar.py     # 【推荐】长文档自适应分句 + 自回归切片高亮演示
│   ├── example_ar.py            # 基础自回归连续切片发射演示
│   ├── example_multispan.py     # DETR-Slot 多切片检测演示
│   ├── example_yolo.py          # YOLO 式语句切片分类演示
│   ├── example_multitask.py     # 【推荐】多任务同屏演示（代词/情绪/归属三类切片）
│   └── example.py               # 基础代词识别分类演示
├── training/                    # 各演进阶段的训练脚本
│   ├── train_multitask.py       # 【推荐】多任务交替联合训练（共享基座 + 3 张任务卡）
│   ├── train_robust_ar.py       # 增强型自回归模型训练（显式反馈 + 因果掩码）
│   ├── train_batched_ar.py      # 基础自回归模型矢量化训练
│   ├── train_fast_multispan.py  # 快速多切片检测训练
│   ├── train_yolo.py            # YOLO 复合损失训练
│   └── train_pronoun.py         # 初级代词分类基准训练
├── src/dtseek/                  # 核心源码包
│   ├── nano_doc_encoder.py      # 【基座】nanoSeek 资产：RMSNorm+RoPE+QK-Norm+SwiGLU+FlashAttn
│   ├── doc_encoder.py           # 旧版简易编码器（保留作对照）
│   ├── robust_ar_model.py       # 增强型自回归切片解码器（Causal Mask + feedback_proj）
│   ├── ar_slice_model.py        # 基础自回归切片解码层
│   ├── segmenter.py             # 长文档自适应分句与全局绝对坐标映射引擎
│   ├── slot_detector.py         # DETR-Slot 动态切片槽位模型
│   ├── slot_loss.py             # 匈牙利二分图匹配与集合损失函数
│   ├── query_projector.py       # 正交 Query 投影层（防止表示退化）
│   ├── rich_ar_dataset.py       # 代词切片数据集（v2：背景句配额 32%）
│   ├── sentiment_dataset.py     # 情绪切片数据集（4 分类均衡）
│   ├── ownership_dataset.py     # 归属人切片数据集（背景句配额 30%）
│   ├── evaluation.py            # 标准多维探针评估套件（Probes + 混淆矩阵）
│   ├── pipeline.py              # DTSeek 顶层执行调度引擎与任务卡注册
│   └── model.py                 # DTSeek 核心解耦决策模型
├── tests/                       # 单元测试用例
│   └── test_model.py            # 张量形状与前向冒烟测试
└── pyproject.toml               # uv / pip 依赖管理配置
```

> **模型权重不入 git**：`checkpoints/` 已被 `.gitignore` 忽略（权重是大体积二进制，
> 且可由训练脚本完整复现）。clone 之后需按下方顺序自行训练产出权重。

---

## 快速上手与使用

### 0. 训练产出权重（首次使用必做）

权重不入库，clone 后先跑一次多任务训练：
```bash
cd /home/vesita/coding/my/DTSeek
uv sync
uv run python training/train_multitask.py
```
训练结束会产出：
- `checkpoints/multitask_v2_dtseek.pt` —— 共享基座 + 3 张任务卡的权重
- `checkpoints/multitask_v2_metrics.json` —— 逐任务可判对错指标
  （首切片类别准确率 / 区间完全命中率 / 背景句误报率）

### 1. 多任务同屏演示（推荐）
一次输入，同时给出人称代词、情绪倾向、发言归属三类切片：
```bash
uv run python examples/example_multitask.py
```

### 2. 交互式切片分类高亮演示
执行增强型自回归切片演示（需先跑 `training/train_robust_ar.py`）：
```bash
uv run python examples/example_robust_ar.py
```

在交互式终端输入任意中文，模型将实时输出切片判定并在原文行内彩色高亮：
```text
DTSeek > 你好，请问你知道我这句话是什么意思吗
 -> 高亮解析: 你(绿)好，请问你(绿)知道我(青)这句话是什么意思吗
    • Step 1: [1:1] '你' (第二人称, 置信度: 1.0) <cont>
    • Step 2: [6:6] '你' (第二人称, 置信度: 1.0) <cont>
    • Step 3: [9:9] '我' (第一人称, 置信度: 1.0) <eos>
```

### 3. 单句命令行测试模式
```bash
uv run python examples/example_robust_ar.py "这期视频我把一个模型接入到了系统里，你看了吗"
```

### 4. 任意长文本（无截断全局高亮）测试
```bash
uv run python examples/example_robust_ar.py "最近 JEV 成了热门模型。这期视频我把与官方完全兼容的模型接入到了系统。这次我也没有只讲参数，如果你感兴趣可以领取体验。"
```
输出将自动通过 `segmenter` 拆分子句，并在 500+ 字的全文上无损精准还原第 17 位“我”、第 48 位“我”、第 65 位“你”的 1-based 闭区间位置！

---

## 运行单元测试
```bash
uv run pytest
```

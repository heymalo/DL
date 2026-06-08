# Batch Size 与 Epoch 的关系

> 记录时间：2026-06-03
>
> 核心问题：`batch_size` 和 `epoch` 是什么关系？它们如何共同决定模型训练过程？

---

## 一、三个核心概念

| 概念 | 英文 | 含义 |
|------|------|------|
| **Epoch** | Epoch | 把整个数据集**完整地过一遍**，称为 1 个 epoch |
| **Batch Size** | Batch Size | 每次送进模型训练的**样本数量** |
| **Iteration / Step** | Iteration / Step | 每完成一次前向+反向传播，称为 1 个 iteration（步） |

---

## 二、数学关系

```
Iteration 次数 = 数据集总样本数 ÷ Batch Size
```

> 如果有余数且 `drop_last=False`，iteration 次数会**向上取整**（最后一批不足 batch_size 也保留）。

---

## 三、具体例子

假设训练集有 **10,000 张图片**：

| Batch Size | 1 个 Epoch 需要的 Iteration 次数 | 说明 |
|-----------|----------------------------------|------|
| 1         | 10,000 次 | 每次 1 张，每张都单独算一次梯度 |
| 4         | 2,500 次  | 每次 4 张 |
| 64        | ~157 次   | 每次 64 张 |
| 256       | ~40 次    | 每次 256 张 |

---

## 四、直观图示

假设数据集有 8 张图片，`batch_size = 2`：

```
数据集: [0] [1] [2] [3] [4] [5] [6] [7]

Epoch 1:
  Iteration 1 → [0, 1]
  Iteration 2 → [2, 3]
  Iteration 3 → [4, 5]
  Iteration 4 → [6, 7]

Epoch 2 (shuffle 打乱后):
  Iteration 1 → [3, 7]
  Iteration 2 → [1, 4]
  Iteration 3 → [0, 5]
  Iteration 4 → [2, 6]

Epoch 3 ...
```

如果训练 **3 个 epoch**，模型总共看过：`3 × 8 = 24` 张（每张重复 3 次）。

---

## 五、代码中的对应关系

```python
# 假设 train_loader 的 batch_size = 64
for epoch in range(10):           # 外层：训练 10 个 epoch（看 10 遍完整数据）
    for batch in train_loader:    # 内层：每个 epoch 内，按 batch 分批取数据
        imgs, labels = batch      # imgs 形状: [64, 3, 32, 32]
        # 1. 前向传播
        # 2. 计算损失
        # 3. 反向传播
        # 4. 更新参数（优化器 step）
```

- **外层循环**：`epoch` 控制"看几遍完整数据"
- **内层循环**：`train_loader` 按 `batch_size` 分批喂数据，每步一个 **iteration**

---

## 六、Batch Size 大小的影响

| Batch Size | 优点 | 缺点 |
|-----------|------|------|
| **小**（如 4, 8, 16） | 内存占用低；梯度噪声大，可能帮助逃离局部最优 | 训练速度慢；梯度波动大，收敛不稳定 |
| **大**（如 256, 512） | 训练速度快；梯度估计更准，收敛更稳定 | 内存占用高；可能陷入尖锐局部最优；泛化能力可能变差 |

> 实际中常用 **32、64、128** 作为起点，根据 GPU 显存和具体任务调整。

---

## 七、Epoch 数量的影响

| 情况 | 表现 | 解决方法 |
|------|------|---------|
| **Epoch 太少** | 欠拟合（underfitting），模型还没学好 | 增加 epoch |
| **Epoch 太多** | 过拟合（overfitting），训练集好但测试集差 | 使用 **Early Stopping**（早停），在验证集性能不再提升时停止 |

---

## 八、一句话总结

> **Epoch 回答"看多少遍"，Batch Size 回答"每次看几张"。两者相除得到"每遍看多少步（Iteration）"。**

它们是两个**独立的超参数**，需要相互配合调优：
- 数据量小 → Batch Size 可以小一点，Epoch 适当多一点
- 数据量大 / GPU 强 → Batch Size 可以大一点，Epoch 适中即可

---

## 九、相关公式速查

```
总样本数 = N
Batch Size = B
Epoch = E

每个 Epoch 的 Iteration 数 = ⌈N / B⌉  （向上取整）
总 Iteration 数 = E × ⌈N / B⌉
```

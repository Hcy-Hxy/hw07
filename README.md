# HW07 胸部X光肺炎影像二分类实战

## 项目结构

```
hw07/
├── train.py              # 训练代码
├── requirements.txt      # 依赖包列表
├── README.md             # 项目说明
├── report.md             # 实验报告
├── evaluation_report.txt # 评估报告（运行后生成）
├── pneumonia_model.h5    # 训练后的模型（运行后生成）
└── figures/              # 图表目录
    ├── sample_images.png          # 样本图像展示
    ├── 全部数据_distribution.png  # 数据集类别分布
    ├── 训练集_distribution.png    # 训练集类别分布
    ├── 验证集_distribution.png    # 验证集类别分布
    ├── training_curves.png        # 训练曲线
    ├── confusion_matrix.png       # 混淆矩阵
    ├── roc_curve.png              # ROC曲线
    └── prediction_examples.png    # 预测示例
```

## 实验背景

医学影像智能处理是人工智能在精准医疗领域最具代表性的落地方向之一。本项目基于 Kaggle 公开数据集 Chest X-Ray Images (Pneumonia)，实现肺炎二分类（Normal vs Pneumonia）实验。

## 数据集说明

### 数据集来源
- **名称**: Chest X-Ray Images (Pneumonia)
- **来源**: [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **提供者**: 圣地亚哥加州大学 (UCSD) 等机构
- **规模**: 约1.15 GB，共5800余张儿童胸部X光影像
- **标注质量**: 所有影像均经过至少两名专家审核，标签可靠

### 数据集结构
```
chest_xray/
├── train/
│   ├── NORMAL/        # 正常胸片（约1340张）
│   └── PNEUMONIA/     # 肺炎胸片（约3875张，含病毒性+细菌性）
├── test/
│   ├── NORMAL/        # 正常胸片（约234张）
│   └── PNEUMONIA/     # 肺炎胸片（约390张）
└── val/               # 原始验证集（仅16张，已忽略）
    ├── NORMAL/
    └── PNEUMONIA/
```

## 环境要求

- **Python**: 3.8+
- **TensorFlow**: 2.10+

### 安装依赖

```bash
pip install -r requirements.txt
```

## 运行说明

### 数据准备

1. 下载 Kaggle 数据集并解压到项目目录
2. 确保目录结构为 `chest_xray/train`, `chest_xray/test`, `chest_xray/val`

### 运行训练

```bash
python train.py
```

### 运行环境建议

| 环境 | 优点 | 推荐指数 |
|------|------|----------|
| Kaggle Notebook | 免费GPU，数据集可直接挂载 | ★★★★★ |
| Google Colab | 免费GPU，需手动下载数据集 | ★★★★☆ |
| 本地环境 | 数据隐私性好 | ★★★☆☆ |

## 模型结构

```
输入层 (224x224x1)
    ↓
Conv2D(32, 3x3) + ReLU + padding='same'
    ↓
MaxPooling(2x2)
    ↓
Conv2D(64, 3x3) + ReLU + padding='same'
    ↓
MaxPooling(2x2)
    ↓
Conv2D(128, 3x3) + ReLU + padding='same'
    ↓
MaxPooling(2x2)
    ↓
Conv2D(256, 3x3) + ReLU + padding='same'
    ↓
MaxPooling(2x2)
    ↓
Flatten
    ↓
Dense(512) + ReLU + Dropout(0.5)
    ↓
Dense(256) + ReLU + Dropout(0.3)
    ↓
Dense(1) + Sigmoid
```

## 训练配置

| 参数 | 值 |
|------|-----|
| 优化器 | Adam |
| 学习率 | 0.0001 |
| 损失函数 | Binary Crossentropy |
| 批量大小 | 32 |
| 训练轮数 | 20 |
| Early Stopping | patience=8 |
| 学习率衰减 | factor=0.5, patience=3 |

## 数据增强策略

| 增强方式 | 参数范围 |
|----------|----------|
| 随机旋转 | ±20° |
| 随机水平平移 | ±20% |
| 随机垂直平移 | ±20% |
| 随机剪切 | ±20% |
| 随机缩放 | ±20% |
| 水平翻转 | 启用 |

## 输出文件

运行后生成的文件：

| 文件 | 说明 |
|------|------|
| `pneumonia_model.h5` | 训练好的模型 |
| `evaluation_report.txt` | 评估指标报告 |
| `figures/sample_images.png` | 样本图像展示 |
| `figures/*_distribution.png` | 类别分布图 |
| `figures/training_curves.png` | 训练/验证准确率和损失曲线 |
| `figures/confusion_matrix.png` | 混淆矩阵可视化 |
| `figures/roc_curve.png` | ROC曲线 |
| `figures/prediction_examples.png` | 预测示例（正确/错误） |

## 评估指标

代码会输出以下评估指标：
- 准确率 (Accuracy)
- 精确率 (Precision)
- 召回率 (Recall)
- F1 分数
- ROC-AUC
- 混淆矩阵

## 参考资料

1. [Kaggle Chest X-Ray Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
2. [TensorFlow 官方文档](https://www.tensorflow.org/)
3. [医学影像深度学习入门](https://arxiv.org/)


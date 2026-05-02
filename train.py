"""
肺炎二分类模型训练 - 胸部X光影像分析
《人工智能导论》课程作业07

数据集：Chest X-Ray Images (Pneumonia)
任务：Normal vs Pneumonia 二分类

代码结构：
1. 数据准备与预处理
2. 数据可视化（样本展示、类别分布）
3. 模型构建（简单CNN）
4. 模型训练与验证
5. 模型评估与结果分析
6. 可视化输出
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# ----------------------
# 1. 配置参数
# ----------------------
DATA_DIR = 'chest_xray'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 0.0001
FIGURES_DIR = 'figures'

os.makedirs(FIGURES_DIR, exist_ok=True)

# ----------------------
# 2. 数据加载与预处理
# ----------------------
def load_dataset(data_dir):
    """加载数据集并返回图像路径和标签"""
    images = []
    labels = []
    filenames = []
    
    for split in ['train', 'test']:
        split_dir = os.path.join(data_dir, split)
        for class_name in ['NORMAL', 'PNEUMONIA']:
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"警告：目录不存在 {class_dir}")
                continue
            
            for filename in os.listdir(class_dir):
                img_path = os.path.join(class_dir, filename)
                images.append(img_path)
                labels.append(1 if class_name == 'PNEUMONIA' else 0)
                filenames.append(filename)
    
    return images, labels, filenames

print("="*50)
print("Loading dataset...")
images, labels, filenames = load_dataset(DATA_DIR)

print(f"总样本数: {len(images)}")
print(f"Normal样本数: {labels.count(0)}")
print(f"Pneumonia样本数: {labels.count(1)}")
print(f"类别比例 (Normal:Pneumonia) = {labels.count(0):d}:{labels.count(1):d}")

train_images, val_images, train_labels, val_labels, train_filenames, val_filenames = train_test_split(
    images, labels, filenames, test_size=0.2, stratify=labels, random_state=42
)

print(f"\n训练集样本数: {len(train_images)}")
print(f"验证集样本数: {len(val_images)}")
print(f"训练集 Normal: {train_labels.count(0)}, Pneumonia: {train_labels.count(1)}")
print(f"验证集 Normal: {val_labels.count(0)}, Pneumonia: {val_labels.count(1)}")
print("="*50)

# ----------------------
# 3. 数据可视化
# ----------------------
def plot_sample_images(images, labels, filenames, num_samples=5):
    """展示样本图像"""
    np.random.seed(42)
    indices = np.random.choice(len(images), num_samples*2, replace=False)
    
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))
    
    normal_indices = [i for i in indices if labels[i] == 0][:num_samples]
    for i, idx in enumerate(normal_indices):
        img = load_img(images[idx], target_size=IMG_SIZE, color_mode='grayscale')
        axes[0, i].imshow(img, cmap='gray')
        axes[0, i].set_title(f"Normal\n{filenames[idx][:20]}...")
        axes[0, i].axis('off')
    
    pneumonia_indices = [i for i in indices if labels[i] == 1][:num_samples]
    for i, idx in enumerate(pneumonia_indices):
        img = load_img(images[idx], target_size=IMG_SIZE, color_mode='grayscale')
        axes[1, i].imshow(img, cmap='gray')
        axes[1, i].set_title(f"Pneumonia\n{filenames[idx][:20]}...")
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'sample_images.png'), dpi=150, bbox_inches='tight')
    print(f"样本图像已保存到 {FIGURES_DIR}/sample_images.png")

def plot_class_distribution(labels, title):
    """绘制类别分布图"""
    plt.figure(figsize=(8, 5))
    counts = [labels.count(0), labels.count(1)]
    plt.bar(['Normal', 'Pneumonia'], counts, color=['#1f77b4', '#ff7f0e'])
    plt.title(f'{title} - 类别分布')
    plt.xlabel('类别')
    plt.ylabel('样本数量')
    plt.grid(axis='y', alpha=0.3)
    
    for i, v in enumerate(counts):
        plt.text(i, v + 10, str(v), ha='center')
    
    plt.savefig(os.path.join(FIGURES_DIR, f'{title}_distribution.png'), dpi=150, bbox_inches='tight')
    print(f"类别分布图已保存到 {FIGURES_DIR}/{title}_distribution.png")

plot_sample_images(train_images, train_labels, train_filenames)
plot_class_distribution(labels, '全部数据')
plot_class_distribution(train_labels, '训练集')
plot_class_distribution(val_labels, '验证集')

# ----------------------
# 4. 数据增强与生成器
# ----------------------
print("\n设置数据增强...")
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

def generate_data(images, labels, datagen, batch_size, img_size):
    """数据生成器"""
    num_samples = len(images)
    while True:
        indices = np.random.permutation(num_samples)
        for i in range(0, num_samples, batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_images = []
            batch_labels = []
            
            for idx in batch_indices:
                img_path = images[idx]
                try:
                    img = load_img(img_path, target_size=img_size, color_mode='grayscale')
                    img = img_to_array(img)
                    img = datagen.random_transform(img)
                    batch_images.append(img)
                    batch_labels.append(labels[idx])
                except Exception as e:
                    print(f"加载图像失败 {img_path}: {e}")
            
            batch_images = np.array(batch_images) / 255.0
            batch_labels = np.array(batch_labels)
            
            yield batch_images, batch_labels

train_generator = generate_data(train_images, train_labels, train_datagen, BATCH_SIZE, IMG_SIZE)
val_generator = generate_data(val_images, val_labels, val_datagen, BATCH_SIZE, IMG_SIZE)

# ----------------------
# 5. 构建CNN模型
# ----------------------
def build_cnn_model(input_shape):
    """构建简单CNN模型"""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

input_shape = (*IMG_SIZE, 1)
model = build_cnn_model(input_shape)

print("\n" + "="*50)
print("模型结构摘要:")
model.summary()
print("="*50)

# ----------------------
# 6. 模型训练
# ----------------------
print("\n设置训练回调函数...")
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=8,
    restore_best_weights=True,
    verbose=1
)

checkpoint = ModelCheckpoint(
    'pneumonia_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-7,
    verbose=1
)

train_steps = len(train_images) // BATCH_SIZE
val_steps = len(val_images) // BATCH_SIZE

print(f"\n开始训练...")
print(f"训练步数: {train_steps}, 验证步数: {val_steps}")
print("="*50)

history = model.fit(
    train_generator,
    steps_per_epoch=train_steps,
    epochs=NUM_EPOCHS,
    validation_data=val_generator,
    validation_steps=val_steps,
    callbacks=[early_stopping, checkpoint, reduce_lr],
    verbose=1
)

# ----------------------
# 7. 模型评估
# ----------------------
print("\n" + "="*50)
print("评估模型...")
print("="*50)

print("\n加载验证集数据...")
val_images_array = []
val_labels_array = []

for img_path, label in zip(val_images, val_labels):
    try:
        img = load_img(img_path, target_size=IMG_SIZE, color_mode='grayscale')
        img = img_to_array(img) / 255.0
        val_images_array.append(img)
        val_labels_array.append(label)
    except Exception as e:
        print(f"加载图像失败 {img_path}: {e}")

val_images_array = np.array(val_images_array)
val_labels_array = np.array(val_labels_array)

print("\n进行预测...")
predictions = model.predict(val_images_array, verbose=1)
predictions_binary = (predictions > 0.5).astype(int).flatten()
predictions_proba = predictions.flatten()

print("\n" + "="*50)
print("分类报告:")
print("="*50)
report = classification_report(val_labels_array, predictions_binary, 
                              target_names=['NORMAL', 'PNEUMONIA'],
                              output_dict=True)
print(classification_report(val_labels_array, predictions_binary, 
                            target_names=['NORMAL', 'PNEUMONIA']))

print("\n" + "="*50)
print("混淆矩阵:")
print("="*50)
cm = confusion_matrix(val_labels_array, predictions_binary)
print(cm)

# ----------------------
# 8. 可视化结果
# ----------------------
def plot_training_curves(history):
    """绘制训练曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(history.history['accuracy'], label='训练准确率', color='#1f77b4')
    axes[0].plot(history.history['val_accuracy'], label='验证准确率', color='#ff7f0e')
    axes[0].set_title('训练与验证准确率', fontsize=12)
    axes[0].set_xlabel('Epoch', fontsize=10)
    axes[0].set_ylabel('准确率', fontsize=10)
    axes[0].legend(fontsize=10)
    axes[0].grid(axis='y', alpha=0.3)
    
    axes[1].plot(history.history['loss'], label='训练损失', color='#1f77b4')
    axes[1].plot(history.history['val_loss'], label='验证损失', color='#ff7f0e')
    axes[1].set_title('训练与验证损失', fontsize=12)
    axes[1].set_xlabel('Epoch', fontsize=10)
    axes[1].set_ylabel('损失', fontsize=10)
    axes[1].legend(fontsize=10)
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'training_curves.png'), dpi=150, bbox_inches='tight')
    print(f"\n训练曲线已保存到 {FIGURES_DIR}/training_curves.png")

def plot_confusion_matrix(cm):
    """绘制混淆矩阵"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['NORMAL', 'PNEUMONIA'],
                yticklabels=['NORMAL', 'PNEUMONIA'],
                annot_kws={'size': 14})
    plt.title('混淆矩阵', fontsize=14)
    plt.xlabel('预测标签', fontsize=12)
    plt.ylabel('真实标签', fontsize=12)
    plt.savefig(os.path.join(FIGURES_DIR, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
    print(f"混淆矩阵已保存到 {FIGURES_DIR}/confusion_matrix.png")

def plot_roc_curve(y_true, y_proba):
    """绘制ROC曲线"""
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='#1f77b4', lw=2, 
             label=f'ROC曲线 (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='#ff7f0e', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title('ROC曲线', fontsize=14)
    plt.xlabel('假阳性率 (FPR)', fontsize=12)
    plt.ylabel('真阳性率 (TPR)', fontsize=12)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(axis='both', alpha=0.3)
    plt.savefig(os.path.join(FIGURES_DIR, 'roc_curve.png'), dpi=150, bbox_inches='tight')
    print(f"ROC曲线已保存到 {FIGURES_DIR}/roc_curve.png")

def plot_prediction_examples(images, true_labels, predictions, filenames, num_samples=6):
    """展示预测示例"""
    fig, axes = plt.subplots(2, num_samples, figsize=(18, 8))
    
    correct_indices = np.where(true_labels == predictions)[0]
    wrong_indices = np.where(true_labels != predictions)[0]
    
    correct_samples = np.random.choice(correct_indices, min(num_samples, len(correct_indices)), replace=False)
    for i, idx in enumerate(correct_samples):
        axes[0, i].imshow(images[idx][:, :, 0], cmap='gray')
        label = 'NORMAL' if true_labels[idx] == 0 else 'PNEUMONIA'
        axes[0, i].set_title(f'正确预测\n真实:{label}', fontsize=10)
        axes[0, i].axis('off')
    
    if len(wrong_indices) > 0:
        wrong_samples = np.random.choice(wrong_indices, min(num_samples, len(wrong_indices)), replace=False)
        for i, idx in enumerate(wrong_samples):
            axes[1, i].imshow(images[idx][:, :, 0], cmap='gray')
            true_label = 'NORMAL' if true_labels[idx] == 0 else 'PNEUMONIA'
            pred_label = 'NORMAL' if predictions[idx] == 0 else 'PNEUMONIA'
            axes[1, i].set_title(f'错误预测\n真实:{true_label} → 预测:{pred_label}', fontsize=10)
            axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'prediction_examples.png'), dpi=150, bbox_inches='tight')
    print(f"预测示例已保存到 {FIGURES_DIR}/prediction_examples.png")

plot_training_curves(history)
plot_confusion_matrix(cm)
plot_roc_curve(val_labels_array, predictions_proba)
plot_prediction_examples(val_images_array, val_labels_array, predictions_binary, val_filenames)

# ----------------------
# 9. 输出评估结果到文件
# ----------------------
def save_evaluation_report(report, cm, roc_auc):
    """保存评估报告到文件"""
    report_text = f"""肺炎二分类模型评估报告
========================================

日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

一、评估指标
------------
总体准确率: {report['accuracy']:.4f}

二、分类报告详情
----------------
              精确率    召回率     F1分数    支持度
NORMAL     : {report['NORMAL']['precision']:.4f}  {report['NORMAL']['recall']:.4f}  {report['NORMAL']['f1-score']:.4f}  {report['NORMAL']['support']}
PNEUMONIA  : {report['PNEUMONIA']['precision']:.4f}  {report['PNEUMONIA']['recall']:.4f}  {report['PNEUMONIA']['f1-score']:.4f}  {report['PNEUMONIA']['support']}
加权平均   : {report['weighted avg']['precision']:.4f}  {report['weighted avg']['recall']:.4f}  {report['weighted avg']['f1-score']:.4f}  {report['weighted avg']['support']}

三、混淆矩阵
------------
          预测
      NORMAL  PNEUMONIA
真实  NORMAL   {cm[0,0]}      {cm[0,1]}
     PNEUMONIA {cm[1,0]}      {cm[1,1]}

四、其他指标
------------
ROC-AUC: {roc_auc:.4f}

========================================
"""
    with open('evaluation_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"\n评估报告已保存到 evaluation_report.txt")

fpr, tpr, _ = roc_curve(val_labels_array, predictions_proba)
roc_auc = auc(fpr, tpr)
save_evaluation_report(report, cm, roc_auc)

print("\n" + "="*50)
print("训练完成！")
print(f"模型已保存为 pneumonia_model.h5")
print(f"所有图表已保存到 {FIGURES_DIR}/ 目录")
print("="*50)
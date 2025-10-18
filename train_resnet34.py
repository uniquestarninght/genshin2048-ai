import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from torchvision.models import ResNet34_Weights
import logging
from datetime import datetime
from tqdm import tqdm  # 新增


def main():
    # 配置日志
    # torch.cuda.empty_cache()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # ----------------------------
    # 配置参数 - 添加类别过滤
    # ----------------------------
    DATA_DIR = "dataset"
    BATCH_SIZE = 64
    NUM_EPOCHS = 15
    LEARNING_RATE = 0.001
    VALIDATION_SPLIT = 0.2
    IMG_SIZE = 184
    NUM_WORKERS = 2


    # 学习率调度器配置
    SCHEDULER_TYPE = "step"  # 可选: "step", "cosine", "exponential", "plateau"

    # ----------------------------
    # 优化的数据预处理 - 添加随机平移和裁剪
    # ----------------------------
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE + 20, IMG_SIZE + 20)),  # 先放大一些用于随机裁剪
        transforms.RandomCrop((IMG_SIZE, IMG_SIZE), padding=4),  # 随机裁剪
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),  # 添加垂直翻转
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 增强颜色抖动
        transforms.RandomRotation(degrees=10),  # 随机旋转
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 随机平移
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),  # 中心裁剪
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 首先加载完整数据集
    full_dataset = datasets.ImageFolder(
        root=DATA_DIR,
        transform=train_transform
    )
    logger.info(f"发现的所有类别: {full_dataset.classes}")

    # 划分训练集和验证集
    val_size = int(VALIDATION_SPLIT * len(full_dataset))
    train_size = len(full_dataset) - val_size

    # 设置随机种子以确保可重现性
    torch.manual_seed(42)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # 创建验证集的transform
    val_dataset_with_transform = datasets.ImageFolder(
        root=DATA_DIR,
        transform=val_transform
    )

    # 创建验证集子集
    val_filtered_subset = torch.utils.data.Subset(val_dataset_with_transform, val_dataset.indices)

    # 🔧 优化数据加载器参数
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=False
    )

    val_loader = DataLoader(
        val_filtered_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=False
    )

    logger.info(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(val_filtered_subset)}")
    logger.info(f"实际类别数量: {len(full_dataset.classes)} - {full_dataset.classes}")

    # ----------------------------
    # 构建模型
    # ----------------------------
    # 在程序开始时就初始化CUDA
    print("开始探测CUDA...")
    start_time = time.time()
    cuda_available = torch.cuda.is_available()
    init_time = time.time() - start_time
    print(f"CUDA探测完成，耗时: {init_time:.2f}秒，可用: {cuda_available}")

    device = torch.device("cuda" if cuda_available else "cpu")
    logger.info(f"使用设备: {device}")

    weights = ResNet34_Weights.IMAGENET1K_V1
    model = models.resnet34(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, len(full_dataset.classes))  # 使用实际的类别数量
    model = model.to(device)

    # ----------------------------
    # 训练配置
    # ----------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 根据配置选择不同的学习率调度器
    if SCHEDULER_TYPE == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)
    elif SCHEDULER_TYPE == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
    elif SCHEDULER_TYPE == "exponential":
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    elif SCHEDULER_TYPE == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                         factor=0.5, patience=5)
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)  # 默认

    logger.info(f"使用学习率调度器: {SCHEDULER_TYPE}")

    # ----------------------------
    # 训练循环 - 添加进度条
    # ----------------------------
    best_val_acc = 0.0
    min_val_loss = float('inf')
    for epoch in range(NUM_EPOCHS):
        # ===== 训练 =====
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        # 🔧 添加进度条
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} [Train]", leave=False)
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total_train += labels.size(0)
            correct_train += predicted.eq(labels).sum().item()

            # 实时更新进度条
            train_acc = 100. * correct_train / total_train
            train_bar.set_postfix(Loss=loss.item(), Acc=f"{train_acc:.2f}%")

        train_acc = 100. * correct_train / total_train

        avg_loss = running_loss / len(train_loader)

        # ===== 验证 =====
        model.eval()
        correct_val = 0
        total_val = 0
        val_loss = 0.0

        val_bar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} [Val]", leave=False)
        with torch.no_grad():
            for inputs, labels in val_bar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = outputs.max(1)
                total_val += labels.size(0)
                correct_val += predicted.eq(labels).sum().item()

                val_acc_so_far = 100. * correct_val / total_val
                val_bar.set_postfix(Acc=f"{val_acc_so_far:.2f}%")

        val_acc = 100. * correct_val / total_val
        avg_val_loss = val_loss / len(val_loader)

        # 更新学习率调度器
        if SCHEDULER_TYPE == "plateau":
            scheduler.step(val_acc)  # 对于ReduceLROnPlateau，使用验证准确率作为指标
        else:
            scheduler.step()  # 对于其他调度器，直接step
        logger.info(f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
                    f"Loss: {avg_loss:.4f} "
                    f"Train Acc: {train_acc:.2f}% "
                    f"Val Loss: {avg_val_loss:.4f} "
                    f"Val Acc: {val_acc:.2f}% "
                    f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # 保存模型时包含类别信息
            save_dict = {
                'model_state_dict': model.state_dict(),
                'class_names': full_dataset.classes,
                'num_classes': len(full_dataset.classes)
            }
            torch.save(save_dict, f"models/resnet34_filtered_max_{max([int(c) for c in full_dataset.classes])}.pth")
            logger.info(f"✅ 模型已保存至 models/resnet34_filtered_max_{max([int(c) for c in full_dataset.classes])}.pth (Val Acc: {val_acc:.2f}%)")
        elif val_acc == best_val_acc:
            if avg_val_loss < min_val_loss:
                min_val_loss = avg_val_loss
                # 保存模型时包含类别信息
                save_dict = {
                    'model_state_dict': model.state_dict(),
                    'class_names': full_dataset.classes,
                    'num_classes': len(full_dataset.classes)
                }
                torch.save(save_dict, f"models/resnet34_filtered_max_{max([int(c) for c in full_dataset.classes])}.pth")
                logger.info(f"✅ 模型已保存至 models/resnet34_filtered_max_{max([int(c) for c in full_dataset.classes])}.pth (Val Acc: {val_acc:.2f}%)")


    logger.info(f"训练完成！最佳验证准确率: {best_val_acc:.2f}%")


if __name__ == '__main__':
    main()

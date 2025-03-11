import os
import pandas as pd
import dataset
from sklearn import model_selection
from torch.utils.data import DataLoader
import torch
import model
from losses import bi_tempered_logistic_loss
import numpy as np
from torch.nn import functional as F
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()

    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    
    return x, y, y[index], lam


def run():
    # 数据目录
    data_path = f'D:\DataSet\木薯叶'

    # 使用albumentations进行数据增强
    train_transform = A.Compose([
        A.RandomResizedCrop(height=224, width=224, scale=(0.8, 1.0)),
        A.Transpose(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5),
        A.OneOf([
            A.MotionBlur(blur_limit=3),
            A.MedianBlur(blur_limit=3),
            A.GaussianBlur(blur_limit=3),
        ], p=0.5),
        A.CoarseDropout(max_holes=8, max_height=8, max_width=8, p=0.5),
        A.Cutout(num_holes=8, max_h_size=8, max_w_size=8, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    test_transform = A.Compose([
        A.Resize(height=224, width=224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    df = pd.read_csv(os.path.join(data_path, 'train.csv'))

    # 设置测试集 训练集
    train_df, test_df = model_selection.train_test_split(
        df,
        test_size=0.1,
        random_state=42,
        shuffle=True,
        stratify=df.label.values)

    # transform数据
    train_data = dataset.dataset(train_df, data_path, transforms=train_transform)
    test_data = dataset.dataset(test_df, data_path, transforms=test_transform)

    # 加载数据
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=16,
        drop_last=True,
        num_workers=8,
        shuffle=True
    )

    test_loader = DataLoader(
        dataset=test_data,
        batch_size=16,
        drop_last=True,
        num_workers=8
    )

    # 使用集成模型
    ensemble_model = model.EnsembleModel()
    ensemble_model = ensemble_model.cuda()

    # 定义优化器
    optimizer = torch.optim.Adam(ensemble_model.parameters(), lr=2e-5)
    
    # 使用Cosine Annealing with Warm Restarts
    # T_0是第一次重启的周期长度，T_mult是每次重启后周期长度的倍增因子
    # 例如，T_0=10, T_mult=2，则重启点为第10、30、70...个epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=10,  # 第一次重启的周期长度
        T_mult=2,  # 每次重启后周期长度的倍增因子
        eta_min=1e-6  # 最小学习率
    )
    
    train_step = 0
    best_accuracy = 0

    for epoch in range(50):
        print(f'第{epoch + 1}/50轮开始训练')
        ensemble_model.train()
        epoch_loss = 0
        
        # 添加tqdm进度条
        train_pbar = tqdm(train_loader, desc=f'训练 Epoch {epoch+1}', leave=True)
        for data in train_pbar:
            image, label = data
            image = image.cuda()
            label = label.cuda()

            # 应用CutMix数据增强
            if np.random.random() > 0.5:
                image, label_a, label_b, lam = cutmix_data(image, label)
                output = ensemble_model(image)
                
                # 计算混合损失
                loss = lam * bi_tempered_logistic_loss(output, label_a, t1=0.8, t2=1.4) + \
                       (1 - lam) * bi_tempered_logistic_loss(output, label_b, t1=0.8, t2=1.4)
            else:
                output = ensemble_model(image)
                loss = bi_tempered_logistic_loss(output, label, t1=0.8, t2=1.4)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            train_step += 1
            
            # 更新进度条信息
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}', 
                'lr': f'{optimizer.param_groups[0]["lr"]:.7f}'
            })

        # 每个epoch结束后更新学习率
        scheduler.step()
        
        avg_loss = epoch_loss / len(train_loader)
        print(f'第{epoch + 1}轮平均训练损失：{avg_loss:.4f}，当前学习率：{optimizer.param_groups[0]["lr"]:.7f}')

        # 验证逻辑
        ensemble_model.eval()
        total_loss = 0
        total_accuracy = 0
        
        # 添加验证进度条
        val_pbar = tqdm(test_loader, desc=f'验证 Epoch {epoch+1}', leave=True)
        with torch.no_grad():
            for data in val_pbar:
                image, label = data
                image = image.cuda()
                label = label.cuda()

                output = ensemble_model(image)
                loss = bi_tempered_logistic_loss(output, label, t1=0.8, t2=1.4)
                total_loss += loss.item()

                accuracy = (output.argmax(1) == label).sum().item()
                total_accuracy += accuracy
                
                # 更新验证进度条信息
                val_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}'
                })

        epoch_accuracy = total_accuracy / len(test_loader.dataset)
        avg_val_loss = total_loss / len(test_loader)
        print(f'第{epoch + 1}轮，验证集上的正确率：{epoch_accuracy:.4f}')
        print(f'第{epoch + 1}轮，验证集上的平均损失：{avg_val_loss:.4f}')
        
        # 保存最佳模型
        if epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy
            torch.save(ensemble_model.state_dict(), 'best_model.pth')
            print(f'保存最佳模型，准确率：{best_accuracy:.4f}')

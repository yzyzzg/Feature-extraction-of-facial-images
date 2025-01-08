from __future__ import print_function
import os
from data import Dataset
import torch
from torch.utils import data
import torch.nn.functional as F
from models import *
import torchvision
from utils import Visualizer, view_model
import torch
import numpy as np
import random
import time
from config import Config
from torch.nn import DataParallel
from torch.optim.lr_scheduler import StepLR
from test import *
import matplotlib.pyplot as plt


def save_model(model, save_path, name, iter_cnt):
    save_name = os.path.join(save_path, name + '_' + str(iter_cnt) + '.pth')
    torch.save(model.state_dict(), save_name)
    return save_name


if __name__ == '__main__':
    # 初始化存储 loss 和 acc 的列表
    epoch_losses = []
    epoch_accuracies = []

    opt = Config()
    if opt.display:
        visualizer = Visualizer()
    device = torch.device("cuda")

    train_dataset = Dataset(opt.train_root, opt.train_list, phase='train', input_shape=opt.input_shape)
    trainloader = data.DataLoader(train_dataset,
                                  batch_size=opt.train_batch_size,
                                  shuffle=True,
                                  num_workers=opt.num_workers)

    identity_list = get_lfw_list(opt.lfw_test_list)
    img_paths = [os.path.join(opt.lfw_root, each) for each in identity_list]

    if opt.loss == 'focal_loss':
        criterion = FocalLoss(gamma=2)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    if opt.backbone == 'resnet18':
        model = resnet_face18(use_se=opt.use_se)
    elif opt.backbone == 'resnet34':
        model = resnet34()
    elif opt.backbone == 'resnet50':
        model = resnet50()

    if opt.metric == 'add_margin':
        metric_fc = AddMarginProduct(512, opt.num_classes, s=30, m=0.35)
    elif opt.metric == 'arc_margin':
        metric_fc = ArcMarginProduct(512, opt.num_classes, s=30, m=0.5, easy_margin=opt.easy_margin)
    elif opt.metric == 'sphere':
        metric_fc = SphereProduct(512, opt.num_classes, m=4)
    else:
        metric_fc = nn.Linear(512, opt.num_classes)

    model.to(device)
    model = DataParallel(model)
    metric_fc.to(device)
    metric_fc = DataParallel(metric_fc)

    if opt.optimizer == 'sgd':
        optimizer = torch.optim.SGD([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                    lr=opt.lr, weight_decay=opt.weight_decay)
    else:
        optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                     lr=opt.lr, weight_decay=opt.weight_decay)
    scheduler = StepLR(optimizer, step_size=opt.lr_step, gamma=0.1)

    for epoch in range(opt.max_epoch):
        scheduler.step()
        model.train()

        total_loss = 0.0
        total_acc = 0.0
        count = 0

        for ii, data in enumerate(trainloader):
            data_input, label = data
            data_input = data_input.to(device)
            label = label.to(device).long()
            feature = model(data_input)
            output = metric_fc(feature, label)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            output = output.data.cpu().numpy()
            output = np.argmax(output, axis=1)
            label = label.data.cpu().numpy()
            acc = np.mean((output == label).astype(int))

            total_loss += loss.item()
            total_acc += acc
            count += 1

        # 记录 epoch 的平均 loss 和 acc
        avg_loss = total_loss / count
        avg_acc = total_acc / count
        epoch_losses.append(avg_loss)
        epoch_accuracies.append(avg_acc)

        print(f"Epoch {epoch+1}/{opt.max_epoch}, Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}")

    # 绘制 loss 和 acc 曲线
    plt.figure(figsize=(12, 6))

    # Loss 曲线
    plt.subplot(1, 2, 1)
    plt.plot(range(1, opt.max_epoch + 1), epoch_losses, label='Loss', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()

    # Accuracy 曲线
    plt.subplot(1, 2, 2)
    plt.plot(range(1, opt.max_epoch + 1), epoch_accuracies, label='Accuracy', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()

    # 显示图像
    plt.tight_layout()
    plt.show()

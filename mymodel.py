import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN1(nn.Module):
    def __init__(self, num_classes=4):
        super(SimpleCNN1, self).__init__()
        # 第一个卷积层：输入通道数为3（RGB图像），输出通道数为16
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)  # 批归一化
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 池化层
        
        # 第二个卷积层：输入通道数为16，输出通道数为32
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        # 第三个卷积层：输入通道数为32，输出通道数为64
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        # 全连接层
        self.fc1 = nn.Linear(64 * 28 * 28, 128)  # 假设输入图像大小为64x64
        self.fc2 = nn.Linear(128, num_classes)  # 输出层，类别数为4

    def forward(self, x):
        #print('111:',x.shape)

        # 第一个卷积块
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # 第二个卷积块
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # 第三个卷积块
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        #print('222:',x.shape)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    



class SimpleCNN2(nn.Module):
    def __init__(self, num_classes=4):
        super(SimpleCNN1, self).__init__()
        # 第一个卷积层：输入通道数为3（RGB图像），输出通道数为16
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)  # 批归一化
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 池化层
        
        # 第二个卷积层：输入通道数为16，输出通道数为32
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        # 第三个卷积层：输入通道数为32，输出通道数为64
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        # 全连接层
        self.fc1 = nn.Linear(64 * 28 * 28, 128)  # 假设输入图像大小为64x64
        self.fc2 = nn.Linear(128, num_classes)  # 输出层，类别数为4

    def forward(self, x):
        #print('111:',x.shape)

        # 第一个卷积块
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # 第二个卷积块
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # 第三个卷积块
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        #print('222:',x.shape)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
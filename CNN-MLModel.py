import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from onnx_coreml import convert


class Net(nn.Module):
    """自建神经网络模型"""
    def __init__(self):
        super(Net, self).__init__()
        # 设置卷积层
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        # 设置池化层
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 设置线性变换函数
        self.lf1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
        self.lf2 = nn.Linear(in_features=120, out_features=84)
        self.lf3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        # 卷积+整流+池化
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        # 转化为1维张量
        x = x.view(-1, 16 * 5 * 5)
        # 线性全连接+整流
        x = F.relu(self.lf1(x))
        x = F.relu(self.lf2(x))
        # 输出
        outputs = self.lf3(x)
        return outputs

def loadData():
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            # transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
            transforms.Normalize((0.5), (0.5))
        ]
    )  # 转化为tensor类型以及将[0,1]的数值正则化为[-1,1]的数值

    train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             transform=transform, download=True)  # 训练集(10个类别)
    train_loader = DataLoader(dataset=train_set,
                              batch_size=4,  # 4个样本一组打包，即将样本容量N减小到N/4，每次4个一起处理
                              shuffle=True, num_workers=2)  # 训练集加载器
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            transform=transform, download=True)  # 测试集
    test_loader = DataLoader(dataset=test_set,
                             batch_size=4, shuffle=False, num_workers=2)  # 测试集加载器
    return train_loader , test_loader

def fit(net,train_loader,criterion,optimizer):
    for epoch in range(10):  # 迭代次数
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data # 获取输入数据，包括样本值和标签：data = [inputs, labels]
            optimizer.zero_grad() # 用零来初始化优化函数的参数

            outputs = net(inputs) # 输入样本，前向传播，获取输出
            loss = criterion(outputs, labels) # 根据期望labels来计算损失，即获得误差
            loss.backward() # 误差逆向传播
            optimizer.step() # 优化一步

            # 训练实时数据
            running_loss += loss.item()
            if i % 2000 == 1999:  # 每2000个样本数据打印一次平均损失
                print('epoch %d, %5d inputs\' average loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    print('Finished Training')

def predict(net,test_loader):
    # 全局准确率测试
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the whole network is: %d %%' % (
            100 * correct / total))

    # 单个类别准确率测试
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):  # 一个mini-batch
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10):
        print('Accuracy of %5s is: %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))

def toOnnx(net):
    # 转化为onnx格式
    dummy_input = torch.rand(1, 3, 32, 32)
    input_names = ['images']
    output_names = ['classLabelProbs']
    onnx_name = 'cifar10_net.onnx'
    torch.onnx.export(net,
                      dummy_input,
                      onnx_name,
                      verbose=True,
                      input_names=input_names,
                      output_names=output_names)

def toMLModel():
    input_names = ['images']
    onnx_name = 'cifar10_net.onnx'
    classes = ['plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck']  # 类别
    # 转化成mlmodel
    mlmodel = convert(model=onnx_name,
                      minimum_ios_deployment_target='13',
                      image_input_names=input_names,
                      mode='classifier',
                      predicted_feature_name='classLabel',
                      class_labels=classes)
    return mlmodel

if __name__ == "__main__":

    # 加载数据，获取训练集和测试集
    train_loader,test_loader = loadData()
    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck') # 类别

    save_path = './cifar10_net.pth' # 模型保存路径

    net = Net() # CNN模型
    net.load_state_dict(torch.load(save_path))  # 加载训练好的模型
    criterion = nn.CrossEntropyLoss() # 交叉熵损失函数，H(p,q)= -Σ(p(i)logq(i)+(1-p())log(1-q(i)))
                                      # 其中p是期望输出（概率分布），q是实际输出（概率分布）
                                      # 则H(p,q)衡量二者之间的信息熵距离，H越小，两个分布越接近
    optimizer = optim.SGD(params=net.parameters(), # 随机梯度下降优化方法，根据net的参数进行设置
                          lr=0.001, # 学习率
                          momentum=0.9) # 动量衰减率，控制最大衰减：v = momentum*v - lr*grad(v)
    # 训练
    fit(net=net,train_loader=train_loader,criterion=criterion,optimizer=optimizer)

    # 测试
    predict(net=net,test_loader=test_loader)

    # 保存网络模型
    torch.save(net.state_dict(), save_path)

    # 将模型转化成onnx格式
    toOnnx(net)

    # 将模型转化为mlmodel格式
    mlmodel = toMLModel()
    mlmodel.save('./cifar10_net.mlmodel')



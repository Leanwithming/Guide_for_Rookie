import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
"""code is for GPU training,because the model is too small, the GPU version is slower than CPU version. 
    delete all .to(device=0)if you want to use CPU version.
"""
transforms = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='.\data', train=True,download=True,transform=transforms)

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transforms)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transforms)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False)

classes = ['plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


import matplotlib.pyplot as plt
import numpy as np

# 展示图片的函数
def imshow(img):
    img = img / 2 + 0.5     # 非归一化
    npimg = img.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()





class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x









#训练模型
net = Net()

# 在 GPU 上训练注意需要将网络和数据放到 GPU 上
net.to(device=0)


dataiter = iter(trainloader)

#使用交叉熵误差
criterion = nn.CrossEntropyLoss()
#使用动量优化器
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

start = time.time()
for epoch in range(2):

    running_loss = 0.0
    for i, data in enumerate(dataiter, 0):
        inputs, lables = data
        # 将训练数据在GPU上
        inputs, lables = inputs.to(device=0), lables.to(device=0)

        #清空梯度
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, lables)
        loss.backward()
        optimizer.step()

        #打印训练信息
        running_loss += loss.item()
        if i % 2000 == 1999:
            # 每 2000 次迭代打印一次信息
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
print('Finished Training! Total cost time: ', time.time() - start)



#测试模型性能
 # 随机获取训练集图片
dataiter = iter(trainloader)
images, labels = dataiter.__next__()
#将测试数据在GPU上
images, lables = images.to(device=0), lables.to(device=0)

# 展示图片
imshow(torchvision.utils.make_grid(images))
# 打印图片类别标签
print('labels:',' '.join('%5s' % classes[labels[j]] for j in range(4)))

#将图片输入网络
outputs = net(images)

#预测结果
_, predicted = torch.max(outputs, 1)
print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

#测试testloader里的10000张图片
correct = 0
total = 0
with torch.no_grad():
    images, lables = data
    # 将测试数据在GPU上
    images, lables = images.to(device=0), lables.to(device=0)

    outputs = net(images)
    _, predicted = torch.max(outputs.data, 1)
    total += lables.size(0)
    correct += (predicted == lables).sum().item()
print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

#分类准确率
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        #一组data为一组为4张图片
        images, labels = data
        # 将测试数据在GPU上
        images, lables = images.to(device=0), lables.to(device=0)
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)


        print(predicted.device, lables.device)
        #检查预测和标签是否相同(squeeze()将转化为一维数组 eg.[False, False, False,  True](一组为4张图片）(使用gpu版本时有莫名bug，虽然预测和标签都在gpu上，但还是报错)
        c = (predicted == labels).squeeze()

        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))



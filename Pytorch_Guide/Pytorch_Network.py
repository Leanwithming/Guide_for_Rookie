import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        #kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        #全连接层 y= kx +b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # max-pooling 采用一个（2，2）的滑动窗口
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # 核(kernel)大小是方形的话，可仅定义一个数字，如 (2,2) 用 2 即可代表
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


    def num_flat_features(self, x):
        #除了 batch 维度外的所有维度
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()

print(net)

params = list(net.parameters())
print('Parameters_Number:', len(params))

#conv1.weight
print('The first parameter size:', params[0].size())

#随机定义一个输入变量(torch.randn[1, 1, 32, 32]表示batch_size=1， 1通道（灰度图像），图片尺寸：32x32
input = torch.randn(1, 1, 32, 32)

#torch.nn 只支持小批量(mini-batches)数据，也就是输入不能是单个样本，比如对于 nn.Conv2d 接收的输入是一个 4 维张量--nSamples * nChannels * Height * Width 。
#所以，如果你输入的是单个样本，需要采用 input.unsqueeze(0) 来扩充一个假的 batch 维度，即从 3 维变为 4 维。

out = net(input)


net.zero_grad()
#保留计算图（前面已经backward一次且内存被释放）使用retain_graph=True解决, 第一个变量是为了让梯度矩阵与此相同size变量进行点乘而得到一个张量
out.backward(torch.randn(1, 10), retain_graph=True)

# 定义伪标签
target = torch.randn((10))
# 调整大小，使得和 output 一样的 size
target = target.view(1, -1)
#采用均方误差
criterion = nn.MSELoss()

loss = criterion(out, target)
print(loss)

#the process: input -> conv2d -> relu -> maxpool2d -> conv2d -> relu -> maxpool2d
     # -> view -> linear -> relu -> linear -> relu -> linear
     # -> MSELoss
     # -> loss


# MSELoss
print(loss.grad_fn)
# Linear layer
print(loss.grad_fn.next_functions[0][0])
# Relu
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])



#针对偏置在反向传播前后的结果
# 清空所有参数的梯度缓存
net.zero_grad()
print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)
#保留计算图（前面已经backward一次且内存被释放）使用retain_graph=True解决
loss.backward(retain_graph=True)

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)


#设置学习率
learing_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learing_rate)



import torch.optim as optim
#创建一个SGD(随机梯度下降)优化器
optimzer = optim.SGD(net.parameters(), lr= 0.01)
#清空梯度缓存
optimzer.zero_grad()
out = net(input)
loss = criterion(out, target)
loss.backward(retain_graph=True)
#更新权重
optimzer.step()








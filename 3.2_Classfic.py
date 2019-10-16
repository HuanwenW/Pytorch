# -*- coding:utf8 -*-
"""
View more, visit my tutorial page: https://morvanzhou.github.io/tutorials/
2019.09.22（10-16更新）
运用神经网络做分类问题
"""
'''
总结：分类和回归所搭建的网络、loss、优化都是一样的，
'''

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


# make fake data 制造数据
n_data = torch.ones(100, 2) # 生成值全部为1的100*2的数组
x0 = torch.normal(2*n_data, 1)      # class0 x data (tensor), shape=(100, 2)
y0 = torch.zeros(100)               # class0 y data (tensor), shape=(100, 1)    标签为0
x1 = torch.normal(-2*n_data, 1)     # class1 x data (tensor), shape=(100, 2)
y1 = torch.ones(100)                # class1 y data (tensor), shape=(100, 1)    标签为1
# torch.cat是将两个张量（tensor）拼接在一起，cat是concatnate的意思，即拼接，联系在一起。
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)  # shape (200, 2) FloatTensor = 32-bit floating合并在一起做数据
y = torch.cat((y0, y1), ).type(torch.LongTensor)    # shape (200,) LongTensor = 64-bit integer合并在一起作为标签

# The code below is deprecated in Pytorch 0.4. Now, autograd directly supports tensors
# x, y = Variable(x), Variable(y)

# plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
# plt.show()

# 定义网络


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.out = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))      # activation function for hidden layer
        x = self.out(x)                 # 输出值，但是这个不是预测值，预测值还需要另外计算
        return x


net = Net(n_feature=2, n_hidden=10, n_output=2) # 输入包含2个特征（一个点的确定由2个坐标决定）几个类别就是几个output（0和1）
# 输出结果的含义解释：1在哪个位置，即被归为哪一类
'''
[0,1]表示分类结果为1
[1,0]表示分类结果为0
[0,0,1]表示分类结果为2，三分类问题
'''

print(net)  # net architecture

optimizer = torch.optim.SGD(net.parameters(), lr=0.02) # 传入net的所有参数和学习率（学习率越小学习的越慢）
loss_func = torch.nn.CrossEntropyLoss()  # the target label is NOT an one-hotted--计算的结果是概率，
# 若为三分类问题，结果为[0.1,0.3,0.6]时，认为是概率最大所在的第3个类
plt.ion()   # something about plotting开始画图

for t in range(100):
    out = net(x)                 # input x and predict based on x
    loss = loss_func(out, y)     # must be (1. nn output, 2. target), the target label is NOT one-hotted

    optimizer.zero_grad()   # clear gradients for next train
    loss.backward()         # backpropagation, compute gradients
    optimizer.step()        # apply gradients

    if t % 2 == 0:
        # plot and show learning process
        plt.cla()
        # 过了这一道softmax的激励函数后的最大概率才是预测值
        prediction = torch.max(out, 1)[1]
      # prediction = torch.max(F.softmax(out), 1)[1] # 定位概率最大值的位置
        pred_y = prediction.data.numpy()
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
        accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size) # 预测值中有多少和真实值一样
        plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color':  'red'})
        plt.pause(0.1)

plt.ioff()# 停止画图
plt.show()
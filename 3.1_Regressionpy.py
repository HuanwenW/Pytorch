# -*- coding:utf8 -*-

'''
运用神经网络，做回归
2019-09-21（10-15第一次复习更新）
'''
import torch
from torch.autograd import Variable
import torch.nn.functional as F # nn是神经网络模块 #
import matplotlib.pyplot as plt # matplotlib 画图模块 #

# torch.manual_seed(1)    # reproducible

# 生成一些假数据来模拟一元二次函数
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data (tensor), shape=(100, 1)从-1到1之间产生100个数
y = x.pow(2) + 0.2*torch.rand(x.size())                 # y^2+噪音  noisy y data (tensor), shape=(100, 1)

'''
# torch can only train on Variable, so convert them to Variable
# The code below is deprecated in Pytorch 0.4. Now, autograd directly supports tensors
# x, y = Variable(x), Variable(y)

# 打印出散点图
plt.scatter(x.data.numpy(), y.data.numpy())
plt.show()

'''

# 定义网络--Net


class Net(torch.nn.Module): # 继承torch的Module
    # 定义所有层的属性（__init__（））
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__() # 继承__init__功能
        # 上面两句的形式是pytorch的标配 ，只需改变参数

        # 定义每层用什么样的形式
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer线性输出
        self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer线性输出
    # 一层层搭建forward（）层与层的关系链接，建立关系的时候会用到激励函数。

    def forward(self, x): # Module中的forward功能
        # 正向传播输入值，神经网络分析出输出值
        x = F.relu(self.hidden(x))      # activation function for hidden layer
        x = self.predict(x)             # linear output
        return x


net = Net(n_feature=1, n_hidden=10, n_output=1) # define the network调用Net

print(net)  # net architecture
print('*'*60)

# 训练网络

# optimizer是训练工具
optimizer = torch.optim.SGD(net.parameters(), lr=0.2) # 传入net的所有参数和学习率
loss_func = torch.nn.MSELoss()  # 预测值和真实值的误差计算公式（均方差）

# 画图
plt.ion()   # something about plotting

for t in range(200):
    prediction = net(x)     # 喂给 net 训练数据x ，（实质是 调用forward（））
    loss = loss_func(prediction, y)  # 计算两者误差 must be (1. nn output, 2. target)两个位置不能反

    optimizer.zero_grad()   # 清空上一步的残余更新参数值
    loss.backward()         # 误差反向传播, 计算参数更新值
    optimizer.step()        # 将参数更新值施加到net的parameters上

    if t % 5 == 0:
        # plot and show learning process
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color':  'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()

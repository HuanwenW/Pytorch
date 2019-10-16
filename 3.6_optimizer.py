# -*- coding:utf8 -*-

'''
多种优化，不是说越先进的优化器，结果越佳，寻找适合自己的数据的才好！
2019-10-16
本代码学习到的优化器有：
SGD 、 Momentum 、 RMSprop 、 Adam
'''


import torch
import torch.utils.data as Data
import torch.nn.functional as F
import matplotlib.pyplot as plt

# 提前定义一些参数
LR = 0.01
BATCH_SIZE = 40
EPOCH = 12

# fake dataset 生成数据 ，构造 y = x^2 的二次函数
x = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim=1)
y = x.pow(2) + 0.1*torch.normal(torch.zeros(*x.size())) # 同样，增加噪音

# plot dataset
plt.scatter(x.numpy(), y.numpy())
plt.show()

# put dateset into torch dataset使用3.5中提到的 data loader
torch_dataset = Data.TensorDataset(x, y) # 先转换成torch能识别的Dataset
loader = Data.DataLoader(dataset=torch_dataset, batch_size=BATCH_SIZE,
                         shuffle=True, num_workers=2,) # 再把dataset放入DataLoader中


# default network定义默认的network
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(1, 20)   # hidden layer
        self.predict = torch.nn.Linear(20, 1)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))      # activation function for hidden layer
        x = self.predict(x)             # linear output
        return x

if __name__ == '__main__':

    # 为每个优化器创建一个网络
    net_SGD = Net()
    net_Momentum = Net()
    net_RMSprop = Net()
    net_Adam = Net()
    nets = [net_SGD, net_Momentum, net_RMSprop, net_Adam]

    # different optimizers
    opt_SGD = torch.optim.SGD(net_SGD.parameters(), lr=LR)
    opt_Momentum = torch.optim.SGD(net_Momentum.parameters(), lr=LR, momentum=0.8)
    opt_RMSprop = torch.optim.RMSprop(net_RMSprop.parameters(), lr=LR, alpha=0.9)
    opt_Adam = torch.optim.Adam(net_Adam.parameters(), lr=LR, betas=(0.9, 0.99))

    optimizers = [opt_SGD, opt_Momentum, opt_RMSprop, opt_Adam]

    loss_func = torch.nn.MSELoss() # 计算误差
    losses_his = [[], [], [], []]  # 记录不同神经网络的loss

    # 训练并画图
    for epoch in range(EPOCH):
        print('Epoch: ', epoch)
        for step, (b_x, b_y) in enumerate(loader):  # for each training step

            # 对于每个优化器，优化属于他的神经网络
            for net, opt, l_his in zip(nets, optimizers, losses_his):
                output = net(b_x)  # get output for every net
                loss = loss_func(output, b_y)  # compute loss for every net
                opt.zero_grad()  # clear gradients for next train
                loss.backward()  # backpropagation, compute gradients
                opt.step()  # apply gradients
                l_his.append(loss.data.numpy())  # loss recoder

    labels = ['SGD', 'Momentum', 'RMSprop', 'Adam']
    for i, l_his in enumerate(losses_his):
        plt.plot(l_his, label=labels[i])
    plt.legend(loc='best')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.ylim((0, 0.2))
    plt.show()




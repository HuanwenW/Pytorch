# -*- coding:utf8 -*-

'''
保存和提取（训练好的）神经网络
2019-10-16
2种保存（提取）网络方法：
第一种是保存（提取）整个网络，第二种是仅保存（提取）训练网络中的参数（优势：速度快，占内存少）
注意：所保存的网络若不指定地址就会默认放在和主程序同一个目录下
'''
"""
Dependencies:
torch: 0.4
"""
import torch
import matplotlib.pyplot as plt

# torch.manual_seed(1)    # reproducible

# fake data
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data (tensor), shape=(100, 1)
y = x.pow(2) + 0.2 * torch.rand(x.size())  # noisy y data (tensor), shape=(100, 1)


# The code below is deprecated in Pytorch 0.4. Now, autograd directly supports tensors
# x, y = Variable(x, requires_grad=False), Variable(y, requires_grad=False)


def save():
    # save net1用3.3所学习的快速构建网络方法定义net
    net1 = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)
    )
    optimizer = torch.optim.SGD(net1.parameters(), lr=0.5)
    loss_func = torch.nn.MSELoss()

    for t in range(100):
        prediction = net1(x)
        loss = loss_func(prediction, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # plot result
    plt.figure(1, figsize=(10, 3))
    plt.subplot(131)
    plt.title('Net1')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)

    # 2 种保存网络的方法
    torch.save(net1, 'net.pkl')  # save entire net保存整个网络，后面参数是为保存的网络 命名
    torch.save(net1.state_dict(), 'net_params.pkl')  # save only the parameters仅保存网络中的参数（速度快）


def restore_net():
    # restore entire net1 to net2 提取之前保存的整个神经网络
    net2 = torch.load('net.pkl')
    prediction = net2(x)

    # plot result
    plt.subplot(132)
    plt.title('Net2')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)


def restore_params():
    # restore only the parameters in net1 to net3仅提取神经网络中的所有参数，
    # 先建一个和net1一样的网络
    net3 = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)
    )

    # copy net1's parameters into net3 将提取的参数放入新建的（net3）神经网络中
    net3.load_state_dict(torch.load('net_params.pkl'))
    prediction = net3(x)

    # plot result
    plt.subplot(133)
    plt.title('Net3')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
    plt.show()


# 调用上面建立的几个功能，然后出图
# save net1
save()

# restore entire net (may slow)
restore_net()

# restore only the net parameters
restore_params()
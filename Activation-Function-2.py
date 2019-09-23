# -*- coding:utf8 -*-
import torch
from torch.autograd import Variable
import torch.nn.functional as F
# nn是神经网络模块 #
import matplotlib.pyplot as plt
# matplotlib 画图模块 #

# ----- fake data ------ #

x = torch.linspace( -5, 5, 200) # 在-5和5之间产生200个数
x = Variable(x)
x_np = x.data.numpy() # 转化为numpy

y_relu = F.relu(x).data.numpy()# 效果首选
y_sigmoid = torch.sigmoid(x).data.numpy()
y_tanh = torch.tanh(x).data.numpy()
y_softplus = F.softplus(x).data.numpy()

#y_softmax  = F.softmax(x).data.numpy() #概率函数，没法作图，但softmax是激活函数

# ------画图----- #

plt.figure(1,figsize=(8,6))
plt.subplot(221)
plt.plot(x_np, y_relu, c='red', label='relu')
plt.ylim(-1,5)
plt.legend(loc='best')


plt.subplot(222)
plt.plot(x_np, y_sigmoid, c='red', label='sigmoid')
plt.ylim(-0.2,1.2)
plt.legend(loc='best')


plt.subplot(223)
plt.plot(x_np, y_tanh, c='red', label='tanh')
plt.ylim(-1.2,1.2)
plt.legend(loc='best')

plt.subplot(224)
plt.plot(x_np, y_softplus, c='red', label='softplus')
plt.ylim(-0.2,6)
plt.legend(loc='best')

plt.show()
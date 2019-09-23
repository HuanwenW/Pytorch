
# -*- coding:utf8 -*-
import torch
import numpy as np


import torchvision
import  torch.nn as nn
import numpy as np
import torchvision.transforms as transforms

# --------- 遗留问题 -------- #

# torch 的 dot 运算报错

#--------- 遗留问题 --------#

# -------   Numpy VS torch ----------- #
'''
np_data = np.arange(6).reshape((2, 3)) # 先产生6个数，再变成 2行3列 的矩阵形式
torch_data = torch.from_numpy(np_data) # 将 np_data 转换为 torch 形式
tensor2array = torch_data.numpy()      # 将 torch_data 转换为 numpy 形式

print(
    '\n numpy',np_data,
    '\ntorch',torch_data,
    '\ntensor2array',tensor2array,
)

#    结果    #

 numpy [[0 1 2]
 [3 4 5]] 
torch tensor([[0, 1, 2],
        [3, 4, 5]]）
tensor2array [[0 1 2]
 [3 4 5]]

# torch_data的输出不一样，torch会声明自己的格式，用tensor()包围数组 #
'''
# ---------  abs 、 mean -------- #
'''
data = [-1, -2, 0, 1, 2]
tensor_data = torch.FloatTensor(data) # 32bit 将 data 转换为tensor的32位

print(
    '\n abs',
    '\n numpy:', np.abs(data),                # [1 2 0 1 2]
    '\n torch_data:', torch.abs(tensor_data),      # tensor([1., 2., 0., 1., 2.])

    '\nmean'
    '\n numpy:', np.mean(data),  # 0.0
    '\n torch_data:', torch.mean(tensor_data),# tensor(0.)
)
'''
# ------------  mm(dot) -----------  #

'''
data_matrix = [[1, 2], [3, 4]]
tensor_data_matrix = torch.FloatTensor(data_matrix) # 32bit 将 data 转换为tensor的32位
data_matrix = np.array(data_matrix)

#  这个方法有问题！ #
print(
    '\n numpy2:', data_matrix.dot(data_matrix),  # 需要将 data_matrix 转换为 np.array 形式才可以用 dot
    '\n tensor_data_matrix2:', tensor_data_matrix.dot(tensor_data_matrix),  # 全部数据展平计算 1*1+2*2+3*3+4*4=30
)

print(
    '\n matual',
    '\n numpy1:', np.matmul(data_matrix,data_matrix),
    '\n tensor_data_matrix1:', torch.mm(tensor_data_matrix,tensor_data_matrix),
)
'''
# ------ Linear regression -----  #
''' 

x = torch.tensor(1.,requires_grad=True)
w = torch.tensor(2.,requires_grad=True)
b = torch.tensor(3.,requires_grad=True)

y = w * x + b

y.backward()

print(x.grad)
print(w.grad)
print(b.grad)
'''

# -----------  variable  ---------- #

from torch.autograd import Variable
tensor = torch.FloatTensor([[1, 2],[3, 4]])
variable = Variable(tensor,requires_grad = True)

t_out = torch.mean(tensor*tensor)  # x^2 = 1*1+2*2+3*3+4*4=30
v_out = torch.mean(variable*variable)

print(t_out)
print(v_out)

v_out.backward()  #反向传递更新

# v_out = 1/4*sum(var*var)
# d(v_out)/d(var) = 1/4*2*variable =variable/2 # 数学概念，可以先放过
print(variable.grad)

print('\n variable:', variable)
print('\n variable.data:', variable.data)
print('\n variable.data.numpy:', variable.data.numpy())
# -*- coding:utf8 -*-

'''
batch用法
2019-10-16
'''
'''
why 分批？
A：所用的数据一次性全部放入训练网络中，机器会很慢，分批训练可以提升效率且不会丢失太多的准确率
注意：
当 所有数据 除以 所分批次BATCH_SIZE不为整数时，最后一次的batch数据仅反馈剩余数据（不足一个batch批次）
'''
import torch
import torch.utils.data as Data #

torch.manual_seed(1)    # reproducible

BATCH_SIZE = 5 # 设置批训练的数据个数
# BATCH_SIZE = 6 # 所设置批次不能被整除情况

x = torch.linspace(1, 10, 10)       # this is x data (torch tensor)从1到10的10个点
y = torch.linspace(10, 1, 10)       # this is y data (torch tensor)从10到1的10个点

# 先转换成torch能识别的Dataset
torch_dataset = Data.TensorDataset(x, y)

# 把dataset放入DataLoader中
loader = Data.DataLoader(
    dataset=torch_dataset,      # torch TensorDataset format
    batch_size=BATCH_SIZE,      # mini batch size
    shuffle=True,               # random shuffle for training 控制是否在训练时要打乱数据再进行抽样（打乱会更接近整体数据效果）
    num_workers=2,              # subprocesses for loading data 多线程来读取数据，提高效率
)


def show_batch():
    for epoch in range(3):   # train entire dataset 3 times 整体训练所有数据（不拆分的）3次
        for step, (batch_x, batch_y) in enumerate(loader):  # 每一次loader释放一小批数据来学习
            # train your data...

            # 打印一些数据，辅助查看
            print('Epoch: ', epoch, '| Step: ', step, '| batch x: ', batch_x.numpy(), '| batch y: ', batch_y.numpy())


if __name__ == '__main__':
    show_batch()
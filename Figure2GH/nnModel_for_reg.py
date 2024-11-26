import warnings

from torch.optim.lr_scheduler import StepLR

warnings.filterwarnings("ignore")  # 不显示warning 方便采集结果（warning就只有版本差异的问题）

import time

import torch
from sklearn.metrics import r2_score
from torch import nn

print(torch.cuda.is_available())

from load_data import load_for_mission1, data_process

# 超参数
num_epochs = int(2e5)
N_HIDDEN1 = 3000
N_HIDDEN2 = 2500
N_HIDDEN3 = 1500
N_HIDDEN4 = 1000
N_HIDDEN5 = 500
N_HIDDEN6 = 200
N_HIDDEN7 = 100
N_HIDDEN8 = 30

N_OUT = 1

lr = 1e-2


def get_batch(x, y, batch_size=40):
    """
    从给定的tensor x和y中随机采样出batch_size大小的数据

    Args:
        x (torch.Tensor): 输入数据的tensor
        y (torch.Tensor): 标签数据的tensor
        batch_size (int): 需要采样的batch大小

    Returns:
        torch.Tensor, torch.Tensor: 采样出的输入数据和标签数据
    """
    N = x.size(0)

    # 随机采样batch_size个index
    sample_idx = torch.randint(0, N, (batch_size,), dtype=torch.long)

    # 根据采样的index获取数据
    x_batch = x[sample_idx]
    y_batch = y[sample_idx]

    return x_batch, y_batch

class Net(torch.nn.Module):

    def __init__(self, n_features, n_hidden1, n_hidden2, n_hidden3, n_hidden4, n_hidden5, n_hidden6, n_hidden7,
                 n_hidden8, n_out):
        super().__init__()

        self.layer = nn.Sequential(  # 这里要写成layer 因为父类的属性名是写死的
            nn.Linear(n_features, n_hidden1),
            nn.BatchNorm1d(n_hidden1),
            nn.ReLU(True),
            nn.Linear(n_hidden1, n_hidden2),
            nn.ReLU(True),
            nn.Linear(n_hidden2, n_hidden3),
            nn.ReLU(True),
            nn.Linear(n_hidden3, n_hidden4),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(n_hidden4, n_hidden5),
            nn.ReLU(True),
            nn.Linear(n_hidden5, n_hidden6),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(n_hidden6, n_hidden7),
            nn.ReLU(True),
            nn.Linear(n_hidden7, n_hidden8),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(n_hidden8, n_out),
        )

    def forward(self, x):
        x = self.layer(x)
        return x

def __load_train(y_i: int):
    # 这个等待被外面修改

    from load_data import Load_for_blood_fat_meta
    l = Load_for_blood_fat_meta()


    x, y, x_cols = l.task2()

    X_train, X_test, y_train, y_test = data_process(x, y[:,y_i], True)


    X_train = torch.tensor(X_train, ).to(torch.float32)
    X_test = torch.tensor(X_test, ).to(torch.float32)
    y_test = torch.tensor(y_test).to(torch.float32)
    y_train = torch.tensor(y_train).to(torch.float32)



    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    X_train, X_test, y_test, y_train = X_train.to(device), X_test.to(device), y_test.to(device), y_train.to(device)

    x = X_train  # torch 默认训练数据类型是float32

    y = y_train
    # 自动选择GPU or CPU
    print('now analysis on ', y_i)
    model = None

    # if torch.cuda.is_available():
    #     model = Net(x.shape[1], N_HIDDEN, 1).cuda()
    #     print("\033[31;1m using gpu.. \033[0m")
    #
    # else:
    #     model = Net(x.shape[1], N_HIDDEN, 1)
    #     print("\033[31;1m using cpu.. \033[0m")




    model = Net(x.shape[1], N_HIDDEN1, N_HIDDEN2, N_HIDDEN3, N_HIDDEN4, N_HIDDEN5, N_HIDDEN6, N_HIDDEN7, N_HIDDEN8,
                n_out=1).to(device)
    if torch.cuda.is_available():

        print("\033[31;1m using gpu.. \033[0m")

    else:

        print("\033[31;1m using cpu.. \033[0m")

    # 3，选择损失函数和优化器
    mse = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    begin = time.time()

    schedu = StepLR(optimizer, step_size=20, gamma=0.9)  # 学习率动态衰减

    best_r2 = -10
    best_r2_train = -10

    for epoch in range(num_epochs):
        x,y = get_batch(x, y)

        inputs = x
        target = y

        # 向前传播
        out = model(inputs)
        loss = mse(out, target)
        # 向后传播
        optimizer.zero_grad()  # 注意每次迭代都需要清零
        loss.backward()
        optimizer.step()
        schedu.step()

        if (epoch + 1) % 200 == 0:
            # if epoch  == num_epochs-1: # 现在我们只显示最后一次结果
            r2_on_train = r2_score(y_train.cpu().detach(), model(X_train).cpu().detach())
            r2_on_test= r2_score(y_test.cpu().detach(), model(X_test).cpu().detach())

            if r2_on_test > best_r2:
                best_r2 = r2_on_test
            if r2_on_train > best_r2_train:
                best_r2_train = r2_on_train

            print(
                'Epoch[{}/{}], loss_on_train:{:.6f},loss_on_tes:{:.6f},r2_on_train:{:.6f},r2_on_test:{:.6f}'.
                    format(epoch + 1, num_epochs,
                           loss.item(),
                           mse(y_test, model(X_test)),
                           r2_on_train,
                           r2_on_test)
            )

    print('best r2:', best_r2)
    print('best r2 on train:', best_r2_train)
    # 将这个结果写入到results/nnModel_for_reg.txt 前面要加上y_i 且区分是训练还是测试的
    with open('results/nnModel_for_reg.txt', 'a') as f:
        f.write(f'{y_i} test {best_r2}\n')
        f.write(f'{y_i} train {best_r2_train}\n')





    end = time.time()

    # print('cost: ', end - begin, 's')

    # if torch.cuda.is_available():
    #     predict = model(x.cuda())  # 使用gpu预测
    #     predict = predict.data.cpu().numpy()  # 再转成numpy()形式
    # else:
    #     predict = model(x)
    #     predict = predict.data.numpy()


def load_train():
    for i in range(5):
        __load_train(i)


if __name__ == '__main__':
    load_train()

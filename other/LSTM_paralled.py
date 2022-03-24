import os
import sys
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from sklearn.metrics import mean_squared_error
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, random_split

import net_utility
import GasModel

warnings.filterwarnings("ignore")

###########################################################################
###########################################################################
# 全局变量

# 加载配置
with open('config.yaml', encoding='utf-8') as f:
    configs = yaml.load(f, Loader=yaml.SafeLoader)
    DATABASE = Path(configs['DATABASE'])  # 数据根目录
    MODEL_DIR = Path(configs['MODEL_DIR'])  # 模型目录
    IMG_DIR = Path(configs['IMG_DIR'])  # 图片目录

    INDUSTRY = configs['INDUSTRY']  # 行业类型
    CLUSTER = configs['CLUSTER']  # 行业分类号
    MODE = configs['MODE']  # 预测模式 [day, month, season, year]

    SEQ_LEN = configs['MODE_SET'][MODE]['SEQ_LEN']
    INPUT_SIZE = configs['MODE_SET'][MODE]['INPUT_SIZE']
    OUTPUT_SIZE = configs['MODE_SET'][MODE]['OUTPUT_SIZE']
    PRE_DAYS = configs['MODE_SET'][MODE]['PRE_DAYS']
    EPOCH = configs['MODE_SET'][MODE]['EPOCH']

    LOOK_BACK = INPUT_SIZE * SEQ_LEN
    BATCH_SIZE = configs['BATCH_SIZE']  # 批大小
    HIDDEN_SIZE = configs['HIDDEN_SIZE']  # 隐藏层
    LEARNING_RATE = configs['LEARNING_RATE']  # 学习率

    COLUMN = configs['COLUMN']  # 预测对象
    USE_PRE = configs['USE_PRE']  # 是否使用历史数据
    USE_ATTN = configs['USE_ATTN']  # 是否使用 attention

    DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")  # 指定GPU
    torch.manual_seed(0)  # 随机种子
    start_time = time.time()

    MODELNAME = f'{INDUSTRY}_{CLUSTER}_{MODE}_LSTM.pkl'
    IMGNAME = f'{INDUSTRY}_{CLUSTER}_{MODE}_LSTM.html'
    if USE_ATTN:
        MODELNAME = MODELNAME.split('.')[0] + '_USE_ATTN.pkl'
        IMGNAME = IMGNAME.split('.')[0] + '_USE_ATTN.html'

###########################################################################
###########################################################################
# DDP

# DDP：从外部得到参数
local_rank = int(os.environ['LOCAL_RANK'])
master_addr = str(os.environ['MASTER_ADDR'])
master_port = str(os.environ['MASTER_PORT'])
rank = int(os.environ['RANK'])
world_size = int(os.environ['WORLD_SIZE'])

# DDP：DDP backend初始化
if torch.cuda.is_available():
    torch.cuda.set_device(local_rank)
    DEVICE = torch.device("cuda", local_rank)
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
else:
    DEVICE = torch.device("cpu")
    dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)

print(f'local_rank: {local_rank}, rank: {rank}, world_size: {world_size}, ',
      f'master_addr: {master_addr}, master_port: {master_port}', f'DEVICE: {DEVICE}')

###########################################################################
###########################################################################
# 数据读取 & 预处理

# 读取同一个聚类的全部数据，合并为一个 DataFrame
# table_name='G_Data0000000177'
df_data = net_utility.get_class_data(industry=INDUSTRY, cluster=CLUSTER, type=MODE)
print('get class data, done!')

# 拼接成星期数据
if MODE == 'day':
    df_data = net_utility.get_weekly_data(df_data)

# 数据标准化
norm_data, norm_data_info, scalers = net_utility.normalized_data(df_data, COLUMN)
print('normalized data, done!')

# 生成数据集
gd_data = GasModel.GasDataset(norm_data, norm_data_info)

try:
    # 将数据集分为训练集和测试集
    n_train = int(len(gd_data) * 0.8)
    n_test = len(gd_data) - n_train
    ds_train, ds_test = random_split(gd_data, [n_train, n_test])

    train_sampler = torch.utils.data.distributed.DistributedSampler(ds_train)
    dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, sampler=train_sampler)
    dl_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    dl_data = DataLoader(gd_data, batch_size=1, shuffle=False, drop_last=False)
    print('dataset, done!')
except:
    print('{}_{} error: 记录数量不足'.format(INDUSTRY, CLUSTER))
    sys.exit(1)

###########################################################################
###########################################################################
# 模型训练


def train(model, dl_train):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_loss = []

    # 开始训练
    print(f'EPOCH = {EPOCH}')
    for e in range(EPOCH):
        # DDP：设置 sampler 的 epoch
        # DistributedSampler 需要这个来指定 shuffle 方式
        dl_train.sampler.set_epoch(e)
        __loss = 0
        for dl in dl_train:
            # 是否使用历史数据
            if PRE_DAYS != 0 and USE_PRE:
                feature, pre, date, label = dl
                feature, label = feature.to(DEVICE), label.to(DEVICE)
                pre, date = feature.to(DEVICE), label.to(DEVICE)

                pre = pre.view(-1, INPUT_SIZE * PRE_DAYS)
                label = label.view(-1, OUTPUT_SIZE)
                feature = feature.view(-1, SEQ_LEN, INPUT_SIZE)
                out = model(feature, pre, date)
            else:
                feature, label = dl
                feature, label = feature.to(DEVICE), label.to(DEVICE)
                label = label.view(-1, OUTPUT_SIZE)
                feature = feature.view(-1, SEQ_LEN, INPUT_SIZE)
                out = model(feature)
            loss = criterion(out, label)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            __loss += loss.item()
        train_loss.append(__loss)
        if (e + 1) % 10 == 0:  # 每 10 次输出结果
            print('Epoch: {}, Loss: {}'.format(e + 1, __loss / len(dl_train)))

    # 保存模型参数
    if not MODEL_DIR.exists():
        MODEL_DIR.mkdir()
    # 保存模型参数, 只需要在进程 0 上保存一次
    if dist.get_rank() == 0:
        torch.save(model.state_dict(), (MODEL_DIR / MODELNAME))
    return model, train_loss


model = GasModel.LSTM().to(DEVICE)
if (MODEL_DIR / MODELNAME).exists() and dist.get_rank() == 0:
    model.load_state_dict(torch.load(MODEL_DIR / MODELNAME))

# DDP: 构造 DDP model(Load 模型要在构造 DDP 模型之前)
if torch.cuda.is_available():
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
else:
    model = DDP(model)

if not (MODEL_DIR / MODELNAME).exists():
    print('start training...')
    model, train_loss = train(model, dl_train)
    print('training, done')

end_time = time.time()
print('time cost: ', end_time - start_time, 's')

###########################################################################
###########################################################################
# 模型测试

model = model.eval()
pred, actual = [], []
for i, dl in enumerate(dl_data):
    # 实际预测，OUTPUT_SIZE 不重复
    if i % OUTPUT_SIZE == 0:
        if PRE_DAYS != 0 and USE_PRE:
            feature, pre, date, label = dl
            feature, label = feature.to(DEVICE), label.to(DEVICE)
            pre, date = feature.to(DEVICE), label.to(DEVICE)

            pre = pre.view(-1, INPUT_SIZE * PRE_DAYS)
            feature = feature.view(-1, SEQ_LEN, INPUT_SIZE)
            out = model(feature, pre, date)
        else:
            feature, label = dl
            feature, label = feature.to(DEVICE), label.to(DEVICE)
            feature = feature.view(-1, SEQ_LEN, INPUT_SIZE)
            out = model(feature)
        pred += out.view(-1).data.cpu().tolist()
        actual += label.view(-1).data.cpu().tolist()

pred, actual = np.array(pred), np.array(actual)
pred[pred < 0] = 0
pred[pred > 1] = 1

###########################################################################
###########################################################################
# 误差 & 可视化


def inverse_transform_respectively(data):
    table_name = list(data['TableName'])[0]
    data[['Actual', 'Pred']] = \
        scalers[table_name].inverse_transform(data[['Actual', 'Pred']])
    return data


# 拼接结果矩阵
result = pd.DataFrame({'Actual': actual, 'Pred': pred})
result = pd.concat([result, norm_data_info[LOOK_BACK:]], axis=1)
result = result.groupby('TableName').apply(inverse_transform_respectively)

result.dropna(inplace=True)
result.reset_index(drop=True, inplace=True)

rmse = np.sqrt(mean_squared_error(result['Actual'], result['Pred']))
print(IMGNAME + "类整体根均方误差(RMSE): " + str(rmse))
# net_utility.plot_result(IMGNAME, result)


def deal_result(data):
    # 计算根均方误差(RMSE)
    table_name = data['TableName'].tolist()[0]
    rmse = np.sqrt(mean_squared_error(data['Actual'], data['Pred']))
    print(table_name + " 根均方误差(RMSE): " + str(rmse))
    # 画图
    net_utility.plot_result(table_name + IMGNAME, data)

    if MODE != 'day':
        # 按 OUTPUT_SIZE 聚合
        data_sum = data[:(len(data) // OUTPUT_SIZE) * OUTPUT_SIZE]
        data_sum = data_sum.groupby(data_sum.index // OUTPUT_SIZE).agg({
            'CreateDate': 'last',
            'TableName': 'last',
            'Actual': 'sum',
            'Pred': 'sum'
        })
        if len(data_sum) < 1:
            return
        rmse = np.sqrt(mean_squared_error(data_sum['Actual'], data_sum['Pred']))
        print(table_name + " 累积根均方误差(RMSE): " + str(rmse))
        # 画图
        net_utility.plot_result(table_name + '_sum_' + IMGNAME, data)


# 绘制类中每个表的数据
grouped = result.groupby('TableName').apply(deal_result)

###########################################################################
###########################################################################
# 运行

#! 单机
#todo(use cuda:0)       torchrun /home/ubuntu/dev/naturegas/cangnm_code/LSTM_paralled.py
#todo(use 4 gpu)        torchrun --nproc_per_node 4 /home/hby/cangnm_code/AE_paralled.py

#! 多机
#! (1) torchrun(每台机器上运行命令相同)  torch.__version__ >= v1.10
#   nnodes: 并行机器数量
#   nproc_per_node: 每台机器上的可用卡
#   rdzv_id: 标识 ID
#   rdzv_backend: 后端
#   rdzv_endpoint: 主节点

#! 指定网卡
#todo(机器 1)   GLOO_SOCKET_IFNAME=enp0s3 torchrun --nnodes=3 --nproc_per_node=1 --rdzv_id=0 --rdzv_backend=c10d --rdzv_endpoint=192.168.56.101:29400 /home/ubuntu/dev/naturegas/cangnm_code/LSTM_paralled.py
#todo(机器 2)   GLOO_SOCKET_IFNAME=enp0s3 torchrun --nnodes=3 --nproc_per_node=1 --rdzv_id=0 --rdzv_backend=c10d --rdzv_endpoint=192.168.56.101:29400 /home/ubuntu/dev/naturegas/cangnm_code/LSTM_paralled.py
#todo(机器 3)   GLOO_SOCKET_IFNAME=enp0s3 torchrun --nnodes=3 --nproc_per_node=1 --rdzv_id=0 --rdzv_backend=c10d --rdzv_endpoint=192.168.56.101:29400 /home/ubuntu/dev/naturegas/cangnm_code/LSTM_paralled.py

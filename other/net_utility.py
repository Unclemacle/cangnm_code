import json
import os
import random
import sys
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import more_itertools as mit
import numpy as np
import pandas as pd
import plotly
import plotly.graph_objs as go
import torch
import yaml
from chinese_calendar import is_workday
from plotly.subplots import make_subplots
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

################################################################
# 全局参数
with open('config.yaml', encoding='utf-8') as f:
    configs = yaml.load(f, Loader=yaml.SafeLoader)
    DATABASE = Path(configs['DATABASE'])  # 数据根目录
    MODEL_DIR = Path(configs['MODEL_DIR'])  # 模型目录
    IMG_DIR = Path(configs['IMG_DIR'])  # 图片目录

################################################################
# 异常处理类


class ErrorWindow():

    def __init__(self, e_s):
        self.e_s = e_s

        self.i_anom = np.array([])  # 窗口中异常点的下标
        self.E_seq = np.array([])  # 连续异常点的首尾坐标
        self.non_anom_max = float('-inf')  # 窗口中非异常点的最大值

        self.sd_lim_min = 3
        self.sd_lim = 12.0  # z 范围的最大值
        self.sd_threshold = self.sd_lim  # 最佳的 z

        self.mean_e_s = np.mean(self.e_s)
        self.sd_e_s = np.std(self.e_s)
        self.epsilon = self.mean_e_s + self.sd_lim * self.sd_e_s  # 阈值

        self.p = 0.05  # 连续误差序列最大值的波动率

    def find_epsilon(self):
        '''寻找最佳的 z 值'''
        e_s = self.e_s
        max_score = float('-inf')

        # 遍历寻找最佳 z
        for z in np.arange(self.sd_lim_min, self.sd_lim, 0.5):
            # 计算阈值
            epsilon = self.mean_e_s + (self.sd_e_s * z)
            # 除去异常点后的序列
            pruned_e_s = e_s[e_s < epsilon]

            # 大于阈值的点的下标
            i_anom = np.argwhere(e_s >= epsilon).reshape(-1)
            # # 设置缓冲区
            # buffer = np.arange(1, 2)
            # # 将每个异常点周围的值（缓冲区）添加到异常序列中
            # i_anom = np.concatenate(
            #     (i_anom, np.array([i + buffer for i in i_anom]).flatten(),
            #      np.array([i - buffer for i in i_anom]).flatten()))
            # i_anom = i_anom[(i_anom < len(e_s)) & (i_anom >= 0)]
            i_anom = np.sort(np.unique(i_anom))

            # 如果存在异常点
            if len(i_anom) > 0:
                # 得到连续异常点的下标
                groups = [list(group) for group in mit.consecutive_groups(i_anom)]
                # 得到连续异常点的首尾坐标
                E_seq = [(g[0], g[-1]) for g in groups if not g[0] == g[-1]]
                # groups: [[1, 2, 3, 4], [6, 7], [9, 10]]
                # E_seq: [(1, 4), (6, 7), (9, 10)]

                # 计算去除异常点前后均值，方差的变化
                mean_delta = (self.mean_e_s - np.mean(pruned_e_s)) / self.mean_e_s
                sd_delta = (self.sd_e_s - np.std(pruned_e_s)) / self.sd_e_s
                # 计算得分
                score = (mean_delta + sd_delta) / (len(E_seq)**2 + len(i_anom))

                if score >= max_score and len(E_seq) < 6 and len(i_anom) < (len(e_s) / 2):
                    max_score = score
                    self.sd_threshold = z
                    self.epsilon = self.mean_e_s + z * self.sd_e_s

        # i_anom = np.argwhere(self.e_s >= self.epsilon).reshape(-1)
        # i_anom = np.sort(np.unique(i_anom))
        # self.i_anom = i_anom

    def compare_to_epsilon(self):
        '''获取当前窗口小于阈值的最大值'''
        e_s = self.e_s
        epsilon = self.epsilon

        # 找到异常点的下标
        i_anom = np.argwhere(e_s >= epsilon).reshape(-1)
        if len(i_anom) == 0:
            return

        # buffer = np.arange(1, 2)
        # i_anom = np.concatenate((i_anom, np.array([i + buffer for i in i_anom]).flatten(),
        #                          np.array([i - buffer for i in i_anom]).flatten()))
        # i_anom = i_anom[(i_anom < len(e_s)) & (i_anom >= 0)]
        i_anom = np.sort(np.unique(i_anom))

        # 获取当前窗口小于阈值的最大值
        window_indices = np.setdiff1d(np.arange(0, len(e_s)), i_anom)
        non_anom_max = np.max(np.take(e_s, window_indices))

        groups = [list(group) for group in mit.consecutive_groups(i_anom)]
        E_seq = [(g[0], g[-1]) for g in groups if not g[0] == g[-1]]

        self.i_anom = i_anom
        self.E_seq = E_seq
        self.non_anom_max = non_anom_max

    def prune_anoms(self):
        E_seq = self.E_seq
        e_s = self.e_s
        non_anom_max = self.non_anom_max

        if len(E_seq) == 0:
            return

        # 得到每个连续异常序列中的最大值
        E_seq_max = np.array([max(e_s[e[0]:e[1] + 1]) for e in E_seq])
        E_seq_max_sorted = np.sort(E_seq_max)[::-1]
        # 每个连续异常序列中的最大值 + 非异常点的最大值
        E_seq_max_sorted = np.append(E_seq_max_sorted, [non_anom_max])

        i_to_remove = np.array([])
        for i in range(0, len(E_seq_max_sorted) - 1):
            # 在异常序列中最大误差之间的最小百分比下降
            if (E_seq_max_sorted[i] - E_seq_max_sorted[i+1]) \
                    / E_seq_max_sorted[i] < self.p:
                i_to_remove = np.append(
                    i_to_remove,
                    np.argwhere(E_seq_max == E_seq_max_sorted[i])).astype(int)
            else:
                i_to_remove = np.array([])
        i_to_remove.sort()

        if len(i_to_remove) > 0:
            E_seq = np.delete(E_seq, i_to_remove, axis=0)

        if len(E_seq) == 0:
            self.i_anom = np.array([])
            return

        indices_to_keep = np.concatenate(
            [range(e_seq[0], e_seq[1] + 1) for e_seq in E_seq])
        mask = np.isin(self.i_anom, indices_to_keep)
        self.i_anom = self.i_anom[mask]


################################################################
# 获取聚类的合并数据
def get_class_data(type, industry=None, cluster=None, table_name=None, freq='1H'):
    '''获取聚类行业数据

    Args:
    -------
    type: str
        数据类型 ['day', 'month', 'season', 'year']
    industry: str
        行业名称, 多表模式使用
    cluster: int
        行业聚类号, 多表模式使用
    table_name: str
        表名, 单表模式使用
    freq: str
        数据频率 ['2H', '1H', '30T', '20T', '15T'], 默认 1H

    Returns:
    -------
    class_data: pd.DataFrame
        合并数据
    '''
    if industry or cluster:
        # 获取分类文件
        cluster_info = json.load(open(DATABASE / 'cluster_info.json', 'r'))
        cluster_info = cluster_info[industry]
        # 获取同行业下的所有表具
        tables = pd.DataFrame(list(cluster_info.items()))
        tables.columns = ['TableName', 'Class']
        tables = tables[tables['Class'] == str(cluster)]['TableName'].to_list()
    if table_name:
        tables = [table_name]

    class_data = pd.DataFrame()
    for table in tables:
        data = pd.read_csv(DATABASE / 'data_resample_1H' / (table + '.csv'))
        data['CreateDate'] = pd.to_datetime(data['CreateDate'])
        data.sort_values(['CreateDate'], inplace=True)

        # data = data[data['Freq'] == freq]
        data.loc[data['DBTotal'] < 0, ['DBTotal']] = 0
        data.loc[data['DGTotal'] < 0, ['DGTotal']] = 0
        if type != 'day':
            # 聚合天数，由 type 指定
            agg_time = {'month': 24, 'season': 24 * 7, 'year': 24 * 30}
            data = data[:(len(data) // agg_time[type]) * agg_time[type]]
            data = data.groupby(data.index // agg_time[type]).agg({
                'CreateDate': 'last',
                'GTotal': 'last',
                'BTotal': 'last',
                'T': 'mean',
                'Pa': 'mean',
                'DGTotal': 'sum',
                'DBTotal': 'sum'
            })
        data['TableName'] = table
        data['IsWorkday'] = data['CreateDate'].apply(lambda x: int(is_workday(x)))
        data.reset_index(drop=True, inplace=True)
        class_data = pd.concat([class_data, data], axis=0, ignore_index=True)

    return class_data


################################################################
# 标准化数据
def normalized_data(data, column):
    '''标准化数据

    Args:
    -------
    data: pd.DataFrame
        待标准化数据
    column: str
        待标准化列名

    Returns:
    -------
    norm_data: tensor
        标准化后的数据
    norm_data_info: pd.DataFrame
        标准化后的数据信息
    scalers: Dict(MinMaxScaler)
        标准化器 
    '''

    def fit_transform_respectively(data):
        table_name = list(data['TableName'])[0]
        scaler = MinMaxScaler()
        data['Norm'] = scaler.fit_transform(data[column])
        scalers[table_name] = scaler
        return data

    # 标准化数据
    scalers = defaultdict(MinMaxScaler)
    norm_data = data.groupby('TableName').apply(fit_transform_respectively)
    norm_data = torch.tensor(norm_data['Norm'], dtype=torch.float32)
    norm_data_info = data[['CreateDate', 'TableName', 'IsWorkday']]

    return norm_data, norm_data_info, scalers


################################################################
# 连续 7 天的特征拼接
def get_weekly_data(data):
    '''将表具数据重新组合为「连续」的周数据

    Args:
    -------
    data: pd.DataFrame
        合并数据

    Returns:
    -------
    weekly_data: pd.DataFrame
        每周数据
    '''
    dates = data['CreateDate']
    l, r = 0, 0
    keep_index = []
    while r < len(data):
        # 找到星期一
        while r < len(data) and dates.iloc[r].weekday() != 0:
            r += 24
        # 记录滑动窗口左端点
        l = r
        # 目标星期，0 - 6 分别代表星期一到星期日
        target = 0
        while r < len(data) and dates.iloc[r].weekday() == target:
            r += 24
            target += 1
        # 如果存在连续的 7 天数据，否则继续向后寻找
        if target == 7:
            # 记录窗口中数据的位置
            keep_index += list(range(l, r))

    data_weekly = data.iloc[keep_index, :]
    data_weekly.reset_index(drop=True, inplace=True)

    return data_weekly


################################################################
# 异常检测
def detect_anomaly(actual, pred, window_size: int):
    '''异常检测

    Args:
    -------
        actual(array): 真实值
        pred(array): 预测值
        window_size(int): 窗口大小
    
    Returns:
    -------
        anomaly_list(List): 异常点序号
    '''
    e = abs(actual - pred)
    smoothing_window = max(int(window_size * 0.05), 1)
    # e_s = pd.DataFrame(e).ewm(span=smoothing_window).mean().values.flatten()
    e_s = e

    anomaly_list = np.array([])
    for i in range(len(e_s) // window_size):
        cur = np.array(e_s[i * window_size:(i + 1) * window_size])
        act = np.array(actual[i * window_size:(i + 1) * window_size])

        window = ErrorWindow(cur)
        window.sd_lim_min = 5
        window_act = ErrorWindow(act)
        window_act.sd_lim_min = 3
        window.find_epsilon()
        window_act.find_epsilon()
        window.compare_to_epsilon()
        # window_act.compare_to_epsilon()

        if len(window.i_anom) == 0:
            continue

        window.prune_anoms()
        # window_act.prune_anoms()

        if len(window.i_anom) == 0:
            continue

        window.i_anom = np.sort(np.unique(window.i_anom))
        window_act.i_anom = np.sort(np.unique(window_act.i_anom))

        con = np.sort(np.unique(np.append(window.i_anom, window_act.i_anom)))
        # print(con)
        # anomaly_list = np.append(anomaly_list,
        #                          window.i_anom + i * window_size).astype('int')
        anomaly_list = np.append(anomaly_list, con + i * window_size).astype('int')
        # anomaly_list = np.delete(anomaly_list, [i for i in anomaly_list if error[i] <= 0.01])
    return anomaly_list.tolist()


def generate_anomaly(norm_data, norm_data_info):
    '''生成异常点

    Args:
    -------
        norm_data(tensor): 标准化数据
        norm_data_info(pd.DataFrame): 数据信息
    
    Returns:
    -------
        anomaly_data(tensor): 异常点
    '''
    n = len(norm_data)
    norm_data_info['Manual'] = 0
    anomaly_data = norm_data.clone()
    for i in random.sample(range(n), int(n * 0.001)):
        norm_data_info.loc[i, 'Manual'] = 1

        # if anomaly_data[i] < 1e-5:
        #     anomaly_data[i] = 1
        # elif anomaly_data[i] < 1e-3:
        #     anomaly_data[i] = max(1, anomaly_data[i] * 1000)
        # elif anomaly_data[i] < 1e-2:
        #     anomaly_data[i] = max(1, anomaly_data[i] * 100)
        # elif anomaly_data[i] < 1e-1:
        #     anomaly_data[i] = max(1, anomaly_data[i] * 10)

        if random.random() > 0.5:
            anomaly_data[i] = 1  # max(1, anomaly_data[i] * 1000)
        else:
            anomaly_data[i] = 0  # min(0, anomaly_data[i] / 10)
    return anomaly_data, norm_data_info

    # n = len(norm_data)
    # norm_data_info['Manual'] = 0
    # anomaly_data = norm_data.clone()

    # operation = {
    #     0: lambda a, b, c, d: a + d * b,
    #     1: lambda a, b, c, d: a + d * c,
    #     2: lambda a, b, c, d: a - d * b,
    #     3: lambda a, b, c, d: a - d * c
    # }
    # for i in random.sample(range(n), int(n * 0.005)):

    #     norm_data_info.loc[i, 'Manual'] = 1
    #     random_seed = random.randint(0, 3)
    #     new = operation[random_seed](norm_data[i], norm_data[i - 1], norm_data[i + 1],
    #                                  random.randint(5, 10))

    #     if (new == norm_data[i]) | (new == 0):
    #         norm_data[i] = max(1, anomaly_data[i] * 10)
    #     else:
    #         norm_data[i] = new

    # return anomaly_data, norm_data_info


################################################################
# 绘图函数
def plot_result(imgname, data, anomaly=None):
    if not IMG_DIR.exists():
        IMG_DIR.mkdir()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[500, 200])
    actual = go.Scatter(name='actual',
                        x=data['CreateDate'],
                        y=data['Actual'],
                        mode='lines')
    prediction = go.Scatter(name='prediction',
                            x=data['CreateDate'],
                            y=data['Pred'],
                            marker=dict(color='orange'),
                            mode='lines')
    error = go.Scatter(name='error',
                       x=data['CreateDate'],
                       y=abs(data['Actual'] - data['Pred']),
                       mode='lines')
    fig.append_trace(actual, 1, 1)
    fig.append_trace(prediction, 1, 1)
    fig.append_trace(error, 2, 1)
    if anomaly is not None:
        anomaly = go.Scatter(name='anomaly',
                             x=anomaly['CreateDate'],
                             y=anomaly['Actual'],
                             marker=dict(color='red'),
                             mode='markers')
        fig.append_trace(anomaly, 1, 1)
    fig.update_layout(title_text=imgname)
    plotly.offline.init_notebook_mode()
    plotly.offline.iplot(fig)
    # plotly.offline.plot(fig, filename=str(IMG_DIR / imgname), auto_open=False)


if __name__ == '__main__':
    pass

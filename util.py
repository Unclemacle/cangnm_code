import os
import yaml
import sys
import logging
from collections import defaultdict

import plotly
import plotly.express as px
import pandas as pd
import numpy as np
import more_itertools as mit
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from chinese_calendar import is_workday


class Config():
    '''加载全局参数
    '''
    def __init__(self, path_to_config, mode) -> None:
        self.path_to_config = path_to_config
        self.mode = mode

        with open(path_to_config, encoding='utf-8') as f:
            self.configs = yaml.load(f, Loader=yaml.SafeLoader)

        for k, v in self.configs.items():
            if k == 'MODE_SET':
                for sk, sv in v[self.mode].items():
                    setattr(self, sk, sv)
            setattr(self, k, v)
        setattr(self, 'LOOK_BACK', self.SEQ_LEN * self.INPUT_SIZE)


def setup_logging():
    '''Configure logging object to track parameter settings, training, and evaluation.
    
    Args:
        config(obj): Global object specifying system runtime params.
    Returns:
    -------
    logger (obj): Logging object

    '''
    logger = logging.getLogger('gasforecast')
    logger.setLevel(logging.INFO)

    # create file handler
    file_handler = logging.FileHandler('gasforecast.log', mode='w')
    file_handler.setLevel(logging.INFO)
    # create consol handler
    consol_handler = logging.StreamHandler()
    consol_handler.setLevel(logging.INFO)

    # set format
    formatter = logging.Formatter(
        '[%(asctime)s %(filename)s line:%(lineno)d] %(levelname)s: %(message)s')
    file_handler.setFormatter(formatter)
    consol_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(consol_handler)
    return logger


def get_dateline_data(data_raw, last=None, date_start=None, date_end=None):
    '''获取指定时间范围内的数据

    Args:
    -------
    data_raw: pd.DataFrame
        原始数据
    last: int
        指定最后 last 天的数据
    date_start: str | datetime64
        指定数据的开始日期
    date_end: str | datetime64
        指定数据的结束日期

    Returns:
    -------
    data: pd.DataFrame
        指定时间范围内的数据
    '''
    if last and (date_start or date_end):
        raise ValueError(
            'last and date_start/date_end can not be specified at the same time')
    if last:
        if not isinstance(last, int):
            raise ValueError('last must be an integer')
    else:
        try:
            # 类型转换
            date_start = pd.Timestamp(date_start)
            date_end = pd.Timestamp(date_end)
        except:
            raise ValueError('date_start/date_end must be in datetime like format')

    data = data_raw.copy()
    data['CreateDate'] = pd.to_datetime(data['CreateDate'])
    data['Date'] = data['CreateDate'].dt.date

    # 获取指定时间范围内的数据
    if date_start and date_end:
        data = data[(data['Date'] >= date_start) & (data['Date'] <= date_end)]

    # 获取最新的 last 天的数据
    elif last:
        data = data[data['Date'] >= data['Date'].max() - pd.Timedelta(days=last)]
    data.drop(columns=['Date'], inplace=True)
    data.reset_index(drop=True, inplace=True)
    return data


def get_sample_freqs(data_raw):
    '''获取原始数据中每天的采样频率

    Args:
    -------
    data_raw: pd.DataFrame
        原始数据

    Returns:
    -------
    freqs: int
        数据采样频率
    '''
    data = data_raw.copy()
    data['CreateDate'] = pd.to_datetime(data['CreateDate'])
    data.drop_duplicates(inplace=True)
    cnts = data.groupby(data['CreateDate'].dt.date).count()['CreateDate']
    if cnts.mean() > 48 and cnts.median() > 48:
        freqs = 3600
    elif cnts.mean() > 24 and cnts.median() > 24:
        freqs = 3600
    elif cnts.mean() > 12 and cnts.median() > 12:
        freqs = 3600 * 2
    elif cnts.mean() > 6 and cnts.median() > 6:
        freqs = 3600 * 4
    elif cnts.mean() > 4 and cnts.median() > 4:
        freqs = 3600 * 6
    else:
        freqs = 3600 * 24
    return freqs


def get_resample_data(data_raw, target, freqs=3600):
    '''重采样数据

    Args:
    -------
    data_raw: pd.DataFrame
        待重采样数据
    target: str
        待重采样的目标列
    freqs: int
        重采样频率(单位: 秒), 默认为 3600s

    Returns:
    -------
    data_resample: pd.DataFrame
        重采样后的数据
    '''
    target = target if target != 'Flow' else 'BTotal'

    data = data_raw.copy()
    data = data[['CreateDate', target]]
    data['CreateDate'] = pd.to_datetime(data['CreateDate'])
    data.drop(data[data['CreateDate'].dt.year < 2014].index, inplace=True)
    data.drop(data[data['CreateDate'].dt.year > 2023].index, inplace=True)
    data.drop_duplicates(inplace=True)
    data.dropna(inplace=True)

    # 重采样时间，精确到小时
    data['CreateDate'] = pd.to_datetime(np.floor(data['CreateDate'].view('int64') \
        // 1e9 // freqs * freqs) * 1e9)

    # 按小时分组，取每个小时用量的平均值
    data_avg = data.groupby('CreateDate').mean()
    data_avg['Date'] = data_avg.index.date
    data_avg.insert(0, 'CreateDate', data_avg.index)

    # 统计每天记录数是否为 select_num 条
    select_num = 24 * 3600 // freqs
    select_date = data_avg.groupby('Date').count()
    select_date = pd.Series(select_date.loc[select_date[target] == select_num].index)

    # 取满足条件的日期
    data_resample = pd.merge(data_avg, select_date)
    data_resample.drop(columns=['Date'], inplace=True)
    data_resample.sort_values(by=['CreateDate'], inplace=True)

    # 计算差分流量
    re_target = 'D' + target
    data_resample[re_target] = data_resample[target].diff()
    data_resample.fillna(0, inplace=True)

    # 日期不连续问题处理
    date_delta = data_resample['CreateDate'].diff().fillna(pd.Timedelta(seconds=freqs))
    date_delta /= pd.Timedelta(seconds=freqs)
    data_resample['DBTotal'] /= date_delta

    # 小于 0 的值置为 0
    if target == 'BTotal':
        data_resample.loc[data_resample[re_target] < 0, re_target] = 0

    # 去除包含异常值的整天数据
    mu, std = data_resample[re_target].mean(), data_resample[re_target].std()
    drop = set(data_resample[data_resample[re_target] > mu + 5 * std].index \
        // select_num * select_num)
    drop = np.array([list(range(i, i + select_num)) for i in drop]).flatten()
    data_resample.drop(index=drop, inplace=True)
    data_resample.reset_index(drop=True, inplace=True)
    return data_resample


def get_distributed_data(data_resample):
    '''获得仪表数据分别在工作日和节假日的流量使用情况

    Args:
    -------
    data_resample: pd.DataFrame
        经过重采样之后的数据

    Returns:
    -------
    distributed: List
        工作日及节假日的数据分布
    '''
    data = data_resample.copy()
    data['CreateDate'] = pd.to_datetime(data['CreateDate'])

    # 按日期（行）和时间（列）展开
    data['Date'] = data['CreateDate'].dt.date
    data['Time'] = data['CreateDate'].dt.time
    data_pivot = pd.pivot_table(data, values='DBTotal', index='Date',
                                columns='Time').copy()
    data_pivot.dropna(inplace=True)

    # 过滤没有用气的日期
    data_pivot = data_pivot[data_pivot.apply(np.sum, axis=1) != 0]

    # 区分工作日和节假日
    data_pivot['IsWorkday'] = data_pivot.index.map(lambda x: is_workday(x))
    data_workday = data_pivot[data_pivot['IsWorkday'] == True]
    data_holiday = data_pivot[data_pivot['IsWorkday'] == False]

    if data_workday.empty and data_holiday.empty:
        return []

    data_workday = list(data_workday.mean())[:-1] if not data_workday.empty else [0] * 24
    data_holiday = list(data_holiday.mean())[:-1] if not data_holiday.empty else [0] * 24

    distributed = data_workday + data_holiday
    distributed = [d / max(distributed) for d in distributed]
    return distributed


def get_agg_data(data_resample, table_name, target, mode):
    '''
    Args:
    -------
    data_resample: pd.DataFrame | List[pd.DataFrame]
        重采样数据
    table_name: str | List[str]
        数据表名
    target: str
        目标列名
    mode: str
        预测模式 ['day', 'month', 'season', 'year']

    Returns:
    -------
    data_agg: pd.DataFrame
        聚合数据
    '''
    data = data_resample.copy()
    if not isinstance(data, list):
        data = [data]
    if not isinstance(table_name, list):
        table_name = [table_name]

    agg_dict = {'CreateDate': 'last'}
    if target == 'Flow':
        agg_dict['BTotal'] = 'last'
        agg_dict['DBTotal'] = 'sum'
    elif target == 'T':
        agg_dict['T'] = 'mean'
        agg_dict['DT'] = 'mean'
    elif target == 'Pa':
        agg_dict['Pa'] = 'mean'
        agg_dict['DPa'] = 'mean'

    data_agg = pd.DataFrame()
    for i, df in enumerate(data):
        if mode != 'day':
            # 聚合天数，由 mode 指定
            agg_time = {'month': 24, 'season': 24 * 7, 'year': 24 * 30}
            df = df[:(len(df) // agg_time[mode]) * agg_time[mode]]
            df = df.groupby(df.index // agg_time[mode]).agg(agg_dict)
        if not df.empty:
            df.reset_index(drop=True, inplace=True)
            df['TableName'] = table_name[i]
            data_agg = pd.concat([data_agg, df], axis=0, ignore_index=True)
    data_agg['IsWorkday'] = data_agg['CreateDate'].apply(lambda x: int(is_workday(x)))
    return data_agg


def get_normalized_data(data_agg, target):
    '''标准化数据

    Args:
    -------
    data_agg: pd.DataFrame
        待标准化数据
    target: str
        待标准化列名

    Returns:
    -------
    data_norm: pd.Series
        标准化后的数据
    data_norm_info: pd.DataFrame
        标准化后的数据信息
    scalers: dict(str: MinMaxScaler)
        标准化器 
    '''
    def fit_transform_respectively(data):
        table_name = list(data['TableName'])[0]
        scaler = MinMaxScaler()
        data[[re_target]] = scaler.fit_transform(data[[re_target]])
        scalers[table_name] = scaler
        return data

    target = target if target != 'Flow' else 'BTotal'
    re_target = target if target == 'Pa' else 'D' + target
    # 标准化数据
    data = data_agg.copy()
    scalers = defaultdict(MinMaxScaler)
    data_norm = data.groupby('TableName').apply(fit_transform_respectively)
    data_norm = data_norm[[re_target]]
    data_norm_info = data[['CreateDate', 'TableName', 'IsWorkday']]

    return data_norm, data_norm_info, scalers


def detect_anomaly(actual, pred, std_min=3, std_max=10):
    '''检测异常值

    Args:
    -------
    actual: np.array
        实际值
    pred: np.array
        预测值
    std_min: int = 3
        z 的最小值
    std_max: int = 10
        z 的最大值

    Returns:
    -------
    anomaly: np.array
        异常值下标
    '''
    if not isinstance(actual, np.ndarray):
        actual = np.array(actual)
    if not isinstance(pred, np.ndarray):
        pred = np.array(pred)
    e = abs(actual - pred)
    mean, std = np.mean(e), np.std(e)
    epsilon = mean + std * std_max

    max_score = float('-inf')
    # 遍历寻找最佳 z
    for z in np.arange(std_min, std_max, 0.5):
        # 计算阈值
        epsilon = mean + std * z
        # 除去异常点后的序列
        pruned_e = e[e < epsilon]
        # 大于阈值的点的下标
        i_anom = np.argwhere(e >= epsilon).reshape(-1)
        # 将每个异常点周围的值（缓冲区）添加到异常序列中
        i_anom = np.concatenate((i_anom, np.array([i + 5 for i in i_anom]).flatten(),
                                 np.array([i - 5 for i in i_anom]).flatten()))
        i_anom = i_anom[(i_anom < len(e)) & (i_anom >= 0)]
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
            mean_delta = (mean - np.mean(pruned_e)) / mean
            sd_delta = (std - np.std(pruned_e)) / std
            # 计算得分
            score = (mean_delta + sd_delta) / (len(E_seq)**2 + len(i_anom))

            if score >= max_score and len(i_anom) < (len(e) / 2):
                max_score = score
                epsilon = mean + z * std

    # 找到异常点的下标
    i_anom = np.argwhere(e >= epsilon).reshape(-1)
    return i_anom if len(i_anom) > 0 else np.array([])


def plot_predict_result(img_path, data, predict_mode, output_size, show):
    '''绘制结果

    Args:
    -------
    img_path: pathlib.PosixPath
        图片保存路径
    data: pd.DataFrame
        数据
    predict_mode: str
        预测模式 ['last', 'all']
    output_size: int
        预测结果的长度
    show: bool
        是否显示图片
    '''
    if not img_path.parent.exists():
        img_path.parent.mkdir()

    data = data.copy()
    if predict_mode == 'all':
        data['Error'] = abs(data['Actual'] - data['Pred'])

        sub1 = px.line(data, x='CreateDate', y=['Actual', 'Pred'])
        sub2 = px.line(data, x='CreateDate', y=['Error'])
        sub1['data'][0]['line']['color'] = "#636EFB"
        sub1['data'][1]['line']['color'] = "#FFA500"
        sub2['data'][0]['line']['color'] = "#00CC96"

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[500, 200])
        fig.add_trace(sub1['data'][0], row=1, col=1)
        fig.add_trace(sub1['data'][1], row=1, col=1)
        fig.add_trace(sub2['data'][0], row=2, col=1)

    elif predict_mode == 'last':
        fig = px.line(data, x='CreateDate', y=['Actual', 'Pred'])
        fig.add_vline(x=data.iloc[-output_size, 0],
                      line_width=3,
                      line_dash="dash",
                      line_color="green")
    if show:
        fig.show()
    if img_path.suffix == '.html':
        fig.write_html(img_path, auto_open=False)
    else:
        fig.write_image(img_path)


def plot_detect_result(img_path, data_anomaly, result, target, show):
    '''绘制结果

    Args:
    -------
    img_path: pathlib.PosixPath
        图片保存路径
    data_anomaly: pd.DataFrame
        带异常的原始数据
    result: pd.DataFrame
        差分预测数据
    target: str
        检测对象, 可选 ['Flow', 'Pa', 'T']
    show: bool
        是否显示图片
    '''
    if not img_path.parent.exists():
        img_path.parent.mkdir()

    data = data_anomaly.copy()
    result = result.copy()
    df_target = 'Anomaly_' + target
    if target == 'Flow':
        re_target = 'DBTotal'
        n = 3
    else:
        re_target = 'D' + target
        n = 2

    fig = make_subplots(rows=n, cols=1, shared_xaxes=True, row_heights=[300] * n)
    if target == 'Flow':
        target = 'GFlow'
        sub_total_anomaly = data[data['Anomaly_Total'] != 0]
        sub_total = px.line(data, x='CreateDate', y=['BTotal'])
        sub_total.add_scatter(x=sub_total_anomaly['CreateDate'],
                              y=sub_total_anomaly['BTotal'],
                              mode='markers',
                              name='BTotal Anomaly')
        sub_total['data'][0]['line']['color'] = "#3366CC"
        sub_total['data'][1]['marker']['color'] = "#DC3912"
        fig.add_trace(sub_total['data'][0], row=n - 2, col=1)
        fig.add_trace(sub_total['data'][1], row=n - 2, col=1)

    sub_anomaly = data[data[df_target] != 0]
    sub = px.line(data, x='CreateDate', y=[target])
    sub.add_scatter(x=sub_anomaly['CreateDate'],
                    y=sub_anomaly[target],
                    mode='markers',
                    name=f'{target} Anomaly')
    sub['data'][0]['line']['color'] = "#636EFA"
    sub['data'][1]['marker']['color'] = "#EF553B"
    fig.add_trace(sub['data'][0], row=n - 1, col=1)
    fig.add_trace(sub['data'][1], row=n - 1, col=1)

    result[re_target] = result['Actual']
    sub_diff_anomaly = result[result['Anomaly'] != 0]
    sub_diff = px.line(result, x='CreateDate', y=[re_target])
    sub_diff.add_scatter(x=sub_diff_anomaly['CreateDate'],
                         y=sub_diff_anomaly[re_target],
                         mode='markers',
                         name=f'{re_target} Anomaly')
    sub_diff['data'][0]['line']['color'] = "#1F77B4"
    sub_diff['data'][1]['marker']['color'] = "#FF7F0E"
    fig.add_trace(sub_diff['data'][0], row=n, col=1)
    fig.add_trace(sub_diff['data'][1], row=n, col=1)

    if show:
        fig.show()
    if img_path.suffix == '.html':
        fig.write_html(img_path, auto_open=False)
    else:
        fig.write_image(img_path)
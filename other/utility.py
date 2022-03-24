#coding=utf-8

import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly
import plotly.graph_objs as go
# import pymssql
import yaml
from chinese_calendar import is_workday
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score

################################################################
# 全局参数

with open('config.yaml', encoding='utf-8') as f:
    configs = yaml.load(f, Loader=yaml.SafeLoader)
    DATABASE = Path(configs['DATABASE'])  # 数据根目录
    MODEL_DIR = Path(configs['MODEL_DIR'])  # 模型目录
    HOST = configs['SQLSERVER']['HOST']
    PORT = configs['SQLSERVER']['PORT']
    USER = configs['SQLSERVER']['USER']
    PASSWORD = configs['SQLSERVER']['PASSWORD']


################################################################
# 连接数据库
def connect(database_name):
    '''连接数据库, 返回连接会话

    Args:
    -------
    database_name: str
        数据库名
    
    Returns:
    -------
    conn: Connection
        连接会话
    '''
    conn = pymssql.connect(host=HOST,
                           port=PORT,
                           user=USER,
                           password=PASSWORD,
                           database=database_name,
                           charset='GBK')
    if conn:
        print("连接数据库 " + database_name + " 成功!")
    else:
        print("连接数据库 " + database_name + " 失败!")
    return conn


################################################################
# 获取数据库所有表名
def get_all_table_name(conn):
    '''获取数据库所有表名, 返回表名列表

    Args: 
    -------
    conn: Connection
        连接会话

    Returns:
    -------
    table_names: list
        数据库表名列表
    '''
    sql = 'SELECT name FROM sys.tables'
    table_names = pd.read_sql(sql, conn)
    table_names = [row[0] for row in table_names.values.tolist()]
    return sorted(table_names)


################################################################
# 获取数据库所有表格数据信息
def export_table_info():
    '''获取数据库所有表格数据长度, 更新频率以及外部信息, 生成文件 table_info.csv 存入 DATABASE

    table_info 内容包括: 
        TableName: 表名
        Length: 表格长度
        Freq: 更新频率
        MeterID: 表具ID
        GPRSCode: 采集仪ID
        Addr: 采集仪对应的表具地址
        RecordPeroid: 记录周期
        PowerMode: 数据传输模式
        Position: 表具位置
        Spec: 表具型号
        UserName: 表具使用者
        MeterName: 表具名
        GPRSName: 采集仪名
        AreaName: 表具所属的地区
        OrgName: 表具所属的公司
        IndustryName3: 表具所属的行业孙类
        IndustryName2: 表具所属的行业子类
        IndustryName1: 表具所属的行业父类

    Returns:
    -------
    table_info: pd.DataFrame
        表格数据长度, 更新频率以及外部信息
    '''
    if (DATABASE / 'table_info.csv').exists():
        print('文件 table_info.csv 已存在')
        table_info = pd.read_csv(DATABASE / 'table_info.csv')
        return table_info

    conn1 = connect('CNMCSPV2020Data')  # 数据库 CNMCSPV2020Data 连接会话
    conn2 = connect('CNMCSPV2020')  # 数据库 CNMCSPV2020 连接会话

    # 获取数据库所有表名
    table_names = get_all_table_name(conn1)

    info = defaultdict(int)
    freq = []
    for i, t_name in enumerate(table_names, 1):
        # 仅匹配 'G_Data' 开头的表
        if not re.match(r'G_Data', t_name):
            continue

        # 获取表数据长度
        sql = 'SELECT CreateDate FROM ' + t_name
        data = pd.read_sql(sql, conn1)
        row, col = data.shape
        info[t_name] = row

        # 获取表更新频率
        if row <= 1:
            freq.append(row)
        else:
            cnt = row / len(data.groupby([data['CreateDate'].dt.date]))
            freq.append(cnt)

        if i % 200 == 0 or i == len(table_names):
            print('当前进度: ' + str(i) + '/' + str(len(table_names)))

    df_len = pd.DataFrame([info.keys(), info.values(), freq],
                          index=['TableName', 'Length', 'Freq']).T

    # 获取数据表的外部信息
    sql = '''SELECT M.TableName, M.MeterID, M.GPRSCode, M.Addr, M.RecordPeroid, G.PowerMode, 
                M.Position, M.Spec, U.UserName, M.Name MeterName, G.Name GPRSName, 
                A.Name AreaName, O.OrgName, I3.Name IndustryName3, I2.Name IndustryName2, I1.Name IndustryName1
        FROM G_Meter M
            LEFT JOIN(
                G_GPRS G
                LEFT JOIN(
                    G_Area A
                    LEFT JOIN WH_Org O ON A.OrgCode=O.OrgCode
                ) ON G.AreaCode=A.Code
            ) ON M.GPRSCode=G.Code
            LEFT JOIN(
                TB_UserMeter UM
                LEFT JOIN(
                    TB_User U
                    LEFT JOIN(
                        TB_Industry I3
                        JOIN(
                            TB_Industry I2
                            JOIN TB_Industry I1 ON I2.ParentID=I1.IndustryID
                        ) ON I3.ParentID=I2.IndustryID
                    ) ON U.IndustryID=I3.IndustryID
                ) ON UM.UserID=U.UserID
            ) ON M.MeterID=UM.MeterID
    '''
    df_info = pd.read_sql(sql, conn2)
    table_info = pd.merge(df_len, df_info, how='left', on='TableName')
    table_info.to_csv(DATABASE / 'table_info.csv', index=False)
    print('生成文件 table_info.csv')
    return table_info


################################################################
# 获取数据库所有表具的参数信息
def export_spec_info():
    '''获取数据库中所有表具的参数信息, 生成文件 spec_info.csv 存入 DATABASE

    spec_info 内容包括:
        TableName: 表名
        Spec: 表具型号
        CName: 表具中文名称
        Remarks: 表具型号
        MinValue: 可测量的瞬时流量下限
        MaxValue: 可测量的瞬时流量上限
        GFlowMin: 人为规定的最小工况流量
        GFlowMax: 人为规定的最大工况流量
        BFlowMin: 人为规定的最小标况流量
        BFlowMax: 人为规定的最大标况流量
        PaMin: 人为规定的最大压强
        PaMax: 人为规定的最小压强
        TMin: 人为规定的最小温度
        TMax: 人为规定的最大温度

    Args: 
    -------
    conn: Connection
        数据库 CNMCSPV2020 连接会话

    Returns:
    -------
    spec_info: pd.DataFrame
        表具参数信息
    '''
    if (DATABASE / 'spec_info.csv').exists():
        print('文件 spec_info.csv 已存在')
        spec_info = pd.read_csv(DATABASE / 'spec_info.csv')
        return spec_info

    conn = connect('CNMCSPV2020')  # 数据库 CNMCSPV2020 连接会话
    sql = '''SELECT M.TableName, M.Spec, S.CName, S.Remarks, S.MinValue, S.MaxValue, M.GFlowMin, M.GFlowMax, 
                M.BFlowMin, M.BFlowMax, M.PaMin, M.PaMax, M.TMin, M.TMax
            FROM G_Meter M, TB_Spec S
            WHERE M.Spec=S.Name
    '''
    spec_info = pd.read_sql(sql, conn)
    spec_info.sort_values(by=['TableName'], inplace=True)
    spec_info.to_csv(DATABASE / 'spec_info.csv', index=False)
    print('生成文件 spec_info.csv')
    return spec_info


################################################################
# 获取所有表具的参数信息
def export_all_spec_info(conn):
    '''获取所有表具的参数信息, 生成文件 all_spec_info.csv 存入 DATABASE

    spec_info 内容包括:
        Spec: 表具型号
        CName: 表具中文名称
        Remarks: 表具型号
        MinValue: 可测量的瞬时流量下限
        MaxValue: 可测量的瞬时流量上限

    Returns:
    -------
    all_spec_info: pd.DataFrame
        表具参数信息
    '''
    if (DATABASE / 'all_spec_info.csv').exists():
        print('文件 all_spec_info.csv 已存在')
        all_spec_info = pd.read_csv(DATABASE / 'all_spec_info.csv')
        return all_spec_info

    conn = connect('CNMCSPV2020')  # 数据库 CNMCSPV2020 连接会话
    sql = '''SELECT Name, CName, Remarks, MinValue, MaxValue FROM TB_Spec'''
    all_spec_info = pd.read_sql(sql, conn)
    all_spec_info.to_csv(DATABASE / 'all_spec_info.csv', index=False)
    print('生成文件 all_spec_info.csv')
    return all_spec_info


################################################################
# 获取数据库所有行业信息
def export_industry_info():
    '''获取数据库所有行业信息, 生成文件 industry_info.csv 存入 DATABASE
    
    Returns:
    -------
    industry_info: pd.DataFrame
        行业信息
    '''
    if (DATABASE / 'industry_info.csv').exists():
        print('文件 industry_info.csv 已存在')
        industry_info = pd.read_csv(DATABASE / 'industry_info.csv')
        return industry_info

    conn = connect('CNMCSPV2020')  # 数据库 CNMCSPV2020 连接会话
    sql = '''SELECT IndustryID, Name, Level, ParentID
            FROM TB_Industry
            ORDER BY IndustryID
    '''
    industry_info = pd.read_sql(sql, conn)
    industry_info.sort_values(by=['IndustryID'], inplace=True)
    industry_info.to_csv(DATABASE / 'industry_info.csv', index=False)
    print('生成文件 industry_info.csv')
    return industry_info


################################################################
# 获取数据库原始数据
def export_raw_data():
    '''获取数据库原始数据, 生成文件存入 DATABASE//data_expotred
    '''
    if not (DATABASE / 'data_expotred').exists():
        (DATABASE / 'data_expotred').mkdir()

    # 获取数据库所有表名
    conn = connect('CNMCSPV2020Data')  # 数据库 CNMCSPV2020Data 连接会话
    table_names = get_all_table_name(conn)

    for i, table in enumerate(table_names, 1):
        if i % 200 == 0 or i == len(table_names):
            print('当前进度: ' + str(i) + '/' + str(len(table_names)))

        # 仅匹配 'G_Data' 开头的表
        if not re.match(r'G_Data', table):
            continue
        if (DATABASE / 'data_expotred' / (table + '.csv')).exists():
            continue

        # 获取表数据
        sql = 'SELECT CreateDate, GTotal, BTotal, GFlow, BFlow, T, Pa, Err, Alarm FROM ' + table
        data = pd.read_sql(sql, conn)
        if not data.empty:
            data.sort_values(by=['CreateDate'], inplace=True)
            data.to_csv(DATABASE / 'data_expotred' / (table + '.csv'), index=False)


################################################################
# 重采样数据
def get_resample_data(data, freqs=3600):
    '''重采样数据

    Args:
    -------
    freqs: int
        重采样频率(单位: 秒), 默认为 3600s
    '''
    # data = data[['CreateDate', 'GTotal', 'BTotal', 'GFlow', 'BFlow', 'T', 'Pa']]
    data = data[['CreateDate', 'GTotal', 'BTotal']]
    data['CreateDate'] = pd.to_datetime(data['CreateDate'])
    data.drop(data[data['CreateDate'].dt.year < 2014].index, inplace=True)
    data.drop(data[data['CreateDate'].dt.year > 2022].index, inplace=True)
    data.drop_duplicates(inplace=True)
    data.dropna(inplace=True)

    # 重采样时间，精确到小时
    data['CreateDate'] = pd.to_datetime(np.floor(data['CreateDate'].view('int64') \
        // 1e9 // freqs * freqs) * 1e9)

    # 按小时分组，取每个小时用量的平均值
    data_avg = data.groupby('CreateDate').mean()
    data_avg['Date'] = data_avg.index.date
    data_avg.insert(0, 'CreateDate', data_avg.index)

    # 统计每天记录数是否为 24 条（保证 1 小时 1 条数据）
    select_num = 24 * 3600 // freqs
    select_date = data_avg.groupby('Date').count()
    select_date = pd.Series(select_date.loc[select_date['BTotal'] == select_num].index)

    data_resample = pd.merge(data_avg, select_date)
    data_resample.drop(columns=['Date'], inplace=True)
    data_resample.sort_values(by=['CreateDate'], inplace=True)

    # 计算差分流量
    data_resample['DGTotal'] = data_resample['GTotal'].diff()
    data_resample['DBTotal'] = data_resample['BTotal'].diff()
    data_resample.fillna(0, inplace=True)

    # 0点重置（日期不连续问题）
    n = len(data_resample)
    data_resample.loc[range(0, n, select_num), ['DGTotal', 'DBTotal']] = \
        data_resample.loc[range(1, n, select_num), ['DGTotal', 'DBTotal']].to_numpy()

    # 小于 0 的值置为 0
    data_resample['DBTotal'] = data_resample['DBTotal'].apply(lambda x: max(x, 0))
    data_resample['DGTotal'] = data_resample['DGTotal'].apply(lambda x: max(x, 0))

    # 去除包含异常值的整天数据
    mu, std = data_resample['DBTotal'].mean(), data_resample['DBTotal'].std()
    drop = set(data_resample[data_resample['DBTotal'] > mu + 5 * std].index // 24 * 24)
    drop = np.array([list(range(i, i + 24)) for i in drop]).flatten()
    data_resample.drop(drop, inplace=True)
    data_resample.reset_index(inplace=True)

    return data_resample


################################################################
# 导出重采样数据
def export_resample_data(freqs=3600):
    '''导出重采样数据

    Args:
    -------
    freqs: int
        重采样频率(单位: 秒), 默认为 3600s
    '''
    if not (DATABASE / 'data_exported').exists():
        print('data_exported 文件夹不存在，请先获取原始数据')
        return
    if not (DATABASE / 'data_resample_1H').exists():
        (DATABASE / 'data_resample_1H').mkdir()

    # 获取所有表名
    tables = os.listdir(DATABASE / 'data_exported')

    for i, table in enumerate(tables):
        if i % 100 == 0:
            print(f'{i}/{len(tables)}')

        if (DATABASE / 'data_resample_1H' / table).exists():
            continue

        data = pd.read_csv(DATABASE / 'data_exported' / table)
        data_resample = get_resample_data(data, freqs)

        if data_resample.empty:
            continue

        data_resample.to_csv(DATABASE / 'data_resample_1H' / table, index=False)
    print('数据重采样完成!')


################################################################
# 获得仪表每日平均数据分布
def get_table_distributed(data):
    '''
    获得仪表数据分别在工作日和节假日的流量使用情况

    Args:
    -------
    data: pd.DataFrame
        经过重采样之后的数据

    Returns:
    -------
    distributed: List
        工作日及节假日的数据分布
    '''
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


################################################################
# 获得行业聚类数据
def get_table_cluster(industry_name, min_cn=5, max_cn=30):
    '''获得行业聚类数据
    
    Args:
    -------
    industry_name: str
        行业名称
    min_cn: float
        最小聚类数
    max_cn: float
        最大聚类数
    
    Returns:
    -------
    cluster_pred: dict
        行业聚类数据
    '''
    # 获取该行业的所有表名
    table_info = pd.read_csv(DATABASE / 'table_info.csv')
    table_info = table_info.loc[(table_info['IndustryName3'] == industry_name)
                                & (table_info['Length'] > 0), 'TableName'].to_numpy()
    data_merge = pd.DataFrame()
    for table in table_info:
        if not (DATABASE / 'data_resample_1H' / (table + '.csv')).exists():
            continue
        data = pd.read_csv(DATABASE / 'data_resample_1H' / (table + '.csv'))

        # 获取每日平均数据分布
        distributed = get_table_distributed(data)
        distributed = pd.Series(distributed, dtype='float64')
        distributed.name = table
        # 拼接
        data_merge = pd.concat([data_merge, distributed], axis=1)
    data_merge = data_merge.T
    data_merge.dropna(inplace=True)
    min_cn = min(min_cn, len(data_merge)) if len(data_merge) > min_cn else 2
    max_cn = min(max_cn, len(data_merge))

    # 聚类数遍历，使轮廓系数
    cluster_num = defaultdict(int)
    for cn in np.arange(min_cn, max_cn + 1, 1):
        pred = KMeans(n_clusters=cn, random_state=0).fit_predict(data_merge)
        if 0 < np.max(pred) < len(data_merge) - 1:
            cluster_num[cn] = silhouette_score(data_merge, pred)

    # 轮廓系数确定最佳聚类数
    best_cn = max(cluster_num, key=cluster_num.get) if cluster_num else len(data_merge)
    cluster = KMeans(n_clusters=best_cn, random_state=0)
    pred = cluster.fit_predict(data_merge)
    joblib.dump(cluster, (MODEL_DIR / f'cluster_{industry_name}.model'))
    data_merge['Pred'] = pred
    cluster_pred = data_merge['Pred'].to_dict()
    print(f'{industry_name} cluster num: {best_cn}')
    return cluster_pred


################################################################
# 导出全行业聚类数据
def export_cluster_info():
    '''导出全行业聚类数据
    '''
    table_info = pd.read_csv(DATABASE / 'table_info.csv')
    industry3 = table_info['IndustryName3'].drop_duplicates().dropna().tolist()
    industry3.remove('其他')

    cluster_info = defaultdict(dict)

    for i, industry_name in enumerate(industry3):
        if i % 10 == 0:
            print(f'{i}/{len(industry3)}')
        cluster_dict = get_table_cluster(industry_name)
        for key, val in cluster_dict.items():
            cluster_dict[key] = str(val)
        cluster_info[industry_name] = cluster_dict

    json_str = json.dumps(cluster_info, indent=4)
    with open(DATABASE / 'cluster_info.json', 'w') as json_file:
        json_file.write(json_str)
    return cluster_info


if __name__ == "__main__":
    pass

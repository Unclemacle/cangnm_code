import os
import warnings
import pandas as pd
import time
import numpy as np
from pathlib import Path
import pyspark
from hdfs.client import Client
from pyspark import SparkContext
from pyspark.sql import functions as fn
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.functions import pandas_udf, PandasUDFType

warnings.filterwarnings("ignore")

spark = SparkSession.builder \
        .appName('resample') \
        .getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("ERROR")
spark.conf.set("spark.sql.execution.arrow.enabled", "true")
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

DATABASE = Path('/home/ubuntu/dev/naturegas/data/')
COLUMNS = ['CreateDate', 'GTotal', 'BTotal', 'GFlow', 'BFlow', 'T', 'Pa']
client = Client('http://192.168.56.101:50070/')


################################################################
# 导出重采样数据
def export_resample_data(freqs=3600):
    if not client.status('/data/data_exported', strict=False):
        print('data_exported 文件夹不存在，请先获取原始数据')
        return
    if not client.status('/data/data_resample_1H', strict=False):
        client.makedirs('/data/data_resample_1H')

    # 获取所有表名
    tables = client.list('/data/data_exported')

    for i, table in enumerate(tables):
        if i % 100 == 0:
            en = time.time()
            print(f'{i}/{len(tables)} time: ' + str(en - st))
            # print(f'{i}/{len(tables)}')

        # if os.path.exists(DATABASE / 'data_resample' / table):
        #     continue

        # table = 'G_Data0000000177.csv'
        data = spark.read.csv('hdfs://spark1:9000/data/data_exported/' + table,
                              header=True,
                              inferSchema=True)
        print('0:' + str(time.time() - st))
        if data.count() < 24:
            continue
        data_resample = get_segment_daily_data(data, freqs)
        if data_resample.count() == 0:
            continue

        # data_resample.write.csv('hdfs://spark1:9000/data/data_resample_1H/' + table)
        data_resample.toPandas().to_csv(DATABASE / 'data_resample_by_spark' / table,
                                        index=False)
    print('数据重采样完成!')


def get_segment_daily_data(data, resample_interval):
    # 差分
    def df_diff(data, column):
        tmp_column = 'tmp_' + column
        data = data.withColumn(tmp_column, fn.lag(data[column]).over(df_window))
        data = data.withColumn(
            "D" + column,
            fn.when(fn.isnull(data[column] - data[tmp_column]),
                    0).otherwise(data[column] - data[tmp_column]))
        data = data.drop(tmp_column)
        return data

    df_window = Window.partitionBy().orderBy("CreateDate")
    data = data.select(COLUMNS)
    data = data.withColumn('CreateDate', fn.to_timestamp('CreateDate'))
    data = data.filter(fn.year('CreateDate') >= 2014)
    data = data.filter(fn.year('CreateDate') <= 2022)
    data = data.drop_duplicates().dropna()

    data = df_diff(data, 'BTotal')
    mu, std = data.select(fn.mean('DBTotal'), fn.stddev('DBTotal')).first()
    data = data.filter(data['DBTotal'] < mu + 10 * std)
    data = data.drop('DBTotal')

    data = data.withColumn(
        'CreateDate',
        fn.to_timestamp(
            fn.floor(fn.unix_timestamp('CreateDate') / resample_interval) *
            resample_interval))

    data_avg = data.groupby('CreateDate').mean()
    data_avg = data_avg.withColumn('Date', fn.to_date('CreateDate'))

    select_date = data_avg.groupby('Date').count().filter(
        fn.col('count') == 24).select('Date')

    data_resample = data_avg.join(select_date, data_avg['Date'] == select_date['Date'])
    data_resample = data_resample.drop('Date')

    for i, old_name in enumerate(data_resample.columns):
        data_resample = data_resample.withColumnRenamed(old_name, COLUMNS[i])

    data_resample = df_diff(data_resample, 'BTotal')
    data_resample = df_diff(data_resample, 'GTotal')
    return data_resample


def get_resample_data(data, freqs=3600):
    '''重采样数据

    Args:
    -------
    freqs: int
        重采样频率(单位: 秒), 默认为 3600s
    '''
    data = data[['CreateDate', 'GTotal', 'BTotal', 'GFlow', 'BFlow', 'T', 'Pa']]
    data['CreateDate'] = pd.to_datetime(data['CreateDate'])
    data.drop(data[data['CreateDate'].dt.year < 2014].index, inplace=True)
    data.drop(data[data['CreateDate'].dt.year > 2022].index, inplace=True)
    data.drop_duplicates(inplace=True)
    data.dropna(inplace=True)

    # 重采样时间
    data['CreateDate'] = pd.to_datetime(np.floor(data['CreateDate'].astype('int64') \
        // 1e9 // freqs * freqs) * 1e9)

    data_avg = data.groupby('CreateDate').mean()
    data_avg['Date'] = data_avg.index.date
    data_avg.insert(0, 'CreateDate', data_avg.index)

    select_date = data_avg.groupby('Date').count()
    select_date = pd.Series(select_date.loc[select_date['T'] == \
        (24 * 3600 // freqs)].index)

    data_resample = pd.merge(data_avg, select_date)
    data_resample.drop(columns=['Date'], inplace=True)
    data_resample.sort_values(by=['CreateDate'], inplace=True)
    data_resample['DGTotal'] = data_resample['GTotal'].diff()
    data_resample['DBTotal'] = data_resample['BTotal'].diff()
    data_resample.fillna(0, inplace=True)

    data_resample['DBTotal'] = data_resample['DBTotal'].apply(lambda x: max(x, 0))
    data_resample['DGTotal'] = data_resample['DGTotal'].apply(lambda x: max(x, 0))

    mu, std = data_resample['DBTotal'].mean(), data_resample['DBTotal'].std()
    drop = set(data_resample[data_resample['DBTotal'] > mu + 10 * std].index // 24 * 24)
    drop = np.array([list(range(i, i + 24)) for i in drop]).flatten()
    data_resample.drop(drop, inplace=True)

    return data_resample


################################################################
# 导出重采样数据
def export_resample_data_by_pandas(freqs=3600):
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
            en = time.time()
            print(f'{i}/{len(tables)} time: ' + str(en - st))

        if (DATABASE / 'data_resample_1H' / table).exists():
            continue

        data = pd.read_csv(DATABASE / 'data_exported' / table)
        data_resample = get_resample_data(data, freqs)

        if data_resample.empty:
            continue

        data_resample.to_csv(DATABASE / 'data_resample_1H' / table, index=False)
    print('数据重采样完成!')


# spark-submit --master spark://192.168.56.101:7077 --executor-memory 6G /home/ubuntu/dev/naturegas/cangnm_code/resample_data.py
# python /home/ubuntu/dev/naturegas/cangnm_code/test.py

if __name__ == '__main__':
    st = time.time()
    export_resample_data()
    en = time.time()
    sc.stop()
    print('last: ' + str(en - st))
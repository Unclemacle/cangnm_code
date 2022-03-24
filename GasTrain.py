import os
import sys
from collections import defaultdict

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, silhouette_score
from torch import nn
from torch.utils.data import DataLoader, random_split

import GasModel
import util
from GasBase import GasBase

logger = util.setup_logging()


class GasTrain(GasBase):
    '''燃气训练类

    Args:
    -------
    config_dir: str
        配置文件路径
    mode: str
        预测时间单位 ['day', 'month', 'season', 'year']
    table_industry: str
        训练行业

    Example:
    -------
    >>> gf = GasTrain(config_dir='/home/hby/cangnm_code/config.yaml',
    ...               mode='day',
    ...               table_industry='玻璃')
    >>> gf.start()
    '''
    def __init__(self, config_dir: str, mode: str, table_industry: str) -> None:
        super().__init__(config_dir, mode)
        self.table_industry = table_industry
        self.target = 'BTotal'

    def start(self):
        '''模型训练主函数

        模型训练流程:
        ------
            1. 获取数据的行业信息
            2. 读取原始数据并重采样
            3. 获取工作日和节假日的流量分布
            4. 进行行业聚类
            5. 根据聚类对表进行划分
            6. 分类训练(n次)
                7. 预处理标准化
                8. 生成数据集
                9. 模型训练
        '''

        # 1. 获取数据的行业信息
        self.load_table_info()

        # 2. 读取原始数据并重采样
        self.load_raw_data_and_resample()

        # 3. 获取工作日和节假日的流量分布
        self.get_distributed_data()

        # 4. 进行行业聚类
        self.generate_cluster_model()

        # 5. 根据聚类对表进行划分
        self.divide_table_by_cluster()

        # 6. 分类训练
        for clu, table_name in self.table_divide.items():
            logger.info(f'Start training: {self.table_industry} {clu} cluster.')
            self.table_cluster = clu
            self.table_name = table_name
            self.data_resample = [self.table_match[i] for i in self.table_name]

            # 7. 预处理标准化
            self.get_normalized_data()

            # 8. 生成数据集
            self.generate_dataset()

            # 9. 模型训练
            self.train_model()

    def load_table_info(self):
        '''获取指定行业的所有表

        可根据具体的数据格式创建子类, 继承 `GasTrain` 进行函数重写

        Returns:
        -------
        table_name: list[str]
            表名列表
        '''
        table_info = pd.read_csv(self.DATABASE / 'table_info.csv')

        # 对象行业的所有表
        self.table_name = table_info.loc[table_info['IndustryName3'] == \
                self.table_industry, 'TableName'].tolist()

        logger.info(f'{sys._getframe().f_code.co_name} done!')

    def load_raw_data_and_resample(self):
        '''读取原始数据并重采样

        可根据具体的数据格式创建子类, 继承 `GasTrain` 进行函数重写

        Args:
        -------
        table_name: list[str]
            表名列表

        Returns:
        -------
        data_resample: list[pd.DataFrame]
            重采样后的数据
        '''
        self.data_resample = []
        for table in self.table_name:
            table_path = self.DATABASE / 'data_exported' / f'{table}.csv'

            if not table_path.exists():
                self.table_name.remove(table)
                logger.info(f'{table_path} is not exist.')
                continue

            # 加载原始数据
            data_raw = pd.read_csv(table_path)
            data_raw = data_raw[['CreateDate', 'GTotal', 'BTotal']]
            data_raw['CreateDate'] = pd.to_datetime(data_raw['CreateDate'])

            if len(data_raw) < 24:
                self.table_name.remove(table)
                logger.info(f'{table_path} does not have enough data.')
                continue

            # 数据重采样
            data_resample = util.get_resample_data(data_raw, self.target, self.freqs)
            self.data_resample.append(data_resample)

        # 剩余可用表名检查
        if not self.table_name:
            raise ValueError(
                f'{self.table_industry} does not have enough table or table does not have enough data.'
            )
        logger.info(f'{sys._getframe().f_code.co_name} done!')

    def get_distributed_data(self):
        '''获得仪表数据分别在工作日和节假日的流量使用情况

        获得每个仪表分别在工作日和节假日的流量使用情况, 即工作日和节假日每个小时平均用量, 以进行聚类分析

        Args:
        -------
        data_resample: list[pd.DataFrame]
            重采样后的数据
        table_name: list[str]
            表名列表

        Returns:
        -------
        data_distributed: pd.DataFrame
            工作日和节假日的流量使用情况
        '''
        distributed_merge = pd.DataFrame()
        for i, data in enumerate(self.data_resample):
            # 获取每日平均数据分布
            distributed = util.get_distributed_data(data)
            distributed = pd.Series(distributed, dtype='float64')
            distributed.name = self.table_name[i]
            # 拼接
            distributed_merge = pd.concat([distributed_merge, distributed], axis=1)

        # 转置，行为表名，列为工作日和节假日每个小时平均用量
        distributed_merge = distributed_merge.T
        distributed_merge.dropna(inplace=True)
        self.data_distributed = distributed_merge
        logger.info(f'{sys._getframe().f_code.co_name} done!')

    def generate_cluster_model(self):
        '''进行行业聚类

        根据 `data_distributed` 进行聚类, 通过轮廓系数确定最佳聚类数

        Args:
        -------
        data_distributed: pd.DataFrame
            工作日和节假日的流量使用情况

        Returns:
        -------
        table_cluster: dict(str: int)
            表名与行业类别的对应关系
        cluster_num: int
            聚类数
        '''
        data_distributed = self.data_distributed

        n = len(data_distributed)
        min_cluster_num, max_cluster_num = self.min_cluster_num, self.max_cluster_num
        min_cluster_num = min(min_cluster_num, n) if n > min_cluster_num else 2
        max_cluster_num = min(max_cluster_num, n)

        # 聚类数遍历，使轮廓系数
        cluster_num = defaultdict(int)
        for cn in np.arange(min_cluster_num, max_cluster_num + 1, 1):
            pred = KMeans(n_clusters=cn, random_state=0).fit_predict(data_distributed)
            if 0 < np.max(pred) < n - 1:
                cluster_num[cn] = silhouette_score(data_distributed, pred)

        # 轮廓系数确定最佳聚类数
        best_cn = max(cluster_num, key=cluster_num.get) if cluster_num else n
        cluster = KMeans(n_clusters=best_cn, random_state=0)
        pred = cluster.fit_predict(data_distributed)
        data_distributed['Pred'] = pred

        # 保存聚类模型
        joblib.dump(cluster, (self.MODEL_DIR / f'cluster_{self.table_industry}.model'))

        self.table_cluster = data_distributed['Pred'].to_dict()
        self.cluster_num = best_cn
        logger.info(f'{self.table_industry} cluster num: {best_cn}')
        logger.info(f'{sys._getframe().f_code.co_name} done!')

    def divide_table_by_cluster(self):
        '''根据聚类对表进行划分

        根据 `table_cluster` 划分表，并保存到 `self.table_cluster_dict`

        Args:
        -------
        table_cluster: dict(str: int)
            表名与行业类别的对应关系
        table_name: list(str)
            表名列表
        data_resample: list(pd.DataFrame)
            重采样后的数据

        Returns:
        -------
        table_divide: dict(int: str)
            行业类别与表名的对应关系
        table_match: dict(str: pd.DataFrame)
            表名与重采样后数据的对应关系
        '''
        table_divide = defaultdict(list)
        for key, val in self.table_cluster.items():
            table_divide[val].append(key)

        self.table_divide = table_divide
        self.table_match = dict(zip(self.table_name, self.data_resample))
        logger.info(f'{sys._getframe().f_code.co_name} done!')

    def get_normalized_data(self):
        '''数据标准化

        - 数据聚合: 根据 `mode` 将重采样后的数据分别聚合为小时数据(day)、天数据(month)、周数据(season)、月数据(year)
        - 数据归一化: 对每个表的数据分别进行归一化

        Args:
        -------
        table_name: list(str)
            表名列表
        data_resample: list(pd.DataFrame)
            重采样后的数据

        Returns:
        -------
        data_norm: torch.tensor
            标准化后的数据
        data_norm_info: pd.DataFrame
            标准化后的数据信息
        scalers: dict(str: MinMaxScaler)
            标准化器字典
        '''

        # 数据聚合
        self.data_agg = util.get_agg_data(self.data_resample, self.table_name,
                                          self.target, self.mode)

        # 数据标准化
        data_norm, data_norm_info, scalers = util.get_normalized_data(
            self.data_agg, self.target)
        self.data_norm = torch.tensor(data_norm.values, dtype=torch.float)
        self.data_norm_info = data_norm_info
        self.scalers = scalers
        logger.info(f'{sys._getframe().f_code.co_name} done!')

    def generate_dataset(self):
        '''生成数据集

        Args:
        -------
        data_norm: torch.tensor
            标准化后的数据
        data_norm_info: pd.DataFrame
            标准化后的数据信息

        Returns:
        -------
        dl_data: DataLoader
            数据集
        '''
        # 加载全局参数
        self.MODEL_NAME = f'{self.table_industry}_{self.table_cluster}_{self.mode}_LSTM.pkl'

        # 将数据集分为训练集和测试集
        gd_data = GasModel.GasDataset(self.data_norm, self.data_norm_info, self.config)

        if len(gd_data) == 0:
            logger.warning(
                f'{self.table_industry}_{self.table_cluster} has not enough data.')
            return

        # 将数据集分为训练集和测试集
        n_train = int(len(gd_data) * 0.8)
        n_test = len(gd_data) - n_train
        ds_train, ds_test = random_split(gd_data, [n_train, n_test])

        # 将数据集转换为 DataLoader
        self.dl_data = DataLoader(ds_train, batch_size=self.config.BATCH_SIZE)
        self.dl_test = DataLoader(ds_test, batch_size=1)

        if len(self.dl_data) == 0:
            logger.warning(
                f'{self.table_industry}_{self.table_cluster} has not enough data.')
            return
        logger.info(f'{sys._getframe().f_code.co_name} done!')

    def train_model(self):
        '''模型训练

        Args:
        -------
        dl_data: DataLoader
            数据集
        '''

        # 模型初始化
        model = GasModel.LSTM(self.config)
        criterion = nn.MSELoss()
        if self.use_gpu:
            model = model.cuda()
            model = torch.nn.DataParallel(model, device_ids=self.gpu_id)
            criterion = criterion.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.LEARNING_RATE)
        self.train_loss = []

        # 开始训练
        logger.info('Start training... Epochs: {}'.format(self.config.EPOCH))
        for e in range(self.config.EPOCH):
            loss = 0
            for dl in self.dl_data:
                # 是否使用历史数据
                if self.config.PRE_DAYS != 0 and self.config.USE_PRE:
                    feature, pre, date, label = dl
                    pre = pre.view(-1, self.config.INPUT_SIZE * self.config.PRE_DAYS)
                    label = label.view(-1, self.config.OUTPUT_SIZE)
                    feature = feature.view(-1, self.config.SEQ_LEN,
                                           self.config.INPUT_SIZE)
                    if self.use_gpu:
                        feature, label = feature.cuda(), label.cuda()
                        pre, date = pre.cuda(), date.cuda()
                    out = model(feature, pre, date)
                else:
                    feature, label = dl
                    label = label.view(-1, self.config.OUTPUT_SIZE)
                    feature = feature.view(-1, self.config.SEQ_LEN,
                                           self.config.INPUT_SIZE)
                    if self.use_gpu:
                        feature, label = feature.cuda(), label.cuda()
                    out = model(feature)
                ls = criterion(out, label)

                # 反向传播
                optimizer.zero_grad()
                ls.backward()
                optimizer.step()
                loss += ls.item()
            self.train_loss.append(loss)
            if e and e % 10 == 0:  # 每 10 次输出结果
                logger.info('Epoch: {}/{}, Loss: {:.4f}'.format(
                    e, self.config.EPOCH, loss / len(self.dl_data)))

        # 保存模型参数
        if not self.MODEL_DIR.exists():
            self.MODEL_DIR.mkdir()
        torch.save(model.state_dict(), (self.MODEL_DIR / self.MODEL_NAME))
        logger.info(f'{sys._getframe().f_code.co_name} done!')

    def plot_train_loss(self):
        '''绘制训练损失
        '''
        fig = px.line(x=np.arange(len(self.train_loss)), y=self.train_loss)
        fig.show()


if __name__ == "__main__":
    # 训练模式
    gf = GasTrain(config_dir='/home/hby/cangnm_code/config.yaml',
                  mode='month',
                  table_industry='玻璃')
    # 设置 gpu
    gf.set_gpu_config(use_gpu=False)
    gf.start()
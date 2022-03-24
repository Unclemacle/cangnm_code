import os
import sys

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

import GasModel
import util
from GasBase import GasBase

logger = util.setup_logging()


class GasForecast(GasBase):
    '''燃气流量预测类

    Args:
    -------
    config_dir: str
        配置文件路径
    mode: str
        预测时间单位 ['day', 'month', 'season', 'year']
    table_name: str
        预测对象

    Example:
    -------
    >>> gf = GasForecast(config_dir='/home/hby/cangnm_code/config.yaml',
    ...                  mode='day',
    ...                  table_name='G_Data0000006723')
    >>> gf.start()
    '''
    def __init__(self, config_dir: str, mode: str, table_name: str) -> None:
        super().__init__(config_dir, mode)
        self.table_name = table_name
        self.predict_mode = 'last'  # ['last', 'all']
        self.target = 'BTotal'

    def set_predict_mode(self, predict_mode='last'):
        '''设置预测模式

        Args:
        -------
        predict_mode: str = 'last'
            预测模式, 可选 ['last', 'all']. last 预测最新一天, all 预测全部历史数据
        '''
        if predict_mode not in ['last', 'all']:
            raise ValueError("predict mode must be 'last' or 'all'.")
        self.predict_mode = predict_mode

    def start(self):
        '''预测主函数

        预测流程:
        -------
            1. 获取数据的行业信息
            2. 读取原始数据并重采样
            3. 获取工作日和节假日的流量分布
            4. 读取聚类模型并判断行业类别
            5. 预处理标准化
            6. 生成数据集
            7. 模型预测
            8. 输出预测结果
        '''

        # 1. 获取数据的行业信息
        self.load_table_info()

        # 2. 读取原始数据并重采样
        self.load_raw_data_and_resample()

        # 3. 获取工作日和节假日的流量分布
        self.get_distributed_data()

        # 4. 读取聚类模型并判断行业类别
        self.get_table_cluster()

        # 5. 预处理标准化
        self.get_normalized_data()

        # 6. 生成数据集
        self.generate_dataset()

        # 7. 读取 LSTM 模型并预测
        self.predict()

        # 8. 输出预测结果
        self.generate_result()

    def load_table_info(self):
        '''获取数据的行业信息

        可根据具体的数据格式创建子类, 继承 `GasForecast` 进行函数重写

        Returns:
        -------
        table_industry: str
            表的行业信息
        '''
        table_info = pd.read_csv(self.DATABASE / 'table_info.csv')

        # 预测对象的行业信息
        self.table_industry = table_info.loc[table_info['TableName'] == \
            self.table_name, 'IndustryName3'].values[0]

        logger.info(f'{sys._getframe().f_code.co_name} done!')

    def load_raw_data_and_resample(self):
        '''读取原始数据并重采样

        可根据具体的数据格式创建子类, 继承 `GasForecast` 进行函数重写

        Args:
        -------
        table_name: str
            表名

        Returns:
        -------
        data_resample: pd.DataFrame
            重采样后的数据
        '''

        table_path = self.DATABASE / 'data_exported' / f'{self.table_name}.csv'

        if not table_path.exists():
            logger.info(f'{table_path} is not exist.')

        # 加载原始数据
        data_raw = pd.read_csv(table_path)
        data_raw = data_raw[['CreateDate', 'GTotal', 'BTotal']]
        data_raw['CreateDate'] = pd.to_datetime(data_raw['CreateDate'])

        if len(data_raw) < 24:
            logger.info(f'{table_path} does not have enough data.')

        # 数据重采样
        data_resample = util.get_resample_data(data_raw, self.target, self.freqs)
        self.data_resample = data_resample

        logger.info(f'{sys._getframe().f_code.co_name} done!')

    def get_distributed_data(self):
        '''获得仪表数据分别在工作日和节假日的流量使用情况

        获得每个仪表分别在工作日和节假日的流量使用情况, 即工作日和节假日每个小时平均用量, 以进行聚类分析

        Args:
        -------
        data_resample: pd.DataFrame
            重采样后的数据
        table_name: str
            表名

        Returns:
        -------
        data_distributed: pd.DataFrame
            工作日和节假日的流量使用情况
        '''
        # 获取每日平均数据分布
        distributed = util.get_distributed_data(self.data_resample)
        distributed = pd.DataFrame({self.table_name: distributed}, dtype='float64')

        # 转置，行为表名，列为工作日和节假日每个小时平均用量
        self.data_distributed = distributed.T
        logger.info(f'{sys._getframe().f_code.co_name} done!')

    def get_table_cluster(self):
        '''读取聚类模型并判断行业类别

        读取预先训练的聚类模型, 根据 `data_distributed` 获取行业类别

        Args:
        -------
        data_distributed: pd.DataFrame
            工作日和节假日的流量使用情况

        Returns:
        -------
        cluster_num: int
            聚类数
        '''

        model_path = self.MODEL_DIR / f'cluster_{self.table_industry}.model'
        if not model_path.exists():
            raise ValueError(f'{model_path} does not exist, please train first.')

        clu = joblib.load(model_path)
        self.table_cluster = clu.predict(self.data_distributed)[0]
        logger.info(f'{sys._getframe().f_code.co_name} done!')

    def get_normalized_data(self):
        '''数据标准化

        - 数据聚合: 根据 `mode` 将重采样后的数据分别聚合为小时数据(day)、天数据(month)、周数据(season)、月数据(year)
        - 数据归一化: 对每个表的数据分别进行归一化

        Args:
        -------
        table_name: str
            表名
        data_resample: pd.DataFrame
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
        self.data_agg = util.get_agg_data(self.data_resample, self.table_name, self.target, self.mode)

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
        self.IMG_NAME = f'{self.table_name}_{self.mode}_LSTM.{self.img_type}'

        # all 模式下，对该表的所有历史数据进行预测
        if self.predict_mode == 'all':
            gd_data = GasModel.GasDataset(self.data_norm, self.data_norm_info,
                                          self.config)

        # last 模式下，预测最近一天的数据
        elif self.predict_mode == 'last':
            # 取最后一个数据长度
            last = self.config.LOOK_BACK * max(1,
                                               self.config.PRE_DAYS * self.config.USE_PRE)
            gd_data = GasModel.GasDataset(self.data_norm[-last:],
                                          self.data_norm_info[-last:],
                                          config=self.config,
                                          last=True)
        self.dl_data = DataLoader(gd_data, batch_size=1)

        if len(self.dl_data) == 0:
            raise ValueError(f'{self.table_name} has not enough data.')
        logger.info(f'{sys._getframe().f_code.co_name} done!')

    def predict(self):
        '''数据预测

        Args:
        -------
        dl_data: DataLoader
            数据集
        '''

        # 模型初始化
        model = GasModel.LSTM(self.config)
        if self.use_gpu:
            model = model.cuda()
            model = torch.nn.DataParallel(model, device_ids=self.gpu_id)

        # 模型加载
        if not (self.MODEL_DIR / self.MODEL_NAME).exists():
            raise RuntimeError(f'{self.MODEL_NAME} is not exist, please train first.')
        model.load_state_dict(torch.load(self.MODEL_DIR / self.MODEL_NAME))
        model = model.eval()

        pred, actual = [], []
        for i, dl in enumerate(self.dl_data):
            # last 模式下, i == 0
            # all 模式下, 一次预测一天的数据
            if i % self.config.OUTPUT_SIZE == 0:
                if self.config.PRE_DAYS != 0 and self.config.USE_PRE:
                    feature, pre, date, label = dl
                    pre = pre.view(-1, self.config.INPUT_SIZE * self.config.PRE_DAYS)
                    feature = feature.view(-1, self.config.SEQ_LEN,
                                           self.config.INPUT_SIZE)
                    if self.use_gpu:
                        feature, label = feature.cuda(), label.cuda()
                        pre, date = pre.cuda(), date.cuda()
                    out = model(feature, pre, date)
                else:
                    feature, label = dl
                    feature = feature.view(-1, self.config.SEQ_LEN,
                                           self.config.INPUT_SIZE)
                    if self.use_gpu:
                        feature, label = feature.cuda(), label.cuda()
                    out = model(feature)
                pred += out.view(-1).data.cpu().tolist()
                actual += label.view(-1).data.cpu().tolist()
        pred, actual = np.array(pred), np.array(actual)
        pred[pred < 0] = 0
        pred[pred > 1] = 1
        self.pred = pred
        self.actual = actual
        logger.info(f'{sys._getframe().f_code.co_name} done!')

    def generate_result(self):
        '''生成结果

        根据 `predict_mode` 对预测结果进行处理, 反归一化

        Args:
        -------
        pred: np.array
            预测数据
        data_norm: torch.tensor
            标准化后的数据
        data_norm_info: pd.DataFrame
            标准化后的数据信息
        scalers: dict(str: MinMaxScaler)
            标准化器字典

        Returns:
        -------
        result: pd.DataFrame
            预测结果汇总
        '''
        table_name = self.table_name

        # last 模式下, len(pred) == OUTPUT_SIZE, len(actual) == LOOK_BACK
        # 使用 nan 填充 pred 的前半部分
        if self.predict_mode == 'last':
            boundary = -self.config.LOOK_BACK
            pred = pd.Series(np.append([np.nan] * self.config.LOOK_BACK, self.pred),
                             name='Pred')

        # all 模式下, len(pred) == len(actual), 直接进行拼接
        elif self.predict_mode == 'all':
            boundary = self.config.LOOK_BACK
            pred = pd.Series(self.pred, name='Pred')

        # 根据 predict_mode 进行截取
        actual = pd.Series(self.data_norm[boundary:].flatten(), name='Actual')
        data_norm_info = self.data_norm_info[boundary:].reset_index(drop=True)

        # 反归一化
        result = pd.concat([data_norm_info, actual, pred], axis=1)
        result[['Actual', 'Pred']] = \
            self.scalers[table_name].inverse_transform(result[['Actual', 'Pred']])

        # last 模式下, 根据相同的频率拓展日期
        if self.predict_mode == 'last':
            time_step = result.iloc[1, 0] - result.iloc[0, 0]
            for i in range(self.config.OUTPUT_SIZE):
                result.iloc[-self.config.OUTPUT_SIZE + i, 0] = \
                    result.iloc[-self.config.OUTPUT_SIZE + i - 1, 0] + time_step
            result['TableName'] = table_name

        self.result = result
        logger.info(f'{sys._getframe().f_code.co_name} done!')

    def plot_result(self, show=False):
        '''绘图

        Args:
        -------
        result: pd.DataFrame
            预测结果汇总
        show: bool = False
            是否显示图片(要求 jupyter notebook 环境)
        '''
        img_path = self.IMG_DIR / self.IMG_NAME
        util.plot_predict_result(img_path, self.result, self.predict_mode,
                                 self.config.OUTPUT_SIZE, show)


if __name__ == "__main__":
    # 预测模式
    gf = GasForecast(config_dir='/home/hby/cangnm_code/config.yaml',
                     mode='month',
                     table_name='G_Data0000007815')
    # 设置预测模式, last 预测最新一天, all 预测全部历史数据
    gf.set_predict_mode('last')
    # 设置 gpu
    gf.set_gpu_config(use_gpu=False)
    gf.start()
    gf.plot_result()

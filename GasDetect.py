import re
import sys

import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split, TensorDataset
from sklearn.ensemble import IsolationForest

import GasModel
import util
from GasBase import GasBase

logger = util.setup_logging()


class GasDetect(GasBase):
    '''燃气检测类

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
    >>> gf = GasDetect(config_dir='/home/hby/cangnm_code/config.yaml',
    ...                mode='day',
    ...                table_industry='玻璃')
    >>> gf.set_gpu_config(use_gpu=True, gpu_id=[0, 1])
    >>> gf.start_train()
    '''
    def __init__(self, config_dir: str, mode: str, table_name: str) -> None:
        super().__init__(config_dir, mode)

        if not isinstance(table_name, str):
            raise ValueError('table_name must be str.')

        self.table_name = table_name
        self.detect_obj = ['Flow', 'Pa', 'T']
        self.target = 'Flow'
        self.detect_mode = 'last-7'
        self.detect_day = 7

    def set_detect_object(self, detect_obj):
        '''指定检测对象

        Args:
        -------
        detect_obj: str | list
            检测对象, 可选 ['Flow', 'Pa', 'T']
        '''
        if not isinstance(detect_obj, (str, list)):
            raise ValueError('detect_obj must be str or list[str].')
        if isinstance(detect_obj, str):
            detect_obj = [detect_obj]

        detect_obj = [obj.lower() for obj in detect_obj]
        for obj in detect_obj:
            if obj not in ['flow', 'pa', 't']:
                raise ValueError('detect_obj must be flow, pa or t.')
        detect_obj.sort()
        # 首字母大写
        self.target = detect_obj[0].title()
        self.detect_obj = detect_obj

    def set_detect_mode(self, detect_mode='last-7'):
        '''设置检测模式

        Args:
        -------
        detect_mode: str = 'last-7'
            检测模式, 可选 ['last-n', 'all']. last-n 检测最新 n 天, all 检测全部历史数据
        '''
        if detect_mode != 'all' and not re.match('last-\d+', detect_mode):
            raise ValueError("predict mode must be 'last-n' or 'all'.")
        self.detect_mode = detect_mode
        if detect_mode != 'all':
            self.detect_day = int(detect_mode.split('-')[-1])

    def start(self):
        '''检测主函数

        检测流程:
        -------
            1. 获取数据的行业信息
            2. 读取原始数据
            3. 流量检测
            4. 气压检测
            5. 温度检测
        '''

        # 1. 获取数据的行业信息
        self.load_table_info()

        # 2. 读取原始数据
        self.load_raw_data()

        # 3. 流量检测
        if 'flow' in self.detect_obj:
            self.detect_flow()

        # 4. 气压检测
        if 'pa' in self.detect_obj:
            self.detect_pa()

        # 5. 温度检测
        if 't' in self.detect_obj:
            self.detect_t()

    def detect_flow(self):
        '''流量检测

        流量检测流程:
        -------
            1. 检测累积流量
            2. 检测瞬时流量
            3. 检测差分流量
        '''
        self.target = 'Flow'

        # 1. 检测累积流量
        self.detect_decrease_total()

        # 2. 检测瞬时流量
        self.detect_by_threshold()

        # 2. 通过指数加权/移动平均检测气压
        self.detect_by_ewm()

        # 3. 检测差分流量
        self.detect_by_diff()

    def detect_pa(self):
        '''气压检测

        气压检测流程:
        -------
            1. 通过阈值检测气压
            2. 通过指数加权/移动平均检测气压
            3. 检测差分气压
        '''
        self.target = 'Pa'

        # 1. 通过阈值检测气压
        self.detect_by_threshold()

        # 2. 通过指数加权/移动平均检测气压
        self.detect_by_ewm()

        # 3. 检测差分气压
        self.detect_by_diff()

    def detect_t(self):
        '''温度检测

        温度检测流程:
        -------
            1. 通过阈值检测温度
            2. 通过指数加权/移动平均检测温度
            3. 检测差分温度
        '''
        self.target = 'T'

        # 1. 通过阈值检测温度
        self.detect_by_threshold()

        # 2. 通过指数加权/移动平均检测温度
        self.detect_by_ewm()

        # 3. 检测差分温度
        self.detect_by_diff()

    def detect_by_diff(self):
        '''检测差分流量

        检测差分流量流程:
        -------
            1. 数据重采样
            2. 预处理标准化
            3. 生成数据集
            4. 模型训练
            5. 检测差分流量
            6. 检测结果生成
        '''
        # 1. 数据重采样
        self.get_resample_data()

        # 2. 预处理标准化
        self.get_normalized_data()

        # 3. 生成数据集
        self.generate_dataset()

        # 4. 模型训练
        self.train_model()

        # 5. 检测差分流量
        self.detect()

        # 6. 检测结果生成
        self.generate_result()

        self.plot_result(show=True)

    def load_table_info(self):
        '''获取数据的行业信息

        可根据具体的数据格式创建子类, 继承 `GasDetect` 进行函数重写

        Returns:
        -------
        table_industry: str
            表的行业信息
        '''
        table_info = pd.read_csv(self.DATABASE / 'table_info.csv')
        try:
            self.table_industry = table_info.loc[table_info['TableName'] == \
                self.table_name, 'IndustryName3'].values[0]
        except:
            raise ValueError('TableName not in table_info.csv')

        # 加载仪表参数信息
        spec_info = pd.read_csv(self.DATABASE / 'spec_info.csv')
        row = spec_info.loc[spec_info['TableName'] == self.table_name, :]
        self.min_value = row['MinValue'].values[0]
        self.max_value = row['MaxValue'].values[0]
        self.gflow_min = row['GFlowMin'].values[0]
        self.gflow_max = row['GFlowMax'].values[0]
        self.bflow_min = row['BFlowMin'].values[0]
        self.bflow_max = row['BFlowMax'].values[0]
        self.pa_min = row['PaMin'].values[0]
        self.pa_max = row['PaMax'].values[0]
        self.t_min = row['TMin'].values[0]
        self.t_max = row['TMax'].values[0]

    def load_raw_data(self):
        '''读取原始数据

        可根据具体的数据格式创建子类, 继承 `GasDetect` 进行函数重写

        Args:
        -------
        table_name: str
            表名

        Returns:
        -------
        data_anomaly: pd.DataFrame
            带异常的原始数据
        '''
        table_path = self.DATABASE / 'data_exported' / f'{self.table_name}.csv'

        if not table_path.exists():
            raise FileNotFoundError(f'{table_path} not found.')

        data_raw = pd.read_csv(table_path)
        data_raw = data_raw[[
            'CreateDate', 'GTotal', 'BTotal', 'GFlow', 'BFlow', 'Pa', 'T'
        ]]
        data_raw['CreateDate'] = pd.to_datetime(data_raw['CreateDate'])

        # 获取指定时间范围内的数据
        if self.detect_mode == 'all':
            data_detect = data_raw
        else:
            if self.detect_day > len(set(data_raw['CreateDate'])):
                raise ValueError(f'{self.detect_day} is too large.')
            data_detect = util.get_dateline_data(data_raw, last=self.detect_day)

        if len(data_raw) < 24:
            raise ValueError(f'{table_path} does not have enough data.')

        self.data_raw = data_raw
        self.data_anomaly = data_detect.copy()
        self.data_anomaly['Anomaly_Flow'] = 0
        self.data_anomaly['Anomaly_Total'] = 0
        self.data_anomaly['Anomaly_Pa'] = 0
        self.data_anomaly['Anomaly_T'] = 0
        self.data_anomaly['Info'] = ''

    def detect_by_threshold(self):
        '''通过阈值进行检测

        通过仪表参数以及人为设定阈值判断是否超上下限

        Args:
        -------
        target: str
            检测目标列
        data_anomaly: pd.DataFrame
            带异常的原始数据
        '''
        target = self.target
        data_anomaly = self.data_anomaly

        df_target = 'Anomaly_' + target
        if target == 'Flow':
            target = 'GFlow'
            max_value = self.max_value
            min_value = 0
        else:
            max_value = eval(f'self.{target.lower()}_max')
            min_value = eval(f'self.{target.lower()}_min')
        info_max = f'{target} 超过当前设定最大阈值 {max_value}'
        info_min = f'{target} 低于当前设定最小阈值 {min_value}'

        data_anomaly.loc[data_anomaly[target] > max_value, df_target] += 1
        data_anomaly.loc[data_anomaly[target] > max_value, 'Info'] += info_max

        data_anomaly.loc[data_anomaly[target] < min_value, df_target] += 1
        data_anomaly.loc[data_anomaly[target] < min_value, 'Info'] += info_min

        # 3σ 检测
        mean, std = np.mean(data_anomaly[target]), np.std(data_anomaly[target])
        info_max = f'{target} 3σ 检测'
        data_anomaly.loc[data_anomaly[target] > mean + 5 * std, df_target] += 1
        data_anomaly.loc[data_anomaly[target] > mean + 5 * std, 'Info'] += info_max

        info_min = f'{target} 3σ 检测'
        data_anomaly.loc[data_anomaly[target] < mean - 5 * std, df_target] += 1
        data_anomaly.loc[data_anomaly[target] < mean - 5 * std, 'Info'] += info_min

        # # 表具参数范围检测
        # info = f'GFlow 超过当前表具最大量程 {self.gflow_max};'
        # data_anomaly.loc[data_anomaly['GFlow'] > self.gflow_max, 'Anomaly_Flow'] += 1
        # data_anomaly.loc[data_anomaly['GFlow'] > self.gflow_max, 'Info'] += info

        # info = f'GFlow 低于当前表具最小量程 {self.gflow_min};'
        # data_anomaly.loc[data_anomaly['GFlow'] < self.gflow_min, 'Anomaly_Flow'] += 1
        # data_anomaly.loc[data_anomaly['GFlow'] < self.gflow_min, 'Info'] += info

        # info = f'BFlow 超过当前表具最大量程 {self.bflow_max};'
        # data_anomaly.loc[data_anomaly['BFlow'] > self.bflow_max, 'Anomaly_Flow'] += 1
        # data_anomaly.loc[data_anomaly['BFlow'] > self.bflow_max, 'Info'] += info

        # info = f'BFlow 低于当前表具最小量程 {self.bflow_min};'
        # data_anomaly.loc[data_anomaly['BFlow'] < self.bflow_min, 'Anomaly_Flow'] += 1
        # data_anomaly.loc[data_anomaly['BFlow'] < self.bflow_min, 'Info'] += info

    def detect_decrease_total(self):
        '''检测累积量是否减少

        差分累积流量, 检测是否存在负值

        Args:
        -------
        data_anomaly: pd.DataFrame
            带异常的原始数据
        '''
        data_anomaly = self.data_anomaly

        info = f'BTotal 小于上一个检测值;'
        data_anomaly.loc[data_anomaly['BTotal'].diff() < 0, 'Anomaly_Total'] += 1
        data_anomaly.loc[data_anomaly['BTotal'].diff() < 0, 'Info'] += info

    def get_resample_data(self):
        '''进行数据重采样

        可根据具体的数据格式创建子类, 继承 `GasDetect` 进行函数重写

        Args:
        -------
        data_raw: pd.DataFrame
            原始数据
        
        Returns:
        -------
        data_resample: pd.DataFrame
            重采样后的数据
        '''
        data_raw = self.data_raw.copy()

        # 分析原始数据的采样频率
        freqs = util.get_sample_freqs(data_raw)
        self.config.INPUT_SIZE = int(self.config.INPUT_SIZE // (freqs / 3600))
        self.config.OUTPUT_SIZE = int(self.config.OUTPUT_SIZE // (freqs / 3600))

        data_resample = util.get_resample_data(data_raw, self.target, freqs)
        if len(data_resample) == 0:
            raise ValueError('重采样后数据为空.')
        self.data_resample = data_resample

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
        self.MODEL_NAME = f'{self.table_industry}_{self.table_name}_{self.mode}_AE.pkl'
        self.IMG_NAME = f'{self.table_industry}_{self.table_name}_{self.mode}_AE.{self.img_type}'

        data_norm = self.data_norm.view(-1, self.config.INPUT_SIZE)

        if self.detect_mode == 'all':
            self.detect_day = len(data_norm)
        gd_data = TensorDataset(data_norm[-self.detect_day:],
                                data_norm[-self.detect_day:])

        if len(gd_data) == 0:
            raise ValueError(f'{self.table_name} does not have enough data.')

        n_train = int(len(gd_data) * 0.8)
        n_test = len(gd_data) - n_train
        ds_train, ds_test = random_split(gd_data, [n_train, n_test])

        self.dl_data = DataLoader(ds_train, batch_size=self.config.BATCH_SIZE)
        self.dl_detect = DataLoader(gd_data, batch_size=1)

    def train_model(self):
        '''模型训练

        Args:
        -------
        dl_data: DataLoader
            数据集
        '''
        model = GasModel.AE(self.config)
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
                feature, label = dl
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

    def detect(self):
        '''差分流量检测

        Args:
        -------
        dl_detect: DataLoader
            数据集

        Returns:
        -------
        actual: np.array
            实际值
        predict: np.array
            预测值
        '''
        # 模型初始化
        model = GasModel.AE(self.config)
        if self.use_gpu:
            model = model.cuda()
            model = torch.nn.DataParallel(model, device_ids=self.gpu_id)

        # 模型加载
        if not (self.MODEL_DIR / self.MODEL_NAME).exists():
            raise RuntimeError(f'{self.MODEL_NAME} is not exist, please train first.')
        model.load_state_dict(torch.load(self.MODEL_DIR / self.MODEL_NAME))
        model = model.eval()

        pred, actual = [], []
        for feature, label in self.dl_detect:
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

        Args:
        -------
        actual: np.array
            实际值
        predict: np.array
            预测值
        data_norm_info: pd.DataFrame
            标准化后的数据信息
        scalers: dict(str: MinMaxScaler)
            标准化器

        Returns:
        -------
        result: pd.DataFrame
            结果
        
        '''
        table_name = self.table_name

        if self.detect_mode == 'all':
            last = 0
        else:
            cnt = int(self.detect_mode.split('-')[-1])
            last = -cnt * self.config.INPUT_SIZE

        data_norm_info = self.data_norm_info[last:].reset_index(drop=True)

        actual = pd.Series(self.actual, name='Actual')
        pred = pd.Series(self.pred, name='Pred')
        result = pd.concat([data_norm_info, actual, pred], axis=1)
        result[['Actual', 'Pred']] = \
            self.scalers[table_name].inverse_transform(result[['Actual', 'Pred']])

        result['Anomaly'] = 0
        error_idx = util.detect_anomaly(result['Actual'], result['Pred'])
        result.loc[error_idx, 'Anomaly'] = 1
        self.result = result
        logger.info(f'{sys._getframe().f_code.co_name} done!')

    def detect_by_ewm(self):
        '''通过指数加权/移动平均检测气压

        Args:
        -------
        data_anomaly: pd.DataFrame
            带异常的原始数据
        '''
        data_anomaly = self.data_anomaly
        df_target = 'Anomaly_' + self.target
        target = self.target if self.target != 'Flow' else 'GFlow'

        # 当设置较小的系数 a 时，得出的均值更大程度上是参考过去的测量值
        data_ewm = data_anomaly[target].ewm(alpha=0.3).mean()
        error_idx = util.detect_anomaly(data_anomaly[target], data_ewm)
        data_anomaly.loc[error_idx, df_target] = 1

    def plot_result(self, show=False):
        '''绘图

        Args:
        -------
        data_anomaly: pd.DataFrame
            带异常的原始数据
        result: pd.DataFrame
            预测结果汇总
        show: bool = False
            是否显示图片(要求 jupyter notebook 环境)
        '''
        img_path = self.IMG_DIR / self.IMG_NAME
        result = util.get_dateline_data(self.result, last=self.detect_day)
        util.plot_detect_result(img_path, self.data_anomaly, result, self.target, show)
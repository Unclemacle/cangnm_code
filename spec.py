import os
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing

warnings.filterwarnings("ignore")

DATABASE = Path("../../data/cangnm_data/")


class Spec:
    def __init__(self):
        self.spec_info = pd.read_csv(DATABASE / "spec_info.csv")

        self.reply_msg = []
        self.new_table = False
        self.recommend = True

        self.rate_seg_sib = 0.2  # 大马拉小车, 前20%区间
        self.rate_seg_bis = 0.8  # 小马拉大车, 后20%区间

        self.rate_over_max = 0.1  # 超过最大值的百分比
        self.rate_over_min = 0.1  # 超过最小值的百分比
        self.rate_sib = 0.8  # 大马拉小车, 指定区间数据占比
        self.rate_bis = 0.8  # 小马拉大车, 指定区间数据占比

    def start_check(self, table_name: str):
        '''检查流程

        1. 加载当前表具信息
        2. 检查表具数据(预处理)
        3. 获取每个区间的数据量
        4. 根据数据量判断是否需要推荐选型
        5. 推荐表具
        '''
        self.table_name = table_name

        print("开始检查 {}".format(self.table_name))

        # 加载当前表具信息
        self.__load_spec_info()
        # 检查表具数据
        self.__load_and_check_data()
        # 获取每个区间的数据量
        self.__get_segment_distribute()
        # 判断选型
        self.__recommend_check()
        # 推荐表具
        self.__recommend_spec()

    def __load_spec_info(self):
        '''加载当前表具信息
        
        1. 若不存在当前表具信息, 则认为是新表, 直接推荐表具
        2. 不考虑阀门
        
        '''
        self.spec_info.set_index('TableName', inplace=True)

        # 若不存在当前表具信息, 则认为是新表
        if self.table_name not in self.spec_info.index:
            self.reply_msg.append("无正在使用的表具信息")
            self.new_table = True
            return

        # 阀门，不考虑
        # if self.spec_info.loc[self.table_name, 'MaxValue'] == 0:
        #     self.reply_msg.append("当前使用表具未设置最大值")
        #     self.new_table = True
        #     return

        self.spec_name = self.spec_info.loc[self.table_name, "Spec"]
        self.max_value = self.spec_info.loc[self.table_name, "MaxValue"]
        self.min_value = self.spec_info.loc[self.table_name, "MinValue"]
        print('当前使用表具名称: ' + self.spec_info.loc[self.table_name, "CName"])
        print('量程范围: {}~{}'.format(self.min_value, self.max_value))

    def __load_and_check_data(self):
        '''加载当前表具数据并检查
        
        1. 数据预处理
        2. 检查正常区间的数据数量, 如果数据量过少，则退出
        '''
        table_name = self.table_name

        raw_data = pd.read_csv(DATABASE / 'data_exported' / (table_name + '.csv'))
        raw_data = raw_data[['CreateDate', 'GFlow', 'BFlow', 'Pa', 'T']]
        raw_data['CreateDate'] = pd.to_datetime(raw_data['CreateDate'])
        self.raw_data = raw_data

        # 数据预处理
        data = raw_data.dropna().drop_duplicates()
        data.drop(data[(data['Pa'] > 1000) | (data['Pa'] < 0)].index, inplace=True)
        data.drop(data[(data['T'] > 80) | (data['T'] < -30)].index, inplace=True)
        data.drop(data[(data['CreateDate'].dt.year < 2014)
                       | (data['CreateDate'].dt.year > 2022)].index,
                  inplace=True)
        data = data[data['GFlow'] > 0]
        mean = data['GFlow'].mean()
        std = data['GFlow'].std()
        self.over_data = data[(data['GFlow'] > mean + std)
                              | (data['BFlow'] / data['GFlow'] > 15)
                              | (data['BFlow'] / data['GFlow'] < 1 / 15)
                              | (data['GFlow'] > 100000)]
        data.drop(self.over_data.index, inplace=True)
        data.reset_index(drop=True, inplace=True)
        self.data = data[['CreateDate', 'GFlow']]

        if len(self.over_data):
            self.reply_msg.append("GFlow 存在异常值: {}".format(self.over_data['GFlow'].max()))

        # 检查正常区间的数据数量, 如果数据量过少，则退出
        if len(data) < 100:
            self.reply_msg.append("{} 数据量太少({})或存在过多异常数据({})".format(
                table_name, len(self.data), len(self.over_data)))
            if len(ts.reply_msg):
                print('异常信息:')
                print('\n'.join(ts.reply_msg))
            sys.exit(1)

    def __get_segment_distribute(self, n=5):
        '''获取每个区间的数据量
        
        1. 0-100分为5个区间
        2. 前后指定区间的数据量
        
        '''
        if self.new_table:
            return

        data = self.data
        seg_point = np.linspace(self.min_value, self.max_value, n + 1)

        # 计算每个区间的数据量
        self.seg_data = [None] * (n + 2)
        self.seg_data_info = defaultdict(int)
        for i in range(1, len(seg_point)):
            seg_name = '{}-{}'.format(seg_point[i - 1], seg_point[i])
            seg_add = data[(data['GFlow'] >= seg_point[i - 1])
                           & (data['GFlow'] < seg_point[i])]
            self.seg_data[i] = seg_add
            self.seg_data_info[seg_name] = len(seg_add)

        # 大马拉小车区间数据
        seg_sib = (self.max_value - self.min_value) * self.rate_seg_sib + self.min_value
        self.data_sib = data[(data['GFlow'] >= self.min_value)
                             & (data['GFlow'] < seg_sib)]
        # 小马拉大车区间数据
        seg_bis = (self.max_value - self.min_value) * self.rate_seg_bis + self.min_value
        self.data_bis = data[(data['GFlow'] >= seg_bis)
                             & (data['GFlow'] <= self.max_value)]

        # 统计超出范围的数据
        over_max = data[data['GFlow'] > self.max_value]
        over_min = data[data['GFlow'] < self.min_value]
        self.seg_data[0] = over_min
        self.seg_data[-1] = over_max
        self.seg_data_info['OverMax'] = len(over_max)
        self.seg_data_info['OverMin'] = len(over_min)
        self.seg_data_info = pd.DataFrame.from_dict(self.seg_data_info,
                                                    orient='index',
                                                    columns=['Count'])
        self.seg_data_info.index.name = '分布'

    def __recommend_check(self):
        '''检查选型
        
        1. 如果  >100 或 <0  的记录条数比重  >10%, 那么我们也认为需要更换表具
        2. 如果     0-20     的记录条数比重  >80%, 那么可以认为是大马拉小车
        3. 如果    80-100    的记录条数比重  >80%, 那么可以认为是小马拉大车

        '''
        if self.new_table:
            return

        data = self.data
        seg_data = self.seg_data
        len_over_max = len(seg_data[-1])
        len_over_min = len(seg_data[0])
        len_sib = len(self.data_sib)
        len_bis = len(self.data_bis)
        # 检查每个区间的数据量
        if len_over_max / len(data) <= self.rate_over_max and \
            len_over_min / len(data) <= self.rate_over_min:

            if len_sib / len(data) > self.rate_sib:
                self.reason = '表具选型异常: 大马拉小车. {:.2f}%的记录使用了仪表量程的前{}%'.format(
                    len_sib / len(data) * 100, self.rate_seg_sib * 100)
            elif len_bis / len(data) > self.rate_bis:
                self.reason = '表具选型异常: 小马拉大车. {:.2f}%的记录使用了仪表量程的后{}%'.format(
                    len_bis / len(data) * 100, (1 - self.rate_seg_bis) * 100)
            else:
                self.recommend = False
        else:
            self.reason = "有{:.2f}%条记录超过当前表具测量上界, 有{:.2f}%条记录超过当前表具测量下界".format(
                len_over_max / len(data) * 100, len_over_min / len(data) * 100)
        self.reply_msg.append(self.reason)

    def __recommend_spec(self):
        '''推荐表具'''
        if self.recommend == False:
            print('当前表具的使用情况如下:')
            print(self.seg_data_info)
            print('不需要更换表具')
            return

        # 所有表具的信息
        all_spec_info = pd.read_csv(DATABASE / "all_spec_info.csv")
        all_spec_info = all_spec_info[all_spec_info['MaxValue'] != 0]
        all_spec_info.dropna(inplace=True)
        all_spec_info.drop_duplicates(inplace=True)
        all_spec_info.reset_index(drop=True, inplace=True)
        self.all_spec_info = all_spec_info

        data = self.data
        mean = data['GFlow'].mean()
        std = data['GFlow'].std()
        data = data[(data['GFlow'] >= mean - 3 * std) & (data['GFlow'] <= mean + 3 * std)]

        print('当前表数据最小值: {}, 最大值: {}'.format(data['GFlow'].min(), data['GFlow'].max()))
        all_spec_info['Mean'] = (all_spec_info['MinValue'] +
                                 all_spec_info['MaxValue']) / 2

        # fig = px.scatter(x=spec_feature[:, 0], y=spec_feature[:, 1])
        # fig.show()

        test = [[data['GFlow'].min(), data['GFlow'].max(), data['GFlow'].mean()]]

        neigh = KNeighborsClassifier(5)
        neigh.fit(all_spec_info[['MinValue', 'MaxValue', 'Mean']], all_spec_info['Name'])
        neigh_dist, neigh_ind = neigh.kneighbors(test)
        rec_list = all_spec_info.loc[neigh_ind.reshape(-1)]
        rec_list.reset_index(drop=True, inplace=True)

        if not self.new_table:
            print('当前表具的使用情况如下:')
            print(self.seg_data_info)
            print('\n推荐更换表具')
            print(rec_list[['Name', 'MinValue', 'MaxValue', 'Mean']])
            print('推荐理由: ' + self.reason)
        else:
            print('无当前表具信息')
            print('\n推荐更换表具')
            print(rec_list[['Name', 'MinValue', 'MaxValue', 'Mean']])

            tmp = Spec()
            tmp.spec_info
            tmp.__load_spec_info()
            tmp.__load_and_check_data()
            print('更换表具后的分布情况如下:')
            print(self.seg_data_info)
            print()

        def __change_check():
            tmp = Spec()
            tmp.__load_spec_info()
            tmp.__load_and_check_data()
            print('更换表具后的分布情况如下:')
            print(self.seg_data_info)
            print()


if __name__ == '__main__':
    ts = Spec()
    # ts.start_check('G_Data0000000177')
    ts.start_check('G_Data0000000664')
    # ts.start_check('G_Data0000005341')
    if len(ts.reply_msg):
        print('\n异常信息:')
        print('\n'.join(ts.reply_msg))

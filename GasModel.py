import torch
import pandas as pd
from torch import nn
from torch.utils.data import Dataset

##############################################################################################################
##############################################################################################################


# 定义数据集
class GasDataset(Dataset):
    def __init__(self, data, data_info, config, last=False) -> None:
        super().__init__()
        self.data = data.clone()
        self.data_info = data_info.copy()
        self.config = config
        self.last = not last

    def __len__(self) -> int:
        dataset_len = len(self.data) - self.config.OUTPUT_SIZE * self.last - \
            self.config.LOOK_BACK * max(1, self.config.PRE_DAYS * self.config.USE_PRE)
        return dataset_len + 1 if dataset_len >= 0 else 0

    def __getitem__(self, idx):
        PRE_DAYS = self.config.PRE_DAYS
        LOOK_BACK = self.config.LOOK_BACK
        OUTPUT_SIZE = self.config.OUTPUT_SIZE

        # 是否使用历史数据
        if self.config.PRE_DAYS != 0 and self.config.USE_PRE:
            # PRE_DAYS 以 label_start_idx 为准
            feature_start_idx = idx + (PRE_DAYS - 1) * LOOK_BACK
            feature_end_idx = idx + PRE_DAYS * LOOK_BACK
            label_start_idx = idx + PRE_DAYS * LOOK_BACK
            label_end_idx = idx + PRE_DAYS * LOOK_BACK + OUTPUT_SIZE * self.last

            feature = self.data[feature_start_idx:feature_end_idx]
            label = self.data[label_start_idx:label_end_idx]

            # 加入历史数据及其工作日信息
            date_info, pre = [], pd.DataFrame()
            for i in range(PRE_DAYS):
                pre_start = idx + i * LOOK_BACK
                pre_end = idx + i * LOOK_BACK + OUTPUT_SIZE

                date_info.append(self.data_info.loc[pre_start, 'IsWorkday'])
                pre = pd.concat([pre, self.data[pre_start:pre_end]])

            # 加入 feature 的工作日信息
            for i in range(self.config.SEQ_LEN):
                date_info.append(self.data_info.loc[feature_start_idx + i * OUTPUT_SIZE,
                                                    'IsWorkday'])
            date_info = torch.tensor(date_info, dtype=torch.float32)
            pre = torch.tensor(pre, dtype=torch.float32)
            return feature, pre, date_info, label
        else:
            feature = self.data[idx:idx + LOOK_BACK]
            label = self.data[idx + LOOK_BACK:idx + LOOK_BACK + OUTPUT_SIZE * self.last]
        return feature, label


##############################################################################################################
##############################################################################################################


class LSTM(nn.Module):
    def __init__(self, config, num_layers=3):
        super(LSTM, self).__init__()
        self.config = config

        self.lstm = nn.LSTM(config.INPUT_SIZE,
                            config.HIDDEN_SIZE,
                            num_layers,
                            batch_first=True)  # lstm
        if config.USE_ATTN:
            self.W_Q = nn.Linear(config.HIDDEN_SIZE, config.HIDDEN_SIZE)
            self.W_K = nn.Linear(config.HIDDEN_SIZE, config.HIDDEN_SIZE)
            self.W_V = nn.Linear(config.HIDDEN_SIZE, config.HIDDEN_SIZE)
            self.mul_attn = nn.MultiheadAttention(
                config.HIDDEN_SIZE, num_heads=4, batch_first=True)  # multi-head attention
        self.linear1 = nn.Linear(config.HIDDEN_SIZE, config.OUTPUT_SIZE)  # 回归
        # 如果使用历史数据，则进行拼接，在放入全连接
        if config.PRE_DAYS != 0 and config.USE_PRE:
            self.linear2 = nn.Linear(
                config.PRE_DAYS * config.OUTPUT_SIZE + config.SEQ_LEN + config.PRE_DAYS,
                config.OUTPUT_SIZE)
            self.linear3 = nn.Linear(config.OUTPUT_SIZE * 2, config.OUTPUT_SIZE)

    def forward(self, input, pre=None, date=None):
        # input shape: (batch, seq_len, input_size)
        output, hidden = self.lstm(input)
        # output shape: (batch, seq_len, hidden_size)
        if self.config.USE_ATTN:
            Q, K, V = self.W_Q(output), self.W_K(output), self.W_V(output)
            output, attn_output_weights = self.mul_attn(Q, K, V)
            # output shape: (batch, seq_len, hidden_size)
        output = self.linear1(output[:, -1, :])

        if pre is not None and date is not None:
            pre_date = torch.cat((pre, date), 1)
            pre_date = self.linear2(pre_date)
            output = self.linear3(torch.cat((output, pre_date), 1))

        return output


##############################################################################################################
##############################################################################################################


class AE(nn.Module):
    def __init__(self, config):
        super(AE, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(config.INPUT_SIZE, 24), nn.Dropout(0.5),
                                     nn.ReLU(True), nn.Linear(24, 10), nn.Dropout(0.5),
                                     nn.ReLU(True))
        self.decoder = nn.Sequential(nn.Linear(10, 24), nn.ReLU(True),
                                     nn.Linear(24, config.OUTPUT_SIZE), nn.Sigmoid())

    def forward(self, x):
        en_out = self.encoder(x)
        de_out = self.decoder(en_out)
        return de_out
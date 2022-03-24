from pathlib import Path

import numpy as np
import pandas as pd
import torch

from util import Config


class GasBase():
    '''燃气基类
    '''
    def __init__(self, config_dir, mode) -> None:
        # 参数检查
        if not Path(config_dir).exists():
            raise FileNotFoundError(f'{config_dir} not found.')
        if mode not in ['day', 'month', 'season', 'year']:
            raise ValueError("mode must be 'day', 'month', 'season' or 'year'.")

        self.config = Config(config_dir, mode)
        self.mode = mode

        # 读取全局参数
        self.DATABASE = Path(self.config.DATABASE)
        self.MODEL_DIR = Path(self.config.MODEL_DIR)
        self.IMG_DIR = Path(self.config.IMG_DIR)

        # 创建目录
        if not self.MODEL_DIR.exists():
            self.MODEL_DIR.mkdir()
        if not self.IMG_DIR.exists():
            self.IMG_DIR.mkdir()

        # 配置其他参数
        self.use_gpu = True if torch.cuda.is_available() else False
        self.gpu_id = [0] if self.use_gpu else None

        self.freqs = 3600
        self.min_cluster_num = 5
        self.max_cluster_num = 30
        self.img_type = 'png'

    def set_gpu_config(self, use_gpu: bool = None, gpu_id=None):
        '''设置 GPU 参数

        Args:
        -------
        use_gpu: bool
            是否使用 GPU
        gpu_id: int | List[int]
            GPU ID 列表

        Example:
        -------
        >>> gas = GasForecast()
        使用 GPU
        >>> gas.set_gpu_config(use_gpu=True)
        使用 GPU, 指定 GPU ID [0, 1]
        >>> gas.set_gpu_config(use_gpu=True, gpu_id=[0, 1])
        '''
        if use_gpu:
            if not torch.cuda.is_available():
                raise ValueError("GPU is not available, please set use_gpu=False.")
            if not isinstance(gpu_id, (int, list)):
                raise ValueError("GPU ID required int or list[int].")
        else:
            if gpu_id:
                raise ValueError("GPU is not available, please not set gpu_id.")

        self.use_gpu = use_gpu
        self.gpu_id = gpu_id if isinstance(gpu_id, list) else [gpu_id]

    def set_resample_freqs(self, freqs: int = 3600):
        '''设置重采样频率

        Args:
        -------
        freqs: int = 3600
            重采样频率(单位: 秒), 默认为 3600s
        '''
        self.freqs = freqs

    def set_cluster_config(self, min_cluster_num: int = 5, max_cluster_num: int = 30):
        '''设置聚类参数

        Args:
        -------
        min_cluster_num: int = 5
            最小聚类数
        max_cluster_num: int = 30
            最大聚类数
        '''
        self.min_cluster_num = min_cluster_num
        self.max_cluster_num = max_cluster_num

    def set_img_type(self, img_type='png'):
        '''设置图片输出格式
        
        Args:
        ------
        img_type: str = 'png'
            图片保存类型, 可选 ['png', 'jpg', 'jpeg', 'webp', 'svg', 'pdf', 'html']
        '''
        if img_type not in ['png', 'jpg', 'jpeg', 'webp', 'svg', 'pdf', 'html']:
            raise ValueError(
                "img type must be 'png', 'jpg', 'jpeg', 'webp', 'svg', 'pdf' or 'html'.")
        self.img_type = img_type

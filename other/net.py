from typing import Optional, Tuple

import numpy as np
import torch
from torch import nn


def split_heads(x: torch.Tensor, n_heads: int) -> torch.Tensor:
    """
    Args:
    -------
    x : torch.Tensor (batch_size, length, dim)
        Input tensor.
    n_heads : int
        attention 头数
    """
    batch_size, dim = x.size(0), x.size(-1)
    x = x.view(batch_size, -1, n_heads,
               dim // n_heads)  # (batch_size, length, n_heads, d_head)
    x = x.transpose(1, 2)  # (batch_size, n_heads, length, d_head)
    return x


def combine_heads(x: torch.Tensor) -> torch.Tensor:
    """
    Args:
    -------
    x : torch.Tensor (batch_size, n_heads, length, d_head)
        Input tensor.
    """
    batch_size, n_heads, d_head = x.size(0), x.size(1), x.size(3)
    x = x.transpose(1, 2).contiguous().view(
        batch_size, -1, d_head * n_heads)  # (batch_size, length, n_heads * d_head)
    return x


def add_mask(x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Mask away by setting such weights to a large negative number, so that they evaluate to 0
    under the softmax.
    Parameters
    -------
    x : torch.Tensor (batch_size, n_heads, *, length) or (batch_size, length)
        Input tensor.
    mask : torch.Tensor, optional (batch_size, length)
        Mask metrix, ``None`` if it is not needed.
    """
    if mask is not None:
        if len(x.size()) == 4:
            expanded_mask = mask.unsqueeze(1).unsqueeze(1)  # (batch_size, 1, 1, length)
        x = x.masked_fill(expanded_mask.bool(), -np.inf)
    return x


class ScaledDotProductAttention(nn.Module):
    '''
    Scaled Dot-Product Attention

    Args:
    -------
    scale: float
        缩放因子
    dropout: float, optional
        dropout比例
    '''
    def __init__(self, scale: float, dropout: float = 0.5) -> None:
        super(ScaledDotProductAttention, self).__init__()

        self.scale = scale
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = None if dropout is None else nn.Dropout(dropout)

    def forward(self,
                Q: torch.Tensor,
                K: torch.Tensor,
                V: torch.Tensor,
                mask: Optional[torch.Tensor] = None):
        '''
        Args:
        -------
        Q : torch.Tensor (batch_size, n_heads, length, d_head)
            Query tensor.
        K : torch.Tensor (batch_size, n_heads, length, d_head)
            Key tensor.
        V : torch.Tensor (batch_size, n_heads, length, d_head)
            Value tensor.
        mask : torch.Tensor (batch_size, 1, 1, length)
            Mask metrix, ``None`` if it is not needed.
        
        Returns:
        --------
        context : torch.Tensor (batch_size, n_heads, length, d_head)
            上下文向量
        attn : torch.Tensor (batch_size, n_heads, length, length)
            Attention weights.
        '''
        # Q·K^T / sqrt(d_head)
        score = torch.matmul(Q / self.scale,
                             K.transpose(2, 3))  # (batch_size, n_heads, length, length)
        score = add_mask(score, mask)

        # softmax
        att = self.softmax(score)  # (batch_size, n_heads, length, length)
        att = att if self.dropout is None else self.dropout(att)
        context = att @ V  # (batch_size, n_heads, length, d_head)

        return context, att


class SelfAttention(nn.Module):
    '''
    Multi-Head Self-Attention

    Args:
    -------
    dim : int
        输入的维度
    n_heads : int
        attention 头数
    simplified : bool, optional, default=False
        是否简化版本
    dropout : float, optional
        dropout比例  
    '''
    def __init__(self,
                 dim: int,
                 n_heads: int = 8,
                 simplified: bool = False,
                 dropout: Optional[float] = None):
        super(SelfAttention, self).__init__()

        assert dim % n_heads == 0

        self.n_heads = n_heads
        self.d_head = dim // n_heads

        self.simplified = simplified

        if not simplified:
            # 线性变换
            self.W_Q = nn.Linear(dim, n_heads * self.d_head)
            self.W_K = nn.Linear(dim, n_heads * self.d_head)
            self.W_V = nn.Linear(dim, n_heads * self.d_head)

        # scaled dot-product attention
        scale = self.d_head**0.5  # scale factor
        self.attention = ScaledDotProductAttention(scale=scale, dropout=dropout)

        self.layer_norm = nn.LayerNorm(dim)
        self.fc = nn.Linear(n_heads * self.d_head, dim)

        self.dropout = None if dropout is None else nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        '''
        Args:
        -------
        x : torch.Tensor (batch_size, length, dim)
            Input tensor.
        mask : torch.Tensor, optional (batch_size, length)
            Mask metrix, ``None`` if it is not needed.

        Returns:
        --------
        out : torch.Tensor (batch_size, length, dim)
            Output tensor.
        '''
        if self.simplified:
            Q = K = V = x
        else:
            Q = self.W_Q(x)
            K = self.W_K(x)
            V = self.W_V(x)

        Q = split_heads(Q, self.n_heads)  # (batch_size, n_heads, length, d_head)
        K = split_heads(K, self.n_heads)
        V = split_heads(V, self.n_heads)

        context, _ = self.attention(Q, K, V,
                                    mask=mask)  # (batch_size, n_heads, length, d_head)
        context = combine_heads(context)  # (batch_size, length, n_heads * d_head)

        out = self.fc(context)  # (batch_size, length, dim)
        out = out if self.dropout is None else self.dropout(out)

        out = out + x  # residual connection
        out = self.layer_norm(out)  # LayerNorm

        return out

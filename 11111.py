import torch
from torch import nn
from torch.nn import Parameter
from torch.nn.functional import multi_head_attention_forward
import torch.nn.functional as F
class MyMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0., bias=True):
        super(MyMultiheadAttention, self).__init__()
        """
        :param embed_dim:   词嵌入的维度，也就是前面的d_model参数，论文中的默认值为512
        :param num_heads:   多头注意力机制中多头的数量，也就是前面的nhead参数， 论文默认值为 8
        :param bias:        最后对多头的注意力（组合）输出进行线性变换时，是否使用偏置
        """
        self.embed_dim = embed_dim  # 前面的d_model参数
        self.head_dim = embed_dim // num_heads  # head_dim 指的就是d_k,d_v
        self.kdim = self.head_dim
        self.vdim = self.head_dim
        self.num_heads = num_heads  # 多头个数
        self.dropout = dropout
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim 除以 num_heads必须为整数"
        # 上面的限制条件就是论文中的  d_k = d_v = d_model/n_head 条件
        self.q_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))  
        # embed_dim = kdim * num_heads
        # 这里第二个维度之所以是embed_dim，实际上这里是同时初始化了num_heads个W_q堆叠起来的, 也就是num_heads个头
        self.k_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))  
        # W_k,  embed_dim = kdim * num_heads
        self.v_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))  
        # W_v,  embed_dim = vdim * num_heads
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        # 最后将所有的Z组合起来的时候，也是一次性完成， embed_dim = vdim * num_heads

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        """
        在论文中，编码时query, key, value 都是同一个输入， 
        解码时 输入的部分也都是同一个输入， 
        解码和编码交互时 key,value指的是 memory, query指的是tgt
        :param query: # [tgt_len, batch_size, embed_dim], tgt_len 表示目标序列的长度
        :param key:  #  [src_len, batch_size, embed_dim], src_len 表示源序列的长度
        :param value: # [src_len, batch_size, embed_dim], src_len 表示源序列的长度
        :param attn_mask: # [tgt_len,src_len] or [num_heads*batch_size,tgt_len, src_len]
                一般只在解码时使用，为了并行一次喂入所有解码部分的输入，所以要用mask来进行掩盖当前时刻之后的位置信息
        :param key_padding_mask: [batch_size, src_len], src_len 表示源序列的长度
        :return:
        attn_output: [tgt_len, batch_size, embed_dim]
        attn_output_weights: # [batch_size, tgt_len, src_len]
        """
        return multi_head_attention_forward(query, key, value, self.num_heads,
                                            self.dropout, self.out_proj.weight, self.out_proj.bias,
                                            training=self.training,
                                            key_padding_mask=key_padding_mask,
                                            q_proj_weight=self.q_proj_weight,
                                            k_proj_weight=self.k_proj_weight,
                                            v_proj_weight=self.v_proj_weight,
                                            attn_mask=attn_mask)

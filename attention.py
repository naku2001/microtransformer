import math
import torch as nn
import torch.nn.functional as F


def naive_attention(q:torch.Tensor,k:torch.Tensor,v:torch.Tensor)-> torch.Tensor:
    scores = torch.matmul(q,k.transpose(-2,-1))
    scores = scores/math.sqrt(q.size(-1))
    attn_weights = F.softmax(scores,dim =-1)
    output = torch.matmul(attn_weights,v)
    return output


def causal_attention(q:torch.Tensor,k:torch.Tensor,v:torch.Tensor) -> torch.Tensor:
    seq_len = q.size(1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
    scores = torch.triu(torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool),diagonal=1)
    scores = scores.masked_fill(mask, float('-inf'))
    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, v)
    return output


class SingleHeadAttention(nn.Module):
    def __init__(self,hidden_dim:int,head_dim:int):
        super().__init__()
        self.head_dim = head_dim
        self.q_proj = nn.Linear(hidden_dim, head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, head_dim, bias=False)
        self.o_proj = nn.Linear(head_dim, hidden_dim, bias=False)
        
    def forward()
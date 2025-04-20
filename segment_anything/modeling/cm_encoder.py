import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

from typing import Type

try:
    from .common import LayerNorm2d, MLPBlock
except:
    from common import LayerNorm2d, MLPBlock
    

class CMEncoder(nn.Module):
    def __init__(
        self,
        in_channel: int = 256,
        norm_dim : int = 256,
        mlp_dim : int = 256,
        mlp_hid_dim : int = 512,
        mlp_act: Type[nn.Module] = nn.GELU,
    ) -> None:
          super().__init__()
          self.in_channel = in_channel
          self.norm_dim = norm_dim
          self.embedding_dim = mlp_dim
          self.mlp_dim = mlp_hid_dim
          self.act = mlp_act
          self.cross_att = CrossAttention(self.in_channel)
          self.norm = LayerNorm2d(self.norm_dim)
          self.mlp = MLPBlock(self.embedding_dim, self.mlp_dim, self.act)
     
    def forward(self, x, y):
        shortcut = x
        x = self.cross_att(x,y)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)  # (1, 64, 64, 256)
        x = x.view(1, 64*64, 256)
        x = self.mlp(x)
        x = x.view(1, 64, 64, 256).permute(0, 3, 1, 2)
        output = shortcut + x
        return output



#交叉注意力层,输入为x(1,256,64,64)，y(1,256,64)->输出为(1,256,64,64)大小
class CrossAttention(nn.Module):
    def __init__(self, embed_dim):
        super(CrossAttention, self).__init__()
        self.query_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)

    def forward(self, x, y):
        """
        x: (B, C, H, W)
        y: (B, C, N)
        return: (B, C, H, W)
        """
        B, C, H, W = x.shape
        _, _, N = y.shape

        # Project queries from x
        Q = self.query_proj(x).flatten(2).transpose(1, 2)  # (B, H*W, C)

        # Project keys and values from y
        y = y.permute(0, 2, 1)  # (B, N, C)
        K = self.key_proj(y)    # (B, N, C)
        V = self.value_proj(y)  # (B, N, C)

        # Attention weights
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (C ** 0.5)  # (B, H*W, N)
        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, H*W, N)

        # Apply attention
        attended = torch.matmul(attn_weights, V)  # (B, H*W, C)
        attended = attended.transpose(1, 2).reshape(B, C, H, W)  # (B, C, H, W)

        # Final output projection
        return self.out_proj(attended)


if __name__ == '__main__':
    x = torch.randn(1, 256, 64, 64).to("cpu")
    y = torch.randn(1, 256, 64).to("cpu")

    model = CMEncoder(in_channel=256, norm_dim=256, mlp_dim=256, mlp_hid_dim=512,).to("cpu")
    out = model(x, y)

    print(out.shape)  # should be (1, 256, 64, 64)
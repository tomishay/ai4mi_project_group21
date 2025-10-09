
import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbed(nn.Module):
    def __init__(self, in_ch: int, embed_dim: int, patch: int = 16):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch, stride=patch)
        self.embed_dim = embed_dim
        self.patch = patch
    def forward(self, x):
        x = self.proj(x)                  # [B,D,Hp,Wp]
        B, D, Hp, Wp = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # [B, N, D]
        return tokens, (Hp, Wp)

class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        hid = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hid)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hid, dim)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x

class Attention(nn.Module):
    def __init__(self, dim, heads=6, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.h = heads
        d = dim // heads
        self.scale = d ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.ad = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.pd = nn.Dropout(proj_drop)
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.h, C // self.h).unbind(2)
        q, k, v = qkv
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.ad(attn.softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.pd(self.proj(x))

class Block(nn.Module):
    def __init__(self, dim, heads=6, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, drop, drop)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio, drop)
    def forward(self, x):
        x = x + self.attn(self.n1(x))
        x = x + self.mlp(self.n2(x))
        return x

def _sincos_pe(Hp, Wp, dim, device):
    assert dim % 4 == 0
    y, x = torch.meshgrid(torch.arange(Hp, device=device),
                          torch.arange(Wp, device=device),
                          indexing="ij")
    d = dim // 4
    omega = 1.0 / (10000 ** (torch.arange(d, device=device) / d))
    oy = torch.einsum('hw,d->hwd', y.float(), omega)
    ox = torch.einsum('hw,d->hwd', x.float(), omega)
    pe = torch.cat([oy.sin(), oy.cos(), ox.sin(), ox.cos()], dim=2)
    return pe.view(1, Hp*Wp, dim)

class TinyViTSeg(nn.Module):
    """
    I/O: [B,C,H,W] -> [B,K,H,W]
    """
    def __init__(self, in_dim: int, out_dim: int,
                 embed_dim: int = 192, depth: int = 6,
                 heads: int = 6, patch: int = 16, drop: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch = patch
        self.embed = PatchEmbed(in_dim, embed_dim, patch)
        self.drop = nn.Dropout(drop)
        self.blocks = nn.ModuleList([Block(embed_dim, heads, 4.0, drop) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        # decoder: x2 * 3 = x8
        self.dec1 = nn.ConvTranspose2d(embed_dim, embed_dim//2, 2, 2)
        self.dec2 = nn.ConvTranspose2d(embed_dim//2, embed_dim//4, 2, 2)
        self.dec3 = nn.ConvTranspose2d(embed_dim//4, embed_dim//8, 2, 2)
        self.head = nn.Sequential(
            nn.Conv2d(embed_dim//8, embed_dim//8, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(embed_dim//8, out_dim, 1),
        )
        print(f"> Initialized TinyViTSeg(in={in_dim}, out={out_dim}, patch={patch}, "
              f"embed={embed_dim}, depth={depth}, heads={heads})")

    def forward(self, x):
        B, C, H, W = x.shape
        tok, (Hp, Wp) = self.embed(x)               # [B,N,D]
        tok = tok + _sincos_pe(Hp, Wp, self.embed_dim, tok.device)
        tok = self.drop(tok)
        for blk in self.blocks:
            tok = blk(tok)
        tok = self.norm(tok)
        feat = tok.transpose(1, 2).reshape(B, self.embed_dim, Hp, Wp)
        y = self.dec1(feat); y = self.dec2(y); y = self.dec3(y)
        if y.shape[-2:] != (H, W):
            y = F.interpolate(y, size=(H, W), mode='bilinear', align_corners=False)
        return self.head(y)

    def init_weights(self, *args, **kwargs):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)

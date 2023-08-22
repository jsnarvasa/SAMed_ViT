from segment_anything.modeling import ImageEncoderViT
import torch
from torch import nn, einsum
from einops import rearrange
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        # print(x.shape)
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        # print(q.shape, k.shape, v.shape)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out
    

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
    

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)
    

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class TimeSeries_ImageEncoderViT(ImageEncoderViT):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.temporal_pos_embedding = nn.Linear(366, 768)

    
    @classmethod
    def create_from_super(cls, superclass_instance: ImageEncoderViT):
        '''
        Initialise a TimeSeries_ImageEncoderViT instance
        based on an existing instance of a ImageEncoderViT class
        '''
        return cls(
            img_size = superclass_instance.img_size,
            patch_size = superclass_instance.patch_embed.proj.kernel_size[0],
            in_chans = superclass_instance.patch_embed.proj.in_channels,
            embed_dim = superclass_instance.patch_embed.proj.out_channels,
            depth = len(superclass_instance.blocks),
            num_heads = superclass_instance.blocks[0].attn.num_heads,
            mlp_ratio = superclass_instance.blocks[0].mlp.lin1.out_features/superclass_instance.blocks[0].mlp.lin1.in_features,
            out_chans = superclass_instance.neck._modules['2'].out_channels,
            qkv_bias = torch.is_tensor(superclass_instance.blocks[0].attn.qkv.bias),
            norm_layer = type(superclass_instance.blocks[0].norm1),
            act_layer = type(superclass_instance.blocks[0].mlp.act),
            use_rel_pos = superclass_instance.blocks[0].attn.use_rel_pos,
            window_size = superclass_instance.blocks[0].window_size,
        )
    

    def create_temporal_encoder(self, dim_size: int) -> None:
        '''
        Create a temporal encoder for the timeseries nature of data
        '''

        self.dim_size = dim_size

        self.temporal_encoder = Transformer(
            dim = self.dim_size,
            depth = 4,
            heads = 4,
            dim_head = 32,
            mlp_dim = self.dim_size * 4,
            dropout = 0,
        )

    
    def forward(self, x: torch.Tensor, doy_batch: torch.Tensor) -> torch.Tensor:
        '''
        Need to override the existing forward method within the superclass
        Such that we are able to insert the timeseries transformer processing here
        '''

        B, T = doy_batch.shape
        doy_batch = doy_batch.to(torch.int64)
        doy_batch_onehot = F.one_hot(doy_batch, num_classes = 366).to(torch.float32)
        doy_batch_onehot = doy_batch_onehot.reshape(-1, 366)
        temporal_pos_embedding = self.temporal_pos_embedding(doy_batch_onehot).reshape(B, T, 768)
        temporal_pos_embedding = temporal_pos_embedding.unsqueeze(2).unsqueeze(2)


        # Need to split the input on a per observation basis, since patch_embed does not expect timeseries
        timeseries_tensors = x.split(split_size = 1, dim = 1)
        timeseries_tensors = [tensor.squeeze(1) for tensor in timeseries_tensors]

        timeseries_tensors = [self.patch_embed(tensor) for tensor in timeseries_tensors]

        # Stack the tensors back together into a single timeseries tensor again
        x = torch.stack(timeseries_tensors, dim = 1)
        # Infusing the temporal_positional embeddings into the input
        x += temporal_pos_embedding
        x = rearrange(x, 'b n (h p1) (w p2) d -> (b h w) n (p1 p2 d)', p1=2, p2=2)

        # Removing this, since it's losing too much information
        # x = rearrange(x, 'b n h w d -> b n (h w d)')
        # Cut the first 2048 tokens, due to memory limits
        # 2048 since we have 8 * 8 * 32 (where 32 was generally the number of tokens representing a pixel in TSViT)
        #x = x[:, :, :self.dim_size]
        
        # Send it to temporal encoder here
        x = self.temporal_encoder(x)

        # Reshaping the tensor to the original shape expected by SAM's image encoder
        # Essentially, reverting the rearrange performed prior to sending to temporal transformer
        # Since the rearrange is only performed to split the 8x8 into smaller 2x2 sub-patches
        # However, we are combining the temporal dimension and the token dimension hence (n d)
        x = rearrange(x, '(b h w) n (p1 p2 d) -> b (h p1) (w p2) (n d)', p1=2, p2=2, h=4, w=4)
        
        # Not required anymore, since fixed with the rearrange operation above
        # x = rearrange(x, 'b n (h w d) -> b h w (n d)', h=8, w=8)
        # Cutting the first 768 tokens, since this is the original size of the patch embedding generated within SAM
        x = x[..., :768]

        if self.pos_embed is not None:
            x = x + self.pos_embed

        for blk in self.blocks:
            x = blk(x)

        x = self.neck(x.permute(0, 3, 1, 2))  # [b, c, h, w], [1, 256, 64, 64]

        return x

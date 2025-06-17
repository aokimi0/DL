import torch
from torch import nn
from einops.layers.torch import Rearrange

class MlpBlock(nn.Module):
    def __init__(self, dim, mlp_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, mlp_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(mlp_dim, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

class MixerBlock(nn.Module):
    def __init__(self, dim, num_patches, tokens_mlp_dim, channels_mlp_dim):
        super().__init__()
        self.token_mix = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n d -> b d n'),
            MlpBlock(num_patches, tokens_mlp_dim),
            Rearrange('b d n -> b n d')
        )
        self.channel_mix = nn.Sequential(
            nn.LayerNorm(dim),
            MlpBlock(dim, channels_mlp_dim)
        )

    def forward(self, x):
        x = x + self.token_mix(x)
        x = x + self.channel_mix(x)
        return x

class MlpMixer(nn.Module):
    def __init__(self, in_channels=1, dim=128, num_blocks=4, patch_size=4,
                 tokens_mlp_dim=64, channels_mlp_dim=256, num_classes=10, image_size=28):
        super().__init__()
        num_patches = (image_size // patch_size) ** 2
        
        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size),
            Rearrange('b c h w -> b (h w) c'),
        )
        
        self.mixer_blocks = nn.Sequential(
            *[MixerBlock(dim, num_patches, tokens_mlp_dim, channels_mlp_dim) for _ in range(num_blocks)]
        )
        
        self.layer_norm = nn.LayerNorm(dim)
        self.classifier = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.mixer_blocks(x)
        x = self.layer_norm(x)
        x = x.mean(dim=1)
        return self.classifier(x) 
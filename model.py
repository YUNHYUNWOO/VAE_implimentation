import yaml

import torch
from torch import nn
from torch.nn import functional as F
from einops.layers.torch import Rearrange
from einops import rearrange, einsum
from torchsummary import summary

def weight_init_xavier_uniform(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.01)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)

class CBRblock(nn.Module):
    def __init__(self,  
                 in_channels: int, 
                 out_channels: int, 
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 1
                 ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.batchNorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.batchNorm(x)
        x = self.relu(x)

        return x


class UPSampler(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size=3, 
                 stride=1, 
                 padding=1,
                 output_padding=1):
        super().__init__()
        self.cbr_block = CBRblock(in_channels=in_channels, 
                                  out_channels=out_channels, 
                                  kernel_size=kernel_size, 
                                  stride=stride, 
                                  padding=padding)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear')
        x = self.cbr_block(x)
        return x
    
class DownSampler(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        kernel_size=3
        stride = 2
        padding = 1

        # self.rearrange = Rearrange(
        #     "b c (h p1) (w p2) -> b (c p1 p2) h w",
        #     p1=divider, p2=divider
        # )
        # self.compress = CBRblock(4*channels, channels, kernel_size=1, stride=1, padding=0)
        self.conv = nn.Conv2d(
                self.in_channels, self.out_channels, kernel_size=kernel_size, stride=stride, padding=padding
            )
        self.batchNorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.batchNorm(x)
        x = self.relu(x)
        return x

class ResBlock(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels,
                 layer_depth= 3,
                 dropout_rate=0.3,
                 ):
        super().__init__()

        self.block_list = [CBRblock(in_channels=in_channels, out_channels=out_channels)]
        self.block_list += [CBRblock(in_channels=out_channels, out_channels=out_channels) for _ in range(layer_depth - 1)]
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.block = nn.Sequential(*self.block_list)

        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else :
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1, stride=1,padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.block(x)
        h = self.dropout(h)
        return h + self.shortcut(x)



class Encoder(nn.Module):
    #hidden_channels must contain input channels
    def __init__(self, 
                 input_shape: torch.Tensor, 
                 hidden_channels: list, 
                 latent_shape: torch.Tensor, 
                 resBlock_depth:int = 3,
                 conv_per_resBlock:int = 3,):
        super().__init__()

        pooling_depth = len(hidden_channels) - 1
        self.pool_list = []
        for i in range(pooling_depth):
            self.pool_list.append(DownSampler(hidden_channels[i], hidden_channels[i + 1]))
            self.pool_list += [ResBlock(in_channels=hidden_channels[i+1], out_channels=hidden_channels[i+1], layer_depth = conv_per_resBlock) for _ in range(resBlock_depth)]
            
        self.pool_layer = nn.Sequential(*self.pool_list)

        flatten_shape = int(hidden_channels[-1] * (input_shape[0] / (2**pooling_depth)) * (input_shape[1] / (2**pooling_depth)))
        self.to_mean_layer = nn.Sequential(nn.Flatten(), 
                                         nn.Linear(in_features=flatten_shape, out_features=latent_shape))
        self.to_log_var_layer = nn.Sequential(nn.Flatten(), 
                                         nn.Linear(in_features=flatten_shape, out_features=latent_shape))
        
    def forward(self, x: torch.Tensor) ->tuple[torch.Tensor, torch.Tensor]:
        x = self.pool_layer(x)
        mean = self.to_mean_layer(x)
        log_var = self.to_log_var_layer(x)

        return mean, log_var

class Decoder(nn.Module):
    #hidden_channels must contain input channels
    def __init__(self, 
                 latent_shape: int, 
                 hidden_channels: list, 
                 output_shape: torch.Tensor, 
                 resBlock_depth:int = 3,
                 conv_per_resBlock:int = 3):
        super().__init__()

        unpooling_depth = len(hidden_channels) - 1
        flatten_shape = int(hidden_channels[0] * (output_shape[0] / (2**unpooling_depth)) * (output_shape[1] / (2**unpooling_depth)))
        self.to_img_layer = nn.Sequential(nn.Linear(in_features=latent_shape, out_features=flatten_shape), 
                                    #마지막을 Sigmoid로 한 것과 비교해볼 것
                                    Rearrange(
                                        "b (c h w) -> b c h w",
                                        c = hidden_channels[0], h = int(output_shape[0] / (2**unpooling_depth)), w = int(output_shape[1] / (2**unpooling_depth))
                                    ),
                                    nn.ReLU()
                                    )
        
        self.unpool_list = []
        for i in range(unpooling_depth):
            self.unpool_list += [ResBlock(in_channels=hidden_channels[i], out_channels=hidden_channels[i], layer_depth = conv_per_resBlock) for _ in range(resBlock_depth)] 
            self.unpool_list.append(UPSampler(in_channels=hidden_channels[i], out_channels=hidden_channels[i + 1]))
        self.unpool_list += [ResBlock(in_channels=hidden_channels[-1], out_channels=hidden_channels[-1], layer_depth = resBlock_depth)]
        self.unpool_layer = nn.Sequential(*self.unpool_list)

        self.to_end_point_layer = nn.Conv2d(in_channels=hidden_channels[-1], out_channels=hidden_channels[-1], kernel_size=3, stride=1, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.to_img_layer(x)
        x = self.unpool_layer(x)
        x = self.to_end_point_layer(x)
        return x

class Vanilla_VAE(nn.Module):
    def __init__(self, 
                 in_channels: int,
                 latent_shape: int,
                 hidden_channels: list,
                 input_shape: torch.Tensor,
                 resBlock_depth: int,
                 conv_per_resBlock:int,
                 recon_loss_type: str,
                 kld_weight: float) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.latent_shape = latent_shape
        self.hidden_channels = hidden_channels

        self.Encoder = Encoder(input_shape=input_shape,
                               hidden_channels=hidden_channels,
                               latent_shape=latent_shape,
                               resBlock_depth=resBlock_depth,
                               conv_per_resBlock=conv_per_resBlock)
        self.Decoder = Decoder(latent_shape=latent_shape,
                               hidden_channels=hidden_channels[::-1],
                               output_shape=input_shape,
                               resBlock_depth=resBlock_depth,
                               conv_per_resBlock=conv_per_resBlock)
        self.apply(weight_init_xavier_uniform)

        self.recon_loss_type = recon_loss_type
        self.kld_weight = kld_weight
        self.device = 'cpu'

    def forward(self, x: torch.Tensor):
        mean, log_var = self.Encoder(x)

        clipped_log_var = torch.clip(log_var, min= -10.0, max = 10.0)
        z = mean + torch.randn_like(mean) * (torch.exp(clipped_log_var / 2.0))
        # z = mean + torch.randn_like(mean) * ((torch.exp(log_var)) ** 0.5)
        sample = self.Decoder(z)

        return {'sample': sample, 
                'x': x, 
                'mean': mean, 
                'log_var': log_var}
    
    def loss_fn(self, sample: torch.Tensor, x: torch.Tensor, mean: torch.Tensor, log_var: torch.Tensor):
        # prior_matching_loss = torch.mean(0.5 * torch.sum(torch.exp(log_var) + mean ** 2 - log_var - 1, dim=1), dim=0)
        prior_matching_loss = torch.mean(0.5 * torch.mean(torch.exp(log_var) + mean ** 2 - log_var - 1, dim=1), dim=0)
        reconstruction_loss = F.binary_cross_entropy_with_logits(sample, x) if self.recon_loss_type == 'bce' else F.mse_loss(sample, x)

        loss = self.kld_weight * prior_matching_loss + reconstruction_loss
        return {'loss' : loss, 'prior_matching_loss': self.kld_weight * prior_matching_loss, 'reconstruction_loss': reconstruction_loss}
    
    #return reconstructed output
    @torch.no_grad
    def reconstruct(self, x: torch.Tensor):
        return self.forward(x)['sample']
    
    # return generate new sample from N(0, I)
    @torch.no_grad
    def sample(self, n_sample):
        z = torch.randn([n_sample, self.latent_shape], device=self.device)
        x = self.Decoder(z)
        if self.recon_loss_type == 'bce':
            x = torch.sigmoid(x)
        elif self.recon_loss_type == 'mse':
            x = torch.clamp(x, 0, 1)
        return x

    def to(self, device):
        self.device = device
        super().to(device)

if __name__ == "__main__":

    with open("./VAE/config/Vanilla_VAE_EXP5.yaml", "r") as file:
        config = yaml.safe_load(file)  # Use safe_load to prevent execution of arbitrary Python objects

    if(config['model']['name'] == 'Vanilla_VAE'):
        model = Vanilla_VAE(**config['model']['model_params'])
    summary(model, input_size=(3, 48, 48), device='cpu')
    print(model.device)
    # 중간중간에 해야겠지 싶은데
    # 막 지금 하는게 루틴이 그다지 안정되지 않으니까

    # model.to('cuda')
    # print(model.device)
    exit()

    #CBRblock test
    x = torch.randn([2, 3, 32, 32])
    cbr_temp = CBRblock(in_channels=3, out_channels=16)
    print(cbr_temp(x).shape)

    #downSample test
    x = torch.randn([2, 16, 32, 32])
    downSampler_temp = DownSampler(in_channels=16, out_channels=32)
    print(downSampler_temp(x).shape)

    #resBlock test
    x = torch.randn([2, 4, 32, 32])
    res_tmp = ResBlock(in_channels=4, out_channels=8, layer_depth=3)
    print(res_tmp(x).shape)

    #upSample test
    x = torch.randn([2, 4, 4, 4])
    upSampler_temp = UPSampler(in_channels=4, out_channels=8)
    print(upSampler_temp(x).shape)

    encoder_temp = Encoder(input_shape=[32, 32],
                        hidden_channels = [3, 8, 8, 16], 
                        latent_shape=10,
                        resBlock_depth=2,
                        conv_per_resBlock=3)
    #encoder test
    x = torch.randn([2, 3, 32, 32])
    for tensor in encoder_temp(x):
        print(tensor.shape)

    #뭔가 너무 많이 할당된다. 
    decoder_temp = Decoder(latent_shape=10,
                        hidden_channels = [16, 8, 8, 3], 
                        output_shape=[32, 32],
                        resBlock_depth=2,
                        conv_per_resBlock=3)
    #decoder test
    x = torch.randn([2, 10])
    print(decoder_temp(x).shape)
    
    summary(encoder_temp, input_size=(3, 32, 32))
    summary(decoder_temp, input_size=(10,))

    vae_temp_list = [Vanilla_VAE(in_channels=3, latent_shape=10, hidden_channels=[3, 16, 16, 16, 16], input_shape=[32, 32], resBlock_depth=3, 
                         recon_loss_type='bce', kld_weight=0.16),
                     Vanilla_VAE(in_channels=3, latent_shape=10, hidden_channels=[3, 16, 16, 16, 16], input_shape=[32, 32], resBlock_depth=3, 
                         recon_loss_type='mse', kld_weight=0.16)]
    for vae_temp in vae_temp_list:
        x = torch.rand([64, 3, 32, 32])
        x = vae_temp(x)
        print(x)
        loss = vae_temp.loss_fn(**x)
        print(loss)
        print(vae_temp.sample(2).shape)

    summary(vae_temp_list[0], input_size=(3, 32, 32) )
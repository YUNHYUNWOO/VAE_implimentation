import yaml

import torch
from torch import nn
from torch.nn import functional as F
from einops.layers.torch import Rearrange
from einops import rearrange, einsum
from torchsummary import summary

from .CodeBook import CodeBook, CodeBook_ema, CodeBook_grad, CodeBook_adap

def weight_init_xavier_uniform(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.01)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)


class ResBlock(nn.Module):
    def __init__(self,  
                 in_channels, 
                 out_channels,
                 dropout_rate=0.0,
                 ):
        super().__init__()

        self.block_1 = nn.Sequential(nn.ReLU(),
                         nn.BatchNorm2d(in_channels), 
                         nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
        
        self.block_2 = nn.Sequential(nn.ReLU(),
                         nn.BatchNorm2d(out_channels), 
                         nn.Dropout2d(p=dropout_rate),
                         nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
        
        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else :
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1, stride=1,padding=0)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        temp = self.block_1(x)
        temp = self.block_2(temp)
        x = self.shortcut(x) + temp
        return x


class UPSampler(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels):
        super().__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels, 
                              kernel_size=3, 
                              stride=1,
                              padding=1)
        

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = nn.functional.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv(x)

        return x
    
class DownSampler(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=2, padding=1
            )


    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.conv(x)

        return x



class Encoder(nn.Module):
    #hidden_channels must contain input channels
    def __init__(self, 
                 input_shape: torch.Tensor, 
                 hidden_channels: list, 
                 D: int, 
                 resBlock_depth: int = 3,
                 dropout_rate: float = 0.1):
        
        super().__init__()

        pooling_depth = len(hidden_channels) - 1
        self.pool_list = []
        for i in range(pooling_depth):
            self.pool_list.append(DownSampler(hidden_channels[i], hidden_channels[i + 1]))
            self.pool_list += [ResBlock(in_channels=hidden_channels[i+1], out_channels=hidden_channels[i+1], dropout_rate=dropout_rate) for _ in range(resBlock_depth)]
            
        self.pool_layer = nn.Sequential(*self.pool_list)

        self.to_latent = ResBlock(in_channels=hidden_channels[-1], out_channels=D, dropout_rate=0.0)

        
    def forward(self, x: torch.Tensor) ->tuple[torch.Tensor, torch.Tensor]:
        x = self.pool_layer(x)
        z = self.to_latent(x)

        return z

class Decoder(nn.Module):
    #hidden_channels must contain input channels
    def __init__(self, 
                 D: int, 
                 hidden_channels: list, 
                 output_shape: torch.Tensor, 
                 resBlock_depth:int = 3,
                 dropout_rate: float = 0.1):
        super().__init__()

        self.to_hidden = ResBlock(in_channels=D, out_channels=hidden_channels[0], dropout_rate=0.0)
        
        unpooling_depth = len(hidden_channels) - 1
        self.unpool_list = []
        for i in range(unpooling_depth):
            self.unpool_list += [ResBlock(in_channels=hidden_channels[i], out_channels=hidden_channels[i], dropout_rate=dropout_rate) for _ in range(resBlock_depth)] 
            self.unpool_list.append(UPSampler(in_channels=hidden_channels[i], out_channels=hidden_channels[i + 1]))
        self.unpool_layer = nn.Sequential(*self.unpool_list)

        self.to_end_point_layer = ResBlock(in_channels=hidden_channels[-1], out_channels=hidden_channels[-1], dropout_rate=dropout_rate)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.to_hidden(x)
        x = self.unpool_layer(x)
        x = self.to_end_point_layer(x)

        return x


class VQ_VAE(nn.Module):
    def __init__(self, 
                 in_channels: int,
                 hidden_channels: list,
                 input_shape: torch.Tensor,
                 resBlock_depth: int,
                 dropout_rate: float,
                 D: int,
                 beta: float,
                 codebook_type: str,
                 codebook_param: dict,
                 recon_loss_type: str = 'bce',
                 ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.D = D
        self.beta = beta

        self.hidden_channels = hidden_channels
        self.Encoder = Encoder(input_shape=input_shape,
                               hidden_channels=hidden_channels,
                               D=D,
                               resBlock_depth=resBlock_depth,
                               dropout_rate=dropout_rate)

        self.codebook_type = codebook_type
        if codebook_type == 'grad':
            self.CodeBook = CodeBook_grad(**codebook_param)
        elif codebook_type == 'ema':
            self.CodeBook = CodeBook_ema(**codebook_param)
        elif codebook_type == 'adap':
            self.CodeBook = CodeBook_adap(**codebook_param)

        self.Decoder = Decoder(D=D,
                               hidden_channels=hidden_channels[::-1],
                               output_shape=input_shape,
                               resBlock_depth=resBlock_depth,
                               dropout_rate=dropout_rate)
        self.apply(weight_init_xavier_uniform)

        self.recon_loss_type = recon_loss_type
        self.device = 'cpu'

    def forward(self, x: torch.Tensor):
        ## To do: change to z_flatten -> CodeBook.get_discrete_token(z_flat)
        z = self.Encoder(x)
        # print(z)
        z_discrete, indices = self.CodeBook.get_discrete_token(z)
        z_discrete_grad = z + (z_discrete - z).detach()
        sample = self.Decoder(z_discrete_grad)

        return {'sample': sample, 
                'x': x,
                'z': z,
                'z_discrete': z_discrete,
                'indices': indices,}
    
    def loss_fn(self, sample: torch.Tensor, x: torch.Tensor, z: torch.Tensor, z_discrete: torch.Tensor, indices: torch.Tensor):

        reconstruction_loss = F.binary_cross_entropy_with_logits(sample, x) if self.recon_loss_type == 'bce' else F.mse_loss(sample, x)
        encoder2codebook_loss = self.beta * F.mse_loss(z, z_discrete.detach())

        loss = reconstruction_loss + encoder2codebook_loss

        codebook_loss, codebook_log_info = self.CodeBook.accumulate(z, z_discrete, indices)
        if isinstance(codebook_loss, torch.Tensor):
            loss += codebook_loss

        if self.training:
            self.CodeBook.update()


        log_info = {'loss' : loss.item(), 
                'reconstruction_loss': reconstruction_loss.item(), 
                'encoder2codebook_loss': encoder2codebook_loss.item(),
                **codebook_log_info}
        
        return loss, log_info
    
    #return reconstructed output
    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor):
        return self.forward(x)['sample']
    
    # return generate new sample from N(0, I)
    @torch.no_grad()
    def sample(self, base_samples):
        x = self.forward(base_samples)['sample']
        if self.recon_loss_type == 'bce':
            x = torch.sigmoid(x)
        elif self.recon_loss_type == 'mse':
            x = torch.clamp(x, 0, 1)
        return x

    def to(self, device):
        self.device = device
        super().to(device)

if __name__ == "__main__":

    with open("./config/VQ_VAE_base_fixed.yaml", "r") as file:
        config = yaml.safe_load(file)  # Use safe_load to prevent execution of arbitrary Python objects

    print(config)
    if(config['model']['name'] == 'VQ_VAE'):
        model_grad = VQ_VAE(**config['model']['model_params'])
    model_grad.to('cuda')
    summary(model_grad, input_size=(3, 48, 48), device='cuda')

    with open("./config/VQ_VAE_ema_fixed.yaml", "r") as file:
        config = yaml.safe_load(file)  # Use safe_load to prevent execution of arbitrary Python objects

    print(config)
    if(config['model']['name'] == 'VQ_VAE'):
        model_ema = VQ_VAE(**config['model']['model_params'])
    model_ema.to('cuda')
    summary(model_ema, input_size=(3, 48, 48), device='cuda')

    with open("./config/VQ_VAE_adap_fixed.yaml", "r") as file:
        config = yaml.safe_load(file)  # Use safe_load to prevent execution of arbitrary Python objects

    print(config)
    if(config['model']['name'] == 'VQ_VAE'):
        model_adap = VQ_VAE(**config['model']['model_params'])
    model_adap.to('cuda')
    summary(model_adap, input_size=(3, 48, 48), device='cuda')

    #ResBlock test
    x = torch.randn([2, 3, 32, 32])
    res_block = ResBlock(in_channels=3, out_channels=16)
    print(f'resBlock: should be (2, 16, 32, 32)')
    print(res_block(x).shape)

    #downSample test
    x = torch.randn([2, 16, 32, 32])
    downSampler = DownSampler(in_channels=16, out_channels=32)
    print(f'downSamler: should be (2, 32, 16, 16)')
    print(downSampler(x).shape)

    #upSample test
    x = torch.randn([2, 4, 4, 4])
    upSampler = UPSampler(in_channels=4, out_channels=8)
    print(f'upSamler: should be (2, 8, 8, 8)')
    print(upSampler(x).shape)

    encoder = Encoder(input_shape=[32, 32],
                        hidden_channels = [3, 8, 8, 16], 
                        D=32,
                        resBlock_depth=2)
    #encoder test
    x = torch.randn([2, 3, 32, 32])
    print(f'Encoder: should be (2, 32, 4, 4)')
    print(encoder(x).shape)

    #뭔가 너무 많이 할당된다. 
    decoder = Decoder(D=32,
                        hidden_channels = [16, 8, 8, 3], 
                        output_shape=[32, 32],
                        resBlock_depth=2)
    # #decoder test
    x = torch.randn([2, 32, 4, 4])
    print(f'Decoder: should be (2, 3, 32, 32)')
    print(decoder(x).shape)


    codebook = CodeBook_grad(K=256, D=32)
    x = torch.randn([2, 32, 4, 4])
    print(f'codebook: should be (2, 32, 4, 4)')
    print(codebook.get_discrete_token(x)[0].shape)

    def test_st_sends_grad_to_encoder(vqvae):
        vqvae.train()
        x = torch.randn(2, vqvae.in_channels, 48, 48, device=vqvae.device)

        out = vqvae(x)
        loss, log_info = vqvae.loss_fn(**out)

        vqvae.zero_grad(set_to_none=True)
        loss.backward()

        # encoder params 중 하나라도 grad가 있어야 함
        enc_grads = [p.grad for p in vqvae.Encoder.parameters() if p.requires_grad]
        assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in enc_grads)

    test_st_sends_grad_to_encoder(model_grad)

    def test_grad_mode_codebook_gets_grad(vqvae_grad):
        vqvae_grad.train()
        x = torch.randn(2, vqvae_grad.in_channels, 48, 48, device=vqvae_grad.device)

        out = vqvae_grad(x)
        loss, log_dict = vqvae_grad.loss_fn(**out)

        vqvae_grad.zero_grad(set_to_none=True)
        loss.backward()

        g = vqvae_grad.CodeBook.embedding.weight.grad
        assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
    
    test_grad_mode_codebook_gets_grad(model_grad)


    def test_ema_codebook_has_no_grad(vqvae_ema):
        vqvae_ema.train()
        x = torch.randn(2, vqvae_ema.in_channels, 48, 48, device=vqvae_ema.device)

        out = vqvae_ema(x)
        loss, log_dict = vqvae_ema.loss_fn(**out)

        vqvae_ema.zero_grad(set_to_none=True)
        loss.backward()

        assert vqvae_ema.CodeBook.embedding.weight.grad is None
    test_ema_codebook_has_no_grad(model_ema)

    def test_ema_updates_buffers_and_weights(vqvae_ema):
        vqvae_ema.train()
        x = torch.randn(2, vqvae_ema.in_channels, 48, 48, device=vqvae_ema.device)

        # 업데이트 전 스냅샷
        cs0 = vqvae_ema.CodeBook.cluster_size.detach().clone()
        es0 = vqvae_ema.CodeBook.embed_sum.detach().clone()
        w0  = vqvae_ema.CodeBook.embedding.weight.detach().clone()

        out = vqvae_ema(x)
        _ = vqvae_ema.loss_fn(**out)

        cs1 = vqvae_ema.CodeBook.cluster_size.detach()
        es1 = vqvae_ema.CodeBook.embed_sum.detach()
        w1  = vqvae_ema.CodeBook.embedding.weight.detach()

        assert not torch.allclose(cs0, cs1), "cluster_size not updated"
        assert not torch.allclose(es0, es1), "embed_sum not updated"
        assert not torch.allclose(w0,  w1),  "embedding weight not updated"
    test_ema_updates_buffers_and_weights(model_ema)

    # summary(encoder_temp, input_size=(3, 32, 32))
    # summary(decoder_temp, input_size=(10,))

    # vae_temp_list = [Vanilla_VAE(in_channels=3, latent_shape=10, hidden_channels=[3, 16, 16, 16, 16], input_shape=[32, 32], resBlock_depth=3, 
    #                      recon_loss_type='bce', kld_weight=0.16),
    #                  Vanilla_VAE(in_channels=3, latent_shape=10, hidden_channels=[3, 16, 16, 16, 16], input_shape=[32, 32], resBlock_depth=3, 
    #                      recon_loss_type='mse', kld_weight=0.16)]
    # for vae_temp in vae_temp_list:
    #     x = torch.rand([64, 3, 32, 32])
    #     x = vae_temp(x)
    #     print(x)
    #     loss = vae_temp.loss_fn(**x)
    #     print(loss)
    #     print(vae_temp.sample(2).shape)

    # summary(vae_temp_list[0], input_size=(3, 32, 32) )
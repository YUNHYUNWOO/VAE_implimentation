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

class CodeBook(nn.Module):
    def __init__(self, K: int, D: int):
        super().__init__()

        self.K = K
        self.D = D
        
        self.embedding = nn.Embedding(self.K, self.D)
        self.embedding.weight.data.uniform_(-1/self.K, 1/self.K)

    def get_discrete_token(self, z: torch.Tensor):
        H, W = z.shape[-2:]

        # print(z[0, :3, :3, :3])
        flat_z = Rearrange("b c h w -> b (h w) c")(z)

        sq_z = (flat_z ** 2).sum(dim=2, keepdim=True)
        # print('sq_z', sq_z.shape)
        sq_e = (self.embedding.weight ** 2).sum(dim=1).unsqueeze(0).unsqueeze(0)
        
        # print(flat_z.shape)
        # print(self.embedding.weight.t().shape)
        d = (sq_z + sq_e) - 2.0 * torch.matmul(flat_z, self.embedding.weight.t()) 
        indices = torch.argmin(d, dim=2)
        # print(indices.shape)
        z_discrete = self.embedding.weight[indices]
        # print(z_discrete.shape)

        z_discrete = Rearrange("b (h w) c -> b c h w",
                            h = H,
                            w = W)(z_discrete)
        indices = Rearrange("b (h w) -> b h w",
                            h = H,
                            w = W)(indices)
        # print(z_discrete.shape)
        # print(z_discrete[0, :3, :3, :3])
        
        return z_discrete, indices

class VQ_VAE(nn.Module):
    def __init__(self, 
                 in_channels: int,
                 hidden_channels: list,
                 input_shape: torch.Tensor,
                 resBlock_depth: int,
                 dropout_rate: float,
                 K: int,
                 D: int,
                 beta: float,
                 codebook_update_type: str,
                 codebook_update_ratio: float | None,
                 recon_loss_type: str = 'bce',
                 ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.K = K
        self.D = D
        self.beta = beta

        self.hidden_channels = hidden_channels
        self.Encoder = Encoder(input_shape=input_shape,
                               hidden_channels=hidden_channels,
                               D=D,
                               resBlock_depth=resBlock_depth,
                               dropout_rate=dropout_rate)

        self.CodeBook = CodeBook(K=K, D=D)
        self.codebook_update_type = codebook_update_type
        self.codebook_update_ratio = codebook_update_ratio
        if codebook_update_type == 'ema':
            self.register_buffer("cluster_size", torch.zeros(K))
            self.register_buffer("embed_sum", torch.zeros(K, D))
            self.CodeBook.embedding.weight.requires_grad_(False)

        self.Decoder = Decoder(D=D,
                               hidden_channels=hidden_channels[::-1],
                               output_shape=input_shape,
                               resBlock_depth=resBlock_depth,
                               dropout_rate=dropout_rate)
        self.apply(weight_init_xavier_uniform)

        self.recon_loss_type = recon_loss_type
        self.device = 'cpu'

    def forward(self, x: torch.Tensor):
        z = self.Encoder(x)
        z_discrete, indices = self.CodeBook.get_discrete_token(z)
        z_discrete_grad = z + (z_discrete - z).detach()
        sample = self.Decoder(z_discrete_grad)

        return {'sample': sample, 
                'x': x,
                'z': z,
                'z_discrete': z_discrete,
                'indices': indices}
    
    def loss_fn(self, sample: torch.Tensor, x: torch.Tensor, z: torch.Tensor, z_discrete: torch.Tensor, indices: torch.Tensor):

        if self.codebook_update_type == 'grad':
            reconstruction_loss = F.binary_cross_entropy_with_logits(sample, x) if self.recon_loss_type == 'bce' else F.mse_loss(sample, x)
            codebook2encoder_loss = F.mse_loss(z.detach(), z_discrete)
            encoder2codebook_loss = self.beta * F.mse_loss(z, z_discrete.detach())

            loss = reconstruction_loss + codebook2encoder_loss + encoder2codebook_loss

            return {'loss' : loss, 
                    'reconstruction_loss': reconstruction_loss, 
                    'codebook2encoder_loss': codebook2encoder_loss, 
                    'encoder2codebook_loss': encoder2codebook_loss}
        elif self.codebook_update_type == 'ema':

            reconstruction_loss = F.binary_cross_entropy_with_logits(sample, x) if self.recon_loss_type == 'bce' else F.mse_loss(sample, x)
            encoder2codebook_loss = self.beta * F.mse_loss(z, z_discrete.detach())

            loss = reconstruction_loss + encoder2codebook_loss

            with torch.no_grad():
                z_flat = Rearrange("b c h w -> (b h w) c")(z.detach())
                indices_flat = Rearrange("b h w -> (b h w)")(indices.detach())

                # n_cluster_size: (K)
                n_cluster_size = torch.sum(F.one_hot(indices_flat, num_classes=self.K), dim=0)
                indices_unique = torch.unique(indices_flat)
                # z_sum: unique_indices x D
                z_sum = torch.stack([torch.sum(z_flat[i == indices_flat], dim=0) for i in indices_unique])

                # n_embed_sum: K x D
                n_embed_sum = torch.zeros_like(self.embed_sum)
                n_embed_sum[indices_unique] = z_sum

                # ema update
                self.cluster_size.mul_(self.codebook_update_ratio).add_(n_cluster_size, alpha=(1 - self.codebook_update_ratio))
                self.embed_sum.mul_(self.codebook_update_ratio).add_(n_embed_sum, alpha=(1 - self.codebook_update_ratio))

                # exploding correction
                eps = 1e-5
                n = self.cluster_size.sum()
                cluster_size_safe = (self.cluster_size + eps) / (n + self.K * eps) * n
                self.CodeBook.embedding.weight.copy_(self.embed_sum / cluster_size_safe.unsqueeze(1))

            return {'loss' : loss, 
                    'reconstruction_loss': reconstruction_loss, 
                    'encoder2codebook_loss': encoder2codebook_loss}
    
    #return reconstructed output
    @torch.no_grad
    def reconstruct(self, x: torch.Tensor):
        return self.forward(x)['sample']
    
    # return generate new sample from N(0, I)
    @torch.no_grad
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

    with open("./config/VQ_VAE_base.yaml", "r") as file:
        config = yaml.safe_load(file)  # Use safe_load to prevent execution of arbitrary Python objects

    print(config)
    if(config['model']['name'] == 'VQ_VAE'):
        model = VQ_VAE(**config['model']['model_params'])
    model.to('cuda')
    summary(model, input_size=(3, 48, 48), device='cuda')

    with open("./config/VQ_VAE_ema.yaml", "r") as file:
        config = yaml.safe_load(file)  # Use safe_load to prevent execution of arbitrary Python objects

    print(config)
    if(config['model']['name'] == 'VQ_VAE'):
        model_ema = VQ_VAE(**config['model']['model_params'])
    model_ema.to('cuda')
    summary(model_ema, input_size=(3, 48, 48), device='cuda')

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


    codebook = CodeBook(K=256, D=32)
    x = torch.randn([2, 32, 4, 4])
    print(f'codebook: should be (2, 32, 4, 4)')
    print(codebook.get_discrete_token(x)[0].shape)

    def test_st_sends_grad_to_encoder(vqvae):
        vqvae.train()
        x = torch.randn(2, vqvae.in_channels, 48, 48, device=vqvae.device)

        out = vqvae(x)
        loss_dict = vqvae.loss_fn(**out)
        loss = loss_dict["loss"]

        vqvae.zero_grad(set_to_none=True)
        loss.backward()

        # encoder params 중 하나라도 grad가 있어야 함
        enc_grads = [p.grad for p in vqvae.Encoder.parameters() if p.requires_grad]
        assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in enc_grads)

    test_st_sends_grad_to_encoder(model)

    def test_grad_mode_codebook_gets_grad(vqvae_grad):
        vqvae_grad.train()
        x = torch.randn(2, vqvae_grad.in_channels, 48, 48, device=vqvae_grad.device)

        out = vqvae_grad(x)
        loss_dict = vqvae_grad.loss_fn(**out)
        loss = loss_dict["loss"]

        vqvae_grad.zero_grad(set_to_none=True)
        loss.backward()

        g = vqvae_grad.CodeBook.embedding.weight.grad
        assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
    
    test_grad_mode_codebook_gets_grad(model)


    def test_ema_codebook_has_no_grad(vqvae_ema):
        vqvae_ema.train()
        x = torch.randn(2, vqvae_ema.in_channels, 48, 48, device=vqvae_ema.device)

        out = vqvae_ema(x)
        loss_dict = vqvae_ema.loss_fn(**out)
        loss = loss_dict["loss"]

        vqvae_ema.zero_grad(set_to_none=True)
        loss.backward()

        assert vqvae_ema.CodeBook.embedding.weight.grad is None
    test_ema_codebook_has_no_grad(model_ema)

    def test_ema_updates_buffers_and_weights(vqvae_ema):
        vqvae_ema.train()
        x = torch.randn(2, vqvae_ema.in_channels, 48, 48, device=vqvae_ema.device)

        # 업데이트 전 스냅샷
        cs0 = vqvae_ema.cluster_size.detach().clone()
        es0 = vqvae_ema.embed_sum.detach().clone()
        w0  = vqvae_ema.CodeBook.embedding.weight.detach().clone()

        out = vqvae_ema(x)
        _ = vqvae_ema.loss_fn(**out)

        cs1 = vqvae_ema.cluster_size.detach()
        es1 = vqvae_ema.embed_sum.detach()
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
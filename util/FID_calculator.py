import numpy as np
import torch
from pytorch_fid.inception import InceptionV3
from pytorch_fid.fid_score import calculate_frechet_distance
from torch.utils.data import Dataset, DataLoader, TensorDataset
from tqdm import tqdm
from contextlib import contextmanager

## Base Dataset에 대해 미리 계산을 해놓음
## 그 이후 비교하려는 Dataset을 
class FID_calculator():
    def __init__(self, base_dataset: Dataset=None, batch_size=128, device='cuda'):
        if not torch.cuda.is_available():
            device = 'cpu'
        self.device = device

        if Dataset == None:
            self.param = None
            return
        self.init_fid_param(base_dataset, batch_size)
        return
    
    def init_fid_param(self, base_dataset: Dataset, batch_size=128):
        mu, sigma = self.calc_fid_param(base_dataset, batch_size)
        self.param = {'mu': mu, 'sigma':sigma}
        return
    
    def calc_fid_param(self, dataset, batch_size=128) -> dict[str, torch.Tensor]:
        
        with torch.device(self.device):
            model = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]], normalize_input=True)
            dataloader = DataLoader(dataset, batch_size=batch_size)

            with torch.no_grad():
                hiddens = []
                for x, *_ in tqdm(dataloader, desc='calculating fid_param'):
                    #print(x.shape)
                    x = x.to(self.device)
                    # model(x)[0] = [ torch.Tensor with batch_size x 2048 x 1 x 1 ]
                    out = model(x)[0].squeeze()
                    hiddens.append(out)

        hiddens = torch.cat(hiddens, dim=0).cpu().numpy()

        mu = np.mean(hiddens, axis=0)
        sigma = np.cov(hiddens, rowvar=False)

        return mu, sigma

    def calc_fid_score(self, dataset, batch_size=64):
        # returns fid score of base_dataset and param_dataset
        mu, sigma = self.calc_fid_param(dataset, batch_size)
        fid_score = calculate_frechet_distance(self.param['mu'], self.param['sigma'], mu, sigma)
        return fid_score

# if __name__ == '__main__':
#     download_celeba()
#     celeba_data = get_celeba_data({'data_path': "VAE/data/celeba-small-images-dataset",
#                                   'train_batch_size': 64,
#                                   'val_batch_size':  64,
#                                   # should match with model's end_point
#                                   'end_point':[0, 1], 
#                                   'patch_size': 48,
#                                   'num_workers': 4,
#                                   'pin_memory': True})

#     fid_calculator = FID_calculator(celeba_data['val']['dataset'], batch_size=64, device='cuda')
#     print(fid_calculator.param)
#     print(type(fid_calculator.param['mu']))
#     print(fid_calculator.param['mu'].shape)


    
import yaml

import torch
from torch import nn
from torch.nn import functional as F
from einops.layers.torch import Rearrange
from einops import rearrange, einsum
from torchsummary import summary


def search_closest_2d(z: torch.Tensor, embedding: torch.Tensor):
    H, W = z.shape[-2:]

    # print(z[0, :3, :3, :3])
    flat_z = rearrange(z, "b c h w -> b (h w) c")

    sq_z = (flat_z ** 2).sum(dim=2, keepdim=True)
    # print('sq_z', sq_z.shape)
    sq_e = (embedding ** 2).sum(dim=1).unsqueeze(0).unsqueeze(0)
    
    # print(flat_z.shape)
    # print(self.embedding.weight.t().shape)
    d_sq = (sq_z + sq_e) - 2.0 * torch.matmul(flat_z, embedding.t()) 
    d_sq, indices = torch.min(d_sq, dim=2)
    # print(indices.shape)
    z_discrete = embedding[indices]
    # print(z_discrete.shape)

    z_discrete = rearrange(z_discrete, 
                        "b (h w) c -> b c h w",
                        h = H,
                        w = W)
    indices = rearrange(indices,
                        "b (h w) -> b h w",
                        h = H,
                        w = W)

    # print(z_discrete.shape)
    # print(z_discrete[0, :3, :3, :3])
    
    return z_discrete, indices

class CodeBook(nn.Module):
    def __init__(self):
        super().__init__()
    
    def get_discrete_token(self, z: torch.Tensor, embedding: torch.Tensor):
        raise NotImplementedError

    def accumulate(self, z: torch.Tensor, z_discrete:torch.Tensor, indices: torch.Tensor):
        '''
        Docstring for accumulate
        calcuate and keep next change.
        call CodeBook.update() to apply changes

        returns loss(if update is not based on gradient: None), log info: dict
        '''
        raise NotImplementedError
    
    def update(self):
        '''
        Docstring for update
        apply changes

        returns None
        '''

        raise NotImplementedError
    

class CodeBook_grad(CodeBook):
    def __init__(self, K: int, D: int):
        super().__init__()

        self.K = K
        self.D = D

        self.embedding = nn.Embedding(self.K, self.D)
        self.embedding.weight.data.uniform_(-1/self.K, 1/self.K)


    def get_discrete_token(self, z: torch.Tensor):
        z_discrete, indices = search_closest_2d(z, self.embedding.weight)
        return z_discrete, indices
    
    def accumulate(self, z: torch.Tensor, z_discrete: torch.Tensor, indices: torch.Tensor):
        codebook2encoder_loss = F.mse_loss(z.detach(), z_discrete)
        return codebook2encoder_loss, {'codebook2encoder_loss': codebook2encoder_loss.item()}

    def update(self):
        pass

class CodeBook_ema(CodeBook):
    def __init__(self, K: int, D: int, update_ratio: float):
        super().__init__()

        self.K = K
        self.D = D
        self.update_ratio = update_ratio

        self.embedding = nn.Embedding(self.K, self.D)
        self.embedding.weight.data.uniform_(-1/self.K, 1/self.K)
        self.embedding.weight.requires_grad_(False)

        self.register_buffer("cluster_size", torch.ones(K))
        self.register_buffer("embed_sum", self.embedding.weight.clone())

        self.register_buffer("cluster_size_buffer", torch.zeros(K))
        self.register_buffer("embed_sum_buffer", torch.zeros(K, D))
        self.register_buffer("embedding_buffer", torch.zeros(K, D))

    def get_discrete_token(self, z: torch.Tensor):
        z_discrete, indices = search_closest_2d(z, self.embedding.weight)
        return z_discrete, indices
    
    def accumulate(self, z: torch.Tensor, z_discrete: torch.Tensor, indices: torch.Tensor):

        # indices는 Valid한 embedding들 안에서만 평가되는 값
        # 즉, max가 K - 1
    
        z_flat = rearrange(z.detach(), "b c h w -> (b h w) c")
        indices_flat = rearrange(indices.detach(), "b h w -> (b h w)")

        # n_cluster_size: (K)
        n_cluster_size = torch.sum(F.one_hot(indices_flat, num_classes=self.K), dim=0)
        indices_unique = torch.unique(indices_flat)
        # z_sum: unique_indices x D
        z_sum = torch.stack([torch.sum(z_flat[i == indices_flat], dim=0) for i in indices_unique])

        # n_embed_sum: K x D
        n_embed_sum = torch.zeros_like(self.embed_sum)
        n_embed_sum[indices_unique] = z_sum

        # ema update
        self.cluster_size_buffer = (self.update_ratio) * self.cluster_size + (1 - self.update_ratio) * n_cluster_size
        self.embed_sum_buffer = (self.update_ratio) * self.embed_sum + (1 - self.update_ratio) * n_embed_sum

        # exploding correction
        eps = 1e-5
        n = self.cluster_size_buffer.sum()
        cluster_size_safe = (self.cluster_size_buffer + eps) / (n + self.K * eps) * n
        self.embedding_buffer = self.embed_sum_buffer / cluster_size_safe.unsqueeze(1)
        
        return None, {}
    
    @torch.no_grad()
    def update(self):
        # ema update
        self.cluster_size.copy_(self.cluster_size_buffer)
        self.embed_sum.copy_(self.embed_sum_buffer)
        self.embedding.weight.copy_(self.embedding_buffer)

class CodeBook_adap(nn.Module):
    def __init__(self, K: int, D: int, 
                 update_ratio: float, 
                 remove_threshold: float, 
                 split_threshold: float,
                 refine_step: int):
        super().__init__()

        self.K = K
        self.D = D
        self.capacity = K
        self.update_ratio = update_ratio
        self.remove_threshold = remove_threshold
        self.split_threshold = split_threshold
        self.refine_step = refine_step
        self.step = 0

        self.embedding = nn.Embedding(self.K, self.D)
        self.embedding.weight.requires_grad_(False)
        self.embedding.weight.uniform_(-1/self.K, 1/self.K)

        self.register_buffer("cluster_size", torch.ones(K))
        self.register_buffer("embed_sum", self.embedding.weight.clone())
        self.register_buffer("d_sq", torch.zeros(K))

        self.register_buffer("cluster_size_buffer", torch.zeros(K))
        self.register_buffer("embed_sum_buffer", torch.zeros(K, D))
        self.register_buffer("d_sq_buffer", torch.zeros(K))
        self.register_buffer("embedding_buffer", torch.zeros(K, D))

    def get_discrete_token(self, z: torch.Tensor):
        z_discrete, indices = search_closest_2d(z, self.embedding.weight[:self.K])
        return z_discrete, indices


    @torch.no_grad()
    def accumulate(self, z: torch.Tensor,  z_discrete: torch.Tensor, indices: torch.Tensor):
        self.assert_invariants()

        # indices는 Valid한 embedding들 안에서만 평가되는 값
        # 즉, max가 K - 1
        
        d_sq = torch.sum((z - z_discrete)**2, dim=1)

        z_flat = rearrange(z.detach(), "b c h w -> (b h w) c")
        indices_flat = rearrange(indices.detach(), "b h w -> (b h w)")
        d_sq_flat = rearrange(d_sq.detach(), "b h w -> (b h w)")

        # n_cluster_size: (K)
        n_cluster_size = torch.sum(F.one_hot(indices_flat, num_classes=self.K), dim=0)
        indices_unique = torch.unique(indices_flat)
        # z_sum: unique_indices x D
        z_sum = torch.stack([torch.sum(z_flat[i == indices_flat], dim=0) for i in indices_unique])
        d_sq_sum = torch.stack([torch.sum(d_sq_flat[i == indices_flat], dim=0) for i in indices_unique])
 
        # n_embed_sum: K x D

        n_embed_sum = torch.zeros_like(self.embed_sum[:self.K])
        n_embed_sum[indices_unique] = z_sum

        n_d_sq = torch.zeros_like(self.d_sq[:self.K])
        n_d_sq[indices_unique] = d_sq_sum

        # ema update

        self.cluster_size_buffer[:self.K] = (self.update_ratio) * self.cluster_size[:self.K] + (1 - self.update_ratio) * n_cluster_size
        self.embed_sum_buffer[:self.K] = (self.update_ratio) * self.embed_sum[:self.K] + (1 - self.update_ratio) * n_embed_sum
        self.d_sq_buffer[:self.K] = (self.update_ratio) * self.d_sq[:self.K] + (1 - self.update_ratio) * n_d_sq
        if self.K < self.capacity:
            self.cluster_size_buffer[self.K:] = 0
            self.embed_sum_buffer[self.K:] = 0
            self.d_sq_buffer[self.K:] = 0

        # exploding correction
        eps = 1e-5
        n = self.cluster_size_buffer[:self.K].sum()
        cluster_size_safe = (self.cluster_size_buffer[:self.K] + eps) / (n + self.K * eps) * n
        self.embedding_buffer[:self.K] = self.embed_sum_buffer[:self.K] / cluster_size_safe.unsqueeze(1)
        if self.K < self.capacity:
            self.embedding_buffer[self.K:] = 0

        self.assert_invariants()

        log_info = {'K': self.K,
                    'sum_cluster_size': self.cluster_size_buffer[:self.K].sum(),
                    'capacity': self.capacity}
        
        return None, log_info

    @torch.no_grad()
    def update(self):
        self.cluster_size.copy_(self.cluster_size_buffer)
        self.embed_sum.copy_(self.embed_sum_buffer)
        self.d_sq.copy_(self.d_sq_buffer)
        self.embedding.weight.copy_(self.embedding_buffer)


        self.step += 1
        if self.step % self.refine_step == 0:
            self.step %= self.refine_step
            self.refine_codebook()

    @torch.no_grad()
    def refine_codebook(self):
        self.assert_invariants()

        device = self.embedding.weight.device
        valid_mask = torch.arange(self.capacity, device=device) < self.K
        
        remove_masks = (self.cluster_size < self.remove_threshold) & valid_mask
        self.remove(remove_masks)

        valid_mask = torch.arange(self.capacity, device=device) < self.K
        eps = 1e-5
        n = self.cluster_size[:self.K].sum()
        cluster_size_safe = (self.cluster_size[:self.K] + eps) / (n + self.K * eps) * n
        split_masks = torch.zeros(self.capacity, dtype=torch.bool, device=device)
        split_masks[:self.K] = (self.d_sq[:self.K] / cluster_size_safe > self.split_threshold)

        # std를 계산해서 Sampling하려 했으나, embedding 자체의 L2 norm이 너무 커서, std가 너무 커짐. 이걸 도입하려면 L2 정규화를 추가해야할듯
        new_embedding = self.embedding.weight[split_masks] + eps * torch.randn(size=(split_masks.sum().item(), self.D), device=self.embedding.weight.device)
        self.embedding.weight[split_masks] += eps * torch.randn(size=(split_masks.sum().item(), self.D), device=self.embedding.weight.device) 

        # 이론적으로 cluster_size는 반으로 나누어가짐
        new_cluster_size = (self.cluster_size[split_masks] / 2.0).clone()
        self.cluster_size[split_masks] /= 2.0
        # embed_sum또한 마찬가지로 나누어가짐
        new_embed_sum = (self.embed_sum[split_masks] / 2.0).clone()
        self.embed_sum[split_masks] /= 2.0
        # d_sq는 제곱승으로 나누어짐
        new_d_sq = (self.d_sq[split_masks] / 2.0).clone()
        self.d_sq[split_masks] /= 2.0

        self.insert(new_embedding, new_cluster_size, new_embed_sum, new_d_sq)
        self.assert_invariants()


    @torch.no_grad()
    def insert(self, new_embedding: torch.Tensor, new_cluster_size: torch.Tensor, new_embed_sum: torch.Tensor, new_d_sq: torch.Tensor):
        self.assert_invariants()

        num_to_save = new_embedding.shape[0]
        if num_to_save <= self.capacity - self.K:
            # empty index들이 나옴

            self.embedding.weight[self.K:self.K + num_to_save] = new_embedding
            self.cluster_size[self.K:self.K + num_to_save] = new_cluster_size
            self.embed_sum[self.K:self.K + num_to_save] = new_embed_sum
            self.d_sq[self.K:self.K + num_to_save] = new_d_sq

        else :
            extended_embedding = torch.cat((self.embedding.weight, torch.zeros_like(self.embedding.weight)))
            self.embedding.weight = torch.nn.Parameter(extended_embedding)
            self.embedding.weight.requires_grad_(False)

            self.cluster_size = torch.cat((self.cluster_size, torch.zeros_like(self.cluster_size)))
            self.embed_sum = torch.cat((self.embed_sum, torch.zeros_like(self.embed_sum)))
            self.d_sq = torch.cat((self.d_sq, torch.zeros_like(self.d_sq)))

            self.embedding.weight[self.K:self.K + num_to_save] = new_embedding
            self.cluster_size[self.K:self.K + num_to_save] = new_cluster_size
            self.embed_sum[self.K:self.K + num_to_save] = new_embed_sum
            self.d_sq[self.K:self.K + num_to_save] = new_d_sq

            # Buffer reassign
            self.cluster_size_buffer = torch.zeros_like(self.cluster_size)
            self.embed_sum_buffer = torch.zeros_like(self.embed_sum)
            self.d_sq_buffer = torch.zeros_like(self.d_sq)
            self.embedding_buffer = torch.zeros_like(self.embedding.weight)

            self.capacity *= 2

        self.K += num_to_save
        self.assert_invariants()


    @torch.no_grad()
    def remove(self, remove_indices):

        self.assert_invariants()

        device = self.embedding.weight.device
        valid_mask = torch.arange(self.capacity, device=device) < self.K
        valid_mask[remove_indices] = False
        
        self.K = int(torch.sum(valid_mask).item())
        #packing
        self.embedding.weight[:self.K] = self.embedding.weight[valid_mask].clone()
        self.cluster_size[:self.K] = self.cluster_size[valid_mask].clone()
        self.embed_sum[:self.K] = self.embed_sum[valid_mask].clone()
        self.d_sq[:self.K] = self.d_sq[valid_mask].clone()

        device = self.embedding.weight.device
        if self.K <= self.capacity // 4:
            new_capacity = max(1, self.capacity // 2)

            # to-do
            # Copying twice causes inefficiency
            self.embedding.weight[:self.K] = self.embedding.weight[:self.K].clone()
            self.embedding.weight = nn.Parameter(self.embedding.weight[:new_capacity].clone())
            self.embedding.weight.requires_grad_(False)

            self.cluster_size[:self.K] = self.cluster_size[:self.K].clone()
            self.cluster_size = self.cluster_size[:new_capacity].clone()

            self.embed_sum[:self.K] = self.embed_sum[:self.K].clone()
            self.embed_sum = self.embed_sum[:new_capacity].clone()

            self.d_sq[:self.K] = self.d_sq[:self.K].clone()
            self.d_sq = self.d_sq[:new_capacity].clone()

            # Buffer reassign
            self.cluster_size_buffer = torch.zeros_like(self.cluster_size)
            self.embed_sum_buffer = torch.zeros_like(self.embed_sum)
            self.d_sq_buffer = torch.zeros_like(self.d_sq)
            self.embedding_buffer = torch.zeros_like(self.embedding.weight)

            self.capacity = new_capacity


        self.assert_invariants()

    def assert_invariants(self):
        assert self.K <= self.capacity
        assert self.embedding.weight.shape[0] == self.capacity
        assert self.cluster_size.shape[0] == self.capacity
        assert self.embed_sum.shape[0] == self.capacity
        assert self.d_sq.shape[0] == self.capacity
        assert self.embedding.weight.requires_grad is False


if __name__ == '__main__' :

    def assert_invariants(cb):
        assert cb.capacity == cb.embedding.weight.shape[0], "capacity != embedding rows"
        assert cb.capacity == cb.cluster_size.shape[0], "capacity != cluster_size len"
        assert cb.capacity == cb.embed_sum.shape[0], "capacity != embed_sum len"
        assert cb.capacity == cb.d_sq.shape[0], "capacity != d_sq len"
        # embedding weight grad off
        assert cb.embedding.weight.requires_grad is False, "embedding.weight should be requires_grad=False"

    def make_payload(n, D, device="cpu", dtype=torch.float32):
        new_embedding = torch.randn(n, D, device=device, dtype=dtype)
        new_cluster_size = torch.arange(1, n + 1, device=device, dtype=dtype)  # unique values
        new_embed_sum = torch.randn(n, D, device=device, dtype=dtype)
        new_d_sq = torch.linspace(0.1, 1.0, steps=n, device=device, dtype=dtype)
        return new_embedding, new_cluster_size, new_embed_sum, new_d_sq

    def test_basic_insert():
        print("=== test_basic_insert ===")
        cb = CodeBook_adap(K=4, D=3, update_ratio=0.9, remove_threshold=0.1, split_threshold=1.0, refine_step=10)
        assert_invariants(cb)

        # Make one empty slot by removing one
        remove_mask = torch.zeros(cb.capacity, dtype=torch.bool)
        remove_mask[1] = True
        cb.remove(remove_mask)
        assert_invariants(cb)
        assert cb.K == 3

        # Insert 1 item -> should fill the empty slot, capacity unchanged
        payload = make_payload(1, cb.D)
        cb.insert(*payload)
        assert_invariants(cb)
        assert cb.capacity == 4
        assert cb.K == 4

        # Check that the inserted values actually landed at the new tail index
        inserted_idx = 3
        new_embedding, new_cluster_size, new_embed_sum, new_d_sq = payload
        assert torch.allclose(cb.embedding.weight[inserted_idx], new_embedding[0]), "embedding not inserted correctly"
        assert torch.allclose(cb.cluster_size[inserted_idx], new_cluster_size[0]), "cluster_size not inserted correctly"
        assert torch.allclose(cb.embed_sum[inserted_idx], new_embed_sum[0]), "embed_sum not inserted correctly"
        assert torch.allclose(cb.d_sq[inserted_idx], new_d_sq[0]), "d_sq not inserted correctly"

        print("PASS\n")

    test_basic_insert()

    def test_overflow_insert_expand():
        print("=== test_overflow_insert_expand ===")
        cb = CodeBook_adap(K=4, D=3, update_ratio=0.9, remove_threshold=0.1, split_threshold=1.0, refine_step=10)
        assert_invariants(cb)

        # Make two empty slots
        remove_mask = torch.zeros(cb.capacity, dtype=torch.bool)
        remove_mask[1] = True
        remove_mask[3] = True
        cb.remove(remove_mask)
        assert_invariants(cb)
        assert cb.K == 2

        # Insert 5 items -> needs expand because empty slots=2 < 5
        payload = make_payload(5, cb.D)
        old_cap = cb.capacity
        cb.insert(*payload)
        assert_invariants(cb)

        assert cb.capacity == old_cap * 2, f"expected capacity double: {old_cap*2}, got {cb.capacity}"
        assert cb.K == 2 + 5
        # ensure all inserted slots became valid

        print("PASS\n")
    test_overflow_insert_expand()

    def test_shrink_remove():
        print("=== test_shrink_remove ===")
        cb = CodeBook_adap(K=8, D=3, update_ratio=0.9, remove_threshold=0.1, split_threshold=1.0, refine_step=10)
        assert_invariants(cb)

        # Remove 7 items => K becomes 1. Since 1 <= 8//4(=2), should shrink to 4
        remove_mask = torch.zeros(cb.capacity, dtype=torch.bool)
        remove_mask[1:] = True  # keep index 0 only
        old_cap = cb.capacity
        cb.remove(remove_mask)

        assert_invariants(cb)
        assert cb.capacity == max(1, old_cap // 2), f"expected shrink to {old_cap//2}, got {cb.capacity}"
        assert cb.K == 1
        # After shrink, convention is first K are True, rest False

        print("PASS\n")
    test_shrink_remove()

    def test_refine_remove_with_shrink():
        print("=== test_refine_remove_with_shrink ===")
        cb = CodeBook_adap(K=8, D=3, update_ratio=0.9, remove_threshold=1.0, split_threshold=1e9, refine_step=10)

        # 7개를 제거 대상으로 만들어 K->1이 되게 하여 shrink 유도
        cb.cluster_size[:] = 0.1
        cb.cluster_size[0] = 2.0  # 하나만 살림

        old_cap = cb.capacity
        cb.refine_codebook()
        assert_invariants(cb)

        assert cb.K == 1
        assert cb.capacity == old_cap // 2

        print("PASS\n")
    test_refine_remove_with_shrink()
        
    def test_refine_only_split_no_expand():
        print("=== test_refine_only_split_no_expand ===")
        # remove는 안 일어나게 remove_threshold를 0으로 작게
        cb = CodeBook_adap(K=8, D=3, update_ratio=0.9, remove_threshold=0.0, split_threshold=1.0, refine_step=10)

        # 빈 슬롯 2개 만들어두기 (split insert가 capacity 안에서 끝나게)
        rm = torch.zeros(cb.capacity, dtype=torch.bool)
        rm[6] = True
        rm[7] = True
        cb.remove(rm)
        assert_invariants(cb)
        assert cb.K == 6

        # split을 특정 valid index들에서만 일어나게 세팅
        # 조건: d_sq / cluster_size_safe > split_threshold
        # cluster_size_safe ~ cluster_size 이 되게 만들기 위해 나머지 통계는 단순화
        cb.cluster_size[:] = 10.0
        cb.embed_sum[:] = torch.randn_like(cb.embed_sum)  # 값은 의미 없음
        cb.d_sq[:] = 0.1
        # split 유도할 두 개 인덱스
        split_full_idx = torch.tensor([0, 3])
        cb.d_sq[split_full_idx] = 1000.0  # 크게 만들어 ratio 초과

        old_cap = cb.capacity
        old_K = cb.K
        old_cluster = cb.cluster_size.clone()
        old_embed_sum = cb.embed_sum.clone()
        old_d_sq = cb.d_sq.clone()

        cb.refine_codebook()
        assert_invariants(cb)

        # split 2개 발생 => K가 +2 증가 (remove 없음)
        assert cb.capacity == old_cap  # 빈 슬롯이 충분해서 expand 안 함
        assert cb.K == old_K + 2

        # split된 원본 통계가 정확히 줄었는지(랜덤 없음 → 정확 비교 가능)
        for j in split_full_idx.tolist():
            assert torch.allclose(cb.cluster_size[j], old_cluster[j] / 2.0)
            assert torch.allclose(cb.embed_sum[j], old_embed_sum[j] / 2.0)
            assert torch.allclose(cb.d_sq[j], old_d_sq[j] / 4.0)

        print("PASS\n")
    test_refine_only_split_no_expand()

    def test_refine_split_causes_expand():
        print("=== test_refine_split_causes_expand ===")
        cb = CodeBook_adap(K=4, D=3, update_ratio=0.9, remove_threshold=0.0, split_threshold=1.0, refine_step=10)

        # 빈 슬롯 없이 full valid 상태에서 split 3개를 유도하면 insert가 expand를 일으켜야 함
        cb.cluster_size[:] = 10.0
        cb.embed_sum[:] = torch.randn_like(cb.embed_sum)
        cb.d_sq[:] = 0.1
        # 3개 split 유도
        cb.d_sq[torch.tensor([0, 1, 2])] = 1000.0

        old_cap = cb.capacity
        old_K = cb.K
        cb.refine_codebook()
        assert_invariants(cb)

        # split 3개 발생 → K +3, 빈 슬롯 없으니 capacity 2배 예상
        assert cb.capacity == old_cap * 2
        assert cb.K == old_K + 3

        print("PASS\n")
    test_refine_split_causes_expand()

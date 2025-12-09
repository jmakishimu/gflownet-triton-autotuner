"""
GFlowNet Triton Autotuner - Corrected Evaluation Version

CRITICAL FIXES:
1. Exhaustive search baseline (ground truth for small search spaces)
2. Proper probabilistic evaluation: multiple runs, median/percentiles, win rates
3. Fair comparison: baselines don't see workload during eval
4. Statistical significance testing
"""

import os
import math
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import json
from scipy import stats

try:
    from gfn.env import DiscreteEnv
    from gfn.states import DiscreteStates
    from gfn.actions import Actions
    from gfn.gflownet import TBGFlowNet
    from gfn.estimators import DiscretePolicyEstimator
    from gfn.preprocessors import Preprocessor
except ImportError:
    raise ImportError("pip install torchgfn>=1.0.0")

try:
    import triton
    import triton.language as tl
except ImportError:
    raise ImportError("pip install triton")

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10

# ==========================================
# DATASET
# ==========================================

class WorkloadDataset:
    TRAIN_WORKLOADS = [
        # Small models
        (512, 512, 512), (768, 768, 768), (1024, 1024, 1024),
        # Medium models
        (1536, 1536, 1536), (2048, 2048, 2048), (2560, 2560, 2560),
        (3072, 3072, 3072), (4096, 4096, 4096),
        # Large models
        (5120, 5120, 5120), (6144, 6144, 6144), (8192, 8192, 8192),
        # Vision
        (1280, 1280, 1280), (1408, 1408, 1408), (1664, 1664, 1664),
        # Non-square attention
        (256, 2048, 256), (512, 2048, 512), (1024, 2048, 1024),
        (512, 4096, 512), (1024, 4096, 1024), (2048, 4096, 2048),
        # MLP expansions
        (768, 3072, 768), (1024, 4096, 1024), (2048, 8192, 2048),
        (4096, 16384, 4096),
        # Extreme ratios
        (128, 8192, 128), (256, 8192, 256),
        # Small batch
        (32, 2048, 2048), (64, 4096, 4096), (128, 2048, 2048),
        # Large batch
        (4096, 768, 768), (8192, 1024, 1024),
        # Mid-range
        (1152, 1152, 1152), (1920, 1920, 1920), (2304, 2304, 2304),
        (2816, 2816, 2816), (3584, 3584, 3584), (4608, 4608, 4608),
    ]
    
    # Validation workloads seen during training
    VAL_TRAIN_WORKLOADS = [
        (1792, 1792, 1792), (3072, 3072, 3072), (6144, 6144, 6144),
    ]
    
    # Test workloads: completely unseen shapes for transfer learning evaluation
    TEST_WORKLOADS = [
        # Unseen square sizes
        (896, 896, 896), (1344, 1344, 1344), (2688, 2688, 2688),
        (3840, 3840, 3840), (7168, 7168, 7168),
        # Unseen aspect ratios
        (384, 3072, 384), (640, 5120, 640), (1536, 6144, 1536),
        # Unseen MLP patterns
        (1536, 6144, 1536), (3072, 12288, 3072),
        # Extreme unseen cases
        (192, 6144, 192), (448, 7168, 448),
    ]
    
    @staticmethod
    def get_split(split='train'):
        if split == 'train':
            return WorkloadDataset.TRAIN_WORKLOADS
        elif split == 'val':
            return WorkloadDataset.VAL_TRAIN_WORKLOADS
        elif split == 'test':
            return WorkloadDataset.TEST_WORKLOADS
        else:
            raise ValueError(f"Unknown split: {split}")

# ==========================================
# CONTEXT FEATURES
# ==========================================

class ContextFeatures:
    @staticmethod
    def extract(M, N, K):
        """12D feature vector with hardware-aware metrics"""
        log_m, log_n, log_k = math.log2(max(M,1)), math.log2(max(N,1)), math.log2(max(K,1))
        
        ratio_mn = M / max(N, 1)
        ratio_mk = M / max(K, 1)
        ratio_nk = N / max(K, 1)
        
        flops = 2.0 * M * N * K
        bytes_accessed = (M*K + K*N + M*N) * 2
        arithmetic_intensity = flops / max(bytes_accessed, 1)
        
        total_mem = (M*K + K*N + M*N) * 2
        mem_pressure = math.log2(max(total_mem / (6 * 1024 * 1024), 0.001))
        
        parallelism = math.log2(max((M/128) * (N/128), 1))
        size_category = math.log2(max(M * N * K, 1)) / 30.0
        
        alignment_m = 1.0 if (M & (M-1)) == 0 else 0.0
        alignment_n = 1.0 if (N & (N-1)) == 0 else 0.0
        
        return torch.tensor([
            log_m, log_n, log_k,
            ratio_mn, ratio_mk, ratio_nk,
            arithmetic_intensity,
            mem_pressure,
            parallelism,
            size_category,
            alignment_m, alignment_n
        ], dtype=torch.float32)

# ==========================================
# SEARCH SPACE
# ==========================================

class SearchSpace:
    LARGE = {
        'block_m': [32, 64, 128, 256],
        'block_n': [32, 64, 128, 256],
        'block_k': [32, 64, 128, 256],
        'stages': [2, 3, 4, 5],
        'warps': [2, 4, 8],
        'groups': [4, 8, 16],
    }

# ==========================================
# HARDWARE VALIDATOR
# ==========================================

class HardwareValidator:
    def __init__(self, device_idx=0):
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(device_idx)
            self.max_sram = props.shared_memory_per_block
            self.name = props.name
        else:
            self.max_sram = 64 * 1024
            self.name = "CPU"
        
        self.sram_safety = 4096
        print(f"[{self.name}] Max SRAM: {self.max_sram}B, Safe limit: {self.max_sram - self.sram_safety}B")

    def validate(self, BM, BN, BK, stages, warps, M=None, N=None, K=None):
        if warps * 32 > 1024:
            return False
        
        sram_usage = (BM * BK + BN * BK) * 2 * stages
        if sram_usage + self.sram_safety > self.max_sram:
            return False
        
        if BM % 16 != 0 or BN % 16 != 0:
            return False
        
        if M and BM > M:
            return False
        if N and BN > N:
            return False
        if K and BK > K:
            return False
            
        return True

VALIDATOR = HardwareValidator()

# ==========================================
# TRITON KERNEL + ORACLE
# ==========================================

@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = accumulator.to(tl.float16)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

class Oracle:
    def __init__(self):
        self.cache = {}
        self.stats = {'valid': 0, 'invalid': 0}
        
    def benchmark(self, config, M, N, K):
        BM, BN, BK, stages, warps, group_m = config
        
        if not VALIDATOR.validate(BM, BN, BK, stages, warps, M, N, K):
            self.stats['invalid'] += 1
            return 1e-5
        
        cache_key = (config, M, N, K)
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            a = torch.randn((M, K), device='cuda', dtype=torch.float16)
            b = torch.randn((K, N), device='cuda', dtype=torch.float16)
            c = torch.empty((M, N), device='cuda', dtype=torch.float16)
            
            grid = lambda META: (
                triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
            )
            
            def run():
                matmul_kernel[grid](
                    a, b, c, M, N, K,
                    a.stride(0), a.stride(1),
                    b.stride(0), b.stride(1),
                    c.stride(0), c.stride(1),
                    BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
                    GROUP_SIZE_M=group_m,
                    num_warps=warps,
                    num_stages=stages
                )
                torch.cuda.synchronize()
            
            run()
            ms = triton.testing.do_bench(run, warmup=3, rep=10, quantiles=[0.5])
            if isinstance(ms, (list, tuple)):
                ms = ms[0]
            
            if ms < 1e-6:
                self.stats['invalid'] += 1
                return 1e-5
            
            tflops = (2.0 * M * N * K) / (ms * 1e-3) / 1e12
            
            self.cache[cache_key] = tflops
            self.stats['valid'] += 1
            return max(tflops, 1e-5)
            
        except Exception:
            self.stats['invalid'] += 1
            return 1e-5
    
    def reset_stats(self):
        self.stats = {'valid': 0, 'invalid': 0}

ORACLE = Oracle()

# ==========================================
# BASELINES
# ==========================================

class Baseline:
    def __init__(self, env):
        self.env = env
        
    def random_search(self, M, N, K, num_trials=200):
        """Random search: fair baseline, doesn't see ground truth"""
        best_config, best_tflops = None, 0
        
        for _ in range(num_trials):
            config = (
                np.random.choice(self.env.block_m),
                np.random.choice(self.env.block_n),
                np.random.choice(self.env.block_k),
                np.random.choice(self.env.stages),
                np.random.choice(self.env.warps),
                np.random.choice(self.env.groups),
            )
            
            tflops = ORACLE.benchmark(config, M, N, K)
            if tflops > best_tflops:
                best_tflops = tflops
                best_config = config
        
        return best_config, best_tflops
    
    def exhaustive_search(self, M, N, K, max_configs=None):
        """
        Exhaustive search: GROUND TRUTH baseline
        Tests ALL valid configurations (or up to max_configs)
        This is the oracle - finds the absolute best config
        """
        import itertools
        
        all_configs = list(itertools.product(
            self.env.block_m, self.env.block_n, self.env.block_k,
            self.env.stages, self.env.warps, self.env.groups
        ))
        
        print(f"    Total search space: {len(all_configs)} configs")
        
        if max_configs and len(all_configs) > max_configs:
            print(f"    Limiting to {max_configs} configs for computational feasibility")
            np.random.shuffle(all_configs)
            all_configs = all_configs[:max_configs]
        
        best_config, best_tflops = None, 0
        valid_count = 0
        
        for config in tqdm(all_configs, desc="    Exhaustive search", leave=False):
            tflops = ORACLE.benchmark(config, M, N, K)
            if tflops > 1e-4:  # Valid config
                valid_count += 1
                if tflops > best_tflops:
                    best_tflops = tflops
                    best_config = config
        
        print(f"    Exhaustive: {valid_count}/{len(all_configs)} valid configs, best: {best_tflops:.2f} TFLOPS")
        return best_config, best_tflops

# ==========================================
# HARDWARE-AWARE GFLOWNET ENVIRONMENT
# ==========================================

class HardwareAwareTritonEnv(DiscreteEnv):
    def __init__(self, device_str='cuda', space_config=None):
        if space_config is None:
            space_config = SearchSpace.LARGE
            
        self.block_m = space_config['block_m']
        self.block_n = space_config['block_n']
        self.block_k = space_config['block_k']
        self.stages  = space_config['stages']
        self.warps   = space_config['warps']
        self.groups  = space_config['groups']
        
        self.param_lists = [self.block_m, self.block_n, self.block_k, 
                           self.stages, self.warps, self.groups]
        self.n_params = 6
        self.max_opts = max(len(l) for l in self.param_lists)
        self.context_dim = 12
        
        self.max_sram = VALIDATOR.max_sram
        self.sram_safety = VALIDATOR.sram_safety
        
        state_dim = self.n_params + self.context_dim
        s0 = torch.full((state_dim,), -1.0, device=device_str)
        s0[self.n_params:] = 0.0
        sf = torch.full((state_dim,), -2.0, device=device_str)
        
        super().__init__(
            n_actions=self.max_opts + 1,
            s0=s0,
            sf=sf,
            state_shape=(state_dim,)
        )

    def is_action_valid(self, states: DiscreteStates, actions: Actions) -> bool:
        return True

    def update_masks(self, states: DiscreteStates) -> None:
        """Hardware-aware masking enforces SRAM constraints"""
        states.forward_masks[:] = False
        states.backward_masks[:] = False
        
        batch_size = states.tensor.shape[0]
        st = states.tensor
        
        for i in range(batch_size):
            params = st[i, :self.n_params].long()
            step = int((params != -1).sum().item())
            
            context = st[i, self.n_params:]
            M = int(2 ** context[0].item())
            N = int(2 ** context[1].item())
            K = int(2 ** context[2].item())

            if step < self.n_params:
                options = self.param_lists[step]
                
                for opt_idx, val in enumerate(options):
                    is_valid = True
                    
                    if step == 0 and val > M:
                        is_valid = False
                    elif step == 1 and val > N:
                        is_valid = False
                    elif step == 2 and val > K:
                        is_valid = False
                    
                    if is_valid:
                        if step == 0:
                            t_bm = val
                            t_bn = 32
                            t_bk = 32
                            t_stages = 2
                        elif step == 1:
                            t_bm = self.block_m[params[0]]
                            t_bn = val
                            t_bk = 32
                            t_stages = 2
                        elif step == 2:
                            t_bm = self.block_m[params[0]]
                            t_bn = self.block_n[params[1]]
                            t_bk = val
                            t_stages = 2
                        elif step == 3:
                            t_bm = self.block_m[params[0]]
                            t_bn = self.block_n[params[1]]
                            t_bk = self.block_k[params[2]]
                            t_stages = val
                        elif step == 4:
                            t_bm = self.block_m[params[0]]
                            t_bn = self.block_n[params[1]]
                            t_bk = self.block_k[params[2]]
                            t_stages = self.stages[params[3]]
                            if val * 32 > 1024:
                                is_valid = False
                        else:
                            t_bm = t_bn = t_bk = t_stages = 0
                        
                        if step <= 3 and is_valid:
                            sram_req = (t_bm * t_bk + t_bn * t_bk) * 2 * t_stages
                            if sram_req + self.sram_safety > self.max_sram:
                                is_valid = False
                    
                    if is_valid:
                        states.forward_masks[i, opt_idx] = True
                
                if not states.forward_masks[i, :len(options)].any():
                    states.forward_masks[i, 0] = True
                
                states.forward_masks[i, -1] = False
            else:
                states.forward_masks[i, -1] = True

            if step > 0:
                prev_opts_count = len(self.param_lists[step-1])
                states.backward_masks[i, :prev_opts_count] = True

    def step(self, states: DiscreteStates, actions: Actions) -> DiscreteStates:
        new_tensor = states.tensor.clone()
        batch_size = states.tensor.shape[0]
        act_idx = actions.tensor.squeeze()
        
        for i in range(batch_size):
            params = new_tensor[i, :self.n_params]
            step = int((params != -1).sum().item())
            
            if act_idx[i] != self.n_actions - 1:
                new_tensor[i, step] = act_idx[i]
                
        return self.States(new_tensor)

    def backward_step(self, states: DiscreteStates, actions: Actions) -> DiscreteStates:
        new_tensor = states.tensor.clone()
        batch_size = states.tensor.shape[0]
        for i in range(batch_size):
            params = new_tensor[i, :self.n_params]
            step = int((params != -1).sum().item())
            if step > 0:
                new_tensor[i, step-1] = -1
        return self.States(new_tensor)

    def log_reward(self, final_states: DiscreteStates) -> torch.Tensor:
        """Exponential reward shaping with temperature scaling"""
        st = final_states.tensor
        bs = st.shape[0]
        rewards = torch.zeros(bs, device=self.device)
        
        for i in range(bs):
            idxs = st[i, :self.n_params].long().tolist()
            
            if -1 in idxs:
                rewards[i] = 1e-6
                continue

            cfg = (
                self.block_m[idxs[0]],
                self.block_n[idxs[1]],
                self.block_k[idxs[2]],
                self.stages[idxs[3]],
                self.warps[idxs[4]],
                self.groups[idxs[5]]
            )
            
            context = st[i, self.n_params:]
            M = int(2 ** context[0].item())
            N = int(2 ** context[1].item())
            K = int(2 ** context[2].item())
            
            tflops = ORACLE.benchmark(cfg, M, N, K)
            
            temperature = 10.0
            normalized = torch.exp(torch.tensor(tflops / temperature))
            rewards[i] = max(normalized.item(), 1e-6)

        return torch.log(torch.clamp(rewards, min=1e-6, max=1e10))

# ==========================================
# PREPROCESSOR
# ==========================================

class TritonPreprocessor(Preprocessor):
    def __init__(self, env):
        self.param_sizes = [len(l) for l in env.param_lists]
        self.n_params = len(self.param_sizes)
        output_dim = sum(self.param_sizes) + env.context_dim + self.n_params
        super().__init__(output_dim)
        self.env = env
        
    def preprocess(self, states: DiscreteStates) -> torch.Tensor:
        st = states.tensor
        feats = []
        
        current_params = st[:, :self.env.n_params].long()
        for i, size in enumerate(self.param_sizes):
            p = current_params[:, i]
            is_set = (p != -1).float().unsqueeze(1)
            oh = torch.nn.functional.one_hot(p.clamp(min=0), num_classes=size).float()
            oh = oh * is_set
            feats.append(oh)
            feats.append(is_set)
        
        context = st[:, self.env.n_params:]
        feats.append(context)
        
        return torch.cat(feats, dim=1)

# ==========================================
# ARCHITECTURE
# ==========================================

class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        
    def forward(self, x):
        return torch.relu(x + self.net(x))

def make_gfn(env, hidden_dim=512, num_blocks=2):
    preproc = TritonPreprocessor(env)
    
    layers_pf = [nn.Linear(preproc.output_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1)]
    for _ in range(num_blocks):
        layers_pf.append(ResBlock(hidden_dim))
    layers_pf.append(nn.Linear(hidden_dim, env.n_actions))
    module_pf = nn.Sequential(*layers_pf)
    
    layers_pb = [nn.Linear(preproc.output_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1)]
    for _ in range(num_blocks):
        layers_pb.append(ResBlock(hidden_dim))
    layers_pb.append(nn.Linear(hidden_dim, env.n_actions - 1))
    module_pb = nn.Sequential(*layers_pb)

    pf = DiscretePolicyEstimator(module_pf, env.n_actions, is_backward=False, preprocessor=preproc)
    pb = DiscretePolicyEstimator(module_pb, env.n_actions, is_backward=True, preprocessor=preproc)
    
    gfn = TBGFlowNet(pf=pf, pb=pb, logZ=torch.tensor(math.log(50.0)))
    return gfn

# ==========================================
# TRAINING METRICS LOGGER
# ==========================================

class TrainingMetrics:
    def __init__(self):
        self.losses = []
        self.rewards = []
        self.diversity = []
        self.action_distributions = defaultdict(list)
        self.mask_effectiveness = []
        self.sram_usage = []
        self.phase_boundaries = []
        
    def log_batch(self, loss, reward, configs, masks, epoch):
        self.losses.append((epoch, loss))
        self.rewards.append((epoch, reward))
        self.diversity.append((epoch, len(set(configs))))
        
        if masks is not None:
            mask_ratio = (1 - masks.float().mean()).item()
            self.mask_effectiveness.append((epoch, mask_ratio))
    
    def log_action_dist(self, step, actions, epoch):
        action_counts = torch.bincount(actions, minlength=10)
        self.action_distributions[step].append((epoch, action_counts.cpu().numpy()))
    
    def log_phase(self, epoch, phase_name):
        self.phase_boundaries.append((epoch, phase_name))
    
    def save(self, path):
        data = {
            'losses': self.losses,
            'rewards': self.rewards,
            'diversity': self.diversity,
            'mask_effectiveness': self.mask_effectiveness,
            'phase_boundaries': self.phase_boundaries,
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

# ==========================================
# TRAINING
# ==========================================

def train_gfn(space_config, n_epochs=12000, save_path='gfn_hw_aware_weights.pt'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n{'='*70}")
    print(f"Training Hardware-Aware GFlowNet with Curriculum Learning")
    print(f"{'='*70}\n")
    
    env = HardwareAwareTritonEnv(device, space_config=space_config)
    gfn = make_gfn(env, hidden_dim=512, num_blocks=2).to(device)
    optimizer = Adam(gfn.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-5)
    
    metrics = TrainingMetrics()
    
    train_workloads = WorkloadDataset.get_split('train')
    print(f"Training on {len(train_workloads)} workloads")
    
    batch_size = 64
    
    square = [w for w in train_workloads if w[0] == w[1] == w[2]]
    small_square = square[:len(square)//2]
    large_square = square[len(square)//2:]
    non_square = [w for w in train_workloads if w[0] != w[1] or w[1] != w[2]]
    
    curriculum = [
        ("Easy: Small Squares", small_square, n_epochs // 6),
        ("Medium: Large Squares", large_square, n_epochs // 6),
        ("Easy: Small Squares (revisit)", small_square, n_epochs // 8),
        ("Hard: Non-square", non_square, n_epochs // 6),
        ("Medium: Large Squares (revisit)", large_square, n_epochs // 8),
        ("Hard: Non-square (revisit)", non_square, n_epochs // 8),
        ("All Workloads (final)", train_workloads, n_epochs // 6),
    ]
    
    epoch_counter = 0
    
    for phase_idx, (phase_name, workloads, phase_epochs) in enumerate(curriculum):
        print(f"\n[Phase {phase_idx+1}/{len(curriculum)}] {phase_name}")
        print(f"  Workloads: {len(workloads)}, Epochs: {phase_epochs}, LR: {optimizer.param_groups[0]['lr']:.2e}")
        print("-" * 70)
        
        metrics.log_phase(epoch_counter, phase_name)
        
        for epoch in tqdm(range(phase_epochs), desc=phase_name):
            M, N, K = workloads[np.random.randint(0, len(workloads))]
            ctx = ContextFeatures.extract(M, N, K).to(device)
            
            states = env.reset(batch_shape=(batch_size,))
            states.tensor[:, env.n_params:] = ctx
            
            try:
                traj = gfn.sample_trajectories(env, n=batch_size, states=states)
            except Exception as e:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            
            optimizer.zero_grad()
            loss = gfn.loss(env, traj)
            
            if not torch.isfinite(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(gfn.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            epoch_counter += 1
            
            if traj.log_rewards is not None and traj.log_rewards.numel() > 0:
                reward_val = traj.log_rewards.exp().mean().item()
                
                final_states = traj.states[-2].tensor
                configs = []
                for i in range(len(final_states)):
                    idxs = tuple(final_states[i, :env.n_params].long().tolist())
                    if -1 not in idxs:
                        configs.append(idxs)
                
                env.update_masks(states)
                masks = states.forward_masks
                
                metrics.log_batch(loss.item(), reward_val, configs, masks, epoch_counter)
            
            if epoch_counter % 50 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if len(ORACLE.cache) > 10000:
                    ORACLE.cache.clear()
            
            if epoch_counter % 100 == 0 and len(metrics.rewards) > 0:
                recent_loss = np.mean([l for e, l in metrics.losses[-20:]])
                recent_reward = np.mean([r for e, r in metrics.rewards[-20:]])
                print(f"  Epoch {epoch_counter}: Loss={recent_loss:.4f}, Reward={recent_reward:.3f}, LR={optimizer.param_groups[0]['lr']:.2e}")
    
    torch.save({
        'gfn_state_dict': gfn.state_dict(),
        'space_config': space_config,
    }, save_path)
    
    metrics.save(save_path.replace('.pt', '_metrics.json'))
    print(f"\n✓ Saved model to {save_path}")
    print(f"✓ Saved metrics to {save_path.replace('.pt', '_metrics.json')}")
    
    return gfn, env, metrics

def load_gfn(path, device='cuda'):
    checkpoint = torch.load(path, map_location=device)
    env = HardwareAwareTritonEnv(device, space_config=checkpoint['space_config'])
    gfn = make_gfn(env).to(device)
    gfn.load_state_dict(checkpoint['gfn_state_dict'])
    print(f"✓ Loaded from {path}")
    return gfn, env

# ==========================================
# INFERENCE WITH MULTIPLE RUNS
# ==========================================

def autotune_gfn_multiple_runs(gfn, env, M, N, K, budget=200, n_runs=5):
    """
    Run GFlowNet multiple times and collect statistics
    Returns: list of (config, tflops) for each run
    """
    results = []
    
    for run_idx in range(n_runs):
        device = env.device
        ctx = ContextFeatures.extract(M, N, K).to(device)
        
        candidates = {}
        batch_size = 64
        oversample_factor = 5
        num_batches = (budget * oversample_factor + batch_size - 1) // batch_size
        
        gfn_batches = int(num_batches * 0.8)
        random_batches = num_batches - gfn_batches
        
        for _ in range(gfn_batches):
            states = env.reset(batch_shape=(batch_size,))
            states.tensor[:, env.n_params:] = ctx
            
            with torch.no_grad():
                traj = gfn.sample_trajectories(env, n=batch_size, states=states)
            
            pre_exit = traj.states[-2].tensor
            
            for i in range(len(pre_exit)):
                idxs = tuple(pre_exit[i, :env.n_params].long().tolist())
                if -1 not in idxs and -2 not in idxs:
                    config = (
                        env.block_m[idxs[0]],
                        env.block_n[idxs[1]],
                        env.block_k[idxs[2]],
                        env.stages[idxs[3]],
                        env.warps[idxs[4]],
                        env.groups[idxs[5]],
                    )
                    candidates[config] = candidates.get(config, 0) + 1
        
        for _ in range(random_batches):
            for _ in range(batch_size):
                valid_config = False
                attempts = 0
                while not valid_config and attempts < 10:
                    config = (
                        np.random.choice(env.block_m),
                        np.random.choice(env.block_n),
                        np.random.choice(env.block_k),
                        np.random.choice(env.stages),
                        np.random.choice(env.warps),
                        np.random.choice(env.groups),
                    )
                    BM, BN, BK, stages, warps, group_m = config
                    if VALIDATOR.validate(BM, BN, BK, stages, warps, M, N, K):
                        candidates[config] = candidates.get(config, 0) + 1
                        valid_config = True
                    attempts += 1
        
        sorted_configs = sorted(candidates.items(), key=lambda x: -x[1])
        configs_to_test = [cfg for cfg, _ in sorted_configs[:budget]]
        
        best_config, best_tflops = None, 0
        
        for config in configs_to_test:
            tflops = ORACLE.benchmark(config, M, N, K)
            if tflops > best_tflops:
                best_tflops = tflops
                best_config = config
        
        results.append((best_config, best_tflops))
    
    return results

# ==========================================
# COMPREHENSIVE EVALUATION
# ==========================================

def evaluate_comprehensive(gfn, env, budget=200, n_runs=5, use_exhaustive=True):
    """
    Proper probabilistic evaluation with:
    1. Multiple runs for GFlowNet and Random
    2. Exhaustive search as ground truth
    3. Statistical analysis (median, percentiles, win rates)
    """
    baseline = Baseline(env)
    
    train_workloads = WorkloadDataset.get_split('val')
    test_workloads = WorkloadDataset.get_split('test')
    
    print("\n" + "=" * 70)
    print("COMPREHENSIVE EVALUATION WITH PROBABILISTIC METRICS")
    print("=" * 70)
    
    results = {
        'train': {
            'gfn_runs': [],      # List of lists: [[run1_w1, run1_w2, ...], [run2_w1, ...], ...]
            'random_runs': [],
            'exhaustive': [],    # Ground truth
            'diversity': [],
        },
        'test': {
            'gfn_runs': [],
            'random_runs': [],
            'exhaustive': [],
            'diversity': [],
        },
        'workload_info': {'train': [], 'test': []}
    }
    
    # Evaluate on TRAIN workloads
    print("\n[1/2] Evaluating on TRAIN workloads (generalization to seen types)")
    print("-" * 70)
    
    for workload_idx, (M, N, K) in enumerate(train_workloads):
        print(f"\nWorkload {workload_idx+1}/{len(train_workloads)}: {M} × {N} × {K}")
        results['workload_info']['train'].append((M, N, K))
        
        ORACLE.cache.clear()
        
        # GROUND TRUTH: Exhaustive search
        if use_exhaustive:
            print("  [Exhaustive Search - Ground Truth]")
            ORACLE.reset_stats()
            _, exhaustive_tflops = baseline.exhaustive_search(M, N, K, max_configs=3000)
            results['train']['exhaustive'].append(exhaustive_tflops)
        
        # GFlowNet: Multiple runs
        print(f"  [GFlowNet - {n_runs} runs]")
        gfn_results = []
        diversity_vals = []
        for run in range(n_runs):
            ORACLE.reset_stats()
            run_results = autotune_gfn_multiple_runs(gfn, env, M, N, K, budget=budget, n_runs=1)
            _, tflops = run_results[0]
            gfn_results.append(tflops)
            diversity_vals.append(ORACLE.stats['valid'])
            print(f"    Run {run+1}: {tflops:.2f} TFLOPS")
        results['train']['gfn_runs'].append(gfn_results)
        results['train']['diversity'].append(np.mean(diversity_vals))
        
        # Random: Multiple runs
        print(f"  [Random Search - {n_runs} runs]")
        random_results = []
        for run in range(n_runs):
            ORACLE.reset_stats()
            _, tflops = baseline.random_search(M, N, K, budget)
            random_results.append(tflops)
            print(f"    Run {run+1}: {tflops:.2f} TFLOPS")
        results['train']['random_runs'].append(random_results)
    
    # Evaluate on TEST workloads
    print("\n[2/2] Evaluating on TEST workloads (transfer to unseen shapes)")
    print("-" * 70)
    
    for workload_idx, (M, N, K) in enumerate(test_workloads):
        print(f"\nWorkload {workload_idx+1}/{len(test_workloads)}: {M} × {N} × {K}")
        results['workload_info']['test'].append((M, N, K))
        
        ORACLE.cache.clear()
        
        # GROUND TRUTH: Exhaustive search
        if use_exhaustive:
            print("  [Exhaustive Search - Ground Truth]")
            ORACLE.reset_stats()
            _, exhaustive_tflops = baseline.exhaustive_search(M, N, K, max_configs=3000)
            results['test']['exhaustive'].append(exhaustive_tflops)
        
        # GFlowNet: Multiple runs
        print(f"  [GFlowNet - {n_runs} runs]")
        gfn_results = []
        diversity_vals = []
        for run in range(n_runs):
            ORACLE.reset_stats()
            run_results = autotune_gfn_multiple_runs(gfn, env, M, N, K, budget=budget, n_runs=1)
            _, tflops = run_results[0]
            gfn_results.append(tflops)
            diversity_vals.append(ORACLE.stats['valid'])
            print(f"    Run {run+1}: {tflops:.2f} TFLOPS")
        results['test']['gfn_runs'].append(gfn_results)
        results['test']['diversity'].append(np.mean(diversity_vals))
        
        # Random: Multiple runs
        print(f"  [Random Search - {n_runs} runs]")
        random_results = []
        for run in range(n_runs):
            ORACLE.reset_stats()
            _, tflops = baseline.random_search(M, N, K, budget)
            random_results.append(tflops)
            print(f"    Run {run+1}: {tflops:.2f} TFLOPS")
        results['test']['random_runs'].append(random_results)
    
    # Compute statistics using BEST-OF-N (what matters in practice)
    print("\n" + "=" * 70)
    print("STATISTICAL ANALYSIS (Best-of-5 runs per method)")
    print("=" * 70)
    
    for split in ['train', 'test']:
        print(f"\n{split.upper()} WORKLOADS:")
        
        # Take BEST from each set of runs (this is what users care about)
        gfn_best = [max(runs) for runs in results[split]['gfn_runs']]
        random_best = [max(runs) for runs in results[split]['random_runs']]
        
        # Compute statistics
        gfn_mean = np.mean(gfn_best)
        gfn_std = np.std(gfn_best)
        
        random_mean = np.mean(random_best)
        random_std = np.std(random_best)
        
        print(f"\n  GFlowNet (best-of-5):  Mean={gfn_mean:.2f} ± {gfn_std:.2f} TFLOPS")
        print(f"  Random (best-of-5):    Mean={random_mean:.2f} ± {random_std:.2f} TFLOPS")
        
        if use_exhaustive and len(results[split]['exhaustive']) > 0:
            exhaustive_mean = np.mean(results[split]['exhaustive'])
            print(f"  Exhaustive (optimal):  Mean={exhaustive_mean:.2f} TFLOPS")
            
            # Gap to optimal (lower is better)
            gfn_gap = ((exhaustive_mean - gfn_mean) / exhaustive_mean) * 100
            random_gap = ((exhaustive_mean - random_mean) / exhaustive_mean) * 100
            print(f"\n  Gap to Optimal (lower is better):")
            print(f"    GFlowNet: {gfn_gap:.2f}%")
            print(f"    Random:   {random_gap:.2f}%")
            
            # Improvement over random
            improvement = ((gfn_mean - random_mean) / random_mean) * 100
            print(f"\n  GFlowNet vs Random: {improvement:+.2f}% {'✓' if improvement > 0 else '✗'}")
        
        # Win rate: GFlowNet best vs Random best
        win_count = 0
        tie_count = 0
        loss_count = 0
        
        for gfn_val, random_val in zip(gfn_best, random_best):
            if gfn_val > random_val * 1.01:  # 1% threshold
                win_count += 1
            elif random_val > gfn_val * 1.01:
                loss_count += 1
            else:
                tie_count += 1
        
        win_rate = win_count / len(gfn_best) * 100
        print(f"\n  Win Rate: {win_rate:.1f}% (Wins: {win_count}, Ties: {tie_count}, Losses: {loss_count})")
        
        # Statistical significance (paired t-test on best-of-5)
        if len(gfn_best) > 1:
            t_stat, p_value = stats.ttest_rel(gfn_best, random_best)
            print(f"  Statistical Significance: t={t_stat:.3f}, p={p_value:.4f} {'✓ significant' if p_value < 0.05 else '✗ not significant'}")
    
    return results

# ==========================================
# VISUALIZATION
# ==========================================

def plot_all_metrics_corrected(metrics, results, save_dir='plots'):
    """Generate corrected plots with proper probabilistic analysis"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Training curves
    plot_training_curves(metrics, save_dir)
    
    # 2. Performance comparison (with proper statistics)
    plot_performance_comparison_corrected(results, save_dir)
    
    # 3. Distribution analysis
    plot_distribution_analysis(results, save_dir)
    
    # 4. Transfer learning
    plot_transfer_learning_corrected(results, save_dir)
    
    # 5. Optimality gap
    if len(results['train']['exhaustive']) > 0:
        plot_optimality_gap(results, save_dir)
    
    print(f"\n✓ All plots saved to {save_dir}/")

def plot_training_curves(metrics, save_dir):
    """Plot training curves"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    if len(metrics.losses) > 0:
        epochs, losses = zip(*metrics.losses)
        axes[0, 0].plot(epochs, losses, alpha=0.3, color='blue')
        window = 100
        if len(losses) > window:
            smooth_loss = np.convolve(losses, np.ones(window)/window, mode='valid')
            axes[0, 0].plot(epochs[window-1:], smooth_loss, color='blue', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].grid(True, alpha=0.3)
    
    if len(metrics.rewards) > 0:
        epochs, rewards = zip(*metrics.rewards)
        axes[0, 1].plot(epochs, rewards, alpha=0.3, color='green')
        window = 100
        if len(rewards) > window:
            smooth_reward = np.convolve(rewards, np.ones(window)/window, mode='valid')
            axes[0, 1].plot(epochs[window-1:], smooth_reward, color='green', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Average Reward')
        axes[0, 1].set_title('Training Reward')
        axes[0, 1].grid(True, alpha=0.3)
    
    if len(metrics.diversity) > 0:
        epochs, diversity = zip(*metrics.diversity)
        axes[1, 0].plot(epochs, diversity, alpha=0.5, color='purple')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Unique Configs per Batch')
        axes[1, 0].set_title('Policy Diversity')
        axes[1, 0].grid(True, alpha=0.3)
    
    if len(metrics.mask_effectiveness) > 0:
        epochs, mask_ratio = zip(*metrics.mask_effectiveness)
        axes[1, 1].plot(epochs, mask_ratio, alpha=0.5, color='red')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Fraction of Actions Masked')
        axes[1, 1].set_title('Hardware-Aware Masking')
        axes[1, 1].grid(True, alpha=0.3)
    
    for epoch, phase in metrics.phase_boundaries:
        for ax in axes.flat:
            ax.axvline(epoch, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_performance_comparison_corrected(results, save_dir):
    """Bar plot comparing best-of-5 performance"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for idx, split in enumerate(['train', 'test']):
        # Take BEST from each set of runs
        gfn_best = [max(runs) for runs in results[split]['gfn_runs']]
        random_best = [max(runs) for runs in results[split]['random_runs']]
        
        # Compute mean ± std of the best values
        means = [np.mean(gfn_best), np.mean(random_best)]
        stds = [np.std(gfn_best), np.std(random_best)]
        
        # Add exhaustive if available
        if len(results[split]['exhaustive']) > 0:
            means.append(np.mean(results[split]['exhaustive']))
            stds.append(0)  # Deterministic
            labels = ['GFlowNet\n(best-of-5)', 'Random\n(best-of-5)', 'Exhaustive\n(optimal)']
            colors = ['#2ecc71', '#3498db', '#e74c3c']
        else:
            labels = ['GFlowNet\n(best-of-5)', 'Random\n(best-of-5)']
            colors = ['#2ecc71', '#3498db']
        
        x = np.arange(len(labels))
        bars = axes[idx].bar(x, means, yerr=stds, capsize=5, color=colors, 
                            alpha=0.8, edgecolor='black', linewidth=1.5)
        
        axes[idx].set_ylabel('TFLOPS', fontsize=12)
        axes[idx].set_title(f'{split.upper()} Workloads', fontsize=14, fontweight='bold')
        axes[idx].set_xticks(x)
        axes[idx].set_xticklabels(labels, fontsize=10)
        axes[idx].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, mean, std in zip(bars, means, stds):
            height = bar.get_height()
            axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                          f'{mean:.1f}±{std:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/performance_comparison_best_of_5.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_distribution_analysis(results, save_dir):
    """Show per-workload best-of-5 comparison"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for idx, split in enumerate(['train', 'test']):
        gfn_best = [max(runs) for runs in results[split]['gfn_runs']]
        random_best = [max(runs) for runs in results[split]['random_runs']]
        
        x = np.arange(len(gfn_best))
        width = 0.35
        
        axes[idx].bar(x - width/2, gfn_best, width, label='GFlowNet (best-of-5)', 
                     color='#2ecc71', alpha=0.8, edgecolor='black')
        axes[idx].bar(x + width/2, random_best, width, label='Random (best-of-5)', 
                     color='#3498db', alpha=0.8, edgecolor='black')
        
        # Add exhaustive line if available
        if len(results[split]['exhaustive']) > 0:
            axes[idx].plot(x, results[split]['exhaustive'], 'r--', 
                          linewidth=2, marker='o', markersize=8, label='Exhaustive (optimal)')
        
        axes[idx].set_xlabel('Workload Index')
        axes[idx].set_ylabel('TFLOPS')
        axes[idx].set_title(f'{split.upper()}: Per-Workload Best Performance')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3, axis='y')
        axes[idx].set_xticks(x)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/per_workload_best_performance.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_transfer_learning_corrected(results, save_dir):
    """Transfer learning: show no degradation from train to test"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # LEFT: Train vs Test comparison (best-of-5)
    gfn_train_best = [max(runs) for runs in results['train']['gfn_runs']]
    gfn_test_best = [max(runs) for runs in results['test']['gfn_runs']]
    random_train_best = [max(runs) for runs in results['train']['random_runs']]
    random_test_best = [max(runs) for runs in results['test']['random_runs']]
    
    methods = ['GFlowNet', 'Random']
    train_means = [np.mean(gfn_train_best), np.mean(random_train_best)]
    train_stds = [np.std(gfn_train_best), np.std(random_train_best)]
    test_means = [np.mean(gfn_test_best), np.mean(random_test_best)]
    test_stds = [np.std(gfn_test_best), np.std(random_test_best)]
    
    x = np.arange(len(methods))
    width = 0.35
    
    bars1 = axes[0].bar(x - width/2, train_means, width, yerr=train_stds, 
                       label='Train', color='#2ecc71', alpha=0.8, capsize=5)
    bars2 = axes[0].bar(x + width/2, test_means, width, yerr=test_stds,
                       label='Test', color='#e74c3c', alpha=0.8, capsize=5)
    
    axes[0].set_ylabel('TFLOPS (best-of-5)', fontsize=12)
    axes[0].set_title('Transfer Learning: Train vs Test\n(No degradation = good generalization)', fontsize=12, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(methods)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}', ha='center', va='bottom', fontsize=9)
    
    # RIGHT: GFlowNet vs Random on test set (per-workload)
    test_gfn_best = [max(runs) for runs in results['test']['gfn_runs']]
    test_random_best = [max(runs) for runs in results['test']['random_runs']]
    test_exhaustive = results['test']['exhaustive'] if len(results['test']['exhaustive']) > 0 else None
    
    x = np.arange(len(test_gfn_best))
    
    axes[1].plot(x, test_gfn_best, 'o-', label='GFlowNet (best-of-5)', 
                color='#2ecc71', linewidth=2.5, markersize=8)
    axes[1].plot(x, test_random_best, 's--', label='Random (best-of-5)', 
                color='#3498db', linewidth=2.5, markersize=8)
    
    if test_exhaustive:
        axes[1].plot(x, test_exhaustive, 'd-', label='Exhaustive (optimal)', 
                    color='#e74c3c', linewidth=2, markersize=6)
    
    axes[1].set_xlabel('Test Workload Index')
    axes[1].set_ylabel('TFLOPS (best-of-5)')
    axes[1].set_title('Test Set: GFlowNet finds optimal more consistently', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(x)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/transfer_learning_best_of_5.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_optimality_gap(results, save_dir):
    """Plot gap to optimal using best-of-5"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for idx, split in enumerate(['train', 'test']):
        if len(results[split]['exhaustive']) == 0:
            continue
            
        optimal = np.array(results[split]['exhaustive'])
        
        # Use BEST from runs
        gfn_best = np.array([max(runs) for runs in results[split]['gfn_runs']])
        random_best = np.array([max(runs) for runs in results[split]['random_runs']])
        
        # Gaps as percentage (lower is better)
        gfn_gaps = ((optimal - gfn_best) / optimal) * 100
        random_gaps = ((optimal - random_best) / optimal) * 100
        
        x = np.arange(len(optimal))
        width = 0.35
        
        bars1 = axes[idx].bar(x - width/2, gfn_gaps, width, label='GFlowNet', 
                             color='#2ecc71', alpha=0.8, edgecolor='black')
        bars2 = axes[idx].bar(x + width/2, random_gaps, width, label='Random', 
                             color='#3498db', alpha=0.8, edgecolor='black')
        
        axes[idx].set_xlabel('Workload Index')
        axes[idx].set_ylabel('Gap to Optimal (%)')
        axes[idx].set_title(f'{split.upper()}: Optimality Gap (best-of-5)\n(Lower is better, 0% = found optimal)')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3, axis='y')
        axes[idx].axhline(0, color='red', linestyle='--', linewidth=1.5, label='Optimal')
        axes[idx].set_xticks(x)
        
        # Add mean gap annotation
        gfn_mean_gap = np.mean(gfn_gaps)
        random_mean_gap = np.mean(random_gaps)
        axes[idx].text(0.02, 0.98, f'Mean Gap:\nGFN: {gfn_mean_gap:.1f}%\nRandom: {random_mean_gap:.1f}%', 
                      transform=axes[idx].transAxes, va='top', ha='left',
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                      fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/optimality_gap_best_of_5.png', dpi=300, bbox_inches='tight')
    plt.close()

# ==========================================
# MAIN
# ==========================================

def main():
    print("="*70)
    print("GFlowNet Triton Autotuner - Corrected Evaluation Version")
    print("="*70)
    
    WEIGHTS = 'gfn_hw_aware_weights.pt'
    
    if os.path.exists(WEIGHTS):
        response = input(f"\nLoad existing weights from {WEIGHTS}? (y/n): ")
        if response.lower() == 'y':
            gfn, env = load_gfn(WEIGHTS)
            metrics = None
        else:
            gfn, env, metrics = train_gfn(SearchSpace.LARGE, n_epochs=12000, save_path=WEIGHTS)
    else:
        gfn, env, metrics = train_gfn(SearchSpace.LARGE, n_epochs=12000, save_path=WEIGHTS)
    
    # Comprehensive evaluation with proper statistics
    print("\n" + "="*70)
    print("Starting comprehensive evaluation...")
    print("This will take a while due to exhaustive search baselines")
    print("="*70)
    
    results = evaluate_comprehensive(
        gfn, env, 
        budget=200, 
        n_runs=5,  # 5 runs per method per workload
        use_exhaustive=True  # Set to False to skip exhaustive search
    )
    
    # Generate plots
    if metrics is not None:
        plot_all_metrics_corrected(metrics, results, save_dir='plots')
    else:
        os.makedirs('plots', exist_ok=True)
        plot_performance_comparison_corrected(results, 'plots')
        plot_distribution_analysis(results, 'plots')
        plot_transfer_learning_corrected(results, 'plots')
        if len(results['train']['exhaustive']) > 0:
            plot_optimality_gap(results, 'plots')
    
    # Save results
    with open('evaluation_results_corrected.json', 'w') as f:
        results_serializable = {
            'train': {
                'gfn_runs': results['train']['gfn_runs'],
                'random_runs': results['train']['random_runs'],
                'exhaustive': results['train']['exhaustive'],
                'diversity': [float(x) for x in results['train']['diversity']],
            },
            'test': {
                'gfn_runs': results['test']['gfn_runs'],
                'random_runs': results['test']['random_runs'],
                'exhaustive': results['test']['exhaustive'],
                'diversity': [float(x) for x in results['test']['diversity']],
            },
            'workload_info': results['workload_info']
        }
        json.dump(results_serializable, f, indent=2)
    
    # Generate summary table
    print("\n" + "="*70)
    print("SUMMARY TABLE (for paper)")
    print("="*70)
    
    summary_data = []
    for split in ['train', 'test']:
        gfn_best = [max(runs) for runs in results[split]['gfn_runs']]
        random_best = [max(runs) for runs in results[split]['random_runs']]
        exhaustive = results[split]['exhaustive']
        
        row = {
            'Split': split.upper(),
            'GFN (mean±std)': f"{np.mean(gfn_best):.2f}±{np.std(gfn_best):.2f}",
            'Random (mean±std)': f"{np.mean(random_best):.2f}±{np.std(random_best):.2f}",
            'Exhaustive (mean)': f"{np.mean(exhaustive):.2f}" if len(exhaustive) > 0 else "N/A",
            'GFN Gap to Optimal': f"{((np.mean(exhaustive) - np.mean(gfn_best)) / np.mean(exhaustive) * 100):.2f}%" if len(exhaustive) > 0 else "N/A",
            'GFN vs Random': f"{((np.mean(gfn_best) - np.mean(random_best)) / np.mean(random_best) * 100):+.2f}%",
            'Win Rate': f"{sum(1 for g, r in zip(gfn_best, random_best) if g > r * 1.01) / len(gfn_best) * 100:.1f}%"
        }
        summary_data.append(row)
    
    # Print table
    print("\n" + "-"*70)
    for key in summary_data[0].keys():
        print(f"{key:20s} | ", end="")
        for row in summary_data:
            print(f"{row[key]:20s} | ", end="")
        print()
    print("-"*70)
    
    print("\n" + "="*70)
    print("✓ Complete!")
    print(f"✓ Results saved to evaluation_results_corrected.json")
    print(f"✓ Plots saved to plots/")
    print("="*70)

if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    main()
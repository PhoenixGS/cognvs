import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union
from functools import partial
import math

import torch
from einops import rearrange, repeat

from ...util import append_dims, default, instantiate_from_config


class Guider(ABC):
    @abstractmethod
    def __call__(self, x: torch.Tensor, sigma: float) -> torch.Tensor:
        pass

    def prepare_inputs(self, x: torch.Tensor, s: float, c: Dict, uc: Dict) -> Tuple[torch.Tensor, float, Dict]:
        pass


class VanillaCFG:
    """
    implements parallelized CFG
    """

    def __init__(self, scale, dyn_thresh_config=None):
        self.scale = scale
        scale_schedule = lambda scale, sigma: scale  # independent of step
        self.scale_schedule = partial(scale_schedule, scale)
        self.dyn_thresh = instantiate_from_config(
            default(
                dyn_thresh_config,
                {"target": "sgm.modules.diffusionmodules.sampling_utils.NoDynamicThresholding"},
            )
        )

    def __call__(self, x, sigma, scale=None):
        x_u, x_c = x.chunk(2)
        scale_value = default(scale, self.scale_schedule(sigma))
        x_pred = self.dyn_thresh(x_u, x_c, scale_value)
        return x_pred

    def prepare_inputs(self, x, s, c, uc):
        c_out = dict()

        for k in c:
            if k in ["vector", "crossattn", "concat", "pl_emb"]:
                c_out[k] = torch.cat((uc[k], c[k]), 0)
            else:
                assert c[k] == uc[k]
                c_out[k] = c[k]
        return torch.cat([x] * 2), torch.cat([s] * 2), c_out


class DynamicCFG(VanillaCFG):
    def __init__(self, scale, exp, num_steps, dyn_thresh_config=None):
        super().__init__(scale, dyn_thresh_config)
        scale_schedule = (
            lambda scale, sigma, step_index: 1 + scale * (1 - math.cos(math.pi * (step_index / num_steps) ** exp)) / 2
        )
        self.scale_schedule = partial(scale_schedule, scale)
        self.dyn_thresh = instantiate_from_config(
            default(
                dyn_thresh_config,
                {"target": "sgm.modules.diffusionmodules.sampling_utils.NoDynamicThresholding"},
            )
        )

    def __call__(self, x, sigma, step_index, scale=None):
        x_u, x_c = x.chunk(2)
        scale_value = self.scale_schedule(sigma, step_index.item())
        x_pred = self.dyn_thresh(x_u, x_c, scale_value)
        return x_pred

class CMGVanillaCFG:
    def __init__(self, scale_t, scale_c, dyn_thresh_config=None):
        self.scale_t = scale_t
        self.scale_c = scale_c
        scale_t_schedule = lambda scale_t, sigma: scale_t
        scale_c_schedule = lambda scale_c, sigma: scale_c
        self.scale_t_schedule = partial(scale_t_schedule, scale_t)
        self.scale_c_schedule = partial(scale_c_schedule, scale_c)
        self.dyn_thresh = instantiate_from_config(
            default(
                dyn_thresh_config,
                {"target": "sgm.modules.diffusionmodules.sampling_utils.CMGThresholding"},
            )
        )

    def __call__(self, x, sigma, scale=None):
        x_00, x_0t, x_ct = x.chunk(3)
        scale_t = default(scale, self.scale_t_schedule(sigma))
        scale_c = default(scale, self.scale_c_schedule(sigma))
        x_pred = self.dyn_thresh(x_00, x_0t, x_ct, scale_t, scale_c)
        return x_pred

    def prepare_inputs(self, x, s, c, uc):
        c_out = dict()

        assert 'pl_emb' in uc
        assert 'pl_emb' in c

        # 00: uc['pl_emb'], uc[others]
        # 0t: uc['pl_emb'], c[others]
        # ct: c['pl_emb'], c[others]
        for k in c:
            if k in ["vector", "crossattn", "concat"]:
                c_out[k] = torch.cat((uc[k], c[k], c[k]), 0)
            elif k == 'pl_emb':
                c_out[k] = torch.cat((uc[k], uc[k], c[k]), 0)
            else:
                assert c[k] == uc[k]
                c_out[k] = c[k]
        return torch.cat([x] * 3), torch.cat([s] * 3), c_out

class CMGDynamicCFG:
    def __init__(self, scale_t, scale_c, exp, num_steps, dyn_thresh_config=None):
        self.scale_t = scale_t
        self.scale_c = scale_c
        scale_t_schedule = (
            lambda scale_t, sigma, step_index: 1 + scale_t * (1 - math.cos(math.pi * (step_index / num_steps) ** exp)) / 2
        )
        scale_c_schedule = (
            lambda scale_c, sigma, step_index: 1 + scale_c * (1 - math.cos(math.pi * (step_index / num_steps) ** exp)) / 2
        )
        self.scale_t_schedule = partial(scale_t_schedule, scale_t)
        self.scale_c_schedule = partial(scale_c_schedule, scale_c)
        self.dyn_thresh = instantiate_from_config(
            default(
                dyn_thresh_config,
                {"target": "sgm.modules.diffusionmodules.sampling_utils.CMGThresholding"},
            )
        )
    
    def __call__(self, x, sigma, step_index, scale=None):
        x_00, x_0t, x_ct = x.chunk(3)
        scale_t = self.scale_t_schedule(sigma, step_index.item())
        scale_c = self.scale_c_schedule(sigma, step_index.item())
        x_pred = self.dyn_thresh(x_00, x_0t, x_ct, scale_t, scale_c)
        return x_pred

    def prepare_inputs(self, x, s, c, uc):
        c_out = dict()
    

        assert 'pl_emb' in uc
        assert 'pl_emb' in c

        # 00: uc['pl_emb'], uc[others]
        # 0t: uc['pl_emb'], c[others]
        # ct: c['pl_emb'], c[others]
        for k in c:
            if k in ["vector", "crossattn", "concat"]:
                c_out[k] = torch.cat((uc[k], c[k], c[k]), 0)
            elif k == 'pl_emb':
                if c[k].dim() == 5 and uc[k].dim() == 5:
                    c_out[k] = torch.cat((uc[k], uc[k], c[k]), 0)
                elif c[k].dim() == 4 and uc[k].dim() == 4:
                    c_out[k] = torch.cat((uc[k].unsqueeze(0), uc[k].unsqueeze(0), c[k].unsqueeze(0)), 0)
                else:
                    raise ValueError(f"Invalid dimension for pl_emb: {c[k].dim()}, {uc[k].dim()}")
            else:
                assert c[k] == uc[k]
                c_out[k] = c[k]
        for k in c_out:
            print(k, c_out[k].shape)
        print(f"??? {torch.cat([x] * 3).shape}")
        return torch.cat([x] * 3), torch.cat([s] * 3), c_out

class IdentityGuider:
    def __call__(self, x, sigma):
        return x

    def prepare_inputs(self, x, s, c, uc):
        c_out = dict()

        for k in c:
            c_out[k] = c[k]

        return x, s, c_out

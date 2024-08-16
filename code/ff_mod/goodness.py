from abc import ABC, abstractmethod
from typing import Any
import torch


class Goodness(ABC):
    def __init__(self,  positive_split : int = 10_000_000, topk_units : int = None) -> None:
        """_summary_

        Args:
            positive_split (int, optional): _description_. Defaults to 10_000_000.
            topk_units (_type_, optional): If int value, defines number of topk_units. Defaults to None.
        """
        self.positive_split = positive_split
        
        self.empty_gx, self.empty_gy = False, False
        
        self.topk_units = topk_units
        
        if self.positive_split == 10_000_000:
            self.empty_gy = True
        elif self.positive_split <= 0:
            self.empty_gx = True
    
    @abstractmethod
    def __call__(self, x):
        """ Returns g_x and g_y"""
        pass
    
    def apply_topk(self, x, k = None):
        
        if self.topk_units is None:
            return x
        
        if k is None:
            k = self.topk_units
        
        if self.positive_split == 10_000_000:
            val_ex, ind_ex = torch.topk(x, k, dim=1)
            x = torch.zeros_like(x).scatter(1, ind_ex, val_ex)
        else:
            val_ex, ind_ex = torch.topk(x[:,:self.positive_split], k, dim=1)
            val_in, ind_in = torch.topk(x[:,self.positive_split:], k, dim=1)
            
            # Create a new vector with those values 
            x = torch.zeros_like(x)
            x = x + torch.zeros_like(x).scatter(1, ind_ex, val_ex)
            x = x + torch.zeros_like(x).scatter(1, self.positive_split + ind_in, val_in)
        
        return x
    
    def normalize(self, x):
        x[:, :self.positive_split] = x[:, :self.positive_split] / (torch.norm(x[:, :self.positive_split], p=2, dim=1).unsqueeze(1) + 1e-5)
        x[:, self.positive_split:] = x[:, self.positive_split:] / (torch.norm(x[:, self.positive_split:], p=2, dim=1).unsqueeze(1) + 1e-5)
        
        return x


class L2_Goodness(Goodness):
    """
        Standard Goodness from original paper.
    """
    def __init__(self, positive_split : int = 10_000_000, use_mean = False, **kwargs) -> None:
        super().__init__(positive_split=positive_split, **kwargs)

        self.use_mean = use_mean
        
        
    def __call__(self, x):
        zero_val = torch.tensor(0, device=x.device, dtype=x.dtype)
        
        if self.topk_units is not None:
            x = self.apply_topk(x)

        if self.use_mean:
            g_x = x[:, :self.positive_split].pow(2).mean(1) if not self.empty_gx else zero_val
            g_y = x[:, self.positive_split:].pow(2).mean(1) if not self.empty_gy else zero_val
        else:
            g_x = x[:, :self.positive_split].pow(2).sum(1) if not self.empty_gx else zero_val
            g_y = x[:, self.positive_split:].pow(2).sum(1) if not self.empty_gy else zero_val

        
        return (g_x, g_y)

class L2_Goodness_SQRT(Goodness):
    """
        Standard Goodness from original paper.
    """
    def __init__(self, positive_split : int = 10_000_000, use_mean = False, **kwargs) -> None:
        super().__init__(positive_split=positive_split, **kwargs)

        self.use_mean = use_mean
        
    def __call__(self, x):
        zero_val = torch.tensor(0, device=x.device, dtype=x.dtype)
        
        if self.topk_units is not None:
            x = self.apply_topk(x)

        if self.use_mean:
            g_x = x[:, :self.positive_split].pow(2).mean(1).sqrt() if not self.empty_gx else zero_val
            g_y = x[:, self.positive_split:].pow(2).mean(1).sqrt() if not self.empty_gy else zero_val
        else:
            g_x = x[:, :self.positive_split].pow(2).sum(1).sqrt() if not self.empty_gx else zero_val
            g_y = x[:, self.positive_split:].pow(2).sum(1).sqrt() if not self.empty_gy else zero_val


        return (g_x, g_y)
    
class Norm_goodness(Goodness):
    def __init__(self, positive_split : int = 10_000_000, use_mean = False, **kwargs) -> None:
        super().__init__(positive_split=positive_split, **kwargs)
        
        self.use_mean = use_mean
        
    def __call__(self, x):
        zero_val = torch.tensor(0, device=x.device, dtype=x.dtype)
        
        if self.topk_units is not None:
            x = self.apply_topk(x)

        g_x = x[:, :self.positive_split].norm(2, dim = 1)
        g_y = x[:, self.positive_split:].norm(2, dim = 1)
        

        return (g_x, g_y)

class L1_Goodness(Goodness):
    def __init__(self, positive_split : int = 10_000_000, use_mean = False, **kwargs) -> None:
        super().__init__(positive_split=positive_split, **kwargs)
        
        self.use_mean = use_mean
        
    def __call__(self, x):
        zero_val = torch.tensor(0, device=x.device, dtype=x.dtype)
        
        if self.topk_units is not None:
            x = self.apply_topk(x)

        if self.use_mean:
            g_x = x[:, :self.positive_split].abs().mean(1) if not self.empty_gx else zero_val
            g_y = x[:, self.positive_split:].abs().mean(1) if not self.empty_gy else zero_val
        else:
            g_x = x[:, :self.positive_split].abs().sum(1) if not self.empty_gx else zero_val
            g_y = x[:, self.positive_split:].abs().sum(1) if not self.empty_gy else zero_val
        
        
        return (g_x, g_y)
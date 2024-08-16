import torch

class kWTA:
    def __init__(self, k = 1):
        self.k = k
    
    def __call__(self, x):
        # X has a shape of ((batch_size, n), (batch_size, n), ...)
        
        for i, subset in enumerate(x):
            # Subset of shape (batch_size, n)
            val_ex, ind_ex = torch.topk(subset, self.k, dim=1)
            
            result = torch.zeros_like(subset).scatter(1, ind_ex, val_ex)
            
            x[i] = result
        
        return x
            
            
            
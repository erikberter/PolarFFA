from typing import Optional
import torch


class ProbabilityBCELoss:
    """
        BCE Loss for probability values of the goodness vector
    """
    def __init__(self):
        super().__init__()
        
    def __call__(self, g_pos, g_neg, **kwargs):
        
        return -torch.log(torch.cat([g_pos, (1-g_neg)])).mean()
    

class BCELoss:
    """
        BCE Loss for probability values of the goodness vector
    """
    def __init__(self, probability_function = None):
        super().__init__()
        self.probability_function = probability_function
        
    def __call__(self, g_pos, g_neg, **kwargs):
        g_pos = self.probability_function(g_pos)
        g_neg = self.probability_function(g_neg)
        
        g_pos = torch.clamp(g_pos, 1e-5, 1-1e-5)
        g_neg = torch.clamp(g_neg, 1e-5, 1-1e-5)
        
        return -torch.log(torch.cat([g_pos, (1-g_neg)])).mean()
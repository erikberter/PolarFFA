from abc import ABC, abstractmethod
from typing import Any
import torch
# Functional
import torch.nn.functional as F



# TODO Create as abstract
class Probability(ABC):
    def __init__(self) -> None:
        pass
    
    @abstractmethod
    def __call__(self, g_x, g_y):
        """

        Args:
            g_x (_type_): Batch of positive goodnesses
            g_y (_type_): Batch of negative goodnesses
        """
        pass

class SigmoidProbability(Probability):
    """
        Standard Probability from original paper.
    """
    
    def __init__(self, theta : float = 2, alpha : float= 1) -> None:
        self.theta = theta
        self.alpha = alpha
    
    def __call__(self, goodness):
        g_x, g_y = goodness
        
        return F.sigmoid(self.alpha * (g_x - g_y - self.theta))
    
class SymmetricFFAProbability(Probability):
    """
        Probability from the Symmetric FFA paper.
    """
    
    def __init__(self, eps : float = 1e-6) -> None:
        self.eps = eps
    
    def __call__(self, goodness):
        g_x, g_y = goodness
        return (g_x + self.eps) / (g_x + g_y + 2 * self.eps)

class ExponentialProbability(Probability):
    """
        Exponential probability.
        
        Problems with negative goodness as e^(g_x - g_y) can be greater than 1.
    """
    
    def __init__(self, alpha : float = 1) -> None:
        self.alpha = alpha
    
    def __call__(self, goodness):
        g_x, g_y = goodness
        
        return 1-torch.exp(-self.alpha * torch.max(0, g_x - g_y))
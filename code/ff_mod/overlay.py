from abc import ABC, abstractmethod
from typing import Any, List
import torch

from torchvision.transforms import GaussianBlur


class Overlay(ABC):
    def __init__(self) -> None:
        pass
    
    @abstractmethod
    def get_positive(self, data, **kwargs):
        """Return positive data with the respective overlay"""
        
    @abstractmethod
    def get_negative(self, data, **kwargs):
        """Return negative data with the respective overlay"""
    
    @abstractmethod
    def save(self, path : str, **kwargs):
        pass
    
    @abstractmethod
    def load(self, path : str, **kwargs):
        pass
    

class MultiOverlay(Overlay):
    def __init__(self):
        self.overlays : List[Overlay] = []
    
    def add_overlay(self, overlay : Overlay):
        self.overlays.append(overlay)
    
    def get_positive(self, data, **kwargs):
        for overlay in self.overlays:
            data = overlay.get_positive(data, **kwargs)
        return data
    
    def get_negative(self, data, **kwargs):
        for overlay in self.overlays:
            data = overlay.get_negative(data, **kwargs)
        return data
    
    def save(self, path: str, **kwargs):
        for i, overlay in enumerate(self.overlays):
            overlay.save(path + f"_multi_{i}", **kwargs)
    
    def load(self, path: str, **kwargs):
        for i, overlay in enumerate(self.overlays):
            overlay.load(path + f"_multi_{i}", **kwargs)
            
            
            
class CornerOverlay(Overlay):
    """
    Original overlaying technique. It puts a one-hot encoding of the label in the left corner of the image.

    Negative samples are obtained by uniformly sampling a label different from the original one.
    
    Args:
        Overlay (_type_): _description_
    """
    def __init__(self, num_classes = 10):
        super().__init__()
        
        self.num_classes = num_classes

    def get_positive(self, data, **kwargs):
        labels = kwargs["labels"]

        if isinstance(labels, int):
            labels = torch.full((data.shape[0],), labels, dtype=torch.long)

        data_ = data.clone()
        
        data_[..., :self.num_classes] *= 0.0
        data_[range(data.shape[0]), ..., labels] = data.max()
        
        return data_, labels
        
    def get_negative(self, data, **kwargs):
        labels = kwargs["labels"]
        
        if isinstance(labels, int):
            labels = torch.full((data.shape[0],), labels, dtype=torch.long)

        labels = (labels + torch.randint(1, self.num_classes, (data.shape[0],), device=data.device)) % self.num_classes
        
        data_ = data.clone()
        
        data_[..., :self.num_classes] *= 0.0
        data_[range(data.shape[0]), ..., labels] = data.max()
        
        return data_, labels
    
    def save(self, path):
        pass
    
    def load(self, path):
        pass


# TODO Currently only valid for 1D data, still requires implementation for other dimensions 
class AppendToEndOverlay(Overlay):
    def __init__(
            self,
            pattern_size: int,
            num_classes: int,
            num_vectors: int = 1,
            p: float = 0.2,
            device: str = "cuda:0"
        ):
        """
        Args:
            pattern_size (int): Size of each pattern
            classes (int): Number of classes
            num_vectors (int): Number of vectors per class. Defaults to 1.
            p (float, optional): Percentaje of ones in the vectors. Defaults to 0.2.
            device (str, optional): Device of the patterns. Defaults to "cuda:0".
        """
        self.pattern_size = pattern_size
        self.num_classes = num_classes
        self.num_vectors = num_vectors
        
        self.device = device

        self.label_vectors = torch.bernoulli(p * torch.ones(num_classes, num_vectors, pattern_size)).to(device)

    def save(self, path):
        # Save the label_vectors tensor to a file
        torch.save(self.label_vectors, path)
    
    def load(self, path):
        # Load the label_vectors tensor from a file
        self.label_vectors = torch.load(path,  map_location=torch.device(self.device))
        
        self.classes = self.label_vectors.shape[0]
        self.num_vectors = self.label_vectors.shape[1]
        self.pattern_size = self.label_vectors.shape[2]

    def get_positive(self, data, **kwargs):
        labels = kwargs["labels"]
        
        batch_label_vectors = self.label_vectors[labels].squeeze(1)
        
        data_ = torch.cat([data, batch_label_vectors], 1).to(data.device)
        
        return data_, labels

    def get_negative(self, data, **kwargs):
        labels = kwargs["labels"]
        
        random_change = torch.randint(1, self.num_classes, (data.shape[0],), device=data.device)
        
        labels = (labels + random_change) % self.num_classes
        
        batch_label_vectors = self.label_vectors[labels].squeeze(1)
        
        data_ = torch.cat([data, batch_label_vectors], 1).to(data.device)
        
        return data_, labels


# TODO
class FussionOverlay(Overlay):
    def __init__(self):
        super().__init__()

    def apply(self, batch, label, steps=7, **kwargs):
        # TODO Improve in future versions
        batch = batch.reshape(batch.shape[0], 1, 28, 28)
        bitmap = torch.randint_like(batch, 0, 2, dtype=torch.long)

        gauss = GaussianBlur(kernel_size=(5, 5))

        for _ in range(steps):
            bitmap = gauss(bitmap)

        permu = torch.randperm(batch.shape[0])

        result = batch * bitmap + batch[permu] * (1 - bitmap)
        result = result.reshape(batch.shape[0], -1)
        return result


    
    
# TODO FloatClassGenerator
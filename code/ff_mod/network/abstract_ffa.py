import torch
import torch.nn as nn
import torch.nn.functional as F

from ff_mod.overlay import Overlay

from abc import ABC, abstractmethod

class AbstractForwardForwardNet(nn.Module):
    """
    Abstract class representing a FeedForward Forward-Forward Neural Network.
    
    Ref:
        - The Forward-Forward Algorithm: Some Preliminary Investigations - G. Hinton (https://arxiv.org/pdf/2212.13345.pdf)
    """
    def __init__(
            self,
            overlay_function : Overlay,
            first_prediction_layer = 0,
            residual_connections = False
        ):
        """
        Args:
            overlay_function (_type_): _description_
            num_classes (_type_): Number of classes of the dataset.
            residual_connections: If True, the input of each layer will be concatenated with the output of the previous layer.
        """
        super().__init__()
        self.overlay_function : Overlay = overlay_function
        
        self.first_prediction_layer = first_prediction_layer
        
        self.layers = torch.nn.ModuleList()
        
        self.residual_connections = residual_connections

    def add_layer(self, layer : nn.Module):
        self.layers.append(layer)
        
        if len(self.layers) == 1:
            layer.is_input_layer = True
        
    
    @abstractmethod
    def prepare_data(self, data):
        """
        Adjust the data based on the network's properties.
        This method needs to be implemented by a subclass.
        """
        
        
    def train_network(self, x_pos, x_neg, labels = None):
        """
        Train the network based on the positive and negative examples.
        """
        
        x_pos = self.prepare_data(x_pos)
        x_neg = self.prepare_data(x_neg)
        
        x_pos_base = x_pos.clone().detach()
        x_neg_base = x_neg.clone().detach()
        
        for i, layer in enumerate(self.layers):
            
            if i > 0 and self.residual_connections:
                x_pos = torch.cat([x_pos, x_pos_base], 1)
                x_neg = torch.cat([x_neg, x_neg_base], 1)
                
            x_pos, x_neg = layer.train_network(x_pos, x_neg, labels = labels)
            
            if i == 0:
                latents_per_class = [x_pos[labels[0] == i].mean(0) for i in range(labels[0].max() + 1)]
                
                overlap = 0
                for i in range(len(latents_per_class)):
                    for j in range(i+1, len(latents_per_class)):
                        overlap += (latents_per_class[i] * latents_per_class[j]).sum()
                
            
            x_pos, x_neg = layer.goodness.normalize(x_pos), layer.goodness.normalize(x_neg)
    
    @torch.no_grad()
    def predict(self, x, total_classes):
        """
        Predict the labels of the input data.

        Args:
            x (torch.Tensor): Input data. 
                Shape: [batch_size, input_size] if the network is not spiking, [batch_size, num_steps, input_size] otherwise.

        Returns:
            torch.Tensor: Predicted labels.
        """
        goodness_scores = []
        
        for label in range(total_classes):
            h, _ = self.overlay_function.get_positive(x, labels = torch.full((x.shape[0],), label, dtype=torch.long))
            h = self.prepare_data(h)
            
            h_base = h.clone().detach()
            
            goodness = torch.zeros(h.shape[0], 1).to(x.device)
            for j, layer in enumerate(self.layers):
                
                if j > 0 and self.residual_connections:
                    h = torch.cat([h, h_base], 1)

                h = layer(h)
                if j >= self.first_prediction_layer:
                    g_pos = layer.goodness(h)
                    goodness += layer.loss.probability_function(g_pos).unsqueeze(1)
            
                h = layer.goodness.normalize(h)
                
            goodness_scores += [goodness]
        return torch.cat(goodness_scores, 1).argmax(1)
    
    # TODO Improve
    @torch.no_grad()
    def get_latent(self, x, labels, depth):
        """Get the latent activation of the network at certain depth.

        Args:
            x (Torch.Tensor): Input data.
            label (_type_): Labels of the input data.
            depth (_type_): Depth of the latent activation.

        Returns:
            _type_: _description_
        """

        h, _ = self.overlay_function.get_positive(x, labels = labels)
        h = self.prepare_data(h)
        
        h_base = h.clone().detach()
        
        final_layer = min(len(self.layers), depth)

        for i in range(final_layer):
            if i > 0:
                h = self.layers[i-1].goodness.normalize(h)
                
            if i > 0 and self.residual_connections:
                h = torch.cat([h, h_base], 1)
            
            h = self.layers[i](h)
            
        return h
    
    def save_network(self, path):
        torch.save(self.state_dict(), path)
        self.overlay_function.save(path + "_overlay_function")
    
    def load_network(self, path):
        self.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        self.overlay_function.load(path + "_overlay_function")
        
        
class AbstractForwardForwardLayer(ABC, nn.Linear):
    """
    Abstract class representing a layer in a Feed-Forward Neural Network.
    """
    def __init__(
            self,
            in_features,
            out_features,
            goodness_function,
            loss_function,
            bias=False,
            device="cuda:0",
            dtype=None
        ):
        super().__init__(in_features, out_features, bias, device, dtype)
        
        self.goodness = goodness_function
        self.loss = loss_function
        
        self.is_input_layer = False

    @abstractmethod
    def forward(self, x):
        pass
    
    def train_network(self, x_pos, x_neg, labels = None):
        y, y_rnd = labels[0], labels[1]
        
        self.opt.zero_grad()
                    
        latent_pos = self.forward(x_pos)
        g_pos = self.goodness(latent_pos)

        latent_neg = self.forward(x_neg)
        g_neg = self.goodness(latent_neg)
        
        loss = self.loss(g_pos, g_neg, latent_pos = latent_pos, latent_neg = latent_neg, labels = y, pre_dim = self.in_features)
        loss.backward()
        
        self.opt.step()
            
        with torch.no_grad():
            return self.forward(x_pos).detach(), self.forward(x_neg).detach()
        

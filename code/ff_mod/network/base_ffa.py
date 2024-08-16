import math
import torch
from torch.optim import Adam

from ff_mod.network.abstract_ffa import AbstractForwardForwardNet, AbstractForwardForwardLayer

from ff_mod.overlay import Overlay

class FFANetwork(AbstractForwardForwardNet):

    def __init__(
            self,
            overlay_function : Overlay,
            first_prediction_layer = 0,
            residual_connections = False
        ):
        
        super().__init__(overlay_function, first_prediction_layer, residual_connections)
        
        
    def prepare_data(self, data):
        # Flatten the data
        data = data.view(data.size(0), -1)
        return data


class FFALayer(AbstractForwardForwardLayer):
    def __init__(
            self,
            in_features,
            out_features,
            goodness_function,
            loss_function,
            activation = torch.nn.ReLU(),
            learning_rate = 0.001,
            **kwargs
        ):
        super().__init__(
            in_features,
            out_features,
            goodness_function=goodness_function,
            loss_function = loss_function,
            **kwargs
        )
        
        self.activation = activation
        
        self.opt = Adam(self.parameters(), lr=learning_rate)
    
    def forward(self, x):
        
        bias_term = 0
        if self.bias is not None:
            bias_term = self.bias.unsqueeze(0)

        result =  self.activation(torch.mm(x, self.weight.T) + bias_term)
        
        return result

class ConvFFALayer(AbstractForwardForwardLayer):
    
    def __init__(self, in_features, out_features, goodness_function, loss_function, activation = torch.nn.ReLU(), input_shape = (28, 28), label_size = -1, kernel_size = 7, stride = 1, learning_rate = 0.01, bias=False, device="cuda:0", dtype=None):
        super().__init__(in_features, out_features, goodness_function, loss_function, bias, device, dtype)
        
        # Assume label is at end
        self.label_size = label_size
        self.kernel_size = kernel_size
        self.stride = stride
        
        self.input_shape = input_shape
        
        self.activation = activation
        
        self.device = device
        
        self.opt = Adam(self.parameters(), lr=learning_rate)
        
    def create_conv_mask(self):
        # Make the weights
        if self.label_size == -1:
            mask = torch.zeros_like(self.weight, device=self.device)
        else:
            mask = torch.zeros_like(self.weight[:, :-self.label_size], device=self.device)
        
        input_x, input_y = self.input_shape
        
        for i in range(self.out_features):
            mask_square = torch.zeros_like(mask[i], device=self.device).reshape(input_x, input_y)
            
            total_elements = math.floor((input_x-self.kernel_size)/self.stride) * math.floor((input_y-self.kernel_size)/self.stride)
            
            total_loops = self.out_features // total_elements
            
            
            w_start = ((i // total_loops) * self.stride) % (input_x-self.kernel_size)
            h_start = (((i // total_loops) // (input_x-self.kernel_size) * self.stride)) % (input_x-self.kernel_size)
            
            mask_square[w_start:w_start+self.kernel_size, h_start:h_start+self.kernel_size] = 1
            
            mask[i] = mask_square.reshape(input_x * input_y)

        
        if self.label_size != -1:
            mask = torch.cat((mask, torch.ones((mask.shape[0], self.label_size), device=self.device)), dim=1)
        
        self.weight_mask = mask.to(self.device)
        
        
        
    def forward(self, x):
        bias_term = 0
        if self.bias is not None:
            bias_term = self.bias.unsqueeze(0)
        
        current_weight = self.weight * self.weight_mask
        
        result =  self.activation(torch.mm(x, current_weight.T) + bias_term)

        return result
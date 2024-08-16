
import torch
import numpy as np
from tqdm import tqdm

from ff_mod.network.abstract_ffa import AbstractForwardForwardNet
from ff_mod.callback.callback import CallbackList, Callback


class Trainer:
    def __init__(self, unsupervised = False, device = "cuda:0", greedy_goodness = False, early_stop = None) -> None:
        self.overlay_function = None
        self.__net : AbstractForwardForwardNet = None
        self.device = device
        
        self.callbacks = CallbackList()
        
        self.train_loader = None
        self.test_loader = None
        
        self.unsupervised = unsupervised
        self.fusion_func = None
        
        self.is_emnist = False
        
        self.greedy_goodness = greedy_goodness
        
        self.early_stop = early_stop
        self.max_accuracy = 0.0
        self.consecutive_no_improvement = 0
    
    def set_network(self, net : AbstractForwardForwardNet) -> None:
        self.__net = net
        self.overlay_function = net.overlay_function

    def get_network(self) -> AbstractForwardForwardNet:
        return self.__net

    def add_callback(self, callback : Callback):
        self.callbacks.add(callback)
    
    def set_fusion_func(self, fusion_func):
        self.fusion_func = fusion_func
    
    def set_dataloader(self, train_loader, test_loader, is_emnist = False):
        self.train_loader = train_loader
        self.test_loader = test_loader
        
        self.is_emnist = is_emnist
        
        # TODO Put the num classes
        self.num_classes = 10
        
        if is_emnist:
            self.num_classes = 26
        
    def train_epoch(self, verbose: int = 1, end_step = None):
        if verbose > 0: print("Train epoch")
            
        self.__net.train()
        for step, (x,y) in  tqdm(enumerate(iter(self.train_loader)), total = len(self.train_loader), leave=True, disable=verbose<2):
            if end_step is not None and step > end_step:
                break
            self.callbacks.on_train_batch_start()

            x, y = x.to(self.device), y.to(self.device)
            
            if self.is_emnist:
                y-=1 # It starts from 1 to 27

            # Prepare data
            x_pos, _ = self.overlay_function.get_positive(x, labels = y, num_classes = self.num_classes)
            x_neg, y_rnd = self.overlay_function.get_negative(x, labels = y, num_classes = self.num_classes)
            
            # Train the network
            self.__net.train_network(x_pos, x_neg, labels = [y,y_rnd])
            
            # Get the predictions
            predicts = self.__net.predict(x, self.num_classes)
            
            self.callbacks.on_train_batch_end(predictions = predicts.cpu(), y = y.cpu())

    @torch.no_grad()
    def get_best_class(self, x, y):
        """ Get the best negative class for each sample in x """
        # TODO Greedy not implemented for subsets 
        alls = torch.zeros((self.num_classes-1, x.size(0)), device = self.device)
        for i in range(1, self.num_classes):
            y_temp = (y+i)%self.num_classes
            
            x_exp = self.__net.get_latent(x, y_temp, 1)
            alls[i-1] = self.__net.layers[0].get_goodness(x_exp)

        return alls.max(0)[1] + 1
    
    @torch.no_grad()
    def test_epoch(self, verbose: int = 1):
        if verbose > 0: print("Test epoch")
        
        accuracy = 0.0
        
        self.__net.eval()
        for step, (x,y) in enumerate(iter(self.test_loader)):
            x, y = x.to(self.device), y.to(self.device)
            #y -= 1
            self.callbacks.on_test_batch_start()
            
            predicts = self.__net.predict(x, self.num_classes)
            accuracy += predicts.eq(y).float().sum().item()
            
            self.callbacks.on_test_batch_end(predictions = predicts.cpu(), y = y.cpu())

        torch.cuda.empty_cache()
        
        return accuracy / len(self.test_loader.dataset)

    def train(self, epochs: int = 2, verbose: int = 1):
        self.max_accuracy = 0.0
        for epoch in range(epochs):
            if verbose > 0: print(f"Epoch {epoch}")
                
            self.train_epoch(verbose=verbose)
            self.callbacks.on_train_epoch_end()
            
            accuracy = self.test_epoch(verbose=verbose)
            self.callbacks.on_test_epoch_end()
            
            if accuracy > self.max_accuracy:
                self.max_accuracy = accuracy
                self.consecutive_no_improvement = 0
            else:
                self.consecutive_no_improvement += 1
                
                if self.early_stop is not None and self.consecutive_no_improvement >= self.early_stop:
                    print("Early stopping")
                    break
            
            self.callbacks.next_epoch()
            torch.cuda.empty_cache()
import os

import torch
import torch.nn as nn
from torch.utils.tensorboard.writer import SummaryWriter

from ff_mod.goodness import L2_Goodness, L1_Goodness, Norm_goodness, L2_Goodness_SQRT
from ff_mod.probability import SigmoidProbability, SymmetricFFAProbability

from ff_mod.network.base_ffa import FFANetwork, FFALayer 

from ff_mod.overlay import AppendToEndOverlay, CornerOverlay
from ff_mod.loss import BCELoss

from ff_mod.callback.accuracy_writer import AccuracyWriter
from ff_mod.callback.best_model_saver import BestModelSaver
from ff_mod.trainer import Trainer

import torchvision

from datetime import datetime

import json

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

BATCH_SIZE = 512
N_EPOCHS = 50
LEARNING_RATE = 0.001

DIM = 1000

NUM_CLASSES = 10

EXPERIMENTAL_FOLDER = "experiments"

mnist_train = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('data', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                                   torchvision.transforms.Lambda(lambda x: x.view(-1))
                               ])),
    batch_size=512, shuffle=True)

mnist_test = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('data', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                                    torchvision.transforms.Lambda(lambda x: x.view(-1))
                               ])),
    batch_size=512, shuffle=True)

kmnist_train = torch.utils.data.DataLoader(
    torchvision.datasets.KMNIST('data', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                                   torchvision.transforms.Lambda(lambda x: x.view(-1))
                               ])),
    batch_size=512, shuffle=True)

kmnist_test = torch.utils.data.DataLoader(
    torchvision.datasets.KMNIST('data', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                                    torchvision.transforms.Lambda(lambda x: x.view(-1))
                               ])),
    batch_size=512, shuffle=True)

fashion_train = torch.utils.data.DataLoader(
    torchvision.datasets.FashionMNIST('data', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                                   torchvision.transforms.Lambda(lambda x: x.view(-1))
                               ])),
    batch_size=512, shuffle=True)

fashion_test = torch.utils.data.DataLoader(
    torchvision.datasets.FashionMNIST('data', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                                    torchvision.transforms.Lambda(lambda x: x.view(-1))
                               ])),
    batch_size=512, shuffle=True)


cifar10_train = torch.utils.data.DataLoader(
    torchvision.datasets.CIFAR10('data', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                                   torchvision.transforms.Lambda(lambda x: x.view(-1))
                               ])),
    batch_size=512, shuffle=True)

cifar10_test = torch.utils.data.DataLoader(
    torchvision.datasets.CIFAR10('data', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                                    torchvision.transforms.Lambda(lambda x: x.view(-1))
                               ])),
    batch_size=512, shuffle=True)

emnist_train = torch.utils.data.DataLoader(
    torchvision.datasets.EMNIST('data', split='letters', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                                   torchvision.transforms.Lambda(lambda x: x.view(-1))
                               ])),
    batch_size=512, shuffle=True)

emnist_test = torch.utils.data.DataLoader(
    torchvision.datasets.EMNIST('data', split='letters', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                                    torchvision.transforms.Lambda(lambda x: x.view(-1))
                               ])),
    batch_size=512, shuffle=True)

datasets = {
    'mnist': (mnist_train, mnist_test, 28*28, 10),
    'kmnist': (kmnist_train, kmnist_test, 28*28, 10),
    'fashion': (fashion_train, fashion_test, 28*28, 10),
    #'emnist': (emnist_train, emnist_test, 28*28, 26),
    #'cifar10': (cifar10_train, cifar10_test, 32*32*3, 10)
}

activations = {
    'ReLU': torch.nn.ReLU(),
    'Sigmoid': torch.nn.Sigmoid(),
    'Tanh': torch.nn.Tanh()
}

goodnesses = {
    'L2M_TK15': L2_Goodness(use_mean=True, topk_units=15),
    'L2M': L2_Goodness(use_mean=True),
    'L1M': L1_Goodness(use_mean=True),
    'L1M_TK15': L1_Goodness(use_mean=True, topk_units=15),
    'L2M_Split': L2_Goodness(positive_split=DIM//2, use_mean=True),
    'L2M_TK15_Split': L2_Goodness(positive_split=DIM//2, use_mean=True, topk_units=15),
    'L1M_Split': L1_Goodness(positive_split=DIM//2, use_mean=True),
    'L1M_TK15_Split': L1_Goodness(positive_split=DIM//2, use_mean=True, topk_units=15),
    'L2S': L2_Goodness(use_mean=False),
    'L2S_TK15': L2_Goodness(use_mean=False, topk_units=15),
    'L1S': L1_Goodness(use_mean=False),
    'L1S_TK15': L1_Goodness(use_mean=False, topk_units=15),
    'L2S_Split': L2_Goodness(positive_split=DIM//2, use_mean=False),
    'L2S_TK15_Split': L2_Goodness(positive_split=DIM//2, use_mean=False, topk_units=15),
    'L1S_Split': L1_Goodness(positive_split=DIM//2, use_mean=False),
    'L1S_TK15_Split': L1_Goodness(positive_split=DIM//2, use_mean=False, topk_units=15),
    'L2Sq' : L2_Goodness_SQRT(use_mean=False),
    'L2Sq_TK15' : L2_Goodness_SQRT(use_mean=False, topk_units=15),
    'L2Sq_Split' : L2_Goodness_SQRT(positive_split=DIM//2, use_mean=False),
    'L2Sq_TK15_Split' : L2_Goodness_SQRT(positive_split=DIM//2, use_mean=False, topk_units=15),
    'L2Mq' :  L2_Goodness_SQRT(use_mean=True),
    'L2Mq_TK15' :  L2_Goodness_SQRT(use_mean=True, topk_units=15),
    'L2Mq_Split' :  L2_Goodness_SQRT(positive_split=DIM//2, use_mean=True),
    'L2Mq_TK15_Split' :  L2_Goodness_SQRT(positive_split=DIM//2, use_mean=True, topk_units=15)
}

probabilities = {
    'SigmoidProbability_Theta_0': SigmoidProbability(theta=0),
    'SigmoidProbability_Theta_2': SigmoidProbability(theta=2),
    'SymmetricFFAProbability': SymmetricFFAProbability()
}

def get_existing_configs(path):
    current_models = []
    
    if not os.path.exists(path):
        return current_models
    
    for folder in os.listdir(path):
        # Read json file config in folder
        with open(os.path.join(path, folder, 'config.json'), 'r') as f:
            config = json.load(f)

            config_str = f"{config['activation']}_{config['goodness']}_{config['probability']}"
            
            current_models += [config_str]
    return current_models

def create_network(goodness, activation, probability, input_size = 784):
    overlay = AppendToEndOverlay(pattern_size=100, num_classes=NUM_CLASSES, p=0.1)
    #overlay = CornerOverlay(num_classes=10)
    network = FFANetwork(overlay)
    
    if probability == 'SigmoidProbability_Theta_2' and "Split" not in goodness and activation == 'Sigmoid':
        if 'L2' in goodness:
            probabilities[probability].theta = 0.2
            probabilities[probability].alpha = 5
        elif 'L1' in goodness:
            probabilities[probability].theta = 0.4
            probabilities[probability].alpha = 2.5
    else:
        probabilities[probability].theta = 2
        probabilities[probability].alpha = 1
        
    loss = BCELoss(probability_function=probabilities[probability])
    
    network.add_layer(FFALayer(input_size + 100, DIM, goodnesses[goodness], loss, activations[activation], learning_rate=LEARNING_RATE))
    network.add_layer(FFALayer(DIM, DIM, goodnesses[goodness], loss, activations[activation], learning_rate=LEARNING_RATE))
    
    return network

def run_experiment(goodness, activation, probability, dataset='mnist'):
    exp_date = datetime.now().strftime("%Y%m%d%H%M%S")
    
    os.makedirs(EXPERIMENTAL_FOLDER, exist_ok=True)
    os.makedirs(f"{EXPERIMENTAL_FOLDER}/{dataset}", exist_ok=True)
    os.makedirs(f"{EXPERIMENTAL_FOLDER}/{dataset}/exp_{exp_date}/", exist_ok=True)
    
    with open(f"{EXPERIMENTAL_FOLDER}/{dataset}/exp_{exp_date}/config.json", "w") as f:
        json.dump({"goodness": goodness, "activation": activation, "probability": probability}, f)
    
    network = create_network(goodness, activation, probability, datasets[dataset][2])
    
    writer = SummaryWriter(f"{EXPERIMENTAL_FOLDER}/{dataset}/exp_{exp_date}/summary/" )
        
    trainer = Trainer(device = 'cuda:0')
    trainer.add_callback(AccuracyWriter(tensorboard=writer))
    trainer.add_callback(BestModelSaver(f"{EXPERIMENTAL_FOLDER}/{dataset}/exp_{exp_date}/", network))
    trainer.set_network(network)
    
    trainer.set_dataloader(datasets[dataset][0], datasets[dataset][1], is_emnist=dataset == 'emnist')
    trainer.num_classes = datasets[dataset][3]
    trainer.train(N_EPOCHS, verbose=-1)
    
    network.save_network(f"{EXPERIMENTAL_FOLDER}/{dataset}/exp_{exp_date}/model")


current_models = []
for dataset in datasets:
    temp = get_existing_configs(f"{EXPERIMENTAL_FOLDER}/{dataset}")
    current_models += [x + "_" + dataset for x in temp]


for dataset in datasets:
    for activation in activations:
        for goodness in goodnesses:
            for probability in probabilities:
                
                # We have to skip some combinations
                # - Symmetric probability only works with split goodness
                # - Split goodness only works with theta 0 probability
                
                if "Symmetric" in probability:
                    if "Split" not in goodness or "TK15" not in goodness:
                        print("Skipping", f"{dataset} | {activation} | {goodness} | {probability} |  it lacks either Split or TK15")
                        continue
                else:
                    if "Split" in goodness and "SigmoidProbability_Theta_0" != probability:
                        # If its split, theta only hinder accuracy
                        print("Skipping", f"{dataset} | {activation} | {goodness} | {probability} |  it has split and theta == 2")
                        continue
                    
                    elif "Split" not in goodness and "SigmoidProbability_Theta_0" == probability:
                        # If not split, theta is needed
                        print("Skipping", f"{dataset} | {activation} | {goodness} | {probability} |  it has no split and theta == 0")
                        continue
                
                if f"{activation}_{goodness}_{probability}_{dataset}" in current_models:
                    print("Skipping", f"{dataset} | {activation} | {goodness} | {probability} |  already exists")
                    continue

                print(f"Starting {goodness} {activation} {probability} {dataset}")
                start_time = datetime.now()
                print(f"Running {goodness} {activation} {probability} {dataset}")
                run_experiment(goodness, activation, probability, dataset=dataset)
                finish_time = datetime.now()
                # Say it in minutes and seconds
                time = [finish_time - start_time]
                
                print(f"Time: {time}")
                
                
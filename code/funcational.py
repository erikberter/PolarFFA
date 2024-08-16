import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm

import math

import os
import json 
from ff_mod.goodness import L2_Goodness, L1_Goodness, Norm_goodness, L2_Goodness_SQRT
from ff_mod.probability import SigmoidProbability, SymmetricFFAProbability

from ff_mod.network.base_ffa import FFANetwork, FFALayer 

from ff_mod.overlay import AppendToEndOverlay, CornerOverlay
from ff_mod.loss import BCELoss


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


datasets = {
    'mnist': (mnist_train, mnist_test, 28*28),
    'kmnist': (kmnist_train, kmnist_test, 28*28),
    'fashion': (fashion_train, fashion_test, 28*28),
    'cifar10': (cifar10_train, cifar10_test, 32*32*3)
}

TEST_MODELS = True

NUM_CLASSES = 10
DIM = 1000
LEARNING_RATE = 0.001

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

def create_network(goodness, activation, probability, input_size=784):
    overlay = AppendToEndOverlay(pattern_size=100, num_classes=NUM_CLASSES, p=0.1)

    network = FFANetwork(overlay)
    
    loss = BCELoss(probability_function=probabilities[probability])
    
    network.add_layer(FFALayer(input_size + 100, DIM, goodnesses[goodness], loss, activations[activation], learning_rate=LEARNING_RATE))
    network.add_layer(FFALayer(DIM, DIM, goodnesses[goodness], loss, activations[activation], learning_rate=LEARNING_RATE))
    
    return network

from ff_mod.trainer import Trainer

trainer = Trainer()

MIN_ACCURATION = 0.00



def normalize_over_mean(latents, batch_size = 512, class_size = 10, dim = 1000):   

    total_batches = latents.shape[0] // (batch_size * class_size)
    
    for batch in range(total_batches):
        skip = batch * batch_size * class_size
        for i in range(batch_size):
            mean_state = np.zeros((dim))
            
            for j in range(class_size):
                #print(f"Spahes: {latents[skip + i + j * batch_size].shape} - {mean_state.shape}")
                mean_state += latents[skip + i + j * batch_size]
            
            mean_state /= class_size
            
            for j in range(10):
                latents[skip + i + j * 512] -= mean_state
    
    return latents

def get_latents(network, total_batches, layer = 1, use_train = True, normalize = False, normalize_mean = False):
    global trainer
    
    latents = []
    labels = []
    positiveness = []
    
    loader = trainer.train_loader if use_train else trainer.test_loader
    
    for i, (data, target) in enumerate(loader):
        data, target = data.to(trainer.device), target.to(trainer.device)
        if i >= total_batches:
            break
        
        for l in range(10):
            latent_t = network.get_latent(data, (target+l)%10, layer)
            target_t = target.clone().detach()
            
            if normalize:
                latent_t = latent_t / (torch.norm(latent_t, dim=1, keepdim=True) + 0.00001)
            
            latents.append(latent_t.detach().cpu().numpy())
            labels.append(target_t.detach().cpu().numpy() * np.ones(latent_t.shape[0]))
            
            if l == 0:
                positiveness.append(np.ones(latent_t.shape[0]))
            else:
                positiveness.append(np.zeros(latent_t.shape[0]))
        
    latents = np.concatenate(latents)
    labels = np.concatenate(labels)
    positiveness = np.concatenate(positiveness)
    
    if normalize_mean:
        latent_t = normalize_over_mean(latents)
        
    return latents, labels, positiveness

def get_all_latents(all_models, use_trains = True, use_normalize = False, normalize_mean = False):
    all_latents = {}
    
    for model in all_models.keys():
        
        all_latents[model] = get_latents(all_models[model], 2, use_train = use_trains, normalize=use_normalize, normalize_mean=normalize_mean)
        
        #if "SymmetricFFAProbability" in model:
        #    all_latents[model + "_Normalized"] = get_latents(all_models[model], 2, use_train = use_trains, normalize = True)
            
    return all_latents

def get_hoyer_distribution(all_latents, eps = 1e-7):
    hoyer_mean = {}
    
    for i, model in enumerate(all_latents.keys()):
        latents, labels, positiveness = all_latents[model]
        
        hoyer = (np.linalg.norm(latents, ord=1, axis=1)+eps) / (np.linalg.norm(latents, ord=2, axis=1)+eps)
        hoyer = (np.sqrt(DIM) - hoyer) / (np.sqrt(DIM) - 1)
        
        hoyer_mean[model] = hoyer.mean()
    
    return hoyer_mean


def get_hoyer_on_positive_but_zero(all_latents, eps=1e-7):
    hoyer_mean = {}
    
    for i, model in enumerate(all_latents.keys()):
        latents, labels, positiveness = all_latents[model]
        
        latents_t = latents[positiveness == 1]
        
        # Filter out the zero latents
        latents_t = latents_t[np.linalg.norm(latents_t, ord=2, axis=1) > 0]
        
        hoyer = (np.linalg.norm(latents_t, ord=1, axis=1)+eps) / (np.linalg.norm(latents_t, ord=2, axis=1) + eps)
        hoyer = (np.sqrt(DIM) - hoyer) / (np.sqrt(DIM) - 1)
        
        hoyer_mean[model] = hoyer.mean()
    
    return hoyer_mean



def get_hoyer_distribution_but_zero(all_latents, eps = 1e-7):
    hoyer_mean = {}
    
    for i, model in enumerate(all_latents.keys()):
        latents, labels, positiveness = all_latents[model]
        
        # Filter out the zero latents
        latents = latents[np.linalg.norm(latents, ord=2, axis=1) > 0]
        
        hoyer = (np.linalg.norm(latents, ord=1, axis=1)+eps) / (np.linalg.norm(latents, ord=2, axis=1) + eps)
        hoyer = (np.sqrt(DIM) - hoyer) / (np.sqrt(DIM) - 1)
        
        hoyer_mean[model] = hoyer.mean()
    
    return hoyer_mean

from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist

def compute_separability(all_latents, kfactor = 5, batch_size = 32, limit = 4000):
    """ Compute the separability of the latents by using the separability index"""
    
    separability = {}
    
    for i, model in enumerate(all_latents.keys()):
        latents_pre, labels_pre, positiveness_pre = all_latents[model]
        
        random_indices = np.random.choice(latents_pre.shape[0], limit, replace=False)
        latents, labels, positiveness = latents_pre[random_indices], labels_pre[random_indices], positiveness_pre[random_indices]
        
        pos_sum, neg_sum = 0, 0

        for i in tqdm(range(math.ceil(latents.shape[0]/batch_size)), leave=False):
            
            # For each batch, compute the indexes of the kfactor nearest neighbors
            indexes = np.argsort(cdist(latents[batch_size * i : batch_size * i + batch_size], latents), axis=1)[:, 1 : 1 + kfactor]
            
            current_positives = positiveness[batch_size * i : batch_size * i + batch_size] == 1
            current_negatives = positiveness[batch_size * i : batch_size * i + batch_size] == 0
            
            pos_sum += np.sum(positiveness[indexes[current_positives]] == 1)
            neg_sum += np.sum(positiveness[indexes[current_negatives]] == 0)
            
        total_pos = np.sum(positiveness == 1) * kfactor
        total_neg = np.sum(positiveness == 0) * kfactor
        
        separability[model] = ((pos_sum / total_pos) + (neg_sum / total_neg))/2
    
    return separability

def compute_neural_usage(all_latents, theta = 0.01):
    neural_usage = {}
    
    for i, model in enumerate(all_latents.keys()):
        # Latent dim (batch, dim)
        latents, labels, positiveness = all_latents[model]
        
        # Sum the absolute value of the latents
        # Dim (dim)
        latent_sum = np.sum(np.abs(latents), axis=0)
        
        threshold = theta * np.max(latent_sum)
        
        neural_usage[model] = np.sum(latent_sum > threshold) / latent_sum.shape[0]
    
    return neural_usage

def compute_neural_usage_hoyer(all_latents, eps = 1e-6):
    neural_usage = {}
    
    for i, model in enumerate(all_latents.keys()):
        # Latent dim (batch, dim)
        latents, labels, positiveness = all_latents[model]
        
        # Apply hoyer
        latent_sum = np.sum(np.abs(latents), axis=0)
        # latent sum dim (DIM)
        hoyer = np.linalg.norm(latent_sum, ord=1) / (np.linalg.norm(latent_sum, ord=2) + eps)
        hoyer = (np.sqrt(DIM) - hoyer) / (np.sqrt(DIM) - 1)

        neural_usage[model] = hoyer
    return neural_usage

from tensorboard.backend.event_processing import event_accumulator
import json

def compute_convergence_area(path, eps=0.005):
    # We recieve the file of the path
    # Calculate the area under the curve of the accuracy
    summary_folder = os.path.join(path, "summary")
    summary_file = os.listdir(summary_folder)[0]
    summary_file = os.path.join(summary_folder, summary_file)

        
    ea = event_accumulator.EventAccumulator(summary_file)
    ea.Reload()
        
    accuracy_list = [x.value for x in ea.Scalars("test_acc")]
    max_acc = max(accuracy_list)
    
    area = 0
    
    for i in range(1, len(accuracy_list)):
        area += max_acc - accuracy_list[i]
        if abs(accuracy_list[i] - max_acc) < eps:
            break
    area = area / len(accuracy_list)
    
    return area    
    
    
def compute_convergente_time(path, eps = 0.005):
    # We recieve the file of the path
    # Calculate the area under the curve of the accuracy
    summary_folder = os.path.join(path, "summary")
    summary_file = os.listdir(summary_folder)[0]
    summary_file = os.path.join(summary_folder, summary_file)

        
    ea = event_accumulator.EventAccumulator(summary_file)
    ea.Reload()
        
    accuracy_list = [x.value for x in ea.Scalars("test_acc")]
    max_acc = max(accuracy_list)
    
    for i in range(1, len(accuracy_list)):
        if abs(accuracy_list[i] - max_acc) < eps:
            break
    return i/len(accuracy_list)


for (use_train, normalize_mean) in [(True, False), (False, False)]:
    results = {}
    names = {}
    current_models = {}
    for dataset in datasets.keys():
        if dataset == "cifar10":
            DIM = 2000
        else:
            DIM = 1000
        # test if folder exists
        EXPERIMENT_FOLDER = f'experiments_train/{dataset}/'
        
        if not os.path.exists(EXPERIMENT_FOLDER):
            print(f"Skipping {dataset}")
            continue
        
        trainer.set_dataloader(datasets[dataset][0], datasets[dataset][1])
        
        # Count number of folders inside
        tot = len(os.listdir(EXPERIMENT_FOLDER))
        
        for i, folder in tqdm(enumerate(os.listdir(EXPERIMENT_FOLDER)), leave=False, total=tot):
            
            current_metrics = []
            
            # Read json file config in folder
            with open(os.path.join(EXPERIMENT_FOLDER, folder, 'config.json'), 'r') as f:
                config = json.load(f)
                
                network = create_network(config['goodness'], config['activation'], config['probability'], input_size=datasets[dataset][2])
                network.load_network(EXPERIMENT_FOLDER+'/' + folder + '/best_model')

                config_str = f"{config['activation']}_{config['goodness']}_{config['probability']}_{dataset}"
                
                if config_str in current_models:
                    act_ind = 1
                    while config_str + f"_{act_ind}" in current_models:
                        act_ind += 1
                    config_str = config_str + f"_{act_ind}"
                
                
                print(f"[{dataset}] {config_str}")
                
                trainer.set_network(network)
                acc = trainer.test_epoch(verbose=0)
                
                current_metrics += [float(acc)]
                current_models[config_str] = True
                
                all_latents = get_all_latents({config_str: network}, use_trains=use_train)
                
                hoyer_mean = get_hoyer_distribution(all_latents)
                current_metrics += [float(hoyer_mean[config_str])]
                
                hoyer_mean_not_zero = get_hoyer_distribution_but_zero(all_latents)
                current_metrics += [float(hoyer_mean_not_zero[config_str])]
                
                hoyer_mean_positive_not_zero = get_hoyer_on_positive_but_zero(all_latents)
                current_metrics += [float(hoyer_mean_positive_not_zero[config_str])]
                
                hoyer_neural_usage = compute_neural_usage_hoyer(all_latents)
                current_metrics += [float(hoyer_neural_usage[config_str])]
                
                neural_usage = compute_neural_usage(all_latents)
                current_metrics += [float(neural_usage[config_str])]
                
                separability_mean = compute_separability(all_latents, batch_size=128, limit=1000)
                current_metrics += [float(separability_mean[config_str])]
                
                convergence_area = compute_convergence_area(EXPERIMENT_FOLDER +'/' + folder)
                current_metrics += [convergence_area]
                
                convergence_time = compute_convergente_time(EXPERIMENT_FOLDER +'/' + folder)
                current_metrics += [convergence_time]
                
                results[config_str] = current_metrics
            
    is_train = "train" if use_train else "test"
    is_train = is_train + "_normalized" if normalize_mean else is_train
    
    names = ['Accuracy', 'Hoyer', 'Hoyer (Non Zero)', 'Hoyer (Positive)', 'Neural Usage (Hoyer)', 'Neural Usage (Percent)', 'Separability', 'Convergence Area', 'Convergence Time']
    
    final = {'names' : names, 'results': results}
    with open(f'experimental_summary_{is_train}.json', 'w') as f:
        json.dump(final, f)
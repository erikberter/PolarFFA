# On the Improvement of Generalization and Stability of Forward-Only Learning via Neural Polarization

**Erik B. Terres-Escudero**[1]  ([e.terres@deusto.es](mailto:e.terres@deusto.es)),
**Javier Del Ser**[2,3] ,**Pablo Garcia-Bringas**[1]  
1. University of Deusto, 48007 Bilbao, Spain
2. TECNALIA, Basque Research & Technology Alliance (BRTA), 48160 Derio, Spain
3. University of the Basque Country (UPV/EHU), 48013 Bilbao, Spain
## Abstract


*Forward-only learning algorithms have recently gained attention as alternatives to gradient backpropagation, replacing the backward step of this latter solver with an additional contrastive forward pass. Among these approaches, the so-called Forward-Forward Algorithm (FFA) has been shown to achieve competitive levels of performance in terms of generalization and complexity. Networks trained using FFA learn to contrastively maximize a layer-wise defined goodness score when presented with real data (denoted as positive samples) and to minimize it when processing synthetic data (corr. negative samples). However, this algorithm still faces weaknesses that negatively affect the model accuracy and training stability, primarily due to a gradient imbalance between positive and negative samples. To overcome this issue, in this work we propose a novel implementation of the FFA algorithm, denoted as Polar-FFA, which extends the original formulation by introducing a neural division (\emph{polarization}) between positive and negative instances. Neurons in each of these groups aim to maximize their goodness when presented with their respective data type, thereby creating a symmetric gradient behavior. To empirically gauge the improved learning capabilities of our proposed Polar-FFA, we perform several systematic experiments using different activation and goodness functions over image classification datasets. Our results demonstrate that Polar-FFA outperforms FFA in terms of accuracy and convergence speed. Furthermore, its lower reliance on hyperparameters reduces the need for hyperparameter tuning to guarantee optimal generalization capabilities, thereby allowing for a broader range of neural network configurations.*


## Suplementary Materials

The file *Supplementary_Material.pdf*, contains all the information regarding the appendixes of this paper. This file is structured as follows:

>A. Instability of Sigmoidal Probability Functions | 1 \
B Proof of Proposition 1 | 1 \
C Proof of Proposition 2 | 2 \
D Experimental Setup: Additional Information | 2 \
E Effect on the Ratio of Positive and Negative Neurons | 3 \
F Additional Results for RQ1: goodness, probability functions and activations | 4 \
G Additional Results for RQ2: Latent Space Taxonomy | 4 

## Code & Results

The code to execute the experiments and the results obtained are contained within the *code* folder. This folder is structured as follows:
* **ff_mod**: Contains all the code required to create Forward-Forward Networks.
* **report_generator.ipynb**: Allows to create report containg information related to the latent space, including sparsity and Persistent Homology Diagrams.
* **test_frog.py**: Visual application to study the different metrics (accuracy, sparsity, convergence speed, ...) allowing to isolate specific configurations (e.g., discard activation functions, specify goodness functions, ...)
* **launch_experiments.py**: Script to launch all training experiments.
* **results**: Folder containing all the details regarding the results form the experiments.

## Acknowledgementes

The authors thank the Basque Government for its funding support via the consolidated research groups MATHMODE (ref. T1256-22) and D4K (ref. IT1528-22), and the colaborative ELKARTEK project KK-2023/00012 (BEREZ-IA). E. B. Terres-Escudero is supported by a PIF research fellowship granted by the University of Deusto.
# Stealing Training Graphs from Graph Neural Networks
This is the light-weight PyTorch implementation of "Stealing Training Graphs from Graph Neural Networks" (KDD 2025). [[paper]](https://arxiv.org/abs/2411.11197)

We will further clean our code as soon as possible, thanks!
## 1. Overview
We organize the structure of our files as follows:
```latex
.
├── configs/                        # Configuration files for experiments and settings
├── data/                        # Configuration files for experiments and settings
├── logs/                           # Directory containing self-reproduced results of GraphSteal (e.g., QM9 dataset results)
├── src/                            # Main source code directory
│   ├── analysis/                   # End-to-end attack implementation
│   ├── checkpoints/                # Directory for storing pretrained graph diffusion model checkpoints
│   ├── classifier/                 # Classification-related codes
│   ├── diffusion/                  # Diffusion-related implementations
│   ├── metrics/                    # Evaluation metrics for experiments
│   ├── models/                     # Model architectures and related files
│   ├── reconstruction/             # Reconstruction-based codes
│   ├── saved_graphs/               # Directory to save intermediate graph outputs
│   ├── diffusion_model_discrete.py # Code for discrete diffusion models
│   ├── diffusion_model.py          # Code for diffusion model implementation
│   ├── run_qm9_classifier.py       # Script to run classification on QM9 dataset
│   ├── run_reconstruct.py          # Script for reconstruction tasks
│   ├── run_train_diffusion.py      # Training script for diffusion-based models
│   └── utils.py                    # General utility functions
└── requirements.txt                # Python dependencies required for the project

```

## 2. Requirements
We follow [DiGress](https://github.com/cvignac/DiGress) to install our environment.

## 3. Run the code
* Step0: Enter `./src` by running `cd ./src`
* Step1: Pre-train graph diffusion model. To train the diffusion model, you can run `python run_train_diffusion.py`. We use [DiGress](https://github.com/cvignac/DiGress) as our graph reconstructor. For more details, please refer to the original repository.  
* Step2: Train the classifier. We provide the implementation of training the GTN classifier on QM9 dataset. You can run ```python run_qm9_classifier.py```. You can refer to `./datasets/qm9_dataset.py` to see the training graph spit setting. 
* Step3: Reconstruct training graphs. You can run `python run_reconstruct.py`. 

## 4. Dataset
The applied dataset (i.e., QM9) will be automatically downloaded to ./data. 

## 5. Citation
If you find this repo to be useful, please cite our paper. Thank you.
```
@inproceedings{lin2025stealing,
author = {Lin, Minhua and Dai, Enyan and Xu, Junjie and Jia, Jinyuan and Zhang, Xiang and Wang, Suhang},
title = {Stealing Training Graphs from Graph Neural Networks},
year = {2025},
booktitle = {Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.1},
pages = {777–788},
}
```

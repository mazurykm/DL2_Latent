# DL2_Latent

# Latent Program Network (LPN)
Code for the paper _Searching Latent Program Spaces_, made as a project for Deep Learning 2 course (2025) at University of Amsterdam. It contains reproduction of - [📄 Paper on arXiv](https://arxiv.org/abs/2411.08706) with original extensions.

Authors and maintainers: Alisia Baielli, Wojciech Kosiuk, Michał Mazuryk, Devin Pereira 

## Overview
The LPN is an architecture for inductive program synthesis that builds in test-time adaption by learning a latent space that can be used for search.
![LPN Diagram](src/figures/lpn_diagram.png)

Main contributions of our extensions:
* Cross-attention as a aggregation strategy for test-time optimization
* Matrix inference mode - latent space modelled as a matrix with sequential decoding of columns leading to iterative refinement


## Installation
Install JAX using the official documentation [here](https://github.com/jax-ml/jax?tab=readme-ov-file#instructions).
Then, install the required packages using the following commands:
```bash
git clone https://github.com/clement-bonnet/lpn
cd lpn
export PYTHONPATH=${PYTHONPATH}:${PWD}
pip install -U -r requirements.txt
```
Add your secrets to the environment variables (HuggingFace token and WandB API key):
```bash
export HF_TOKEN=...
export WANDB_API_KEY=...
```


## Repository Structure
```
lpn/src/
├── configs/        # Configuration files (use with hydra)
├── datasets/       # Data processing utilities (re-arc)
├── models/         # Neural network architectures (lpn)
└── train.py        # Main training script
```


## Usage
To train a model, run the following command (replace the config name with the desired configuration):
```bash
python src/train.py --config-name pattern_2d
```


## License
This project is licensed under the open-source Apache 2.0 License. See the [LICENSE](LICENSE) file for more details.


```

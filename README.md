# 🌿 iNaturalist Species Classification

This repository contains deep learning experiments using the [iNaturalist](https://www.inaturalist.org/) dataset, focusing on image classification with Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs). It includes W&B integration for experiment tracking and Bayesian sweeps for hyperparameter tuning.

---

## 📁 Repository Structure
## 🧠 Part 1: CNN-Based Classification

### 📜 Description

A custom Convolutional Neural Network is trained on the iNaturalist dataset. The model is defined in `model.py` and trained via `train.py`. Hyperparameter tuning is done using W&B Bayesian sweeps defined in `sweep.py`.

### ⚙️ CNN Sweep Configuration Highlights

```python
sweep_configuration = {
    'method': 'bayes',
    'name': 'CNet Optim Sweep',
    'metric': {'goal': 'maximize', 'name': 'val_accuracy'},
    'parameters': {
        'num_layers': {'values': [5]},
        'filter_size': {'values': [[7, 5, 5, 3, 3], [5, 5, 5, 5, 5], [7, 7, 5, 3, 3]]},
        'n_filter': {'values': [[32, 64, 128, 256, 512], [256, 128, 64, 32, 16], [64, 64, 64, 64, 64]]},
        'dense_size': {'values': [512, 1024]},
        'conv_activation': {'values': ['GELU', 'ReLU', 'SiLU']},
        'batch_size': {'values': [64, 128, 256]},
        'learning_rate': {'values': [0.001, 0.0001]},
        'weight_decay': {'values': [0, 0.005]},
        'dropout_prob': {'values': [0.3, 0.5]},
        'batch_norm': {'values': [True, False]},
        'weight_init': {'values': ['xavier', 'kaiming']},
        'optimizer': {'values': ['adam']},
        'n_epochs': {'values': [10, 15]},
        'data_aug': {'values': [True, False]},
        'dense_activation': {'values': ['ReLU']}
    }
}

cd PART_FIRST
python train.py \
  --in_dims 256 \
  --n_epochs 10 \
  --learning_rate 0.0001 \
  --weight_decay 5e-5 \
  --batch_size 64 \
  --conv_activation GELU \
  --dense_activation ReLU \
  --dense_size 1024 \
  --filter_size 7 5 5 3 3 \
  --n_filter 32 64 128 256 512 \
  --filter_org same \
  --batch_norm True \
  --dropout_prob 0.3 \
  --data_aug True


## 🧠 Part 2: Fine-Tuning ViT on iNaturalist

### 📜 Description

This section focuses on fine-tuning a pre-trained Vision Transformer (ViT) model on the iNaturalist dataset for image classification. It includes two scripts:

- `finetune.py`: Standard fine-tuning without sweeps.
- `fine_tune_sweeps.py`: Hyperparameter tuning using W&B sweeps.

### 📂 Folder Structure


### 🔧 CLI Arguments (finetune.py)

You can fine-tune the ViT model using the following command:

```bash
cd PART_SECOND
python finetune.py \
  --batch_size 64 \
  --learning_rate 3e-5 \
  --weight_decay 0.01 \
  --n_epochs 10 \
  --dropout 0.1 \
  --scheduler cosine \
  --freeze_backbone False





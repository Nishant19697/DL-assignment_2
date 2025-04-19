import wandb
from train import train_and_eval
import argparse

def train_wrapper():
    wandb.init()
    config = wandb.config

    if wandb.config.n_filter[0] == wandb.config.n_filter[1]:
        channel_org = f"same_{wandb.config.n_filter[0]}"
    elif wandb.config.n_filter[0] > wandb.config.n_filter[1]:
        channel_org = f"half_{wandb.config.n_filter[0]}"
    elif wandb.config.n_filter[0] < wandb.config.n_filter[1]:
        channel_org = f"double_{wandb.config.n_filter[0]}"

    run_name = f"n_filter-{channel_org}_LR{config.learning_rate}_BS{config.batch_size}_WD{config.weight_decay}_DO{config.dropout_prob}_{config.conv_activation}"
    wandb.run.name = run_name

    args = argparse.Namespace(
        num_layers=config.num_layers,
        dense_size=config.dense_size,
        conv_activation=config.conv_activation,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        dropout_prob=config.dropout_prob,
        batch_norm=config.batch_norm,
        weight_init=config.weight_init,
        optimizer=config.optimizer,
        n_epochs=config.n_epochs,
        data_aug = config.data_aug,
        n_filter = config.n_filter,
        filter_size = config.filter_size,
        dense_activation = config.dense_activation
    )

    train_and_eval(args, logging=True)

sweep_configuration = {
    'method': 'bayes',
    'name': 'CNet Optim Sweep',
    'metric': {'goal': 'maximize', 'name': 'val_accuracy'},
    'parameters': {
        'num_layers': {'values': [5]},
        'filter_size': {'values': [[7, 5, 5, 3, 3], [5, 5, 5, 5, 5], [7, 7, 5, 3, 3]]},
        'n_filter' : {"values" : [[32, 64, 128, 256, 512], [ 256, 128, 64, 32,16], [64, 64, 64, 64, 64]]},
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

if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep=sweep_configuration, project="finetu")
    wandb.agent(sweep_id, function=train_wrapper, count = 150)

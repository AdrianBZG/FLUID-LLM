"""
Main entrypoint for training
"""
import os
import sys
import argparse
import logging
import torch
import wandb
from accelerate import Accelerator
from tqdm import trange, tqdm

from trainer import Trainer, get_data_loader
from utils import set_seed, load_yaml_from_file, get_available_device, get_accelerator, make_save_folder, save_cfg
from models.model import MultivariateTimeLLM

logging.basicConfig(level=logging.INFO,
                    format=f'[{__name__}:%(levelname)s] %(message)s')


def run_train_epoch(dataloader, trainer: Trainer, optimizer, scheduler, accelerator: Accelerator):
    trainer.model.train()

    metrics_per_epoch = []
    dataloader_iterator = tqdm(dataloader, desc="Iterating batches", leave=False)
    for batch_idx, batch in enumerate(dataloader_iterator):
        with accelerator.accumulate(trainer.model):
            states, diffs, bc_mask, position_ids = batch

            with accelerator.autocast():
                loss, log_metrics_dict = trainer.run_train_step(states, diffs, bc_mask, position_ids)

                # Backpropagation
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(trainer.model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            dataloader_iterator.set_description(f"Iterating batches (Batch Idx: {batch_idx + 1} | Loss: {log_metrics_dict['train_loss']:.3g})")
            dataloader_iterator.refresh()

            # Keep track of metrics
            metrics_per_epoch.append(log_metrics_dict)

    # === Aggregate metrics across iterations in the epoch ===
    metrics_names = metrics_per_epoch[0].keys()
    metrics_agg = {f"train/{metric_name}": sum(d[metric_name]
                                               for d in metrics_per_epoch) / len(metrics_per_epoch)
                   for metric_name in metrics_names}
    metrics_agg['train/LR'] = optimizer.param_groups[0]['lr']
    return metrics_agg


def main(args):
    set_seed()
    training_params = load_yaml_from_file(args.config_path)
    logging.info(f"Parameters for training: {training_params}")

    # Make save folder and save config
    save_path = make_save_folder(training_params['checkpoint_save_path'], args.save_folder)
    logging.info(f"Saving checkpoints to: {save_path}")
    save_cfg(args.config_path, save_path)  # WandB saves it, but make another copy anyway.

    # Prepare accelerator
    accelerator = get_accelerator()

    # Get the model
    model = MultivariateTimeLLM(training_params, device_map=get_available_device())

    # Get the train data loader
    train_dataloader = get_data_loader(training_params)
    trainer = Trainer(params=training_params,
                      model=model,
                      device=get_available_device())

    optimizer, scheduler = trainer.prepare_optimizers()

    # Wandb
    if training_params['enable_wandb'] is False:
        os.environ['WANDB_MODE'] = 'disabled'

    wandb.init(project="llm4multivariatets", entity="adrianbzgteam", config=training_params)

    # Prepare model, optimizer and dataloader for accelerate training
    model, optimizer, train_dataloader, scheduler = accelerator.prepare(model, optimizer, train_dataloader, scheduler)
    trainer.model = model

    epoch_iterator = trange(training_params["num_epochs"], desc="Training", position=0, leave=True)
    for epoch_idx, epoch in enumerate(epoch_iterator):
        train_log_metrics = run_train_epoch(dataloader=train_dataloader,
                                            trainer=trainer,
                                            optimizer=optimizer,
                                            scheduler=scheduler,
                                            accelerator=accelerator)

        wandb.log(train_log_metrics, step=epoch_idx)

        epoch_iterator.set_description(f"Training (Epoch: {epoch_idx + 1} | Loss: {train_log_metrics['train/train_loss']})")
        epoch_iterator.refresh()

        # Save model checkpoint
        if training_params['save_model_each'] > 0 and epoch_idx % training_params['save_model_each'] == 0 and epoch_idx > 0:
            accelerator.wait_for_everyone()
            checkpoint_file_path = os.path.join(save_path, f'step_{epoch_idx}.pth')

            checkpoint = {'params': training_params,
                          'state_dict': trainer.model.state_dict(),
                          'optimizer': optimizer.state_dict()}

            logging.info(f"Saving model checkpoint at epoch {epoch_idx} to {checkpoint_file_path}")
            torch.save(checkpoint, checkpoint_file_path)

    # Close wandb
    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path',
                        default="configs/training1.yaml",
                        help='Path to the json config for training')
    parser.add_argument('--save_folder',
                        help='Path to save model checkpoints. Defaults to time', default=None)

    args = parser.parse_args(sys.argv[1:])
    main(args)


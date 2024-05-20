"""
Testing loading a saved model and running the test_generate function
"""
import os
import logging
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from cprint import c_print
import numpy as np
import torch.nn.functional as F

from utils import set_seed, load_yaml_from_file, get_available_device, get_save_folder, get_accelerator
from utils_model import calc_n_rmse, patch_to_img, get_data_loader
from models.model import MultivariateTimeLLM

from dataloader.synthetic.synth_dl import SynthDS

logging.basicConfig(level=logging.INFO,
                    format=f'[{__name__}:%(levelname)s] %(message)s')


def get_eval_dl(model, bs, seq_len):
    ds = SynthDS(

        patch_size=model.config['patch_size'],
        stride=model.config['stride'],
        seq_len=seq_len,
        mode='valid',
        # normalize=model.config['normalize_ds']
    )

    dl = DataLoader(ds, batch_size=bs, num_workers=8, pin_memory=True)
    return dl


def plot_set(plot_step, true_states, pred_states, title):
    fig, axs = plt.subplots(2, 2, figsize=(10, 9))
    fig.suptitle(f'{title}')
    for i, ax in enumerate(axs):
        img_1 = true_states[plot_step, i].cpu()
        img_2 = pred_states[plot_step, i].cpu()

        ax[0].imshow(img_1.T)  # Initial image
        ax[1].imshow(img_2.T)  # Predictions
        ax[0].axis('off'), ax[1].axis('off')
    fig.tight_layout()
    fig.show()


@torch.inference_mode()
def test_generate(model: MultivariateTimeLLM, dl, plot_step, batch_num=0):
    model.eval()

    start_step = 5
    ctx_states = 1
    pred_steps = 10
    start_cut = start_step - ctx_states
    end_state = pred_steps + ctx_states if ctx_states == 1 else pred_steps + ctx_states - 1

    # Keep the first batch for plotting
    first_batch = None
    N_rmses = []
    # Get batch and run through model
    for i, batch in enumerate(dl):
        # Filter out start
        batch = [b[:, start_cut:] for b in batch]
        batch = [t.cuda() for t in batch]

        states, _, diffs, bc_mask, position_ids = batch

        # bs, seq_len, N_patch, channel, px, py = states.shape
        pred_states, pred_diffs = model.gen_seq(batch, pred_steps=pred_steps, start_state=ctx_states)
        pred_states = pred_states[:, :-1]

        # print(f'{pred_states.shape = }')
        # print(pred_states.std(dim=(0, 2, 3, 4)) )

        true_states = patch_to_img(states, model.ds_props)
        true_diffs = patch_to_img(diffs, model.ds_props)
        bc_mask = patch_to_img(bc_mask.float(), model.ds_props).bool()

        true_states = true_states[:, :end_state]
        bc_mask = bc_mask[:, :end_state]
        # print(f'{pred_states.shape = }, {true_states.shape = }, {bc_mask.shape = }')
        N_rmse = calc_n_rmse(pred_states, true_states, bc_mask)
        N_rmses.append(N_rmse)

        if first_batch is None:
            first_batch = (true_states, true_diffs, pred_states, pred_diffs)

        # break

    N_rmses = torch.cat(N_rmses, dim=0)
    N_rmse = torch.mean(N_rmses, dim=0)
    first_rmses = torch.mean(N_rmse[:15])
    c_print(f"First 15 N_RMSE: {first_rmses:.3g}", color='green')
    c_print(f"Standard N_RMSE: {N_rmse[ctx_states:]}, Mean: {N_rmse.mean().item():.3g}", color='cyan')

    # Plotting
    plot_nums = np.array([0, 4]) + ctx_states
    for plot_step in plot_nums:
        true_states, true_diffs, pred_states, pred_diffs = first_batch

        # # Plot diffs
        # plot_set(plot_step, true_diffs[batch_num], pred_diffs[batch_num], 'Differences')

        # Plot states
        plot_set(plot_step, true_states[batch_num], pred_states[batch_num], f'States, step {plot_step-ctx_states}')


def main():
    load_no = -1
    save_epoch = 180
    seq_len = 23
    bs = 16

    plot_step = -1
    plot_batch_num = 0

    set_seed()

    # Load the checkpoint
    load_path = get_save_folder("./model_checkpoints", load_no=load_no)
    checkpoint_file_path = os.path.join(load_path, f'step_{save_epoch}.pth')
    logging.info(f"Loading checkpoint from: {checkpoint_file_path}")

    if not os.path.exists(checkpoint_file_path):
        raise ValueError(f"Checkpoint file not found at {checkpoint_file_path}")

    ckpt = torch.load(checkpoint_file_path)
    ckpt_state_dict = ckpt['state_dict']
    ckpt_params = load_yaml_from_file(f'{load_path}/training1.yaml')
    # ckpt_params['compile'] = False

    # Get dataloader
    ckpt_params['seq_len'] = ckpt_params['autoreg_seq_len']
    _, ds_props = get_data_loader(ckpt_params, mode="valid")

    # Get the model
    model = MultivariateTimeLLM(ckpt_params, ds_props=ds_props, device_map=get_available_device())
    # Load weights
    model.load_state_dict(ckpt_state_dict)
    accelerator = get_accelerator(use_deepspeed=False, precision='bf16')
    model = accelerator.prepare(model)

    # Val dataloader
    dl = get_eval_dl(model, bs, seq_len)

    # Run test_generate
    test_generate(model, dl, plot_step, plot_batch_num)


if __name__ == '__main__':
    main()

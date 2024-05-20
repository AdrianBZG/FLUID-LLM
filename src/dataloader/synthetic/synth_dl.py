"""Plots a CFD trajectory rollout."""

import numpy as np
import os
import pickle
import random
from cprint import c_print
import time
from dataloader.synthetic.solver_node import WaveConfig, PDESolver2D
import torch
from torch.utils.data import Dataset, DataLoader
from dataloader.mesh_utils import to_grid, get_mesh_interpolation


def num_patches(dim_size, kern_size, stride, padding=0):
    """
    Returns the number of patches that can be extracted from an image
    """
    return (dim_size + 2 * padding - kern_size) // stride + 1


class SynthDS(Dataset):
    """ Load a single timestep from the dataset."""

    def __init__(self, patch_size: tuple, stride: tuple, seq_len: int,
                 pad=True, mode="train", normalize=True):
        super().__init__()

        assert mode in ["train", "valid", "test"]

        self.mode = mode
        self.patch_size = patch_size
        self.stride = stride
        self.pad = pad
        self.seq_len = seq_len
        self.normalize = normalize
        self.start_step = 0

        wave_cfg = WaveConfig(seq_len + 10)
        self.data_gen = PDESolver2D(wave_cfg)

        # Load a random file to get min and max values and patch size
        ys, bc_mask = self._load_step()
        state, _ = self._get_step(ys, bc_mask, 5)

        # Calculate number of patches, assuming stride = patch_size
        x_px, y_px = state.shape[1:]

        self.N_x_patch, self.N_y_patch = num_patches(x_px, patch_size[0], stride[0]), num_patches(y_px, patch_size[1], stride[1])
        self.N_patch = self.N_x_patch * self.N_y_patch

    def __getitem__(self, idx):
        """
        Returns as all patches as a single sequence for file with index idx, ready to be encoded by the LLM as a single element of batch.
        """
        return self.ds_get()

    def ds_get(self):
        """
        Returns as all patches as a single sequence, ready to be encoded by the LLM as a single element of batch.
        Return:
             states.shape = ((seq_len - 1), num_patches, 3, H, W)
             states.shape = ((seq_len - 1), num_patches, 3, H, W)
             patch_idx: [x_idx, y_idx, t_idx] for each patch
        """
        sol, bc_mask = self._load_step()

        to_patches = []
        for i in range(self.start_step, self.seq_len + self.start_step):
            state, mask = self._get_step(sol, bc_mask, step_num=i)

            # Patch mask with state
            to_patch = np.concatenate([state, mask[None, :, :]], axis=0)
            to_patches.append(to_patch)

        to_patches = np.stack(to_patches)
        states, next_state, diffs, mask = self._ds_get_pt(to_patches)

        position_ids = self._get_pos_id()

        return states, next_state, diffs, mask, position_ids

    def _ds_get_pt(self, to_patches: np.ndarray):
        """ Pytorch section of ds_get to avoid multiprocessing bug.
            to_patches.shape = (seq_len, C+1, H, W) where last channel dim is mask.
        """
        to_patches = torch.from_numpy(to_patches).float()
        patches = self._patch(to_patches)

        states = patches[:, :-1]
        masks = patches[:, -1]

        # Permute to (seq_len, num_patches, C, H, W)
        states = torch.permute(states, [0, 4, 1, 2, 3])
        masks = torch.permute(masks, [0, 3, 1, 2])

        if self.normalize:
            states = self._normalize(states)

        target = states[1:] - states[:-1]  # shape = (seq_len, num_patches, C, H, W)

        # Compute targets and discard last state that has no diff
        next_state = states[1:]
        states = states[:-1]

        # Reshape mask. All masks are the same
        masks = masks[:-1].unsqueeze(-3).repeat(1, 1, 3, 1, 1)
        return states, next_state, target, masks.bool()

    def _normalize(self, states):
        """ states.shape = [seq_len, N_patch, 3, patch_x, patch_y] """
        # State 0: -0.0005653, 0.2364
        # Diff 0: 0.000548, 0.03722
        # State 1:  0.01035, 0.7167
        # Diff 1: 8.94e-05, 0.1617
        # State 2: -0.0005653, 0.2364
        # Diff 2: 0.000548, 0.03722

        s0_mean, s0_var = -0.0005653, 0.2364
        s1_mean, s1_var = 0.01035, 0.7167
        s2_mean, s2_var = -0.01235, 0.2364

        means = torch.tensor([s0_mean, s1_mean, s2_mean]).reshape(1, 1, 3, 1, 1)
        stds = torch.tensor([s0_var, s1_var, s2_var]).reshape(1, 1, 3, 1, 1)

        # Normalise states
        states = states - means
        states = states / stds

        return states

    def _get_step(self, ys, bc_mask, step_num):
        """
        Returns all interpolated measurements for a given step, including padding.
        """
        ys = ys[step_num]
        ys = torch.cat((ys, ys[0:1]), dim=0)

        if self.pad:
            step_state, P_mask = self._pad(ys, bc_mask)

        return step_state, P_mask

    def _patch(self, states: torch.Tensor):
        """
        Patches a batch of images.
        Returns a tensor of shape (bs, C, patch_h, patch_w, num_patches)
        """
        bs, C, _, _ = states.shape
        ph, pw = self.patch_size

        st = time.time()
        patches = torch.nn.functional.unfold(states, kernel_size=self.patch_size, stride=self.stride)

        if time.time() - st > 0.1:
            c_print(f"Time to patch: {time.time() - st:.3g}s", 'green')

        # Reshape patches to (bs, N, C, ph, pw, num_patches)
        patches_reshaped = patches.view(bs, C, ph, pw, patches.size(2))
        return patches_reshaped

    def _pad(self, state, mask):
        """ Pad state and mask so they can be evenly patched."""
        _, w, h = state.shape
        pad_width = (-w % self.patch_size[0])
        pad_height = (-h % self.patch_size[1])

        padding = (
            (0, 0),  # No padding on channel dimension
            (pad_width // 2, pad_width - pad_width // 2),  # Left, Right padding
            (pad_height // 2, pad_height - pad_height // 2),  # Top, Bottom padding
        )

        padding = np.array(padding)
        state_pad = np.pad(state, padding, mode='constant', constant_values=0)
        mask_pad = np.pad(mask, padding[1:], mode='constant', constant_values=1)
        return state_pad, mask_pad

    def _load_step(self):
        """ Load save file from disk and calculate mesh interpolation triangles"""

        self.data_gen.set_init_cond()
        sol, bc_mask = self.data_gen.solve()
        return sol, bc_mask

    def __len__(self):
        if self.mode == "train":
            return 1000
        else:
            return 128

    def _get_pos_id(self):
        # Get positions / times for each patch
        seq_dim = (self.seq_len - 1) * self.N_patch
        arange = np.arange(seq_dim)
        x_idx = arange % self.N_x_patch
        y_idx = (arange // self.N_x_patch) % self.N_y_patch
        t_idx = arange // self.N_patch
        position_ids = np.stack([x_idx, y_idx, t_idx], axis=1).reshape(self.seq_len - 1, self.N_patch, 3)
        return torch.from_numpy(position_ids)


def plot_all_patches():
    patch_size, stride = (16, 16), (16, 16)

    seq_dl = SynthDS(patch_size=patch_size, stride=stride,
                     seq_len=10)
    ds = DataLoader(seq_dl, batch_size=1)

    for batch in ds:
        state, next_state, diffs, mask, pos_id = batch
        if state.max() > 100:
            print(state.max())
        break

    x_count, y_count = seq_dl.N_x_patch, seq_dl.N_y_patch
    N_patch = seq_dl.N_patch

    show_dim = 0
    p_shows = state[0, :, :, show_dim]  # shape = (N_patch, seq_len, 16, 16)
    p_shows = p_shows.reshape(-1, N_patch, 16, 16)
    vmin, vmax = p_shows.min(), p_shows.max()
    print(f'{vmin = :.2g}, {vmax = :.2g}')
    for show_step in range(0, 9):
        fig, axes = plt.subplots(y_count, x_count, figsize=(16, 16))
        for i in range(y_count):
            for j in range(x_count):
                p_show = p_shows[show_step, i + j * y_count].numpy()
                p_show = p_show.T
                axes[i, j].imshow(p_show[:, :], vmin=vmin, vmax=vmax)
                axes[i, j].axis('off')

        fig.tight_layout()
    plt.show()

    # p_shows = diffs[0]
    # fig, axes = plt.subplots(y_count, x_count, figsize=(16, 4))
    # for i in range(y_count):
    #     for j in range(x_count):
    #         p_show = p_shows[i + j * y_count + show_step*N_patch].numpy()
    #         p_show = np.transpose(p_show, (2, 1, 0))
    #
    #         min, max = -0.005, 0.005  # seq_dl.ds_min_max[0]
    #
    #         axes[i, j].imshow(p_show[:, :, 0], vmin=min, vmax=max)
    #         axes[i, j].axis('off')
    # plt.tight_layout()
    # plt.show()


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from utils import set_seed

    set_seed(1)
    plot_all_patches()

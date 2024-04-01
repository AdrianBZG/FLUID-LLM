"""Plots a CFD trajectory rollout."""

import numpy as np
import os
import pickle
import random
from cprint import c_print
import time

import torch
from torch.utils.data import Dataset, DataLoader
from dataloader.mesh_utils import to_grid, get_mesh_interpolation


def num_patches(dim_size, kern_size, stride, padding=0):
    """
    Returns the number of patches that can be extracted from an image
    """
    return (dim_size + 2 * padding - kern_size) // stride + 1


class MGNDataset(Dataset):
    """ Load a single timestep from the dataset."""

    def __init__(self, load_dir, resolution: int, patch_size: tuple, stride: tuple, seq_len: int, seq_interval=1,
                 pad=True, mode="train"):
        super(MGNDataset, self).__init__()

        assert mode in ["train", "valid", "test"]

        self.mode = mode
        self.load_dir = load_dir
        self.resolution = resolution
        self.patch_size = patch_size
        self.stride = stride
        self.pad = pad
        self.seq_len = seq_len
        self.seq_interval = seq_interval
        self.max_step_num = 600 - self.seq_len * self.seq_interval

        self.save_files = sorted([f for f in os.listdir(f"{self.load_dir}/") if f.endswith('.pkl')])

        # Load a random file to get min and max values and patch size
        triang, tri_index, grid_x, grid_y, save_data = self._load_step(self.save_files[1])
        state, _ = self._get_step(triang, tri_index, grid_x, grid_y, save_data, step_num=20)

        # Get min and max values for each channel
        self.ds_min_max = [(state[0].min(), state[0].max()), (state[1].min(), state[1].max()), (state[2].min(), state[2].max())]

        # Calculate number of patches, assuming stride = patch_size
        x_px, y_px = state.shape[1:]

        self.N_x_patch, self.N_y_patch = num_patches(x_px, patch_size[0], stride[0]), num_patches(y_px, patch_size[1], stride[1])
        self.N_patch = self.N_x_patch * self.N_y_patch

    def __getitem__(self, idx):
        """
        Returns as all patches as a single sequence for file with index idx, ready to be encoded by the LLM as a single element of batch.
        Return:
             state.shape = ((seq_len - 1) * num_patches, 3, H, W)
             diff.shape = ((seq_len - 1)  * num_patches, 3, H, W)
             patch_idx: [x_idx, y_idx, t_idx] for each patch

        """
        # Time sampling is random during training, but set to a fix value during test and valid, to ensure repeatability.
        step_num = random.randint(1, self.max_step_num)
        step_num = 550 if self.mode in ["test", "valid"] else step_num
        return self.ds_get(save_file=self.save_files[idx], step_num=step_num)

    def ds_get(self, save_file=None, step_num=None):
        """
        Returns as all patches as a single sequence, ready to be encoded by the LLM as a single element of batch.
        Return:
             state.shape = ((seq_len - 1) * num_patches, 3, H, W)
             diff.shape = ((seq_len - 1)  * num_patches, 3, H, W)
             patch_idx: [x_idx, y_idx, t_idx] for each patch

        """

        to_patches = self._get_full_seq(save_file, step_num)

        states, diffs, mask = self._ds_get_pt(to_patches)

        # Get positions / times for each patch
        seq_dim = (self.seq_len - 1) * self.N_patch
        arange = np.arange(seq_dim)
        x_idx = arange % self.N_x_patch
        y_idx = (arange // self.N_x_patch) % self.N_y_patch
        t_idx = arange // self.N_patch

        position_ids = np.stack([x_idx, y_idx, t_idx], axis=1)

        return states, diffs, mask, torch.from_numpy(position_ids)

    def _get_step(self, triang, tri_index, grid_x, grid_y, save_data, step_num):
        """
        Returns all interpolated measurements for a given step, including padding.
        """
        Vx = save_data['velocity'][step_num][:, 0]
        Vy = save_data['velocity'][step_num][:, 1]
        P = save_data['pressure'][step_num][:, 0]

        Vx_interp, Vx_mask = to_grid(Vx, grid_x, grid_y, triang, tri_index)
        Vy_interp, Vy_mask = to_grid(Vy, grid_x, grid_y, triang, tri_index)
        P_interp, P_mask = to_grid(P, grid_x, grid_y, triang, tri_index)

        step_state = np.stack([Vx_interp, Vy_interp, P_interp], axis=0)

        if self.pad:
            step_state, P_mask = self._pad(step_state, P_mask)

        return step_state, P_mask

    def _patch(self, states: torch.Tensor):
        """
        Patches a batch of images.
        Returns a tensor of shape (bs, C, patch_h, patch_w, num_patches)
        """
        bs, C, _, _ = states.shape
        ph, pw = self.patch_size
        # states = states.unsqueeze(0)

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

    def _load_step(self, save_file):
        """ Load save file from disk and calculate mesh interpolation triangles"""

        with open(f"{self.load_dir}/{save_file}", 'rb') as f:
            save_data = pickle.load(f)  # ['faces', 'mesh_pos', 'velocity', 'pressure']
        pos = save_data['mesh_pos']
        faces = save_data['cells']

        triang, tri_index, grid_x, grid_y = get_mesh_interpolation(pos, faces, self.resolution)

        return triang, tri_index, grid_x, grid_y, save_data

    def _get_full_seq(self, save_file=None, step_num=None):
        """ Returns numpy arrays of sequence, ready to be patched.
            Required to avoid pytorch multiprocessing bug.

            Return shape: (seq_len, C+1, H, W)
        """
        if save_file is None:
            save_file = random.choice(self.save_files)

        if step_num is None:
            step_num = np.random.randint(0, self.max_step_num)
        if step_num > self.max_step_num:
            c_print(f"Step number {step_num} too high, setting to max step number {self.max_step_num}", 'red')
            step_num = self.max_step_num

        triang, tri_index, grid_x, grid_y, save_data = self._load_step(save_file)

        to_patches = []
        for i in range(step_num, step_num + self.seq_len * self.seq_interval, self.seq_interval):
            state, mask = self._get_step(triang, tri_index, grid_x, grid_y, save_data, step_num=i)

            # Patch mask with state
            to_patch = np.concatenate([state, mask[None, :, :]], axis=0)
            to_patches.append(to_patch)

        return np.stack(to_patches)

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

        # Compute diffs and discard last state that has no diff
        #diffs = states[1:] - states[:-1]  # shape = (seq_len, num_patches, C, H, W)
        target = states[1:]
        states = states[:-1]

        # Reshape into a continuous sequence
        seq_dim = (self.seq_len - 1) * self.N_patch
        states = states.reshape(seq_dim, 3, self.patch_size[0], self.patch_size[1])
        target = target.reshape(seq_dim, 3, self.patch_size[0], self.patch_size[1])

        # Reshape mask. All masks are the same
        masks = masks[:-1].reshape(seq_dim, 1, self.patch_size[0], self.patch_size[1]).repeat(1, 3, 1, 1)

        return states, target, masks.bool()

    def __len__(self):
        return len(self.save_files)


def plot_all_patches():
    load_no = 1
    step_num = 100
    patch_size, stride = (16, 16), (16, 16)

    seq_dl = MGNDataset(load_dir="/home/bubbles/Documents/LLM_Fluid/ds/MGN/cylinder_dataset", resolution=240, patch_size=patch_size, stride=stride,
                        seq_len=10, seq_interval=2)

    ds = DataLoader(seq_dl, batch_size=8, num_workers=8, prefetch_factor=2, shuffle=True)

    st = time.time()
    for batch in ds:
        state, diffs, mask, pos_id = batch
        print(f"Time to get sequence: {time.time() - st:.3g}s")
        st = time.time()

    x_count, y_count = seq_dl.N_x_patch, seq_dl.N_y_patch

    p_shows = state[0]
    fig, axes = plt.subplots(y_count, x_count, figsize=(16, 4))
    for i in range(y_count):
        for j in range(x_count):
            p_show = p_shows[i + j * y_count].numpy()
            p_show = np.transpose(p_show, (2, 1, 0))

            min, max = seq_dl.ds_min_max[0]

            axes[i, j].imshow(p_show[:, :, 0], vmin=min * 1.5, vmax=max * 1.5)
            axes[i, j].axis('off')
    plt.tight_layout()
    plt.show()

    # p_shows = diffs
    #
    # fig, axes = plt.subplots(y_count, x_count, figsize=(16, 4))
    # for i in range(y_count):
    #     for j in range(x_count):
    #         p_show = p_shows[i + j * y_count].numpy()
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

    # plot_patches(None, 10, 20)
    plot_all_patches()
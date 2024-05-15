import torch

# Example 5D tensors
states = torch.rand(2, 3, 4, 5, 6)  # 5D tensor with shape [2, 3, 4, 5, 6]
masks = torch.rand(2, 3, 4, 5, 6) > 0.5  # Random boolean mask of the same shape

# Mean tensor that is smaller and needs to be broadcasted
# For example, mean for each channel only
mean_value = 0.5
means = torch.tensor([mean_value]).view(1, 1, 1, 1, 1)  # Shape [1, 1, 1, 1, 1]

# Expand means to the same shape as states
expanded_means = means.expand_as(states)

# Apply the operation
states[masks] = states[masks] - expanded_means[masks]

print(states)
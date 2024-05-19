import torch
from matplotlib import pyplot as plt

def custom_transform(x, C=1.0):
    """
    Transforms x to approximate x for small values and C for large values.

    Args:
    - x (Tensor): Input tensor.
    - C (float): The constant value that the function should approach for large x.
    - k (float): Scaling factor for the sigmoid function.

    Returns:
    - Tensor: Transformed tensor.
    """
    sigmoid = torch.sigmoid(x/C) - 0.5
    transformed_output = sigmoid * C * 4
    return transformed_output


# Example usage
x = torch.linspace(-50, 50, 100)
transformed_x = custom_transform(x, C=30.0)
plt.plot(x, transformed_x)

plt.plot(x, x)
plt.show()

print(transformed_x)
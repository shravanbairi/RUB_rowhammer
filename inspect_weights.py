import torch

# Path to your checkpoint
checkpoint_path = "fashionmnist_cnn_earlyexit.pt"

# Load the checkpoint
checkpoint = torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=True
)

# Display the keys
print("Checkpoint contents:")
print(checkpoint.keys())
state_dict = checkpoint["model_state_dict"]

print("\nAvailable model parameters:")

for name, parameter in state_dict.items():
    print(name, parameter.shape)
print(state_dict["conv2.bias"])  # Display the weights of conv2
import torch
import torch.nn as nn

# Example model definition (use your actual architecture)
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        return x

# Step 1: Create the model instance
model = UNet()

# Step 2: Load the state dictionary
state_dict = torch.load('leukemia_cells_unet.pt', map_location=torch.device('cpu'), weights_only=True)

# Step 3: Load the weights into the model instance
model.load_state_dict(state_dict)

# Step 4: Set the model to evaluation mode
model.eval()

import torch
import torch.nn as nn

class CatsDogsCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1=nn.Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=3
     )
        self.relu = nn.ReLU()
        
        self.pool = nn.MaxPool2d(
        kernel_size=2,
        stride=2
)
        
        self.conv2=nn.Conv2d(
            in_channels=16,   # channels 16 because passed down through the first convolutional layer
            out_channels=32,  # asking the network to learn 32 different feature detectors at this stage
            kernel_size=3
     )
        
        self.fc1 = nn.Linear(
            in_features=28800,
            out_features=128
   )
        
        self.fc2 = nn.Linear(
            in_features=128,
            out_features=2
    )

    def forward(self, x):
       # print("Before convolution:", x.shape) 
        x = self.conv1(x)
       # print("After convolution:", x.shape)
        x = self.relu(x) # adding Relu function ensures shape stays the same after convolution
        # print("After ReLU:", x.shape)
        x = self.pool(x)
        # print("After max pooling:", x.shape) # Batch size (number of images in this batch), Number of feature maps (channels),Height, Width
        x = self.conv2(x)
        # print("After convolution:", x.shape)
        x = self.relu(x) # adding Relu function ensures shape stays the same after convolution
        # print("After ReLU:", x.shape)
        x = self.pool(x)
        # print("After max pooling:", x.shape)
        x = torch.flatten(x, start_dim=1)
        # print("After flatten:", x.shape)
        x = self.fc1(x)
        # print("After first linear layer:", x.shape)
        x = self.relu(x)
        # print("After ReLU:", x.shape)
        x = self.fc2(x)
        # print("Final output shape:", x.shape)
        return x



import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, conv_layers=2, num_classes=10, conv_filters=[32, 64], kernel_sizes=[3, 3]):
        """
        CNN model for CIFAR-10 classification.

        :param conv_layers: Number of convolutional layers.
        :param num_classes: Number of output classes (CIFAR-10 has 10).
        :param conv_filters: List of filter sizes for each conv layer.
        :param kernel_sizes: List of kernel sizes for each conv layer.
        """
        super(CNN, self).__init__()

        if(len(conv_filters) != conv_layers):
            conv_filters = [64]*conv_layers
        if(len(kernel_sizes) != conv_layers):
            kernel_sizes = [3]*conv_layers        
        # assert len(conv_filters) == conv_layers, "conv_filters length must match conv_layers"
        # assert len(kernel_sizes) == conv_layers, "kernel_sizes length must match conv_layers"
        
        self.convs = nn.ModuleList()
        in_channels = 3  # CIFAR-10 images have 3 channels (RGB)

        for i in range(conv_layers):
            self.convs.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=conv_filters[i],
                    kernel_size=kernel_sizes[i],
                    padding=1
                )
            )
            in_channels = conv_filters[i]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        sample_input = torch.randn(1, 3, 32, 32)  # CIFAR-10 input size
        sample_output = self._get_feature_size(sample_input)

        self.fc1 = nn.Linear(sample_output, 128)  
        self.fc2 = nn.Linear(128, num_classes)

        
    def _get_feature_size(self, x):
        """ Helper function to compute flattened size dynamically """
        for conv in self.convs:
            x = F.relu(conv(x))
            x = self.pool(x)  # Apply pooling after every conv layer
        return x.numel()  # Returns flattened feature size

    def forward(self, x):
        for conv in self.convs:
            x = F.relu(conv(x))
            x = self.pool(x)

        x = torch.flatten(x, start_dim=1)  # Flatten before FC layer
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

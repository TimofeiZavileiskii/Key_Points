from torch import nn


class LinearBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.relu = nn.LeakyReLU(0.1)
        self.norm = nn.BatchNorm1d(out_features)

    def forward(self, x):
        x = self.linear(x)
        x = self.norm(x)
        x = self.relu(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.conv = nn.Conv2d(in_features, out_features, 3, padding=1)
        self.batchnorm = nn.BatchNorm2d(out_features)
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        return x


class ConvResBlock(nn.Module):
    def __init__(self, in_features, out_features, num_layers):
        super().__init__()
        self.entry_batchnorm = nn.BatchNorm2d(in_features)
        self.expansion_conv = ConvBlock(in_features, out_features)
        self.conv_blocks = nn.ModuleList(
            [ConvBlock(out_features, out_features) for _ in range(num_layers - 1)]
        )
        self.batchnorms = nn.ModuleList(
            [nn.BatchNorm2d(out_features) for _ in range(num_layers - 1)]
        )

    def forward(self, x):
        x = self.entry_batchnorm(x)
        x = self.expansion_conv(x)
        for conv_block, batchnorm in zip(self.conv_blocks, self.batchnorms):
            x = x + conv_block(x)
            x = batchnorm(x)
        return x


class LinearResBlock(nn.Module):
    def __init__(self, in_features, out_features, num_layers):
        super().__init__()
        self.entry_batchnorm = nn.BatchNorm1d(in_features)
        self.expansion_linear = LinearBlock(in_features, out_features)
        self.linear_blocks = nn.ModuleList(
            [LinearBlock(out_features, out_features) for _ in range(num_layers - 1)]
        )
        self.batchnorms = nn.ModuleList(
            [nn.BatchNorm1d(out_features) for _ in range(num_layers - 1)]
        )

    def forward(self, x):
        x = self.entry_batchnorm(x)
        x = self.expansion_linear(x)
        for linear_block, batchnorm in zip(self.linear_blocks, self.batchnorms):
            x = x + linear_block(x)
            x = batchnorm(x)
        return x


# Define model
class NeuralNetwork(nn.Module):
    def __init__(self, conv_per_layer=1, linear_per_layer=1):
        super().__init__()

        self.conv_stack = nn.Sequential(
            ConvResBlock(1, 2, conv_per_layer),
            ConvResBlock(2, 4, conv_per_layer),
            nn.MaxPool2d(2),
            ConvResBlock(4, 8, conv_per_layer),
            ConvResBlock(8, 16, conv_per_layer),
            nn.MaxPool2d(2),
            ConvResBlock(16, 16, conv_per_layer),
            ConvResBlock(16, 32, conv_per_layer),
            nn.MaxPool2d(2),
        )
        self.flatten = nn.Flatten()
        self.block1 = LinearResBlock(4608, 4608 * 2, linear_per_layer)
        self.block2 = LinearResBlock(4608 * 2, 4608, linear_per_layer)
        self.block3 = LinearBlock(4608, 30)

    def forward(self, x):
        x = self.conv_stack(x)
        x = self.flatten(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x

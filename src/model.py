import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_


class ConvolutionBlock(nn.Module):
    def __init__(self, input_dim=128, n_filters=256, kernel_size=3, padding=1, stride=1, shortcut=False, downsample=None):
        super(ConvolutionBlock, self).__init__()
        self.downsample = downsample
        self.shortcut = shortcut

        self.conv1 = nn.Conv1d(input_dim, n_filters, kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn1 = nn.BatchNorm1d(n_filters)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(n_filters, n_filters, kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn2 = nn.BatchNorm1d(n_filters)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.shortcut:
            if self.downsample is not None:
                residual = self.downsample(x)
            out += residual

        out = self.relu(out)
        return out


class VDCNN(nn.Module):
    def __init__(self, n_classes=2, num_embedding=69, embedding_dim=16, depth=9, n_fc_neurons=2048, shortcut=False, maxpooling=False):
        super(VDCNN, self).__init__()

        cnn_layers = []
        fc_layers = []

        self.embedding = nn.Embedding(num_embedding, embedding_dim, padding_idx=0)
        cnn_layers.append(nn.Conv1d(embedding_dim, 64, kernel_size=3, padding=1))

        if depth == 9:
            n_conv_block_64, n_conv_block_128, n_conv_block_256, n_conv_block_512 = 1, 1, 1, 1
        elif depth == 17:
            n_conv_block_64, n_conv_block_128, n_conv_block_256, n_conv_block_512 = 2, 2, 2, 2
        elif depth == 29:
            n_conv_block_64, n_conv_block_128, n_conv_block_256, n_conv_block_512 = 5, 5, 2, 2
        elif depth == 49:
            n_conv_block_64, n_conv_block_128, n_conv_block_256, n_conv_block_512 = 8, 8, 5, 3

        cnn_layers.append(ConvolutionBlock(input_dim=64, n_filters=64, shortcut=shortcut))
        for _ in range(n_conv_block_64 - 1):
            cnn_layers.append(ConvolutionBlock(input_dim=64, n_filters=64, shortcut=shortcut))
        cnn_layers.append(nn.MaxPool1d(kernel_size=3, stride=2, padding=1))

        ds = nn.Sequential(nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1), nn.BatchNorm1d(128))
        cnn_layers.append(ConvolutionBlock(input_dim=64, n_filters=128, shortcut=shortcut, downsample=ds))
        for _ in range(n_conv_block_128 - 1):
            cnn_layers.append(ConvolutionBlock(input_dim=128, n_filters=128, shortcut=shortcut))
        cnn_layers.append(nn.MaxPool1d(kernel_size=3, stride=2, padding=1))

        ds = nn.Sequential(nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1), nn.BatchNorm1d(256))
        cnn_layers.append(ConvolutionBlock(input_dim=128, n_filters=256, shortcut=shortcut, downsample=ds))
        for _ in range(n_conv_block_256 - 1):
            cnn_layers.append(ConvolutionBlock(input_dim=256, n_filters=256, shortcut=shortcut))
        cnn_layers.append(nn.MaxPool1d(kernel_size=3, stride=2, padding=1))

        ds = nn.Sequential(nn.Conv1d(in_channels=256, out_channels=512, kernel_size=1), nn.BatchNorm1d(512))
        cnn_layers.append(ConvolutionBlock(input_dim=256, n_filters=512, shortcut=shortcut, downsample=ds))
        for _ in range(n_conv_block_512 - 1):
            cnn_layers.append(ConvolutionBlock(input_dim=512, n_filters=512, shortcut=shortcut))
        cnn_layers.append(nn.MaxPool1d(kernel_size=3, stride=2, padding=1))

        if maxpooling:
            cnn_layers.append(nn.AdaptiveMaxPool1d(8))
        else:
            cnn_layers.append(nn.AdaptiveAvgPool1d(8))
        fc_layers.extend([nn.Linear(8 * 512, n_fc_neurons), nn.ReLU(), nn.Dropout(0.3)])
        fc_layers.extend([nn.Linear(n_fc_neurons, n_fc_neurons), nn.ReLU(), nn.Dropout(0.3)])
        fc_layers.extend([nn.Linear(n_fc_neurons, n_classes)])

        self.cnn_layers = nn.Sequential(*cnn_layers)
        self.fc_layers = nn.Sequential(*fc_layers)

        self.__init_weights()

    def __init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        out = self.embedding(x)
        out = out.transpose(1, 2)
        out = self.cnn_layers(out)
        out = out.view(out.size(0), -1)
        out = self.fc_layers(out)
        return out


if __name__ == '__main__':
    input = torch.rand(8, 1014).type(torch.int32)
    conv = VDCNN(n_classes=4)
    output = conv(input)
    print(output.size())

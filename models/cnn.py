from torch import nn


class CNN1(nn.Module):
    def __init__(self, num_classes=10, client_id=None):
        super(CNN1, self).__init__()
        self.maxpool = nn.MaxPool2d(2, 2)
        if client_id is not None:
            channel_dif = (5 - client_id) * 2
            channel_num = [32 + channel_dif * 1, 64 + channel_dif * 2, 128 + channel_dif * 4]
        else:
            channel_num = [32, 64, 128]

        self.conv1 = nn.Sequential(nn.Conv2d(3, channel_num[0], 3), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(channel_num[0], channel_num[1], 3), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(channel_num[1], channel_num[2], 3), nn.ReLU())

        self.fc = nn.Linear(channel_num[2] * 4 * 4, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class CNN1_BN(nn.Module):
    def __init__(self, num_classes=10, client_id=None):
        super(CNN1_BN, self).__init__()
        self.maxpool = nn.MaxPool2d(2, 2)
        if client_id is not None:
            channel_dif = (5 - client_id) * 2
            channel_num = [32 + channel_dif * 1, 64 + channel_dif * 2, 128 + channel_dif * 4]
        else:
            channel_num = [32, 64, 128]

        self.conv1 = nn.Sequential(nn.Conv2d(3, channel_num[0], 3), nn.BatchNorm2d(channel_num[0]), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(channel_num[0], channel_num[1], 3), nn.BatchNorm2d(channel_num[1]),
                                   nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(channel_num[1], channel_num[2], 3), nn.BatchNorm2d(channel_num[2]),
                                   nn.ReLU())

        self.fc = nn.Linear(channel_num[2] * 4 * 4, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class CNN2(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN2, self).__init__()
        self.maxpool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Sequential(nn.Conv2d(3, 64, 3), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, 3), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(128, 256, 3), nn.ReLU())

        self.fc = nn.Linear(256 * 4 * 4, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class CNN2_BN(nn.Module):
    def __init__(self, num_classes=10, client_id=None):
        super(CNN2_BN, self).__init__()
        self.maxpool = nn.MaxPool2d(2, 2)

        if client_id is not None:
            channel_dif = (5 - client_id) * 2 - 1
            print(channel_dif)
            channel_num = [64 + channel_dif * 1, 128 + channel_dif * 2, 256 + channel_dif * 4]
        else:
            channel_num = [64, 128, 256]

        self.conv1 = nn.Sequential(nn.Conv2d(3, channel_num[0], 3), nn.BatchNorm2d(channel_num[0]), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(channel_num[0], channel_num[1], 3), nn.BatchNorm2d(channel_num[1]),
                                   nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(channel_num[1], channel_num[2], 3), nn.BatchNorm2d(channel_num[2]),
                                   nn.ReLU())

        self.fc = nn.Linear(channel_num[2] * 4 * 4, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

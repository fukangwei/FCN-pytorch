import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models.vgg import VGG
from BagData import dataloader
import numpy as np
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
import os
import shutil

epoch = 100
use_gpu = True if torch.cuda.is_available() else False


class FCNs(nn.Module):

    def __init__(self, pretrained_net, n_class):
        super().__init__()
        self.n_class = n_class
        self.pretrained_net = pretrained_net
        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)

    def forward(self, x):
        output = self.pretrained_net(x)
        x5 = output['x5']  # size = (N, 512, x.H/32, x.W/32)
        x4 = output['x4']  # size = (N, 512, x.H/16, x.W/16)
        x3 = output['x3']  # size = (N, 256, x.H/8,  x.W/8)
        x2 = output['x2']  # size = (N, 128, x.H/4,  x.W/4)
        x1 = output['x1']  # size = (N, 64, x.H/2,  x.W/2)

        score = self.bn1(self.relu(self.deconv1(x5)))  # size = (N, 512, x.H/16, x.W/16)
        score = score + x4  # element-wise add, size = (N, 512, x.H/16, x.W/16)
        score = self.bn2(self.relu(self.deconv2(score)))  # size = (N, 256, x.H/8, x.W/8)
        score = score + x3  # element-wise add, size = (N, 256, x.H/8, x.W/8)
        score = self.bn3(self.relu(self.deconv3(score)))  # size = (N, 128, x.H/4, x.W/4)
        score = score + x2  # element-wise add, size = (N, 128, x.H/4, x.W/4)
        score = self.bn4(self.relu(self.deconv4(score)))  # size = (N, 64, x.H/2, x.W/2)
        score = score + x1  # element-wise add, size = (N, 64, x.H/2, x.W/2)
        score = self.bn5(self.relu(self.deconv5(score)))  # size = (N, 32, x.H, x.W)
        score = self.classifier(score)  # size = (N, n_class, x.H/1, x.W/1)

        return score  # size = (N, n_class, x.H/1, x.W/1)


class VGGNet(VGG):
    def __init__(self, pretrained=True, model='vgg16', requires_grad=True, remove_fc=True, show_params=False):
        super().__init__(make_layers(cfg[model]))
        self.ranges = ranges[model]

        if pretrained:
            exec("self.load_state_dict(models.%s(pretrained=True).state_dict())" % model)

        if not requires_grad:
            for param in super().parameters():
                param.requires_grad = False

        if remove_fc:  # delete redundant fully-connected layer params, can save memory
            del self.classifier

        if show_params:
            for name, param in self.named_parameters():
                print(name, param.size())

    def forward(self, x):
        output = {}
        for idx in range(len(self.ranges)):  # get the output of each maxpooling layer (5 maxpool in VGG net)
            for layer in range(self.ranges[idx][0], self.ranges[idx][1]):
                x = self.features[layer](x)
            output["x%d" % (idx + 1)] = x

        return output


ranges = {
    'vgg11': ((0, 3), (3, 6), (6, 11), (11, 16), (16, 21)),
    'vgg13': ((0, 5), (5, 10), (10, 15), (15, 20), (20, 25)),
    'vgg16': ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31)),
    'vgg19': ((0, 5), (5, 10), (10, 19), (19, 28), (28, 37))
}

cfg = {  # cropped version from https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


if __name__ == "__main__":
    if os.path.exists("./checkpoints") is True:  # delete the former model file
        shutil.rmtree("./checkpoints")

    os.mkdir("./checkpoints")

    if os.path.exists("./runs") is True:  # delete the former TensorBoard info
        shutil.rmtree("./runs")

    vgg_model = VGGNet(requires_grad=True)
    fcn_model = FCNs(pretrained_net=vgg_model, n_class=2)

    dummy_input = torch.Tensor(4, 3, 160, 160)

    writer = SummaryWriter()
    writer.add_graph(fcn_model, dummy_input)

    if use_gpu:
        fcn_model = fcn_model.cuda()
        criterion = nn.BCELoss().cuda()
    else:
        criterion = nn.BCELoss()

    optimizer = optim.SGD(fcn_model.parameters(), lr=1e-2, momentum=0.7)
    saving_index = 0
    show_index = 0

    for epo in range(epoch):
        saving_index += 1
        index = 0
        epo_loss = 0
        for item in dataloader:
            index += 1
            input = item['A']
            y = item['B']
            input = torch.autograd.Variable(input)
            y = torch.autograd.Variable(y)

            if use_gpu:
                input = input.cuda()
                y = y.cuda()

            optimizer.zero_grad()
            output = fcn_model(input)
            output = torch.sigmoid(output)
            loss = criterion(output, y)
            loss.backward()
            iter_loss = loss.item()
            epo_loss += iter_loss
            optimizer.step()

            output_np = output.cpu().data.numpy().copy()
            output_np = np.argmin(output_np, axis=1)
            y_np = y.cpu().data.numpy().copy()
            y_np = np.argmin(y_np, axis=1)
            if np.mod(index, 20) == 1:
                connect_image = np.concatenate((y_np[:, None, :, :], output_np[:, None, :, :]))
                connect_image = torch.from_numpy(connect_image)
                x = vutils.make_grid(connect_image, normalize=True)
                writer.add_image('Label_Pred', x, show_index)
                show_index += 1
                print('epoch {}, {}/{}, loss is {}'.format(epo, index, len(dataloader), iter_loss))

        print('epoch loss = %f' % (epo_loss / len(dataloader)))
        writer.add_scalar('iter_loss', epo_loss / len(dataloader), epo)

        if np.mod(saving_index, 5) == 1:
            torch.save(fcn_model, 'checkpoints/fcn_model_{}.pt'.format(epo))
            print('saveing checkpoints/fcn_model_{}.pt'.format(epo))

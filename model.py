import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.optim as optim


def _make_divisible(v, divisor=8, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class h_sigmoid(nn.Module):
    """
    This function approximates a sigmoid function with a linear function ReLU6. Reduces costs of computation
    related to non-linearity
    """
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        out = self.relu(x+3)/6
        return out

class h_swish(nn.Module):
    """
    This function is a hard-swish (linear) version of a swish function.
    Uses the fact that we have approximated the sigmoid with a hard sigmoid.
    """
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        out = x*self.sigmoid(x)
        return out

#creating additional Squeeze and Excite layer to add to the bottleneck with residuals
class SE_layer(nn.Module):
    """
    Here we recreate the squeeze and excite layer. Main layer that was added in MBL3 (Figure 4)
    """
    def __init__(self, exp_size, divide=4):
        super(SE_layer, self).__init__()
        # just how it is written in paper
        self.dense_SE = nn.Sequential(
            nn.Linear(exp_size, exp_size//divide),
            nn.ReLU(inplace=True),
            nn.Linear(exp_size//divide, exp_size),
            h_sigmoid()
        )

    def forward(self, x):
        batch, n_channels, h, w = x.size()
        out = F.avg_pool2d(x, kernel_size = [h,w]).view(batch, -1)
        out = self.dense_SE(out)
        out = out.view(batch, n_channels, 1, 1)
        out = out*x
        return out

# I will use a shortcut IRLB Layer to denote inverted residual and linear bottleneck layer defined
# in MobileNetV2
class IRLB(nn.Module):
    """
    Inverted Residual and Linear bottleneck layer from MobileNetV2.
    Consists of 1 Convolutional, 1 Depthwise, 1 Pointwise layer.
    We add activation parameter to differentiate between two parameters...
    """
    def __init__(self, n_in, n_out, kernal_size, stride, activation_function, exp_size):
        super(IRLB, self).__init__()

        self.n_out = n_out
        self.activation_function = activation_function

        padding = (kernal_size - 1) // 2

        # checking type of connection
        self.connection = (stride == 1 and n_in == n_out)

        if self.activation_function == "RE":
            activation = nn.ReLU
        else:
            activation = h_swish

        self.conv = nn.Sequential(
            nn.Conv2d(n_in, exp_size, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(exp_size),
            activation(inplace=True)
        )
        self.depth_conv = nn.Sequential(
            nn.Conv2d(exp_size, exp_size, kernel_size=kernal_size, stride=stride, padding=padding, groups=exp_size),
            nn.BatchNorm2d(exp_size),
        )

        self.point_conv = nn.Sequential(
            nn.Conv2d(exp_size, n_out, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(n_out),
            activation(inplace=True)
        )


    def forward(self, x):
        # Taken from MobileNetV2
        out = self.conv(x)
        out = self.depth_conv(out)
        out = self.point_conv(out) # point-wise conv

        # connection
        if self.connection:
            return x + out
        else:
            return out



class IRLB_SE(nn.Module):
    """
    Inverted Residual and Linear bottleneck layer with Squeeze and Excite Layer
    Main difference in architecture of 2nd model from the 3rd.
    Consists of 1 Convolutional, 1 Depthwise, 1 SE, 1 Pointwise layer.
    We add activation parameter to differentiate between two parameters...
    """
    def __init__(self, n_in, n_out, kernal_size, stride, activation_function, exp_size):
        super(IRLB_SE, self).__init__()

        self.n_out = n_out
        self.activation_function = activation_function

        padding = (kernal_size - 1) // 2

        # checking type of connection
        self.connection = (stride == 1 and n_in == n_out)

        if self.activation_function == "RE":
            activation = nn.ReLU
        else:
            activation = h_swish

        self.conv = nn.Sequential(
            nn.Conv2d(n_in, exp_size, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(exp_size),
            activation(inplace=True)
        )
        self.depth_conv = nn.Sequential(
            nn.Conv2d(exp_size, exp_size, kernel_size=kernal_size, stride=stride, padding=padding, groups=exp_size),
            nn.BatchNorm2d(exp_size),
        )

        self.squeeze_block = SE_layer(exp_size)

        self.point_conv = nn.Sequential(
            nn.Conv2d(exp_size, n_out, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(n_out),
            activation(inplace=True)
        )


    def forward(self, x):
        # Taken from MobileNetV2
        out = self.conv(x)
        out = self.depth_conv(out)
        out = self.squeeze_block(out)
        out = self.point_conv(out) # point-wise conv

        # connection
        if self.connection:
            return x + out
        else:
            return out


class MobileNetV3_large(nn.Module):
    """
    MobileNetV3_large model taken from Table 1, of the paper of interest
    """
    def __init__(self, num_classes=10, multiplier=1.0, dropout_rate=0.0):
        super(MobileNetV3_large, self).__init__()
        self.num_classes = num_classes
        init_conv_out = _make_divisible(16 * multiplier)

        # first layer
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=init_conv_out, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(init_conv_out),
            h_swish(inplace=True)
        )

        self.block = nn.Sequential(
            IRLB(_make_divisible(16), _make_divisible(16), 3, 1, "RE", _make_divisible(16)),
            IRLB(_make_divisible(16), _make_divisible(24), 3, 2, "RE", _make_divisible(64)),
            IRLB(_make_divisible(24), _make_divisible(24), 3, 1, "RE", _make_divisible(72)),
            IRLB_SE(_make_divisible(24), _make_divisible(40), 5, 2, "RE", _make_divisible(72)),
            IRLB_SE(_make_divisible(40), _make_divisible(40), 5, 1, "RE", _make_divisible(120)),
            IRLB_SE(_make_divisible(40), _make_divisible(40), 5, 1, "RE", _make_divisible(120)),
            IRLB(_make_divisible(40), _make_divisible(80), 3, 2, "HS", _make_divisible(240)),
            IRLB(_make_divisible(80), _make_divisible(80), 3, 1, "HS", _make_divisible(200)),
            IRLB(_make_divisible(80), _make_divisible(80), 3, 1, "HS", _make_divisible(184)),
            IRLB(_make_divisible(80), _make_divisible(80), 3, 1, "HS", _make_divisible(184)),
            IRLB_SE(_make_divisible(80), _make_divisible(112), 3, 1, "HS", _make_divisible(480)),
            IRLB_SE(_make_divisible(112), _make_divisible(112), 3, 1, "HS", _make_divisible(672)),
            IRLB_SE(_make_divisible(112), _make_divisible(160), 5, 1, "HS", _make_divisible(672)),
            IRLB_SE(_make_divisible(160), _make_divisible(160), 5, 2, "HS", _make_divisible(672)),
            IRLB_SE(_make_divisible(160), _make_divisible(160), 5, 1, "HS", _make_divisible(960))
        )

        out_conv1_in = _make_divisible(160 * multiplier)
        out_conv1_out = _make_divisible(960 * multiplier)
        self.out_conv1 = nn.Sequential(
            nn.Conv2d(out_conv1_in, out_conv1_out, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_conv1_out),
            h_swish(inplace=True),
        )

        out_conv2_in = _make_divisible(960 * multiplier)
        out_conv2_out = _make_divisible(1280 * multiplier)
        self.out_conv2 = nn.Sequential(
            nn.Conv2d(out_conv2_in, out_conv2_out, kernel_size=1, stride=1),
            h_swish(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(out_conv2_out, self.num_classes, kernel_size=1, stride=1),
        )


    def forward(self, x):
        out = self.init_conv(x)
        out = self.block(out)
        out = self.out_conv1(out)
        batch, channels, height, width = out.size()
        out = F.avg_pool2d(out, kernel_size=[height, width])
        out = self.out_conv2(out).view(batch, -1)

        return out


class MobileNetV3_small(nn.Module):
    """
    MobileNetV3_small model taken from Table 2, of the paper of interest
    """
    def __init__(self, num_classes=10, multiplier=1.0, dropout_rate=0.0):
        super(MobileNetV3_small, self).__init__()
        self.num_classes = num_classes
        init_conv_out = _make_divisible(16 * multiplier)

        # first layer
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=init_conv_out, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(init_conv_out),
            h_swish(inplace=True)
        )

        self.block = nn.Sequential(
            IRLB_SE(_make_divisible(16), _make_divisible(16), 3, 2, "RE", _make_divisible(16)),
            IRLB(_make_divisible(16), _make_divisible(24), 3, 2, "RE", _make_divisible(72)),
            IRLB(_make_divisible(24), _make_divisible(24), 3, 1, "RE", _make_divisible(88)),
            IRLB_SE(_make_divisible(24), _make_divisible(40), 5, 2, "RE", _make_divisible(96)),
            IRLB_SE(_make_divisible(40), _make_divisible(40), 5, 1, "RE", _make_divisible(240)),
            IRLB_SE(_make_divisible(40), _make_divisible(40), 5, 1, "RE", _make_divisible(240)),
            IRLB_SE(_make_divisible(40), _make_divisible(48), 5, 1, "HS", _make_divisible(120)),
            IRLB_SE(_make_divisible(48), _make_divisible(48), 5, 1, "HS", _make_divisible(144)),
            IRLB_SE(_make_divisible(48), _make_divisible(96), 5, 2, "HS", _make_divisible(288)),
            IRLB_SE(_make_divisible(96), _make_divisible(96), 5, 1, "HS", _make_divisible(576)),
            IRLB_SE(_make_divisible(96), _make_divisible(96), 5, 1, "HS", _make_divisible(576))
        )

        out_conv1_in = _make_divisible(96 * multiplier)
        out_conv1_out = _make_divisible(576 * multiplier)

        self.out_conv1 = nn.Sequential(
            nn.Conv2d(out_conv1_in, out_conv1_out, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_conv1_out),
            h_swish(inplace=True),
        )

        out_conv2_in = _make_divisible(576 * multiplier)
        out_conv2_out = _make_divisible(1280 * multiplier)
        self.out_conv2 = nn.Sequential(
            nn.Conv2d(out_conv2_in, out_conv2_out, kernel_size=1, stride=1),
            h_swish(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(out_conv2_out, self.num_classes, kernel_size=1, stride=1),
        )


    def forward(self, x):
        out = self.init_conv(x)
        out = self.block(out)
        out = self.out_conv1(out)
        batch, channels, height, width = out.size()
        out = F.avg_pool2d(out, kernel_size=[height, width])
        out = self.out_conv2(out).view(batch, -1)

        return out

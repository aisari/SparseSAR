import math
import torch as th
import torchbox as tb


def raw2img(y, Phi):
    if Phi is None:
        return y
    HW = Phi.shape[1]
    H = W = int(math.sqrt(HW))
    N = y.shape[0]
    Phi = tb.r2c(Phi, cdim=-1)
    # Phi = th.pinverse(Phi)
    # Phi = th.inverse(Phi)
    Phi = Phi.t().conj()
    y = tb.r2c(y, cdim=-1)
    y = y.reshape(N, -1)
    x = (Phi @ y.T).T
    x = x.reshape(N, H, W)
    x = tb.c2r(x, cdim=-1)
    return x


class ResBlock(th.nn.Module):

    def __init__(self, channels=64, kernel_size=3, iscomplex=False):
        super(ResBlock, self).__init__()
        self.iscomplex = iscomplex

        if iscomplex:
            conv2d = tb.ComplexConv2d
            relu = tb.ComplexReLU
            IN = tb.ComplexBatchNorm2d
        else:
            conv2d = th.nn.Conv2d
            relu = th.nn.ReLU
            IN = th.nn.InstanceNorm2d

        self.conv1 = conv2d(channels, channels, kernel_size=kernel_size,
                            padding=kernel_size // 2, bias=True)
        # self.conv2 = conv2d(channels, channels, kernel_size=kernel_size,
        #                     padding=kernel_size // 2, bias=True)
        self.IN = IN(channels)
        self.relu = relu(inplace=False)

        self.actname = 'relu'

        self._initialize_weights()

    def forward(self, x):
        out = self.conv1(x)
        out = self.IN(out)
        out = self.relu(out)
        # out = self.conv2(out)
        # return out + x
        return out

    def _initialize_weights(self):
        if self.iscomplex:
            th.nn.init.orthogonal_(self.conv1.convr.weight, th.nn.init.calculate_gain(self.actname))
            th.nn.init.orthogonal_(self.conv1.convi.weight, th.nn.init.calculate_gain(self.actname))
            # th.nn.init.orthogonal_(self.conv2.convr.weight, th.nn.init.calculate_gain(self.actname))
            # th.nn.init.orthogonal_(self.conv2.convi.weight, th.nn.init.calculate_gain(self.actname))
        else:
            th.nn.init.orthogonal_(self.conv1.weight, th.nn.init.calculate_gain(self.actname))
            # th.nn.init.orthogonal_(self.conv2.weight, th.nn.init.calculate_gain(self.actname))


class Net(th.nn.Module):

    def __init__(self, input_channels=2, kernel_size=3, channels=64, iscomplex=False, Phi=None):
        super(Net, self).__init__()

        self.iscomplex = iscomplex

        if iscomplex:
            conv2d = tb.ComplexConv2d
            relu = tb.ComplexReLU
        else:
            conv2d = th.nn.Conv2d
            IN = th.nn.InstanceNorm2d
            relu = th.nn.ReLU
        self.Phi = Phi

        self.entry = conv2d(input_channels, channels, kernel_size=kernel_size,
                            padding=kernel_size // 2, bias=True)

        self.resblock1 = ResBlock(channels=channels, kernel_size=kernel_size, iscomplex=iscomplex)
        self.resblock2 = ResBlock(channels=channels, kernel_size=kernel_size, iscomplex=iscomplex)
        self.resblock3 = ResBlock(channels=channels, kernel_size=kernel_size, iscomplex=iscomplex)
        self.resblock4 = ResBlock(channels=channels, kernel_size=kernel_size, iscomplex=iscomplex)
        self.resblock5 = ResBlock(channels=channels, kernel_size=kernel_size, iscomplex=iscomplex)

        self.out = conv2d(channels, input_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=True)
        self.IN = IN(channels)
        self.relu = relu(inplace=False)

        self.actname = 'relu'

        self._initialize_weights()

    def forward(self, x):
        xinput = x
        x = self.IN(self.entry(x))
        # x = self.entry(x)
        x = self.relu(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        x = self.resblock5(x)

        x = self.out(x)

        # return x + xinput
        return x

    def _initialize_weights(self):
        if self.iscomplex:
            th.nn.init.orthogonal_(self.entry.convr.weight, th.nn.init.calculate_gain(self.actname))
            th.nn.init.orthogonal_(self.entry.convi.weight, th.nn.init.calculate_gain(self.actname))
            th.nn.init.orthogonal_(self.out.convr.weight, th.nn.init.calculate_gain(self.actname))
            th.nn.init.orthogonal_(self.out.convi.weight, th.nn.init.calculate_gain(self.actname))
        else:
            th.nn.init.orthogonal_(self.entry.weight, th.nn.init.calculate_gain(self.actname))
            th.nn.init.orthogonal_(self.out.weight, th.nn.init.calculate_gain(self.actname))

    def get_params_number(self):
        num_params = 0
        for param in self.parameters():
            num_params += param.nelement()
        print('# of params:', num_params)
        return num_params


if __name__ == '__main__':

    nb = 4
    xinput = th.randn(nb, 3, 32, 32)
    net = Net(input_channels=3, iscomplex=False)
    net.get_params_number()
    out = net(xinput)
    print(out.shape)

    xinput = th.randn(nb, 1, 32, 32) + 1j * th.randn(nb, 1, 32, 32)
    xinput = th.view_as_real(xinput)
    net = Net(input_channels=1, iscomplex=True)
    net.get_params_number()
    out = net(xinput)
    print(out.shape)

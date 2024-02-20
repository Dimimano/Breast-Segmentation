import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn.modules.pooling import _MaxUnpoolNd
from torch.nn.modules.utils import _pair

class MaxUnpool2dop(Function):
    """We warp the `torch.nn.functional.max_unpool2d`
    with an extra `symbolic` method, which is needed while exporting to ONNX.
    Users should not call this function directly.
    """

    @staticmethod
    def forward(ctx, input, indices, kernel_size, stride, padding,
                output_size):
        """Forward function of MaxUnpool2dop.
        Args:
            input (Tensor): Tensor needed to upsample.
            indices (Tensor): Indices output of the previous MaxPool.
            kernel_size (Tuple): Size of the max pooling window.
            stride (Tuple): Stride of the max pooling window.
            padding (Tuple): Padding that was added to the input.
            output_size (List or Tuple): The shape of output tensor.
        Returns:
            Tensor: Output tensor.
        """
        return F.max_unpool2d(input, indices, kernel_size, stride, padding,
                              output_size)


    @staticmethod
    def symbolic(g, input, indices, kernel_size, stride, padding, output_size):
        # get shape
        input_shape = g.op('Shape', input)
        const_0 = g.op('Constant', value_t=torch.tensor(0))
        const_1 = g.op('Constant', value_t=torch.tensor(1))
        output_size_list = list(output_size)
        const_size = g.op('Constant', value_t=torch.tensor(output_size_list))
        batch_size = g.op('Gather', input_shape, const_0, axis_i=0)
        channel = g.op('Gather', input_shape, const_1, axis_i=0)

        # height = (height - 1) * stride + kernel_size
        height = g.op(
            'Gather',
            input_shape,
            g.op('Constant', value_t=torch.tensor(2)),
            axis_i=0)
        height = g.op('Sub', height, const_1)
        height = g.op('Mul', height,
                      g.op('Constant', value_t=torch.tensor(stride[1])))
        height = g.op('Add', height,
                      g.op('Constant', value_t=torch.tensor(kernel_size[1])))

        # width = (width - 1) * stride + kernel_size
        width = g.op(
            'Gather',
            input_shape,
            g.op('Constant', value_t=torch.tensor(3)),
            axis_i=0)
        width = g.op('Sub', width, const_1)
        width = g.op('Mul', width,
                     g.op('Constant', value_t=torch.tensor(stride[0])))
        width = g.op('Add', width,
                     g.op('Constant', value_t=torch.tensor(kernel_size[0])))

        # step of channel
        channel_step = g.op('Mul', height, width)
        # step of batch
        batch_step = g.op('Mul', channel_step, channel)

        # channel offset
        range_channel = g.op('Range', const_0, channel, const_1)
        range_channel = g.op(
            'Reshape', range_channel,
            g.op('Constant', value_t=torch.tensor([1, -1, 1, 1])))
        range_channel = g.op('Mul', range_channel, channel_step)
        range_channel = g.op('Cast', range_channel, to_i=7)  # 7 is int64

        # batch offset
        range_batch = g.op('Range', const_0, batch_size, const_1)
        range_batch = g.op(
            'Reshape', range_batch,
            g.op('Constant', value_t=torch.tensor([-1, 1, 1, 1])))
        range_batch = g.op('Mul', range_batch, batch_step)
        range_batch = g.op('Cast', range_batch, to_i=7)  # 7 is int64

        # update indices
        indices = g.op('Add', indices, range_channel)
        indices = g.op('Add', indices, range_batch)

        return g.op(
            'MaxUnpool',
            input,
            indices,
            const_size,
            kernel_shape_i=kernel_size,
            strides_i=stride)

class MaxUnpool2d(_MaxUnpoolNd):
    """This module is modified from Pytorch `MaxUnpool2d` module.
    Args:
      kernel_size (int or tuple): Size of the max pooling window.
      stride (int or tuple): Stride of the max pooling window.
          Default: None (It is set to `kernel_size` by default).
      padding (int or tuple): Padding that is added to the input.
          Default: 0.
    """

    def __init__(self, kernel_size, stride=None, padding=0):
        super(MaxUnpool2d, self).__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride or kernel_size)
        self.padding = _pair(padding)

    def forward(self, input, indices, output_size=None):
        """Forward function of MaxUnpool2d.
        Args:
            input (Tensor): Tensor needed to upsample.
            indices (Tensor): Indices output of the previous MaxPool.
            output_size (List or Tuple): The shape of output tensor.
                Default: None.
        Returns:
            Tensor: Output tensor.
        """
        if output_size is not None and isinstance(output_size, torch.Size):
          output_size = tuple(s.item() for s in output_size)

        return MaxUnpool2dop.apply(input, indices, self.kernel_size,
                                   self.stride, self.padding, output_size)

class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()
		 
    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))
    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        return x, p
		
class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c+out_c, out_c)
    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        return x

class SegmentationHeadCNN(nn.Module):
    def __init__(self, chin, chout, kernel_size):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels=chin, out_channels=chout, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.seq(x)
        return y
		
class ImageSegmentationUnet4(nn.Module):
    def __init__(self):
        super().__init__()
        """ Encoder """
        self.e1 = encoder_block(3, 64)
        self.e2 = encoder_block(64, 128)
        self.e3 = encoder_block(128, 256)
        self.e4 = encoder_block(256, 512)
        """ Bottleneck """
        self.b = conv_block(512, 1024)
        """ Decoder """
        self.d1 = decoder_block(1024, 512)
        self.d2 = decoder_block(512, 256)
        self.d3 = decoder_block(256, 128)
        self.d4 = decoder_block(128, 64)
        self.segHead = SegmentationHeadCNN(64, 1, kernel_size=3)
		
    def forward(self, inputs):
        """ Encoder """
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)
        """ Bottleneck """
        b = self.b(p4)
        """ Decoder """
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)
        d = self.segHead(d4)
		
        return d

class ImageSegmentationUnet5(nn.Module):
    def __init__(self):
        super().__init__()
        """ Encoder """
        self.e1 = encoder_block(3, 64)
        self.e2 = encoder_block(64, 128)
        self.e3 = encoder_block(128, 256)
        self.e4 = encoder_block(256, 512)
        self.e5 = encoder_block(512, 1024)
        """ Bottleneck """
        self.b = conv_block(1024, 2048)
        """ Decoder """
        self.d1 = decoder_block(2048, 1024)
        self.d2 = decoder_block(1024, 512)
        self.d3 = decoder_block(512, 256)
        self.d4 = decoder_block(256, 128)
        self.d5 = decoder_block(128, 64)
        self.segHead = SegmentationHeadCNN(64, 1, kernel_size=3)
		
    def forward(self, inputs):
        """ Encoder """
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)
        s5, p5 = self.e5(p4)
        """ Bottleneck """
        b = self.b(p5)
        """ Decoder """
        d1 = self.d1(b, s5)
        d2 = self.d2(d1, s4)
        d3 = self.d3(d2, s3)
        d4 = self.d4(d3, s2)
        d5 = self.d4(d4, s1)
        d = self.segHead(d5)
		
        return d
    # end def
# end class

class DownConv2(nn.Module):
    def __init__(self, chin, chout, kernel_size):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels=chin, out_channels=chout, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(chout),
            nn.ReLU(),
            nn.Conv2d(in_channels=chout, out_channels=chout, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(chout),
            nn.ReLU(),
        )
        self.mp = nn.MaxPool2d(kernel_size=2, return_indices=True)

    def forward(self, x):
        y = self.seq(x)
        pool_shape = y.shape
        y, indices = self.mp(y)
        return y, indices, pool_shape

class DownConv3(nn.Module):
    def __init__(self, chin, chout, kernel_size):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels=chin, out_channels=chout, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(chout),
            nn.ReLU(),
            nn.Conv2d(in_channels=chout, out_channels=chout, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(chout),
            nn.ReLU(),
            nn.Conv2d(in_channels=chout, out_channels=chout, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(chout),
            nn.ReLU(),
        )
        self.mp = nn.MaxPool2d(kernel_size=2, return_indices=True)

    def forward(self, x):
        y = self.seq(x)
        pool_shape = y.shape
        y, indices = self.mp(y)
        return y, indices, pool_shape

class UpConv2(nn.Module):
    def __init__(self, chin, chout, kernel_size):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels=chin, out_channels=chin, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(chin),
            nn.ReLU(),
            nn.Conv2d(in_channels=chin, out_channels=chout, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(chout),
            nn.ReLU(),
        )
        self.mup = nn.MaxUnpool2d(kernel_size=2)

    def forward(self, x, indices, output_size):
        y = self.mup(x, indices, output_size=output_size)
        y = self.seq(y)
        return y

class UpConv3(nn.Module):
    def __init__(self, chin, chout, kernel_size):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels=chin, out_channels=chin, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(chin),
            nn.ReLU(),
            nn.Conv2d(in_channels=chin, out_channels=chin, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(chin),
            nn.ReLU(),
            nn.Conv2d(in_channels=chin, out_channels=chout, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(chout),
            nn.ReLU(),
        )
        self.mup = nn.MaxUnpool2d(kernel_size=2)

    def forward(self, x, indices, output_size):
        y = self.mup(x, indices, output_size=output_size)
        y = self.seq(y)
        return y
    
class ImageSegmentationBasic3(torch.nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.out_channels = 3
        self.bn_input = nn.BatchNorm2d(3)
        self.dc1 = DownConv2(3, 64, kernel_size=kernel_size)
        self.dc2 = DownConv2(64, 128, kernel_size=kernel_size)
        self.dc3 = DownConv2(128, 256, kernel_size=kernel_size)
        #self.dc4 = DownConv3(256, 512, kernel_size=kernel_size)
        # self.dc5 = DownConv3(512, 512, kernel_size=kernel_size)

        # self.uc5 = UpConv3(512, 512, kernel_size=kernel_size)
        #self.uc4 = UpConv3(512, 256, kernel_size=kernel_size)
        self.uc3 = UpConv2(256, 128, kernel_size=kernel_size)
        self.uc2 = UpConv2(128, 64, kernel_size=kernel_size)
        self.uc1 = UpConv2(64, 3, kernel_size=kernel_size)
        self.segHead = SegmentationHeadCNN(3, 1, kernel_size=kernel_size)

    def forward(self, batch: torch.Tensor):
        x = self.bn_input(batch)
        # x = batch
        # SegNet Encoder
        x, mp1_indices, shape1 = self.dc1(x)
        x, mp2_indices, shape2 = self.dc2(x)
        x, mp3_indices, shape3 = self.dc3(x)
        #x, mp4_indices, shape4 = self.dc4(x)
        # Our images are 128x128 in dimension. If we run 4 max pooling
        # operations, we are down to 128/16 = 8x8 activations. If we
        # do another down convolution, we'll be at 4x4 and at that point
        # in time, we may lose too much spatial information as a result
        # of the MaxPooling operation, so we stop at 4 down conv
        # operations.
        # x, mp5_indices, shape5 = self.dc5(x)

        # SegNet Decoder
        # x = self.uc5(x, mp5_indices, output_size=shape5)
        #x = self.uc4(x, mp4_indices, output_size=shape4)
        x = self.uc3(x, mp3_indices, output_size=shape3)
        x = self.uc2(x, mp2_indices, output_size=shape2)
        x = self.uc1(x, mp1_indices, output_size=shape1)
        x = self.segHead(x)

        return x
    # end def
# end class

class ImageSegmentationBasic4(torch.nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.out_channels = 3
        self.bn_input = nn.BatchNorm2d(3)
        self.dc1 = DownConv2(3, 64, kernel_size=kernel_size)
        self.dc2 = DownConv2(64, 128, kernel_size=kernel_size)
        self.dc3 = DownConv2(128, 256, kernel_size=kernel_size)
        self.dc4 = DownConv3(256, 512, kernel_size=kernel_size)
        # self.dc5 = DownConv3(512, 512, kernel_size=kernel_size)

        # self.uc5 = UpConv3(512, 512, kernel_size=kernel_size)
        self.uc4 = UpConv3(512, 256, kernel_size=kernel_size)
        self.uc3 = UpConv2(256, 128, kernel_size=kernel_size)
        self.uc2 = UpConv2(128, 64, kernel_size=kernel_size)
        self.uc1 = UpConv2(64, 3, kernel_size=kernel_size)
        self.segHead = SegmentationHeadCNN(3, 1, kernel_size=kernel_size)

    def forward(self, batch: torch.Tensor):
        x = self.bn_input(batch)
        # x = batch
        # SegNet Encoder
        x, mp1_indices, shape1 = self.dc1(x)
        x, mp2_indices, shape2 = self.dc2(x)
        x, mp3_indices, shape3 = self.dc3(x)
        x, mp4_indices, shape4 = self.dc4(x)
        # Our images are 128x128 in dimension. If we run 4 max pooling
        # operations, we are down to 128/16 = 8x8 activations. If we
        # do another down convolution, we'll be at 4x4 and at that point
        # in time, we may lose too much spatial information as a result
        # of the MaxPooling operation, so we stop at 4 down conv
        # operations.
        # x, mp5_indices, shape5 = self.dc5(x)

        # SegNet Decoder
        # x = self.uc5(x, mp5_indices, output_size=shape5)
        x = self.uc4(x, mp4_indices, output_size=shape4)
        x = self.uc3(x, mp3_indices, output_size=shape3)
        x = self.uc2(x, mp2_indices, output_size=shape2)
        x = self.uc1(x, mp1_indices, output_size=shape1)
        x = self.segHead(x)

        return x
    # end def
# end class

class ImageSegmentationBasic5(torch.nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.out_channels = 3
        self.bn_input = nn.BatchNorm2d(3)
        self.dc1 = DownConv2(3, 64, kernel_size=kernel_size)
        self.dc2 = DownConv2(64, 128, kernel_size=kernel_size)
        self.dc3 = DownConv3(128, 256, kernel_size=kernel_size)
        self.dc4 = DownConv3(256, 512, kernel_size=kernel_size)
        self.dc5 = DownConv3(512, 512, kernel_size=kernel_size)

        self.uc5 = UpConv3(512, 512, kernel_size=kernel_size)
        self.uc4 = UpConv3(512, 256, kernel_size=kernel_size)
        self.uc3 = UpConv3(256, 128, kernel_size=kernel_size)
        self.uc2 = UpConv2(128, 64, kernel_size=kernel_size)
        self.uc1 = UpConv2(64, 3, kernel_size=kernel_size)
        self.segHead = SegmentationHeadCNN(3, 1, kernel_size=kernel_size)

    def forward(self, batch: torch.Tensor):
        x = self.bn_input(batch)
        # x = batch
        # SegNet Encoder
        x, mp1_indices, shape1 = self.dc1(batch)
        x, mp2_indices, shape2 = self.dc2(x)
        x, mp3_indices, shape3 = self.dc3(x)
        x, mp4_indices, shape4 = self.dc4(x)
        # Our images are 128x128 in dimension. If we run 4 max pooling
        # operations, we are down to 128/16 = 8x8 activations. If we
        # do another down convolution, we'll be at 4x4 and at that point
        # in time, we may lose too much spatial information as a result
        # of the MaxPooling operation, so we stop at 4 down conv
        # operations.
        x, mp5_indices, shape5 = self.dc5(x)

        # SegNet Decoder
        x = self.uc5(x, mp5_indices, output_size=shape5)
        x = self.uc4(x, mp4_indices, output_size=shape4)
        x = self.uc3(x, mp3_indices, output_size=shape3)
        x = self.uc2(x, mp2_indices, output_size=shape2)
        x = self.uc1(x, mp1_indices, output_size=shape1)
        x = self.segHead(x)

        return x

class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, bias=True):
        super().__init__()
        # The depthwise conv is basically just a grouped convolution in PyTorch with
        # the number of distinct groups being the same as the number of input (and output)
        # channels for that layer.
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, bias=bias, groups=in_channels)
        # The pointwise convolution stretches across all the output channels using
        # a 1x1 kernel.
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class DownDSConv2(nn.Module):
    def __init__(self, chin, chout, kernel_size):
        super().__init__()
        self.seq = nn.Sequential(
            DepthwiseSeparableConv2d(in_channels=chin, out_channels=chout, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(chout),
            nn.ReLU(),
            DepthwiseSeparableConv2d(in_channels=chout, out_channels=chout, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(chout),
            nn.ReLU(),
        )
        self.mp = nn.MaxPool2d(kernel_size=2, return_indices=True)

    def forward(self, x):
        y = self.seq(x)
        pool_shape = y.shape
        y, indices = self.mp(y)
        return y, indices, pool_shape

class DownDSConv3(nn.Module):
    def __init__(self, chin, chout, kernel_size):
        super().__init__()
        self.seq = nn.Sequential(
            DepthwiseSeparableConv2d(in_channels=chin, out_channels=chout, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(chout),
            nn.ReLU(),
            DepthwiseSeparableConv2d(in_channels=chout, out_channels=chout, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(chout),
            nn.ReLU(),
            DepthwiseSeparableConv2d(in_channels=chout, out_channels=chout, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(chout),
            nn.ReLU(),
        )
        self.mp = nn.MaxPool2d(kernel_size=2, return_indices=True)

    def forward(self, x):
        y = self.seq(x)
        pool_shape = y.shape
        y, indices = self.mp(y)
        return y, indices, pool_shape

class UpDSConv2(nn.Module):
    def __init__(self, chin, chout, kernel_size, export):
        super().__init__()
        self.seq = nn.Sequential(
            DepthwiseSeparableConv2d(in_channels=chin, out_channels=chin, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(chin),
            nn.ReLU(),
            DepthwiseSeparableConv2d(in_channels=chin, out_channels=chout, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(chout),
            nn.ReLU(),
        )
        if export:
            self.mup = MaxUnpool2d(kernel_size=2)
        else:
            self.mup = nn.MaxUnpool2d(kernel_size=2)

    def forward(self, x, indices, output_size):
        y = self.mup(x, indices, output_size=output_size)
        y = self.seq(y)
        return y

class UpDSConv3(nn.Module):
    def __init__(self, chin, chout, kernel_size, export):
        super().__init__()
        self.seq = nn.Sequential(
            DepthwiseSeparableConv2d(in_channels=chin, out_channels=chin, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(chin),
            nn.ReLU(),
            DepthwiseSeparableConv2d(in_channels=chin, out_channels=chin, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(chin),
            nn.ReLU(),
            DepthwiseSeparableConv2d(in_channels=chin, out_channels=chout, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(chout),
            nn.ReLU(),
        )
        if export:
            print('yo')
            self.mup = MaxUnpool2d(kernel_size=2)
        else:
            self.mup = nn.MaxUnpool2d(kernel_size=2)

    def forward(self, x, indices, output_size):
        y = self.mup(x, indices, output_size=output_size)
        y = self.seq(y)
        return y

class SegmentationHead(nn.Module):
    def __init__(self, chin, chout, kernel_size):
        super().__init__()
        self.seq = nn.Sequential(
            DepthwiseSeparableConv2d(in_channels=chin, out_channels=chout, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.seq(x)
        return y
    
class ImageSegmentationDSC4_95(torch.nn.Module):
    def __init__(self, kernel_size, export):
        super().__init__()

        self.out_channels = 3
        self.bn_input = nn.BatchNorm2d(3)
        self.dc1 = DownDSConv2(3, 3, kernel_size=kernel_size)
        self.dc2 = DownDSConv2(3, 6, kernel_size=kernel_size)
        self.dc3 = DownDSConv2(6, 12, kernel_size=kernel_size)
        self.dc4 = DownDSConv3(12, 25, kernel_size=kernel_size)
        # self.dc5 = DownConv3(512, 512, kernel_size=kernel_size)

        # self.uc5 = UpConv3(512, 512, kernel_size=kernel_size)
        self.uc4 = UpDSConv3(25, 12, kernel_size=kernel_size, export=export)
        self.uc3 = UpDSConv2(12, 6, kernel_size=kernel_size, export=export)
        self.uc2 = UpDSConv2(6, 3, kernel_size=kernel_size, export=export)
        self.uc1 = UpDSConv2(3, 3, kernel_size=kernel_size, export=export)
        self.segHead = SegmentationHead(3, 1, kernel_size=kernel_size)

    def forward(self, batch: torch.Tensor):
        x = self.bn_input(batch)
        # x = batch
        # SegNet Encoder
        x, mp1_indices, shape1 = self.dc1(x)
        x, mp2_indices, shape2 = self.dc2(x)
        x, mp3_indices, shape3 = self.dc3(x)
        x, mp4_indices, shape4 = self.dc4(x)
        # Our images are 128x128 in dimension. If we run 4 max pooling
        # operations, we are down to 128/16 = 8x8 activations. If we
        # do another down convolution, we'll be at 4x4 and at that point
        # in time, we may lose too much spatial information as a result
        # of the MaxPooling operation, so we stop at 4 down conv
        # operations.
        # x, mp5_indices, shape5 = self.dc5(x)

        # SegNet Decoder
        # x = self.uc5(x, mp5_indices, output_size=shape5)
        x = self.uc4(x, mp4_indices, output_size=shape4)
        x = self.uc3(x, mp3_indices, output_size=shape3)
        x = self.uc2(x, mp2_indices, output_size=shape2)
        x = self.uc1(x, mp1_indices, output_size=shape1)
        x = self.segHead(x)

        return x
    # end def
# end class
    
class ImageSegmentationDSC4_90(torch.nn.Module):
    def __init__(self, kernel_size, export):
        super().__init__()

        self.out_channels = 3
        self.bn_input = nn.BatchNorm2d(3)
        self.dc1 = DownDSConv2(3, 6, kernel_size=kernel_size)
        self.dc2 = DownDSConv2(6, 12, kernel_size=kernel_size)
        self.dc3 = DownDSConv2(12, 25, kernel_size=kernel_size)
        self.dc4 = DownDSConv3(25, 51, kernel_size=kernel_size)
        # self.dc5 = DownConv3(512, 512, kernel_size=kernel_size)

        # self.uc5 = UpConv3(512, 512, kernel_size=kernel_size)
        self.uc4 = UpDSConv3(51, 25, kernel_size=kernel_size, export=export)
        self.uc3 = UpDSConv2(25, 12, kernel_size=kernel_size, export=export)
        self.uc2 = UpDSConv2(12, 6, kernel_size=kernel_size, export=export)
        self.uc1 = UpDSConv2(6, 3, kernel_size=kernel_size, export=export)
        self.segHead = SegmentationHead(3, 1, kernel_size=kernel_size)

    def forward(self, batch: torch.Tensor):
        x = self.bn_input(batch)
        # x = batch
        # SegNet Encoder
        x, mp1_indices, shape1 = self.dc1(x)
        x, mp2_indices, shape2 = self.dc2(x)
        x, mp3_indices, shape3 = self.dc3(x)
        x, mp4_indices, shape4 = self.dc4(x)
        # Our images are 128x128 in dimension. If we run 4 max pooling
        # operations, we are down to 128/16 = 8x8 activations. If we
        # do another down convolution, we'll be at 4x4 and at that point
        # in time, we may lose too much spatial information as a result
        # of the MaxPooling operation, so we stop at 4 down conv
        # operations.
        # x, mp5_indices, shape5 = self.dc5(x)

        # SegNet Decoder
        # x = self.uc5(x, mp5_indices, output_size=shape5)
        x = self.uc4(x, mp4_indices, output_size=shape4)
        x = self.uc3(x, mp3_indices, output_size=shape3)
        x = self.uc2(x, mp2_indices, output_size=shape2)
        x = self.uc1(x, mp1_indices, output_size=shape1)
        x = self.segHead(x)

        return x
    # end def
# end class
    
class ImageSegmentationDSC4_85(torch.nn.Module):
    def __init__(self, kernel_size, export):
        super().__init__()

        self.out_channels = 3
        self.bn_input = nn.BatchNorm2d(3)
        self.dc1 = DownDSConv2(3, 8, kernel_size=kernel_size)
        self.dc2 = DownDSConv2(8, 19, kernel_size=kernel_size)
        self.dc3 = DownDSConv2(19, 38, kernel_size=kernel_size)
        self.dc4 = DownDSConv3(38, 77, kernel_size=kernel_size)
        # self.dc5 = DownConv3(512, 512, kernel_size=kernel_size)

        # self.uc5 = UpConv3(512, 512, kernel_size=kernel_size)
        self.uc4 = UpDSConv3(77, 38, kernel_size=kernel_size, export=export)
        self.uc3 = UpDSConv2(38, 19, kernel_size=kernel_size, export=export)
        self.uc2 = UpDSConv2(19, 8, kernel_size=kernel_size, export=export)
        self.uc1 = UpDSConv2(8, 3, kernel_size=kernel_size, export=export)
        self.segHead = SegmentationHead(3, 1, kernel_size=kernel_size)

    def forward(self, batch: torch.Tensor):
        x = self.bn_input(batch)
        # x = batch
        # SegNet Encoder
        x, mp1_indices, shape1 = self.dc1(x)
        x, mp2_indices, shape2 = self.dc2(x)
        x, mp3_indices, shape3 = self.dc3(x)
        x, mp4_indices, shape4 = self.dc4(x)
        # Our images are 128x128 in dimension. If we run 4 max pooling
        # operations, we are down to 128/16 = 8x8 activations. If we
        # do another down convolution, we'll be at 4x4 and at that point
        # in time, we may lose too much spatial information as a result
        # of the MaxPooling operation, so we stop at 4 down conv
        # operations.
        # x, mp5_indices, shape5 = self.dc5(x)

        # SegNet Decoder
        # x = self.uc5(x, mp5_indices, output_size=shape5)
        x = self.uc4(x, mp4_indices, output_size=shape4)
        x = self.uc3(x, mp3_indices, output_size=shape3)
        x = self.uc2(x, mp2_indices, output_size=shape2)
        x = self.uc1(x, mp1_indices, output_size=shape1)
        x = self.segHead(x)

        return x
    # end def
# end class
    
class ImageSegmentationDSC4_80(torch.nn.Module):
    def __init__(self, kernel_size, export):
        super().__init__()

        self.out_channels = 3
        self.bn_input = nn.BatchNorm2d(3)
        self.dc1 = DownDSConv2(3, 12, kernel_size=kernel_size)
        self.dc2 = DownDSConv2(12, 25, kernel_size=kernel_size)
        self.dc3 = DownDSConv2(25, 51, kernel_size=kernel_size)
        self.dc4 = DownDSConv3(51, 102, kernel_size=kernel_size)
        # self.dc5 = DownConv3(512, 512, kernel_size=kernel_size)

        # self.uc5 = UpConv3(512, 512, kernel_size=kernel_size)
        self.uc4 = UpDSConv3(102, 51, kernel_size=kernel_size, export=export)
        self.uc3 = UpDSConv2(51, 25, kernel_size=kernel_size, export=export)
        self.uc2 = UpDSConv2(25, 12, kernel_size=kernel_size, export=export)
        self.uc1 = UpDSConv2(12, 3, kernel_size=kernel_size, export=export)
        self.segHead = SegmentationHead(3, 1, kernel_size=kernel_size)

    def forward(self, batch: torch.Tensor):
        x = self.bn_input(batch)
        # x = batch
        # SegNet Encoder
        x, mp1_indices, shape1 = self.dc1(x)
        x, mp2_indices, shape2 = self.dc2(x)
        x, mp3_indices, shape3 = self.dc3(x)
        x, mp4_indices, shape4 = self.dc4(x)
        # Our images are 128x128 in dimension. If we run 4 max pooling
        # operations, we are down to 128/16 = 8x8 activations. If we
        # do another down convolution, we'll be at 4x4 and at that point
        # in time, we may lose too much spatial information as a result
        # of the MaxPooling operation, so we stop at 4 down conv
        # operations.
        # x, mp5_indices, shape5 = self.dc5(x)

        # SegNet Decoder
        # x = self.uc5(x, mp5_indices, output_size=shape5)
        x = self.uc4(x, mp4_indices, output_size=shape4)
        x = self.uc3(x, mp3_indices, output_size=shape3)
        x = self.uc2(x, mp2_indices, output_size=shape2)
        x = self.uc1(x, mp1_indices, output_size=shape1)
        x = self.segHead(x)

        return x
    # end def
# end class
    
class ImageSegmentationDSC4_75(torch.nn.Module):
    def __init__(self, kernel_size, export):
        super().__init__()

        self.out_channels = 3
        self.bn_input = nn.BatchNorm2d(3)
        self.dc1 = DownDSConv2(3, 16, kernel_size=kernel_size)
        self.dc2 = DownDSConv2(16, 32, kernel_size=kernel_size)
        self.dc3 = DownDSConv2(32, 64, kernel_size=kernel_size)
        self.dc4 = DownDSConv3(64, 128, kernel_size=kernel_size)
        # self.dc5 = DownConv3(512, 512, kernel_size=kernel_size)

        # self.uc5 = UpConv3(512, 512, kernel_size=kernel_size)
        self.uc4 = UpDSConv3(128, 64, kernel_size=kernel_size, export=export)
        self.uc3 = UpDSConv2(64, 32, kernel_size=kernel_size, export=export)
        self.uc2 = UpDSConv2(32, 16, kernel_size=kernel_size, export=export)
        self.uc1 = UpDSConv2(16, 3, kernel_size=kernel_size, export=export)
        self.segHead = SegmentationHead(3, 1, kernel_size=kernel_size)

    def forward(self, batch: torch.Tensor):
        x = self.bn_input(batch)
        # x = batch
        # SegNet Encoder
        x, mp1_indices, shape1 = self.dc1(x)
        x, mp2_indices, shape2 = self.dc2(x)
        x, mp3_indices, shape3 = self.dc3(x)
        x, mp4_indices, shape4 = self.dc4(x)
        # Our images are 128x128 in dimension. If we run 4 max pooling
        # operations, we are down to 128/16 = 8x8 activations. If we
        # do another down convolution, we'll be at 4x4 and at that point
        # in time, we may lose too much spatial information as a result
        # of the MaxPooling operation, so we stop at 4 down conv
        # operations.
        # x, mp5_indices, shape5 = self.dc5(x)

        # SegNet Decoder
        # x = self.uc5(x, mp5_indices, output_size=shape5)
        x = self.uc4(x, mp4_indices, output_size=shape4)
        x = self.uc3(x, mp3_indices, output_size=shape3)
        x = self.uc2(x, mp2_indices, output_size=shape2)
        x = self.uc1(x, mp1_indices, output_size=shape1)
        x = self.segHead(x)

        return x
    # end def
# end class
    
class ImageSegmentationDSC4_50(torch.nn.Module):
    def __init__(self, kernel_size, export):
        super().__init__()

        self.out_channels = 3
        self.bn_input = nn.BatchNorm2d(3)
        self.dc1 = DownDSConv2(3, 32, kernel_size=kernel_size)
        self.dc2 = DownDSConv2(32, 64, kernel_size=kernel_size)
        self.dc3 = DownDSConv2(64, 128, kernel_size=kernel_size)
        self.dc4 = DownDSConv3(128, 256, kernel_size=kernel_size)
        # self.dc5 = DownConv3(512, 512, kernel_size=kernel_size)

        # self.uc5 = UpConv3(512, 512, kernel_size=kernel_size)
        self.uc4 = UpDSConv3(256, 128, kernel_size=kernel_size, export=export)
        self.uc3 = UpDSConv2(128, 64, kernel_size=kernel_size, export=export)
        self.uc2 = UpDSConv2(64, 32, kernel_size=kernel_size, export=export)
        self.uc1 = UpDSConv2(32, 3, kernel_size=kernel_size, export=export)
        self.segHead = SegmentationHead(3, 1, kernel_size=kernel_size)

    def forward(self, batch: torch.Tensor):
        x = self.bn_input(batch)
        # x = batch
        # SegNet Encoder
        x, mp1_indices, shape1 = self.dc1(x)
        x, mp2_indices, shape2 = self.dc2(x)
        x, mp3_indices, shape3 = self.dc3(x)
        x, mp4_indices, shape4 = self.dc4(x)
        # Our images are 128x128 in dimension. If we run 4 max pooling
        # operations, we are down to 128/16 = 8x8 activations. If we
        # do another down convolution, we'll be at 4x4 and at that point
        # in time, we may lose too much spatial information as a result
        # of the MaxPooling operation, so we stop at 4 down conv
        # operations.
        # x, mp5_indices, shape5 = self.dc5(x)

        # SegNet Decoder
        # x = self.uc5(x, mp5_indices, output_size=shape5)
        x = self.uc4(x, mp4_indices, output_size=shape4)
        x = self.uc3(x, mp3_indices, output_size=shape3)
        x = self.uc2(x, mp2_indices, output_size=shape2)
        x = self.uc1(x, mp1_indices, output_size=shape1)
        x = self.segHead(x)

        return x
    # end def
# end class
    
class ImageSegmentationDSC4_25(torch.nn.Module):
    def __init__(self, kernel_size, export):
        super().__init__()

        self.out_channels = 3
        self.bn_input = nn.BatchNorm2d(3)
        self.dc1 = DownDSConv2(3, 48, kernel_size=kernel_size)
        self.dc2 = DownDSConv2(48, 96, kernel_size=kernel_size)
        self.dc3 = DownDSConv2(96, 192, kernel_size=kernel_size)
        self.dc4 = DownDSConv3(192, 384, kernel_size=kernel_size)
        # self.dc5 = DownConv3(512, 512, kernel_size=kernel_size)

        # self.uc5 = UpConv3(512, 512, kernel_size=kernel_size)
        self.uc4 = UpDSConv3(384, 192, kernel_size=kernel_size, export=export)
        self.uc3 = UpDSConv2(192, 96, kernel_size=kernel_size, export=export)
        self.uc2 = UpDSConv2(96, 48, kernel_size=kernel_size, export=export)
        self.uc1 = UpDSConv2(48, 3, kernel_size=kernel_size, export=export)
        self.segHead = SegmentationHead(3, 1, kernel_size=kernel_size)

    def forward(self, batch: torch.Tensor):
        x = self.bn_input(batch)
        # x = batch
        # SegNet Encoder
        x, mp1_indices, shape1 = self.dc1(x)
        x, mp2_indices, shape2 = self.dc2(x)
        x, mp3_indices, shape3 = self.dc3(x)
        x, mp4_indices, shape4 = self.dc4(x)
        # Our images are 128x128 in dimension. If we run 4 max pooling
        # operations, we are down to 128/16 = 8x8 activations. If we
        # do another down convolution, we'll be at 4x4 and at that point
        # in time, we may lose too much spatial information as a result
        # of the MaxPooling operation, so we stop at 4 down conv
        # operations.
        # x, mp5_indices, shape5 = self.dc5(x)

        # SegNet Decoder
        # x = self.uc5(x, mp5_indices, output_size=shape5)
        x = self.uc4(x, mp4_indices, output_size=shape4)
        x = self.uc3(x, mp3_indices, output_size=shape3)
        x = self.uc2(x, mp2_indices, output_size=shape2)
        x = self.uc1(x, mp1_indices, output_size=shape1)
        x = self.segHead(x)

        return x
    # end def
# end class
    
class ImageSegmentationDSC3(torch.nn.Module):
    def __init__(self, kernel_size, export):
        super().__init__()
        self.out_channels = 3
        self.bn_input = nn.BatchNorm2d(3)
        self.dc1 = DownDSConv2(3, 64, kernel_size=kernel_size)
        self.dc2 = DownDSConv2(64, 128, kernel_size=kernel_size)
        self.dc3 = DownDSConv2(128, 256, kernel_size=kernel_size)
        #self.dc4 = DownDSConv3(256, 512, kernel_size=kernel_size)
        # self.dc5 = DownConv3(512, 512, kernel_size=kernel_size)

        # self.uc5 = UpConv3(512, 512, kernel_size=kernel_size)
        #self.uc4 = UpDSConv3(512, 256, kernel_size=kernel_size, export=export)
        self.uc3 = UpDSConv2(256, 128, kernel_size=kernel_size, export=export)
        self.uc2 = UpDSConv2(128, 64, kernel_size=kernel_size, export=export)
        self.uc1 = UpDSConv2(64, 3, kernel_size=kernel_size, export=export)
        self.segHead = SegmentationHead(3, 1, kernel_size=kernel_size)

    def forward(self, batch: torch.Tensor):
        x = self.bn_input(batch)
        # x = batch
        # SegNet Encoder
        x, mp1_indices, shape1 = self.dc1(x)
        x, mp2_indices, shape2 = self.dc2(x)
        x, mp3_indices, shape3 = self.dc3(x)
        #x, mp4_indices, shape4 = self.dc4(x)
        # Our images are 128x128 in dimension. If we run 4 max pooling
        # operations, we are down to 128/16 = 8x8 activations. If we
        # do another down convolution, we'll be at 4x4 and at that point
        # in time, we may lose too much spatial information as a result
        # of the MaxPooling operation, so we stop at 4 down conv
        # operations.
        # x, mp5_indices, shape5 = self.dc5(x)

        # SegNet Decoder
        # x = self.uc5(x, mp5_indices, output_size=shape5)
        #x = self.uc4(x, mp4_indices, output_size=shape4)
        x = self.uc3(x, mp3_indices, output_size=shape3)
        x = self.uc2(x, mp2_indices, output_size=shape2)
        x = self.uc1(x, mp1_indices, output_size=shape1)
        x = self.segHead(x)

        return x
    # end def
# end class
    

class ImageSegmentationDSC4(torch.nn.Module):
    def __init__(self, kernel_size, export):
        super().__init__()
        self.out_channels = 3
        self.bn_input = nn.BatchNorm2d(3)
        self.dc1 = DownDSConv2(3, 64, kernel_size=kernel_size)
        self.dc2 = DownDSConv2(64, 128, kernel_size=kernel_size)
        self.dc3 = DownDSConv2(128, 256, kernel_size=kernel_size)
        self.dc4 = DownDSConv3(256, 512, kernel_size=kernel_size)
        # self.dc5 = DownConv3(512, 512, kernel_size=kernel_size)

        # self.uc5 = UpConv3(512, 512, kernel_size=kernel_size)
        self.uc4 = UpDSConv3(512, 256, kernel_size=kernel_size, export=export)
        self.uc3 = UpDSConv2(256, 128, kernel_size=kernel_size, export=export)
        self.uc2 = UpDSConv2(128, 64, kernel_size=kernel_size, export=export)
        self.uc1 = UpDSConv2(64, 3, kernel_size=kernel_size, export=export)
        self.segHead = SegmentationHead(3, 1, kernel_size=kernel_size)

    def forward(self, batch: torch.Tensor):
        x = self.bn_input(batch)
        # x = batch
        # SegNet Encoder
        x, mp1_indices, shape1 = self.dc1(x)
        x, mp2_indices, shape2 = self.dc2(x)
        x, mp3_indices, shape3 = self.dc3(x)
        x, mp4_indices, shape4 = self.dc4(x)
        # Our images are 128x128 in dimension. If we run 4 max pooling
        # operations, we are down to 128/16 = 8x8 activations. If we
        # do another down convolution, we'll be at 4x4 and at that point
        # in time, we may lose too much spatial information as a result
        # of the MaxPooling operation, so we stop at 4 down conv
        # operations.
        # x, mp5_indices, shape5 = self.dc5(x)

        # SegNet Decoder
        # x = self.uc5(x, mp5_indices, output_size=shape5)
        x = self.uc4(x, mp4_indices, output_size=shape4)
        x = self.uc3(x, mp3_indices, output_size=shape3)
        x = self.uc2(x, mp2_indices, output_size=shape2)
        x = self.uc1(x, mp1_indices, output_size=shape1)
        x = self.segHead(x)

        return x
    # end def
# end class

class ImageSegmentationDSC5(torch.nn.Module):
    def __init__(self, kernel_size, export):
        super().__init__()
        self.out_channels = 3
        self.bn_input = nn.BatchNorm2d(3)
        self.dc1 = DownDSConv2(3, 64, kernel_size=kernel_size)
        self.dc2 = DownDSConv2(64, 128, kernel_size=kernel_size)
        self.dc3 = DownDSConv3(128, 256, kernel_size=kernel_size)
        self.dc4 = DownDSConv3(256, 512, kernel_size=kernel_size)
        self.dc5 = DownDSConv3(512, 512, kernel_size=kernel_size)

        self.uc5 = UpDSConv3(512, 512, kernel_size=kernel_size, export=export)
        self.uc4 = UpDSConv3(512, 256, kernel_size=kernel_size, export=export)
        self.uc3 = UpDSConv3(256, 128, kernel_size=kernel_size, export=export)
        self.uc2 = UpDSConv2(128, 64, kernel_size=kernel_size, export=export)
        self.uc1 = UpDSConv2(64, 3, kernel_size=kernel_size, export=export)
        self.segHead = SegmentationHead(3, 1, kernel_size=kernel_size)

    def forward(self, batch: torch.Tensor):
        x = self.bn_input(batch)
        # x = batch
        # SegNet Encoder
        x, mp1_indices, shape1 = self.dc1(x)
        x, mp2_indices, shape2 = self.dc2(x)
        x, mp3_indices, shape3 = self.dc3(x)
        x, mp4_indices, shape4 = self.dc4(x)
        # Our images are 128x128 in dimension. If we run 4 max pooling
        # operations, we are down to 128/16 = 8x8 activations. If we
        # do another down convolution, we'll be at 4x4 and at that point
        # in time, we may lose too much spatial information as a result
        # of the MaxPooling operation, so we stop at 4 down conv
        # operations.
        x, mp5_indices, shape5 = self.dc5(x)

        # SegNet Decoder
        x = self.uc5(x, mp5_indices, output_size=shape5)
        x = self.uc4(x, mp4_indices, output_size=shape4)
        x = self.uc3(x, mp3_indices, output_size=shape3)
        x = self.uc2(x, mp2_indices, output_size=shape2)
        x = self.uc1(x, mp1_indices, output_size=shape1)
        x = self.segHead(x)

        return x
			
class DepthwiseSeparableConv2dMobile(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, bias=True):
        super().__init__()
        # The depthwise conv is basically just a grouped convolution in PyTorch with
        # the number of distinct groups being the same as the number of input (and output)
        # channels for that layer.
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, bias=bias, groups=in_channels)
        # The pointwise convolution stretches across all the output channels using
        # a 1x1 kernel.
        self.BN = nn.BatchNorm2d(in_channels)
        self.RELU = nn.ReLU()
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.BN(x)
        x = self.RELU(x)
        x = self.pointwise(x)
        return x

class DownDSConv2Mobile(nn.Module):
    def __init__(self, chin, chout, kernel_size):
        super().__init__()
        self.seq = nn.Sequential(
            DepthwiseSeparableConv2dMobile(in_channels=chin, out_channels=chout, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(chout),
            nn.ReLU(),
            DepthwiseSeparableConv2dMobile(in_channels=chout, out_channels=chout, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(chout),
            nn.ReLU(),
        )
        self.mp = nn.MaxPool2d(kernel_size=2, return_indices=True)

    def forward(self, x):
        y = self.seq(x)
        pool_shape = y.shape
        y, indices = self.mp(y)
        return y, indices, pool_shape

class DownDSConv3Mobile(nn.Module):
    def __init__(self, chin, chout, kernel_size):
        super().__init__()
        self.seq = nn.Sequential(
            DepthwiseSeparableConv2dMobile(in_channels=chin, out_channels=chout, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(chout),
            nn.ReLU(),
            DepthwiseSeparableConv2dMobile(in_channels=chout, out_channels=chout, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(chout),
            nn.ReLU(),
            DepthwiseSeparableConv2dMobile(in_channels=chout, out_channels=chout, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(chout),
            nn.ReLU(),
        )
        self.mp = nn.MaxPool2d(kernel_size=2, return_indices=True)

    def forward(self, x):
        y = self.seq(x)
        pool_shape = y.shape
        y, indices = self.mp(y)
        return y, indices, pool_shape

class UpDSConv2Mobile(nn.Module):
    def __init__(self, chin, chout, kernel_size):
        super().__init__()
        self.seq = nn.Sequential(
            DepthwiseSeparableConv2dMobile(in_channels=chin, out_channels=chin, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(chin),
            nn.ReLU(),
            DepthwiseSeparableConv2dMobile(in_channels=chin, out_channels=chout, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(chout),
            nn.ReLU(),
        )
        self.mup = nn.MaxUnpool2d(kernel_size=2)

    def forward(self, x, indices, output_size):
        y = self.mup(x, indices, output_size=output_size)
        y = self.seq(y)
        return y

class UpDSConv3Mobile(nn.Module):
    def __init__(self, chin, chout, kernel_size):
        super().__init__()
        self.seq = nn.Sequential(
            DepthwiseSeparableConv2dMobile(in_channels=chin, out_channels=chin, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(chin),
            nn.ReLU(),
            DepthwiseSeparableConv2dMobile(in_channels=chin, out_channels=chin, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(chin),
            nn.ReLU(),
            DepthwiseSeparableConv2dMobile(in_channels=chin, out_channels=chout, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(chout),
            nn.ReLU(),
        )
        self.mup = nn.MaxUnpool2d(kernel_size=2)

    def forward(self, x, indices, output_size):
        y = self.mup(x, indices, output_size=output_size)
        y = self.seq(y)
        return y

class SegmentationHeadMobile(nn.Module):
    def __init__(self, chin, chout, kernel_size):
        super().__init__()
        self.seq = nn.Sequential(
            DepthwiseSeparableConv2dMobile(in_channels=chin, out_channels=chout, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.seq(x)
        return y
		
class ImageSegmentationDSCMobile5(torch.nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.out_channels = 3
        self.bn_input = nn.BatchNorm2d(3)
        self.dc1 = DownDSConv2Mobile(3, 64, kernel_size=kernel_size)
        self.dc2 = DownDSConv2Mobile(64, 128, kernel_size=kernel_size)
        self.dc3 = DownDSConv2Mobile(128, 256, kernel_size=kernel_size)
        self.dc4 = DownDSConv2Mobile(256, 512, kernel_size=kernel_size)
        self.dc5 = DownDSConv3Mobile(512, 512, kernel_size=kernel_size)

        self.uc5 = UpDSConv3Mobile(512, 512, kernel_size=kernel_size)
        self.uc4 = UpDSConv2Mobile(512, 256, kernel_size=kernel_size)
        self.uc3 = UpDSConv2Mobile(256, 128, kernel_size=kernel_size)
        self.uc2 = UpDSConv2Mobile(128, 64, kernel_size=kernel_size)
        self.uc1 = UpDSConv2Mobile(64, 3, kernel_size=kernel_size)
        self.segHead = SegmentationHeadMobile(3, 1, kernel_size=kernel_size)

    def forward(self, batch: torch.Tensor):
        x = self.bn_input(batch)
        # x = batch
        # SegNet Encoder
        x, mp1_indices, shape1 = self.dc1(x)
        x, mp2_indices, shape2 = self.dc2(x)
        x, mp3_indices, shape3 = self.dc3(x)
        x, mp4_indices, shape4 = self.dc4(x)
        # Our images are 128x128 in dimension. If we run 4 max pooling
        # operations, we are down to 128/16 = 8x8 activations. If we
        # do another down convolution, we'll be at 4x4 and at that point
        # in time, we may lose too much spatial information as a result
        # of the MaxPooling operation, so we stop at 4 down conv
        # operations.
        x, mp5_indices, shape5 = self.dc5(x)

        # SegNet Decoder
        x = self.uc5(x, mp5_indices, output_size=shape5)
        x = self.uc4(x, mp4_indices, output_size=shape4)
        x = self.uc3(x, mp3_indices, output_size=shape3)
        x = self.uc2(x, mp2_indices, output_size=shape2)
        x = self.uc1(x, mp1_indices, output_size=shape1)
        x = self.segHead(x)

        return x
		
class ImageSegmentationDSCMobile4(torch.nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.out_channels = 3
        self.bn_input = nn.BatchNorm2d(3)
        self.dc1 = DownDSConv2Mobile(3, 64, kernel_size=kernel_size)
        self.dc2 = DownDSConv2Mobile(64, 128, kernel_size=kernel_size)
        self.dc3 = DownDSConv2Mobile(128, 256, kernel_size=kernel_size)
        self.dc4 = DownDSConv3Mobile(256, 512, kernel_size=kernel_size)
        #self.dc5 = DownDSConv3Mobile(512, 512, kernel_size=kernel_size)

        #self.uc5 = UpDSConv3Mobile(512, 512, kernel_size=kernel_size)
        self.uc4 = UpDSConv3Mobile(512, 256, kernel_size=kernel_size)
        self.uc3 = UpDSConv2Mobile(256, 128, kernel_size=kernel_size)
        self.uc2 = UpDSConv2Mobile(128, 64, kernel_size=kernel_size)
        self.uc1 = UpDSConv2Mobile(64, 3, kernel_size=kernel_size)
        self.segHead = SegmentationHeadMobile(3, 1, kernel_size=kernel_size)

    def forward(self, batch: torch.Tensor):
        x = self.bn_input(batch)
        # x = batch
        # SegNet Encoder
        x, mp1_indices, shape1 = self.dc1(x)
        x, mp2_indices, shape2 = self.dc2(x)
        x, mp3_indices, shape3 = self.dc3(x)
        x, mp4_indices, shape4 = self.dc4(x)
        # Our images are 128x128 in dimension. If we run 4 max pooling
        # operations, we are down to 128/16 = 8x8 activations. If we
        # do another down convolution, we'll be at 4x4 and at that point
        # in time, we may lose too much spatial information as a result
        # of the MaxPooling operation, so we stop at 4 down conv
        # operations.
        # x, mp5_indices, shape5 = self.dc5(x)

        # SegNet Decoder
        # x = self.uc5(x, mp5_indices, output_size=shape5)
        x = self.uc4(x, mp4_indices, output_size=shape4)
        x = self.uc3(x, mp3_indices, output_size=shape3)
        x = self.uc2(x, mp2_indices, output_size=shape2)
        x = self.uc1(x, mp1_indices, output_size=shape1)
        x = self.segHead(x)

        return x
    
class DownResDSConv2(nn.Module):
    def __init__(self, chin, chout, kernel_size):
        super().__init__()
        self.seq = nn.Sequential(
            DepthwiseSeparableConv2d(in_channels=chin, out_channels=chout, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(chout),
            nn.ReLU(),
            DepthwiseSeparableConv2d(in_channels=chout, out_channels=chout, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(chout),
        )
        self.relu = nn.ReLU()
        self.downsample = nn.Sequential(
                nn.Conv2d(chin, chout, padding=kernel_size//2, bias=False, kernel_size=3),
                nn.BatchNorm2d(chout),
            )
        self.mp = nn.MaxPool2d(kernel_size=2, return_indices=True)

    def forward(self, x):
        residual = x
        y = self.seq(x)
        #print(y.shape,self.downsample(residual).shape)
        out = y + self.downsample(residual)
        y = self.relu(out)
        pool_shape = y.shape
        y, indices = self.mp(y)
        return y, indices, pool_shape

class DownResDSConv3(nn.Module):
    def __init__(self, chin, chout, kernel_size):
        super().__init__()
        self.seq = nn.Sequential(
            DepthwiseSeparableConv2d(in_channels=chin, out_channels=chout, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(chout),
            nn.ReLU(),
            DepthwiseSeparableConv2d(in_channels=chout, out_channels=chout, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(chout),
            nn.ReLU(),
            DepthwiseSeparableConv2d(in_channels=chout, out_channels=chout, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(chout),
        )
        self.relu = nn.ReLU()
        self.downsample = nn.Sequential(
                nn.Conv2d(chin, chout, padding=kernel_size//2, bias=False, kernel_size=3),
                nn.BatchNorm2d(chout),
            )
        self.mp = nn.MaxPool2d(kernel_size=2, return_indices=True)

    def forward(self, x):
        residual = x
        y = self.seq(x)
        #print(y.shape,self.downsample(residual).shape)
        out = y + self.downsample(residual)
        y = self.relu(out)
        pool_shape = y.shape
        y, indices = self.mp(y)
        return y, indices, pool_shape
    
class ImageSegmentationResDSC4(torch.nn.Module):
    def __init__(self, kernel_size, export):
        super().__init__()
        self.out_channels = 3
        self.bn_input = nn.BatchNorm2d(3)
        self.dc1 = DownResDSConv2(3, 32, kernel_size=kernel_size)
        self.dc2 = DownResDSConv2(32, 64, kernel_size=kernel_size)
        self.dc3 = DownResDSConv3(64, 128, kernel_size=kernel_size)
        self.dc4 = DownResDSConv3(128, 256, kernel_size=kernel_size)
        # self.dc5 = DownConv3(512, 512, kernel_size=kernel_size)

        # self.uc5 = UpConv3(512, 512, kernel_size=kernel_size)
        self.uc4 = UpDSConv3(256, 128, kernel_size=kernel_size, export=export)
        self.uc3 = UpDSConv2(128, 64, kernel_size=kernel_size, export=export)
        self.uc2 = UpDSConv2(64, 32, kernel_size=kernel_size, export=export)
        self.uc1 = UpDSConv2(32, 3, kernel_size=kernel_size, export=export)
        self.segHead = SegmentationHead(3, 1, kernel_size=kernel_size)

    def forward(self, batch: torch.Tensor):
        x = self.bn_input(batch)
        # x = batch
        # SegNet Encoder
        x, mp1_indices, shape1 = self.dc1(x)
        x, mp2_indices, shape2 = self.dc2(x)
        x, mp3_indices, shape3 = self.dc3(x)
        x, mp4_indices, shape4 = self.dc4(x)
        # Our images are 128x128 in dimension. If we run 4 max pooling
        # operations, we are down to 128/16 = 8x8 activations. If we
        # do another down convolution, we'll be at 4x4 and at that point
        # in time, we may lose too much spatial information as a result
        # of the MaxPooling operation, so we stop at 4 down conv
        # operations.
        # x, mp5_indices, shape5 = self.dc5(x)

        # SegNet Decoder
        # x = self.uc5(x, mp5_indices, output_size=shape5)
        x = self.uc4(x, mp4_indices, output_size=shape4)
        x = self.uc3(x, mp3_indices, output_size=shape3)
        x = self.uc2(x, mp2_indices, output_size=shape2)
        x = self.uc1(x, mp1_indices, output_size=shape1)
        x = self.segHead(x)

        return x
    # end def
# end class
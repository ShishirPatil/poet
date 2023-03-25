from poet.power_computation import *
from power_computation_transformer import *


def stablediffusion_256(training, iterations, text_encoding, noise=None, image=(3, 256, 256)):
    layers = [InputLayer(*image)] if training else [InputLayer(*noise)]
    if training:
        vaeencodeblock(layers)
    for _ in range(iterations):
        unet(layers, text_encoding)
    if not training:
        vaedecodeblock(layers)
    return layers


def vaeencodeblock(layers):
    layers.append(Conv2dLayer(3, 3, (3,3), 1, (1, 1), layers[-1]))
    layers.extend(vaeresnetshort(3, 32, 1, (1,1), layers[-1]))
    layers.extend(vaeresnet(32, 32, 1, (1,1), layers[-1]))
    layers.extend(vaeresnet(32, 32, 1, (1,1), layers[-1]))
    layers.append(Conv2dLayer(32, 32, (3,3), 2, (0, 0), layers[-1]))
    layers.extend(vaeresnetshort(32, 64, 1, (1,1), layers[-1]))
    layers.extend(vaeresnet(64, 64, 1, (1,1), layers[-1]))
    layers.extend(vaeresnet(64, 64, 1, (1,1), layers[-1]))
    layers.append(Conv2dLayer(64, 64, (3,3), 2, (0, 0), layers[-1]))
    layers.extend(vaeresnetshort(64, 128, 1, (1,1), layers[-1]))
    layers.extend(vaeresnet(128, 128, 1, (1,1), layers[-1]))
    layers.extend(vaeresnet(128, 128, 1, (1,1), layers[-1]))
    layers.append(Conv2dLayer(128, 128, (3,3), 2, (0, 0), layers[-1]))
    layers.extend(vaeresnet(128, 128, 1, (1,1), layers[-1]))
    layers.extend(vaeresnet(128, 128, 1, (1,1), layers[-1]))
    layers.extend(vaeresnet(128, 128, 1, (1,1), layers[-1]))
    layers.append(Conv2dLayer(128, 128, (3,3), 2, (0, 0), layers[-1]))
    layers.extend(vaeresnet(128, 128, 1, (1,1), layers[-1]))
    attention_block(layers)
    layers.extend(vaeresnet(128, 128, 1, (1,1), layers[-1]))
    layers.append(BatchNorm2d(layers[-1]))
    layers.append(Conv2dLayer(128, 128, (3,3), 1, (1, 1), layers[-1]))
    return 

def vaedecodeblock(layers):
    layers.append(Conv2dLayer(128, 128, (3,3), 1, (1, 1), layers[-1]))
    layers.extend(vaeresnet(128, 128, 1, (1,1), layers[-1]))
    attention_block(layers)
    layers.extend(vaeresnet(128, 128, 1, (1,1), layers[-1]))
    layers.extend(vaeresnet(128, 128, 1, (1,1), layers[-1]))
    layers.extend(vaeresnet(128, 128, 1, (1,1), layers[-1]))
    layers.append(Conv2dLayer(128, 128, (3,3), 2, (0, 0), layers[-1]))
    layers.extend(vaeresnetshort(128, 64, 1, (1,1), layers[-1]))
    layers.extend(vaeresnet(64, 64, 1, (1,1), layers[-1]))
    layers.extend(vaeresnet(64, 64, 1, (1,1), layers[-1]))
    layers.append(Conv2dLayer(64, 64, (3,3), 2, (0, 0), layers[-1]))
    layers.extend(vaeresnetshort(64, 32, 1, (1,1), layers[-1]))
    layers.extend(vaeresnet(32, 32, 1, (1,1), layers[-1]))
    layers.extend(vaeresnet(32, 32, 1, (1,1), layers[-1]))
    layers.append(Conv2dLayer(32, 32, (3,3), 2, (0, 0), layers[-1]))
    layers.extend(vaeresnetshort(32, 3, 1, (1,1), layers[-1]))
    layers.extend(vaeresnet(3, 3, 1, (1,1), layers[-1]))
    layers.extend(vaeresnet(3, 3, 1, (1,1), layers[-1]))
    layers.append(Conv2dLayer(3, 3, (3,3), 2, (0, 0), layers[-1]))
    layers.append(BatchNorm2d(layers[-1]))
    layers.append(Conv2dLayer(3, 3, (3,3), 1, (1, 1), layers[-1]))
    return 

def unet(layers, attention_input):
    #to add: proper timestep embedding structure
    layers.append(LinearLayer(128, 512, None))
    layers.append(SigmoidLayer(layers[-1]))
    layers.append(LinearLayer(512, 512, layers[-1]))
    #Downsampling
    timestepembed = layers[-1]
    layers.extend(resnetblock(128, 128, 1, (1,1), layers[-4], timestepembed))
    layers.extend(resnetblock(128, 128, 1, (1,1), layers[-1], timestepembed))
    layers.append(Conv2dLayer(128, 128, (3,3), 2, (0, 0), layers[-1]))
    layers.extend(resnetblock(128, 128, 1, (1,1), layers[-1], timestepembed))
    layers.extend(resnetblock(128, 128, 1, (1,1), layers[-1], timestepembed))
    layers.append(skip128 := Conv2dLayer(128, 128, (3,3), 2, (0, 0), layers[-1]))
    layers.extend(resnetshortblock(128, 256, 1, (1,1), layers[-1], timestepembed))
    layers.extend(resnetblock(256, 256, 1, (1,1), layers[-1], timestepembed))
    layers.append(Conv2dLayer(256, 256, (3,3), 2, (0, 0), layers[-1]))
    layers.extend(resnetblock(256, 256, 1, (1,1), layers[-1], timestepembed))
    layers.extend(resnetblock(256, 256, 1, (1,1), layers[-1], timestepembed))
    layers.append(Conv2dLayer(256, 256, (3,3), 2, (0, 0), layers[-1]))
    layers.extend(resnetblock(256, 256, 1, (1,1), layers[-1], timestepembed))
    layers.extend(resnetblock(256, 256, 1, (1,1), layers[-1], timestepembed))
    layers.append(skip256 := Conv2dLayer(256, 256, (3,3), 2, (0, 0), layers[-1]))
    attention_block(layers)
    layers.extend(resnetshortblock(256, 512, 1, (1,1), layers[-1], timestepembed))
    attention_block(layers)
    layers.extend(resnetblock(512, 512, 1, (1,1), layers[-1], timestepembed))
    layers.append(Conv2dLayer(512, 512, (3,3), 2, (0, 0), layers[-1]))
    layers.extend(resnetblock(512, 512, 1, (1,1), layers[-1], timestepembed))
    layers.extend(skip512 := resnetblock(512, 512, 1, (1,1), layers[-1], timestepembed))

    #Middle Segment
    layers.extend(resnetblock(512, 512, 1, (1,1), layers[-1], timestepembed))
    attention_block(layers)
    layers.extend(resnetblock(512, 512, 1, (1,1), layers[-1], timestepembed))

    #Upsampling
    layers.append(CropConcatLayer(layers[-1], skip512))
    layers.extend(resnetshortblock(1024, 512, 1, (1,1), layers[-1], timestepembed))
    layers.append(CropConcatLayer(layers[-1], skip512))
    layers.extend(resnetshortblock(1024, 512, 1, (1,1), layers[-1], timestepembed))
    layers.append(CropConcatLayer(layers[-1], skip512))
    layers.extend(resnetshortblock(1024, 512, 1, (1,1), layers[-1], timestepembed))
    layers.append(Conv2dLayer(512, 512, (3,3), 1, (1, 1), layers[-1]))
    attention_block(layers)
    layers.append(CropConcatLayer(layers[-1], skip512))
    layers.extend(resnetshortblock(1024, 512, 1, (1,1), layers[-1], timestepembed))
    attention_block(layers)
    layers.append(CropConcatLayer(layers[-1], skip512))
    layers.extend(resnetshortblock(1024, 512, 1, (1,1), layers[-1], timestepembed))
    attention_block(layers)
    layers.append(CropConcatLayer(layers[-1], skip256))
    layers.extend(resnetshortblock(768, 512, 1, (1,1), layers[-1], timestepembed))
    layers.append(Conv2dLayer(512, 512, (3,3), 1, (1, 1), layers[-1]))
    layers.append(CropConcatLayer(layers[-1], skip256))
    layers.extend(resnetshortblock(768, 256, 1, (1,1), layers[-1], timestepembed))
    layers.append(CropConcatLayer(layers[-1], skip256))
    layers.extend(resnetshortblock(512, 256, 1, (1,1), layers[-1], timestepembed))
    layers.append(CropConcatLayer(layers[-1], skip256))
    layers.extend(resnetshortblock(512, 256, 1, (1,1), layers[-1], timestepembed))
    layers.append(Conv2dLayer(256, 256, (3,3), 1, (1, 1), layers[-1]))
    layers.append(CropConcatLayer(layers[-1], skip256))
    layers.extend(resnetshortblock(512, 256, 1, (1,1), layers[-1], timestepembed))
    layers.append(CropConcatLayer(layers[-1], skip256))
    layers.extend(resnetshortblock(512, 256, 1, (1,1), layers[-1], timestepembed))
    layers.append(CropConcatLayer(layers[-1], skip128))
    layers.extend(resnetshortblock(384, 256, 1, (1,1), layers[-1], timestepembed))
    layers.append(Conv2dLayer(256, 256, (3,3), 1, (1, 1), layers[-1]))
    layers.append(CropConcatLayer(layers[-1], skip128))
    layers.extend(resnetshortblock(384, 128, 1, (1,1), layers[-1], timestepembed))
    layers.append(CropConcatLayer(layers[-1], skip128))
    layers.extend(resnetshortblock(256, 128, 1, (1,1), layers[-1], timestepembed))
    layers.append(CropConcatLayer(layers[-1], skip128))
    layers.extend(resnetshortblock(256, 128, 1, (1,1), layers[-1], timestepembed))
    layers.append(Conv2dLayer(128, 128, (3,3), 1, (1, 1), layers[-1]))
    layers.append(CropConcatLayer(layers[-1], skip128))
    layers.extend(resnetshortblock(256, 128, 1, (1,1), layers[-1], timestepembed))
    layers.append(CropConcatLayer(layers[-1], skip128))
    layers.extend(resnetshortblock(256, 128, 1, (1,1), layers[-1], timestepembed))
    layers.append(CropConcatLayer(layers[-1], skip128))
    layers.extend(resnetshortblock(256, 128, 1, (1,1), layers[-1], timestepembed))
    layers.append(BatchNorm2d(layers[-1]))
    layers.append(SigmoidLayer(layers[-1]))
    return

def attention_block(layers, SEQ_LEN=512, HIDDEN_DIM=768, I=64, HEADS=12):
    input_layer = layers[-1]
    layers.append(QueryKeyValueMatrix(SEQ_LEN, HIDDEN_DIM, I, HEADS, layers[-1]))  # Calculate Query
    layers.append(QKTMatrix(SEQ_LEN=SEQ_LEN, HIDDEN_DIM=I, I=SEQ_LEN, ATTN_HEADS=HEADS, input=layers[-1]))  # QK^T
    layers.append(QKTVMatrix(SEQ_LEN, SEQ_LEN, I, HEADS, layers[-1]))  # QK^TV
    layers.append(LinearLayer(I * HEADS, HIDDEN_DIM, layers[-1]))
    # Residual
    layers.append(SkipAddLayer(input_layer, layers[-1]))
    # FFNs
    layers.append(LinearLayerReLU(HIDDEN_DIM, HIDDEN_DIM * 4, layers[-1]))
    layers.append(LinearLayer(HIDDEN_DIM * 4, HIDDEN_DIM, layers[-1]))
    layers.append(SkipAddLayer(layers[-4], layers[-1]))
    return

#Resnet with shorting built in
def resnetshortblock(in_planes, planes, stride, padding, x, time_emb):
    kernel_size = (3, 3)
    bn1 = BatchNorm2d(x)
    conv1 = Conv2dLayer(in_planes, planes, kernel_size, stride, padding, bn1)
    skipconv = Conv2dLayer(in_planes, planes, (1, 1), stride, padding, x)
    linear = LinearLayer(in_planes, in_planes//2, time_emb)
    skip1 = SkipAddLayer(conv1, linear)
    bn2 = BatchNorm2d(skip1)
    dropout = DropoutLayer(bn2)
    conv2 = Conv2dLayer(planes, planes, kernel_size, 1, padding, dropout)
    skip2 = SkipAddLayer(skipconv, conv2)
    silu = SigmoidLayer(skip2)
    return [bn1, conv1, skipconv, linear, skip1, bn2, dropout, conv2, skip2, silu]


def resnetblock(in_planes, planes, stride, padding, x, time_emb):
    kernel_size = (3, 3)
    bn1 = BatchNorm2d(x)
    conv1 = Conv2dLayer(in_planes, planes, kernel_size, stride, padding, bn1)
    linear = LinearLayer(in_planes, in_planes//2, time_emb)
    skip1 = SkipAddLayer(conv1, linear)
    bn2 = BatchNorm2d(skip1)
    dropout = DropoutLayer(bn2)
    conv2 = Conv2dLayer(planes, planes, kernel_size, 1, padding, dropout)
    silu = SigmoidLayer(conv2)
    return [bn1, conv1, linear, skip1, bn2, dropout, conv2, silu]

def vaeresnet(in_planes, planes, stride, padding, x):
    kernel_size = (3, 3)
    bn1 = BatchNorm2d(x)
    conv1 = Conv2dLayer(in_planes, planes, kernel_size, stride, padding, bn1)
    bn2 = BatchNorm2d(conv1)
    dropout = DropoutLayer(bn2)
    conv2 = Conv2dLayer(planes, planes, kernel_size, 1, padding, dropout)
    silu = SigmoidLayer(conv2)
    return [bn1, conv1, bn2, dropout, conv2, silu]

def vaeresnetshort(in_planes, planes, stride, padding, x):
    kernel_size = (3, 3)
    bn1 = BatchNorm2d(x)
    conv1 = Conv2dLayer(in_planes, planes, kernel_size, stride, padding, bn1)
    skipconv = Conv2dLayer(in_planes, planes, (1, 1), stride, padding, x)
    bn2 = BatchNorm2d(conv1)
    dropout = DropoutLayer(bn2)
    conv2 = Conv2dLayer(planes, planes, kernel_size, 1, padding, dropout)
    skip2 = SkipAddLayer(skipconv, conv2)
    silu = SigmoidLayer(skip2)
    return [bn1, conv1, skipconv, bn2, dropout, conv2, skip2, silu]
'''
#Block for general the general UNet node operation of two 3x3 Conv2d interleaved with two ReLUs
def unet_block(in_planes, planes, x):
    kernel_size = (3, 3)
    conv1 = Conv2dLayer(in_planes, planes, kernel_size, 1, (1,1), x)
    relu1 = ReLULayer(conv1)
    conv2 = Conv2dLayer(planes, planes, kernel_size, 1, (1,1), relu1)
    relu2 = ReLULayer(conv2)
    return [conv1, relu1, conv2, relu2]


def unet_256(batch_size, input_shape=(3, 256, 256)):
    layers = [InputLayer((batch_size, *input_shape))]

    #Encoding layers all the way to the bottom, alternating between UNet blocks and MaxPools
    #Layer 1 1x572x572 -> 64x284x284
    layers.extend(block_64 := unet_block(1, 64, layers[-1]))
    layers.append(MaxPool2d((2,2), 2, layers[-1]))
    #Layer 2 64x284x284 -> 128x140x140
    layers.extend(block_128 := unet_block(64, 128, layers[-1]))
    layers.append(MaxPool2d((2,2), 2, layers[-1]))
    #Layer 3 128x140x140 -> 256x68x68
    layers.extend(block_256 := unet_block(128, 256, layers[-1]))
    layers.append(MaxPool2d((2,2), 2, layers[-1]))
    #Layer 4 256x68x68 -> 512x32x32
    layers.extend(block_512 := unet_block(256, 512, layers[-1]))
    layers.append(MaxPool2d((2,2), 2, layers[-1]))
    
    #Bottom Layer 512x32x32->1024x28x28
    layers.extend(unet_block(512, 1024, layers[-1]))

    #Encoding layers back to the top, cycling between upconv, concatenation, and UNet blocks
    #Layer 4 1024x28x28 -> 512x56x56 + 512x56x56 -> 512x52x52
    layers.append(Conv2dLayer(1024, 512, (2,2), 2, (1,1), layers[-1]))
    layers.append(CropConcatLayer(block_512, layers[-1]))
    layers.extend(unet_block(1024, 512, layers[-1]))
    #Layer 3 512x52x52 -> 256x104x104 + 256x104x104 -> 256x100x100
    layers.append(Conv2dLayer(512, 256, (2,2), 2, (1,1), layers[-1]))
    layers.append(CropConcatLayer(block_256, layers[-1]))
    layers.extend(unet_block(512, 256, layers[-1]))
    #Layer 2 256x100x100 -> 128x200x200 + 128x200x200 -> 128x196x196
    layers.append(Conv2dLayer(256, 128, (2,2), 2, (1,1), layers[-1]))
    layers.append(CropConcatLayer(block_128, layers[-1]))
    layers.extend(unet_block(256, 128, layers[-1]))
    #Layer 1 128x196x196 -> 64x392x392 + 64x392x392 -> 64x388x388
    layers.append(Conv2dLayer(128, 64, (2,2), 2, (1,1), layers[-1]))
    layers.append(CropConcatLayer(block_64, layers[-1]))
    layers.extend(unet_block(128, 64, layers[-1]))

    #Final 1x1 conv
    layers.append(Conv2dLayer(64, 2, (1,1), 1, (1,1), layers[-1]))

    return layers
'''
from keras.layers import Input, Dense, Flatten, BatchNormalization, Conv2D, MaxPooling2D, ZeroPadding2D
import pdb
from keras.models import Model

def conv_block(prev_layer, layers, filters):
    start = prev_layer
    for i in range(layers):
        zero_padding = ZeroPadding2D((1,1))(start)
        start = Conv2D(filters, 3, strides=(1,1),activation="relu")(zero_padding)
        #start = BatchNormalization(axis=1)(conv_layer)
    return MaxPooling2D(pool_size=(2,2), strides=(2,2) )(start)

def fc_block(prev_layer):
    return Dense(4096, activation='relu')(prev_layer)

def vgg16bn(output_classes):
    inps = Input(shape=(4, 256, 256))
    input_layer = BatchNormalization(axis=1)(inps)

    conv_layer_1   = conv_block(input_layer, 2, 64)
    conv_layer_2 = conv_block(conv_layer_1, 2, 128)
    conv_layer_3 = conv_block(conv_layer_2, 2, 256)
    conv_layer_4 = conv_block(conv_layer_3, 3, 512)
    conv_layer_5 = conv_block(conv_layer_4, 3, 512)
    pdb.set_trace()
    flattened = Flatten(input_shape=conv_layer_5.output_shape[1:])(conv_layer_5)
    prev_5 = fc_block(flattened)
    prev_6 = fc_block(prev_5)
    outs = [ Dense(1,activation='sigmoid', name=classification)(flattened) for classification in output_classes]
    return Model(inputs=inps, outputs=outs)

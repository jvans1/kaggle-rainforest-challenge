from keras.layers import Input, Dense, Flatten, BatchNormalization, Conv2D, MaxPooling2D
from keras.models import Model
classifications = [
    "agriculture",
    "artisinal_mine",
    "bare_ground",
    "blooming",
    "blow_down",
    "clear",
    "cloudy",
    "conventional_mine",
    "cultivation",
    "habitation",
    "haze",
    "partly_cloudy",
    "primary",
    "road",
    "selective_logging",
    "slash_burn",
    "water"
]
def conv_layer(prev, filters, kernel_size):
    conv_layer = Conv2D(filters, kernel_size, strides=(1,1), padding='same', data_format='channels_first', activation='relu', use_bias=True)(prev)
    bn_conv = BatchNormalization(axis=1)(conv_layer)
    max_pool   = MaxPooling2D(pool_size=(2,2), strides=None)(bn_conv)
    return max_pool

def dense_layer(prev, neurons):
    dense = Dense(200, activation='relu')(prev)
    return BatchNormalization()(dense)



def create_model():
    inps = Input(shape=(4, 256, 256))
    xinps = BatchNormalization(axis=1)(inps)

    conv_layer_1 = conv_layer(xinps, 32, (3,3))
    conv_layer_2 = conv_layer(conv_layer_1, 64, (3,3))
    flattened = Flatten()(conv_layer_2)

    dense_1 = dense_layer(flattened, 200)
    dense_2 = dense_layer(dense_1, 200)
    outs = [ Dense(1,activation='sigmoid', name=classification)(dense_2) for classification in classifications]
    model = Model(inputs=inps, outputs=outs)
    return  model


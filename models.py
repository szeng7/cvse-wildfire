import tensorflow as tf
from tensorflow.keras import layers, Model
from cnn_autoencoder_model import create_model


def MLP_CNN(input_shape, hidden_dim=64):
    inputs = layers.Input(shape=input_shape)

    #simple cnn
    x = layers.Conv2D(32, kernel_size=3, padding='same', activation='relu')(inputs)
    x = layers.MaxPooling2D(pool_size=2)(x)  #downsample
    x = layers.Conv2D(64, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.UpSampling2D(size=2)(x)  #upsample

    #project to per pixel MLP
    x = layers.Conv2D(hidden_dim, kernel_size=1, activation='relu')(x)
    x = layers.Conv2D(1, kernel_size=1, activation='sigmoid')(x)  #binary mask

    return Model(inputs, x)

def conv_block(x, filters, use_pool=True):
    x = layers.Conv2D(filters, kernel_size=3, strides=1, padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.1)(x)
    
    if use_pool:
        skip = x
        x = layers.MaxPooling2D(pool_size=2)(x)
        return x, skip
    else:
        return x, None

def res_block(x, filters):
    shortcut = x
    x = layers.Conv2D(filters, kernel_size=3, strides=1, padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Conv2D(filters, kernel_size=3, strides=1, padding='same')(x)

    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, kernel_size=1, strides=1, padding='same')(shortcut)

    x = layers.Add()([shortcut, x])
    x = layers.LeakyReLU()(x)
    return x

def CAE(input_shape):
    inputs = layers.Input(shape=input_shape)

    # Encoder
    x, skip1 = conv_block(inputs, 16)
    x, skip2 = conv_block(x, 32)

    # Bottleneck
    x = res_block(x, 32)
    x = res_block(x, 32)

    # Decoder
    x = layers.UpSampling2D()(x)
    x = res_block(x, 16)
    
    x = layers.UpSampling2D()(x)
    x = layers.Conv2D(16, kernel_size=3, strides=1, padding='same')(x)
    x = layers.LeakyReLU()(x)
    
    # Output
    outputs = layers.Conv2D(1, kernel_size=1, activation='sigmoid')(x)

    model = Model(inputs, outputs)
    return model

def NDWS_CAE(input_shape):
    inputs = tf.keras.Input(shape=input_shape)

    encoder_layers = [16, 32]  # based on the wildfire paper (2 levels)
    decoder_layers = [16, 16]  # decoder has same or fewer layers
    encoder_pools = [2, 2]     # maxpool of size 2x2 after each encoder block
    decoder_pools = [2, 2]     # upsample of size 2x2 after each decoder block

    outputs = create_model(
        input_tensor=inputs,
        num_out_channels=1,
        encoder_layers=encoder_layers,
        decoder_layers=decoder_layers,
        encoder_pools=encoder_pools,
        decoder_pools=decoder_pools,
        dropout=0.1,                 # they used 0.1 dropout
        batch_norm='none',            # you can try 'none' or 'all'
        l1_regularization=0.0,
        l2_regularization=0.0
    )

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model
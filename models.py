import tensorflow as tf
from tensorflow.keras import layers, Model

def CNN_MLP(input_shape, hidden_dim=64):
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
import tensorflow as tf
from tensorflow.keras import layers, models

def build_kws_model(input_shape, num_classes):
    """
    Builds a small Depthwise Separable CNN (DS-CNN) suitable for keyword spotting
    and exportable to TensorFlow Lite Micro.
    
    Architecture:
    Conv2D -> BatchNorm -> ReLU
    DepthwiseConv2D -> BatchNorm -> ReLU -> Conv2D -> BatchNorm -> ReLU (DS-Conv)
    DepthwiseConv2D -> BatchNorm -> ReLU -> Conv2D -> BatchNorm -> ReLU (DS-Conv)
    GlobalAveragePooling2D
    Dense -> Softmax
    """
    inputs = layers.Input(shape=input_shape)
    
    from tensorflow.keras import regularizers
    
    # 1. Standard Convolution
    x = layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same', use_bias=False, kernel_regularizer=regularizers.l2(1e-4))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.2)(x)
    
    # 2. Depthwise Separable Convolution 1
    x = layers.DepthwiseConv2D((3, 3), strides=(2, 2), padding='same', use_bias=False, depthwise_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(64, (1, 1), padding='same', use_bias=False, kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.2)(x)
    
    # 3. Depthwise Separable Convolution 2
    x = layers.DepthwiseConv2D((3, 3), padding='same', use_bias=False, depthwise_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(64, (1, 1), padding='same', use_bias=False, kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.2)(x)
    
    # 4. Pooling and Output
    # The shape is now (13, 5, 64). We can flatten instead of GAP to retain temporal info.
    x = layers.Flatten()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
    outputs = layers.Dense(num_classes, activation='softmax', kernel_regularizer=regularizers.l2(1e-4))(x)
    
    model = models.Model(inputs, outputs, name="kws_ds_cnn")
    return model

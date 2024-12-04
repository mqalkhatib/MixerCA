import tensorflow as tf
from tensorflow.keras import layers
import math
from keras import backend as K


from keras.layers import  Permute, Concatenate, Conv2D, Add, Activation, Lambda, multiply, ReLU
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Reshape, AveragePooling2D 
from keras.layers import concatenate, BatchNormalization

def no_Attention(X):
    '''
    No Attention is used
    '''
    return X

def ECA(x, gamma=2, b=1):
    """
    Efficient Channel Attention (ECA).
    Args:
        x: Input tensor with shape (batch_size, height, width, channels).
        gamma: Control parameter for kernel size calculation.
        b: Offset for kernel size calculation.
        
    Returns:
        output: Tensor with channel attention applied.
    """
    N, H, W, C = x.shape
    
    # Calculate kernel size based on the number of channels
    t = int(abs((math.log(C, 2) + b) / gamma))  # Logarithmic scaling with respect to channel size
    k = t if t % 2 == 1 else t + 1  # Ensure k is odd for symmetric padding
    
    # Global Average Pooling (across spatial dimensions H, W)
    avg_pool = tf.reduce_mean(x, axis=[1, 2], keepdims=True)  # Shape: (N, 1, 1, C)
    
    # Reshape for 1D convolution along the channel dimension (C)
    avg_pool_reshaped = tf.squeeze(avg_pool, axis=[1, 2])  # Shape: (N, C)
    
    # 1D Convolution on the channel dimension
    conv = layers.Conv1D(filters=1, kernel_size=k, padding='same', use_bias=False)
    y = conv(tf.expand_dims(avg_pool_reshaped, axis=-1))  # Shape after conv: (N, C, 1)
    
    # Apply a sigmoid activation to normalize the attention weights
    y = tf.nn.sigmoid(y)  # Attention values in the range [0, 1]
    
    # Reshape back to the original shape for multiplication with the input tensor
    y = tf.expand_dims(y, axis = 2 )
    y = tf.transpose(y, (0, 2, 3, 1))
    
    # Scale the input tensor by the channel-wise attention
    output = x * y  # Element-wise multiplication: (N, H, W, C)
    
    return output


def SE(xin, se_ratio=8):
    """
   Squeeze-and-Excitation (SE).
   Args:
       xin: Input tensor with shape (batch_size, height, width, channels).
       se_ratio: Reduction ratio for the squeeze operation (default is 8).
       
   Returns:
       output: Tensor with channel attention applied. The output shape is the same as the input shape.
   """
    # Global Average Pooling along spatial dimensions
    xin_gap = GlobalAveragePooling2D()(xin)
    
    # Squeeze Path
    sqz = Dense(xin.shape[-1] // se_ratio, activation='relu')(xin_gap)
    
    # Excitation Path
    excite = Dense(xin.shape[-1], activation='sigmoid')(sqz)
    
    # Multiply the input by the excitation weights
    out = multiply([xin, tf.keras.layers.Reshape((1, 1, xin.shape[-1]))(excite)])

    return out


def CBAM(input_feature, ratio=8):
    """
    CBAM (Convolutional Block Attention Module).
    Combines Channel and Spatial Attention mechanisms.
    
    Args:
        input_feature: Input tensor with shape (batch_size, height, width, channels).
        ratio: Reduction ratio for the channel attention block.
        
    Returns:
        output: Tensor with channel and spatial attention applied.
    """
    
    # Channel Attention
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    channel = input_feature.shape[channel_axis]  # Number of channels

    shared_layer_one = Dense(channel // ratio,
                             activation='relu',
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    shared_layer_two = Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')

    # Global Average Pooling
    avg_pool = GlobalAveragePooling2D()(input_feature)
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    avg_pool = shared_layer_one(avg_pool)
    avg_pool = shared_layer_two(avg_pool)

    # Global Max Pooling
    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1, 1, channel))(max_pool)
    max_pool = shared_layer_one(max_pool)
    max_pool = shared_layer_two(max_pool)

    # Combine average and max pooling results, then apply sigmoid activation
    channel_attention = Add()([avg_pool, max_pool])
    channel_attention = Activation('sigmoid')(channel_attention)

    # Apply channel attention to the input feature map
    cbam_feature = multiply([input_feature, channel_attention])

    # Spatial Attention
    kernel_size = 7
    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((2, 3, 1))(cbam_feature)

    # Average and Max Pooling across the channel dimension
    avg_pool = Lambda(lambda x: K.mean(x, axis=-1, keepdims=True))(cbam_feature)
    max_pool = Lambda(lambda x: K.max(x, axis=-1, keepdims=True))(cbam_feature)

    # Concatenate pooled features along the channel dimension
    concat = Concatenate(axis=-1)([avg_pool, max_pool])

    # Apply convolution to learn spatial attention
    spatial_attention = Conv2D(filters=1,
                               kernel_size=kernel_size,
                               strides=1,
                               padding='same',
                               activation='sigmoid',
                               kernel_initializer='he_normal',
                               use_bias=False)(concat)

    if K.image_data_format() == "channels_first":
        spatial_attention = Permute((3, 1, 2))(spatial_attention)

    # Apply spatial attention to the feature map
    cbam_feature = multiply([cbam_feature, spatial_attention])

    return cbam_feature

def CA(x, reduction=32, bn_trainable=False):
    """
    Coordinate Attention (CA).
    
    Args:
        x (Tensor): Input tensor with shape (batch_size, height, width, channels).
        reduction (int): Reduction ratio for computing the bottleneck channels.
        bn_trainable (bool): Whether the batch normalization layers are trainable.
        
    Returns:
        Tensor: Output tensor with applied coordinate attention.
    """
    def coord_act(x):
        tmpx = (ReLU(max_value=6)(x + 3)) / 6
        x = x * tmpx
        return x

    x_shape = x.shape.as_list()
    [b, h, w, c] = x_shape
    x_h = AveragePooling2D(pool_size=(1, w), strides=(1, 1), data_format='channels_last')(x)
    x_w = AveragePooling2D(pool_size=(h, 1), strides=(1, 1), data_format='channels_last')(x)
    x_w = K.permute_dimensions(x_w, [0, 2, 1, 3])
    y = concatenate(inputs=[x_h, x_w], axis=1)
    mip = max(8, c // reduction)
    y = Conv2D(filters=mip, kernel_size=(1, 1), strides=(1, 1), padding='valid')(y)
    y = BatchNormalization(trainable=bn_trainable)(y)
    y = coord_act(y)
    x_h, x_w = Lambda(tf.split, arguments={'axis': 1, 'num_or_size_splits': [h, w]})(y)
    x_w = K.permute_dimensions(x_w, [0, 2, 1, 3])
    a_h = Conv2D(filters=c, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation="sigmoid")(x_h)
    a_w = Conv2D(filters=c, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation="sigmoid")(x_w)
    out = x * a_h * a_w
    return out


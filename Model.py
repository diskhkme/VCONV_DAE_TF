from keras.models import Model
from keras.layers import Input, Conv3D, Conv3DTranspose, Reshape, Activation, Dropout, Dense

def build_model_30(input_voxel_size=[30,30,30],
                augmenting_dropout_rate=.5,
                conv_num_filters=[64, 256, 256, 64, 1],
                conv_filter_sizes=[9,4, 5, 6],
                conv_strides=[3,2, 2, 3],
                desc_dims=[6912,6912],
                mode='train'):
    # Input --------------------------------------------------------------------
    x = Input(shape=(input_voxel_size[0], input_voxel_size[1], input_voxel_size[2], 1))

    # Augmenting Dropout -------------------------------------------------------
    drop1 = Dropout(augmenting_dropout_rate)(x)

    # Encoding Layers ----------------------------------------------------------
    conv1 = Conv3D(conv_num_filters[0],
                   kernel_size=(conv_filter_sizes[0],conv_filter_sizes[0],conv_filter_sizes[0]),
                   strides=(conv_strides[0],conv_strides[0],conv_strides[0]),
                   data_format='channels_last',
                   name='conv1')(drop1)
    conv1 = Activation('relu')(conv1)
    conv2 = Conv3D(conv_num_filters[1],
                   kernel_size=(conv_filter_sizes[1], conv_filter_sizes[1], conv_filter_sizes[1]),
                   strides=(conv_strides[1], conv_strides[1], conv_strides[1]),
                   data_format='channels_last',
                   name='conv2')(conv1)
    conv2 = Activation('relu')(conv2)

    reshape1 = Reshape((desc_dims[0],), name='reshape1')(conv2)
    dense1 = Dense(desc_dims[1])(reshape1)

    # Bottleneck(?) Layer -------------------------------------------------------
    bottleneck = Activation('relu', name='bottleneck')(dense1)
    # ---------------------------------------------------------------------------

    drop2 = Dropout(augmenting_dropout_rate)(bottleneck)
    reshape2 = Reshape((3,3,3,conv_num_filters[2]), name='reshape2')(drop2)

    # Decoding Layers ----------------------------------------------------------
    deconv1 = Conv3DTranspose(conv_num_filters[3],
                              kernel_size=(conv_filter_sizes[2],conv_filter_sizes[2],conv_filter_sizes[2]),
                              strides=(conv_strides[2],conv_strides[2],conv_strides[2]),
                              data_format='channels_last',
                              name='deconv1')(reshape2)
    deconv1 = Activation('relu')(deconv1)
    deconv2 = Conv3DTranspose(conv_num_filters[4],
                              kernel_size=(conv_filter_sizes[3],conv_filter_sizes[3],conv_filter_sizes[3]),
                              strides=(conv_strides[3],conv_strides[3],conv_strides[3]),
                              data_format='channels_last',
                              name='deconv2')(deconv1)
    reshape3 = Reshape((input_voxel_size[0], input_voxel_size[1], input_voxel_size[2], 1))(deconv2)

    # Output -------------------------------------------------------------------
    out = Activation('sigmoid', name='output')(reshape3)

    model = Model(x, out)

    return model


def build_model_128(input_voxel_size=[128,128,128],
                augmenting_dropout_rate=.5,
                conv_num_filters=[64, 256, 256, 256, 256, 64, 1],
                conv_filter_sizes=[9,4,4, 5,5, 6],
                conv_strides=[3,2,2, 2,2, 3],
                desc_dims=[131072,131072],
                mode='train'):
    # Input --------------------------------------------------------------------
    x = Input(shape=(input_voxel_size[0], input_voxel_size[1], input_voxel_size[2], 1))

    # Augmenting Dropout -------------------------------------------------------
    drop1 = Dropout(augmenting_dropout_rate)(x)

    # Encoding Layers ----------------------------------------------------------
    conv1 = Conv3D(conv_num_filters[0],
                   kernel_size=(conv_filter_sizes[0],conv_filter_sizes[0],conv_filter_sizes[0]),
                   strides=(conv_strides[0],conv_strides[0],conv_strides[0]),
                   data_format='channels_last',
                   name='conv1')(drop1)
    conv1 = Activation('relu')(conv1)
    conv2 = Conv3D(conv_num_filters[1],
                   kernel_size=(conv_filter_sizes[1], conv_filter_sizes[1], conv_filter_sizes[1]),
                   strides=(conv_strides[1], conv_strides[1], conv_strides[1]),
                   data_format='channels_last',
                   name='conv2')(conv1)
    conv2 = Activation('relu')(conv2)
    conv3 = Conv3D(conv_num_filters[2],
                   kernel_size=(conv_filter_sizes[2], conv_filter_sizes[2], conv_filter_sizes[2]),
                   strides=(conv_strides[2], conv_strides[2], conv_strides[2]),
                   data_format='channels_last',
                   name='conv3')(conv2)
    conv3 = Activation('relu')(conv3)

    reshape1 = Reshape((desc_dims[0],), name='reshape1')(conv3)
    dense1 = Dense(desc_dims[1])(reshape1)

    # Bottleneck(?) Layer -------------------------------------------------------
    bottleneck = Activation('relu', name='bottleneck')(dense1)
    # ---------------------------------------------------------------------------

    drop2 = Dropout(augmenting_dropout_rate)(bottleneck)
    reshape2 = Reshape((conv3.shape[1],conv3.shape[2],conv3.shape[3],conv_num_filters[3]), name='reshape2')(drop2)

    # Decoding Layers ----------------------------------------------------------
    deconv1 = Conv3DTranspose(conv_num_filters[4],
                              kernel_size=(conv_filter_sizes[3],conv_filter_sizes[3],conv_filter_sizes[3]),
                              strides=(conv_strides[3],conv_strides[3],conv_strides[3]),
                              data_format='channels_last',
                              name='deconv1')(reshape2)
    deconv1 = Activation('relu')(deconv1)
    deconv2 = Conv3DTranspose(conv_num_filters[5],
                              kernel_size=(conv_filter_sizes[4],conv_filter_sizes[4],conv_filter_sizes[4]),
                              strides=(conv_strides[4],conv_strides[4],conv_strides[4]),
                              data_format='channels_last',
                              name='deconv2')(deconv1)
    deconv2 = Activation('relu')(deconv2)
    deconv3 = Conv3DTranspose(conv_num_filters[6],
                              kernel_size=(conv_filter_sizes[5], conv_filter_sizes[5], conv_filter_sizes[5]),
                              strides=(conv_strides[5], conv_strides[5], conv_strides[5]),
                              data_format='channels_last',
                              name='deconv3')(deconv2)

    reshape3 = Reshape((input_voxel_size[0], input_voxel_size[1], input_voxel_size[2], 1))(deconv3)

    # Output -------------------------------------------------------------------
    out = Activation('sigmoid', name='output')(reshape3)

    model = Model(x, out)

    return model

def build_model_64(input_voxel_size=[64,64,64],
                augmenting_dropout_rate=.5,
                conv_num_filters=[64, 256, 256, 256, 256, 64, 1],
                conv_filter_sizes=[9,4,4, 5,5, 6],
                conv_strides=[3,2,2, 2,2, 3],
                desc_dims=[131072,131072],
                mode='train'):
    # Input --------------------------------------------------------------------
    x = Input(shape=(input_voxel_size[0], input_voxel_size[1], input_voxel_size[2], 1))

    # Augmenting Dropout -------------------------------------------------------
    drop1 = Dropout(augmenting_dropout_rate)(x)

    # Encoding Layers ----------------------------------------------------------
    conv1 = Conv3D(conv_num_filters[0],
                   kernel_size=(conv_filter_sizes[0],conv_filter_sizes[0],conv_filter_sizes[0]),
                   strides=(conv_strides[0],conv_strides[0],conv_strides[0]),
                   data_format='channels_last',
                   name='conv1',
                   padding='same')(drop1)
    conv1 = Activation('relu')(conv1)
    conv2 = Conv3D(conv_num_filters[1],
                   kernel_size=(conv_filter_sizes[1], conv_filter_sizes[1], conv_filter_sizes[1]),
                   strides=(conv_strides[1], conv_strides[1], conv_strides[1]),
                   data_format='channels_last',
                   name='conv2')(conv1)
    conv2 = Activation('relu')(conv2)
    conv3 = Conv3D(conv_num_filters[2],
                   kernel_size=(conv_filter_sizes[2], conv_filter_sizes[2], conv_filter_sizes[2]),
                   strides=(conv_strides[2], conv_strides[2], conv_strides[2]),
                   data_format='channels_last',
                   name='conv3')(conv2)
    conv3 = Activation('relu')(conv3)

    reshape1 = Reshape((desc_dims[0],), name='reshape1')(conv3)
    dense1 = Dense(desc_dims[1])(reshape1)

    # Bottleneck(?) Layer -------------------------------------------------------
    bottleneck = Activation('relu', name='bottleneck')(dense1)
    # ---------------------------------------------------------------------------

    drop2 = Dropout(augmenting_dropout_rate)(bottleneck)
    reshape2 = Reshape((conv3.shape[1],conv3.shape[2],conv3.shape[3],conv_num_filters[3]), name='reshape2')(drop2)

    # Decoding Layers ----------------------------------------------------------
    deconv1 = Conv3DTranspose(conv_num_filters[4],
                              kernel_size=(conv_filter_sizes[3],conv_filter_sizes[3],conv_filter_sizes[3]),
                              strides=(conv_strides[3],conv_strides[3],conv_strides[3]),
                              data_format='channels_last',
                              name='deconv1')(reshape2)
    deconv1 = Activation('relu')(deconv1)
    deconv2 = Conv3DTranspose(conv_num_filters[5],
                              kernel_size=(conv_filter_sizes[4],conv_filter_sizes[4],conv_filter_sizes[4]),
                              strides=(conv_strides[4],conv_strides[4],conv_strides[4]),
                              data_format='channels_last',
                              name='deconv2')(deconv1)
    deconv2 = Activation('relu')(deconv2)
    deconv3 = Conv3DTranspose(conv_num_filters[6],
                              kernel_size=(conv_filter_sizes[5], conv_filter_sizes[5], conv_filter_sizes[5]),
                              strides=(conv_strides[5], conv_strides[5], conv_strides[5]),
                              data_format='channels_last',
                              name='deconv3')(deconv2)

    reshape3 = Reshape((input_voxel_size[0], input_voxel_size[1], input_voxel_size[2], 1))(deconv3)

    # Output -------------------------------------------------------------------
    out = Activation('sigmoid', name='output')(reshape3)

    model = Model(x, out)

    return model
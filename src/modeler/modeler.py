from keras import Input
from keras import layers
from keras.layers import Activation, Dense
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, BatchNormalization, AveragePooling2D
from keras.models import Model as ModelKeras


class Modeler(object):
    def __init__(self, *args, **kwargs):
        pass

    @staticmethod
    def _conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(1, 1), name=None):
        x = Conv2D(filters, (num_row, num_col), strides=strides, padding=padding, use_bias=False)(x)
        x = BatchNormalization(axis=3, scale=False)(x)
        x = Activation('relu', name=name)(x)
        return x

    def get_model(self, input_shape: list, *args, **kwargs):
        pass


class InceptionV3Modeler(Modeler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_model(self,
                  input_shape: list,
                  include_top: bool=True,
                  pooling: str='max',
                  classes: int=2):
        img_input = Input(shape=input_shape)

        x = self._conv2d_bn(img_input, 32, 3, 3, strides=(2, 2), padding='valid')
        x = self._conv2d_bn(x, 32, 3, 3, padding='valid')
        x = self._conv2d_bn(x, 64, 3, 3)
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)

        x = self._conv2d_bn(x, 80, 1, 1, padding='valid')
        x = self._conv2d_bn(x, 192, 3, 3, padding='valid')
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)
        # mixed 0, 1, 2: 35 x 35 x 256

        branch1x1 = self._conv2d_bn(x, 64, 1, 1)

        branch5x5 = self._conv2d_bn(x, 48, 1, 1)
        branch5x5 = self._conv2d_bn(branch5x5, 64, 5, 5)

        branch3x3dbl = self._conv2d_bn(x, 64, 1, 1)
        branch3x3dbl = self._conv2d_bn(branch3x3dbl, 96, 3, 3)
        branch3x3dbl = self._conv2d_bn(branch3x3dbl, 96, 3, 3)

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = self._conv2d_bn(branch_pool, 32, 1, 1)
        x = layers.concatenate(
            [branch1x1, branch5x5, branch3x3dbl, branch_pool],
            axis=3,
            name='mixed0')

        # mixed 1: 35 x 35 x 256
        branch1x1 = self._conv2d_bn(x, 64, 1, 1)

        branch5x5 = self._conv2d_bn(x, 48, 1, 1)
        branch5x5 = self._conv2d_bn(branch5x5, 64, 5, 5)

        branch3x3dbl = self._conv2d_bn(x, 64, 1, 1)
        branch3x3dbl = self._conv2d_bn(branch3x3dbl, 96, 3, 3)
        branch3x3dbl = self._conv2d_bn(branch3x3dbl, 96, 3, 3)

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = self._conv2d_bn(branch_pool, 64, 1, 1)
        x = layers.concatenate(
            [branch1x1, branch5x5, branch3x3dbl, branch_pool],
            axis=3,
            name='mixed1')

        # mixed 2: 35 x 35 x 256
        branch1x1 = self._conv2d_bn(x, 64, 1, 1)

        branch5x5 = self._conv2d_bn(x, 48, 1, 1)
        branch5x5 = self._conv2d_bn(branch5x5, 64, 5, 5)

        branch3x3dbl = self._conv2d_bn(x, 64, 1, 1)
        branch3x3dbl = self._conv2d_bn(branch3x3dbl, 96, 3, 3)
        branch3x3dbl = self._conv2d_bn(branch3x3dbl, 96, 3, 3)

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = self._conv2d_bn(branch_pool, 64, 1, 1)
        x = layers.concatenate(
            [branch1x1, branch5x5, branch3x3dbl, branch_pool],
            axis=3,
            name='mixed2')

        # mixed 3: 17 x 17 x 768
        branch3x3 = self._conv2d_bn(x, 384, 3, 3, strides=(2, 2), padding='valid')

        branch3x3dbl = self._conv2d_bn(x, 64, 1, 1)
        branch3x3dbl = self._conv2d_bn(branch3x3dbl, 96, 3, 3)
        branch3x3dbl = self._conv2d_bn(
            branch3x3dbl, 96, 3, 3, strides=(2, 2), padding='valid')

        branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
        x = layers.concatenate(
            [branch3x3, branch3x3dbl, branch_pool], axis=3, name='mixed3')

        # mixed 4: 17 x 17 x 768
        branch1x1 = self._conv2d_bn(x, 192, 1, 1)

        branch7x7 = self._conv2d_bn(x, 128, 1, 1)
        branch7x7 = self._conv2d_bn(branch7x7, 128, 1, 7)
        branch7x7 = self._conv2d_bn(branch7x7, 192, 7, 1)

        branch7x7dbl = self._conv2d_bn(x, 128, 1, 1)
        branch7x7dbl = self._conv2d_bn(branch7x7dbl, 128, 7, 1)
        branch7x7dbl = self._conv2d_bn(branch7x7dbl, 128, 1, 7)
        branch7x7dbl = self._conv2d_bn(branch7x7dbl, 128, 7, 1)
        branch7x7dbl = self._conv2d_bn(branch7x7dbl, 192, 1, 7)

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = self._conv2d_bn(branch_pool, 192, 1, 1)
        x = layers.concatenate(
            [branch1x1, branch7x7, branch7x7dbl, branch_pool],
            axis=3,
            name='mixed4')

        # mixed 5, 6: 17 x 17 x 768
        for i in range(2):
            branch1x1 = self._conv2d_bn(x, 192, 1, 1)

            branch7x7 = self._conv2d_bn(x, 160, 1, 1)
            branch7x7 = self._conv2d_bn(branch7x7, 160, 1, 7)
            branch7x7 = self._conv2d_bn(branch7x7, 192, 7, 1)

            branch7x7dbl = self._conv2d_bn(x, 160, 1, 1)
            branch7x7dbl = self._conv2d_bn(branch7x7dbl, 160, 7, 1)
            branch7x7dbl = self._conv2d_bn(branch7x7dbl, 160, 1, 7)
            branch7x7dbl = self._conv2d_bn(branch7x7dbl, 160, 7, 1)
            branch7x7dbl = self._conv2d_bn(branch7x7dbl, 192, 1, 7)

            branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
            branch_pool = self._conv2d_bn(branch_pool, 192, 1, 1)
            x = layers.concatenate(
                [branch1x1, branch7x7, branch7x7dbl, branch_pool],
                axis=3,
                name='mixed' + str(5 + i))

        # mixed 7: 17 x 17 x 768
        branch1x1 = self._conv2d_bn(x, 192, 1, 1)

        branch7x7 = self._conv2d_bn(x, 192, 1, 1)
        branch7x7 = self._conv2d_bn(branch7x7, 192, 1, 7)
        branch7x7 = self._conv2d_bn(branch7x7, 192, 7, 1)

        branch7x7dbl = self._conv2d_bn(x, 192, 1, 1)
        branch7x7dbl = self._conv2d_bn(branch7x7dbl, 192, 7, 1)
        branch7x7dbl = self._conv2d_bn(branch7x7dbl, 192, 1, 7)
        branch7x7dbl = self._conv2d_bn(branch7x7dbl, 192, 7, 1)
        branch7x7dbl = self._conv2d_bn(branch7x7dbl, 192, 1, 7)

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = self._conv2d_bn(branch_pool, 192, 1, 1)
        x = layers.concatenate(
            [branch1x1, branch7x7, branch7x7dbl, branch_pool],
            axis=3,
            name='mixed7')

        # mixed 8: 8 x 8 x 1280
        branch3x3 = self._conv2d_bn(x, 192, 1, 1)
        branch3x3 = self._conv2d_bn(branch3x3, 320, 3, 3, strides=(2, 2), padding='valid')

        branch7x7x3 = self._conv2d_bn(x, 192, 1, 1)
        branch7x7x3 = self._conv2d_bn(branch7x7x3, 192, 1, 7)
        branch7x7x3 = self._conv2d_bn(branch7x7x3, 192, 7, 1)
        branch7x7x3 = self._conv2d_bn(
            branch7x7x3, 192, 3, 3, strides=(2, 2), padding='valid')

        branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
        x = layers.concatenate(
            [branch3x3, branch7x7x3, branch_pool], axis=3, name='mixed8')

        # mixed 9: 8 x 8 x 2048
        for i in range(2):
            branch1x1 = self._conv2d_bn(x, 320, 1, 1)

            branch3x3 = self._conv2d_bn(x, 384, 1, 1)
            branch3x3_1 = self._conv2d_bn(branch3x3, 384, 1, 3)
            branch3x3_2 = self._conv2d_bn(branch3x3, 384, 3, 1)
            branch3x3 = layers.concatenate(
                [branch3x3_1, branch3x3_2], axis=3, name='mixed9_' + str(i))

            branch3x3dbl = self._conv2d_bn(x, 448, 1, 1)
            branch3x3dbl = self._conv2d_bn(branch3x3dbl, 384, 3, 3)
            branch3x3dbl_1 = self._conv2d_bn(branch3x3dbl, 384, 1, 3)
            branch3x3dbl_2 = self._conv2d_bn(branch3x3dbl, 384, 3, 1)
            branch3x3dbl = layers.concatenate(
                [branch3x3dbl_1, branch3x3dbl_2], axis=3)

            branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
            branch_pool = self._conv2d_bn(branch_pool, 192, 1, 1)
            x = layers.concatenate(
                [branch1x1, branch3x3, branch3x3dbl, branch_pool],
                axis=3,
                name='mixed' + str(9 + i))
        if include_top:
            x = GlobalAveragePooling2D(name='avg_pool')(x)
            x = Dense(classes, activation='sigmoid', name='predictions')(x)
        else:
            if pooling == 'avg':
                x = GlobalAveragePooling2D()(x)
            elif pooling == 'max':
                x = GlobalMaxPooling2D()(x)

        model = ModelKeras(img_input, x, name='inception_v3')
        model.compile(loss='binary_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])

        return model

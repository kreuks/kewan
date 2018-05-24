from keras import Input
from keras import layers
from keras.layers import Activation, Dense, Dropout, Flatten
from keras.layers import (Conv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, BatchNormalization,
                          AveragePooling2D)
from keras.models import Model as ModelKeras


class Modeler(object):
    def __init__(self, *args, **kwargs):
        pass

    @staticmethod
    def _conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(1, 1), name=None, axis=3):
        x = Conv2D(filters, (num_row, num_col), strides=strides, padding=padding, use_bias=False)(x)
        x = BatchNormalization(axis=axis, scale=False)(x)
        x = Activation('relu', name=name)(x)
        return x

    def get_model(self,
                  input_shape: list,
                  *args,
                  **kwargs):
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


class InceptionV4Modeler(Modeler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __inception_stem(self, input_):
        x = self._conv2d_bn(input_, 32, 3, 3, padding='valid', strides=(2, 2), axis=-1)
        x = self._conv2d_bn(x, 32, 3, 3, padding='valid', axis=-1)
        x = self._conv2d_bn(x, 64, 3, 3, axis=-1)

        x1 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid', axis=-1)(x)
        x2 = self._conv2d_bn(x, 96, 3, 3, strides=(2, 2), padding='valid', axis=-1)

        x = layers.concatenate([x1, x2], axis=-1)

        x1 = self._conv2d_bn(x, 64, 1, 1, axis=-1)
        x1 = self._conv2d_bn(x1, 96, 3, 3, padding='valid', axis=-1)

        x2 = self._conv2d_bn(x, 64, 1, 1, axis=-1)
        x2 = self._conv2d_bn(x2, 64, 1, 7, axis=-1)
        x2 = self._conv2d_bn(x2, 64, 7, 1, axis=-1)
        x2 = self._conv2d_bn(x2, 96, 3, 3, padding='valid', axis=-1)

        x = layers.concatenate([x1, x2], axis=-1)

        x1 = self._conv2d_bn(x, 192, 3, 3, strides=(2, 2), padding='valid', axis=-1)
        x2 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(x)

        x = layers.concatenate([x1, x2], axis=-1)
        return x

    def __inception_A(self, input_):
        a1 = self._conv2d_bn(input_, 96, 1, 1, axis=-1)

        a2 = self._conv2d_bn(input_, 64, 1, 1, axis=-1)
        a2 = self._conv2d_bn(a2, 96, 3, 3, axis=-1)

        a3 = self._conv2d_bn(input_, 64, 1, 1, axis=-1)
        a3 = self._conv2d_bn(a3, 96, 3, 3, axis=-1)
        a3 = self._conv2d_bn(a3, 96, 3, 3, axis=-1)

        a4 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(input_)
        a4 = self._conv2d_bn(a4, 96, 1, 1, axis=-1)

        m = layers.concatenate([a1, a2, a3, a4], axis=-1)
        return m

    def __inception_B(self, input_):
        b1 = self._conv2d_bn(input_, 384, 1, 1, axis=-1)

        b2 = self._conv2d_bn(input_, 192, 1, 1, axis=-1)
        b2 = self._conv2d_bn(b2, 224, 1, 7, axis=-1)
        b2 = self._conv2d_bn(b2, 256, 7, 1, axis=-1)

        b3 = self._conv2d_bn(input_, 192, 1, 1, axis=-1)
        b3 = self._conv2d_bn(b3, 192, 7, 1, axis=-1)
        b3 = self._conv2d_bn(b3, 224, 1, 7, axis=-1)
        b3 = self._conv2d_bn(b3, 224, 7, 1, axis=-1)
        b3 = self._conv2d_bn(b3, 256, 1, 7, axis=-1)

        b4 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(input_)
        b4 = self._conv2d_bn(b4, 128, 1, 1, axis=-1)

        m = layers.concatenate([b1, b2, b3, b4], axis=-1)
        return m

    def __inception_C(self, input_):
        c1 = self._conv2d_bn(input_, 256, 1, 1, axis=-1)

        c2 = self._conv2d_bn(input_, 384, 1, 1, axis=-1)
        c2_1 = self._conv2d_bn(c2, 256, 1, 3, axis=-1)
        c2_2 = self._conv2d_bn(c2, 256, 3, 1, axis=-1)
        c2 = layers.concatenate([c2_1, c2_2], axis=-1)

        c3 = self._conv2d_bn(input_, 384, 1, 1, axis=-1)
        c3 = self._conv2d_bn(c3, 448, 3, 1, axis=-1)
        c3 = self._conv2d_bn(c3, 512, 1, 3, axis=-1)
        c3_1 = self._conv2d_bn(c3, 256, 1, 3, axis=-1)
        c3_2 = self._conv2d_bn(c3, 256, 3, 1, axis=-1)
        c3 = layers.concatenate([c3_1, c3_2], axis=-1)

        c4 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(input_)
        c4 = self._conv2d_bn(c4, 256, 1, 1, axis=-1)

        m = layers.concatenate([c1, c2, c3, c4], axis=-1)
        return m

    def __reduction_A(self, input_):
        r1 = self._conv2d_bn(input_, 384, 3, 3, strides=(2, 2), padding='valid', axis=-1)

        r2 = self._conv2d_bn(input_, 192, 1, 1, axis=-1)
        r2 = self._conv2d_bn(r2, 224, 3, 3, axis=-1)
        r2 = self._conv2d_bn(r2, 256, 3, 3, strides=(2, 2), padding='valid', axis=-1)

        r3 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(input_)

        m = layers.concatenate([r1, r2, r3], axis=-1)
        return m

    def __reduction_B(self, input_):
        r1 = self._conv2d_bn(input_, 192, 1, 1, axis=-1)
        r1 = self._conv2d_bn(r1, 192, 3, 3, strides=(2, 2), padding='valid', axis=-1)

        r2 = self._conv2d_bn(input_, 256, 1, 1, axis=-1)
        r2 = self._conv2d_bn(r2, 256, 1, 7, axis=-1)
        r2 = self._conv2d_bn(r2, 320, 7, 1, axis=-1)
        r2 = self._conv2d_bn(r2, 320, 3, 3, strides=(2, 2), padding='valid', axis=-1)

        r3 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(input_)

        m = layers.concatenate([r1, r2, r3], axis=-1)
        return m

    def get_model(self,
                  input_shape: list,
                  pooling: str = 'avg',
                  classes: int = 2):
        img_input = Input(shape=input_shape)

        x = self.__inception_stem(img_input)

        for _ in range(4):
            x = self.__inception_A(x)

        x = self.__reduction_A(x)

        for _ in range(7):
            x = self.__inception_B(x)

        x = self.__reduction_B(x)

        for _ in range(3):
            x = self.__inception_C(x)

        if pooling == 'avg':
            x = AveragePooling2D((8, 8))(x)
        elif pooling == 'max':
            x = MaxPooling2D((8, 8))(x)

        x = Dropout(0.8)(x)
        x = Flatten()(x)

        x = Dense(classes, activation='sigmoid', name='prediction')(x)

        model = ModelKeras(img_input, x, name='inception_v4')
        model.compile(loss='binary_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])
        return model

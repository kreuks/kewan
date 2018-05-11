import ast
import os

import keras.backend as K
import numpy as np
import pandas as pd
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator, Iterator


def _get_train_or_validation_samples(split, num_files):
    """
    Get train or validation samples.

    :param split: tuple
        Tuple of fraction.
    :param num_files: int
    :return: int
    """
    if split:
        start, stop = int(split[0] * num_files), int(split[1] * num_files)
    else:
        start, stop = 0, num_files
    return stop - start


class KewanImageDataGenerator(ImageDataGenerator):
    """Generate minibatches of image data with real-time data augmentation.

    Arguments:
        featurewise_center: set input mean to 0 over the dataset.
        samplewise_center: set each sample mean to 0.
        featurewise_std_normalization: divide inputs by std of the dataset.
        samplewise_std_normalization: divide each input by its std.
        zca_whitening: apply ZCA whitening.
        zca_epsilon: epsilon for ZCA whitening. Default is 1e-6.
        rotation_range: degrees (0 to 180).
        width_shift_range: fraction of total width, if < 1, or pixels if >= 1.
        height_shift_range: fraction of total height, if < 1, or pixels if >= 1.
        brightness_range: the range of brightness to apply
        shear_range: shear intensity (shear angle in degrees).
        zoom_range: amount of zoom. if scalar z, zoom will be randomly picked
            in the range [1-z, 1+z]. A sequence of two can be passed instead
            to select this range.
        channel_shift_range: shift range for each channel.
        fill_mode: points outside the boundaries are filled according to the
            given mode ('constant', 'nearest', 'reflect' or 'wrap'). Default
            is 'nearest'.
            Points outside the boundaries of the input are filled according to the
              given mode:
                'constant': kkkkkkkk|abcd|kkkkkkkk (cval=k)
                'nearest':  aaaaaaaa|abcd|dddddddd
                'reflect':  abcddcba|abcd|dcbaabcd
                'wrap':  abcdabcd|abcd|abcdabcd
        cval: value used for points outside the boundaries when fill_mode is
            'constant'. Default is 0.
        horizontal_flip: whether to randomly flip images horizontally.
        vertical_flip: whether to randomly flip images vertically.
        rescale: rescaling factor. If None or 0, no rescaling is applied,
            otherwise we multiply the data by the value provided. This is
            applied after the `preprocessing_function` (if any provided)
            but before any other transformation.
        preprocessing_function: function that will be implied on each input.
            The function will run before any other modification on it.
            The function should take one argument:
            one image (Numpy tensor with rank 3),
            and should output a Numpy tensor with the same shape.
        data_format: 'channels_first' or 'channels_last'. In 'channels_first'
          mode, the channels dimension
            (the depth) is at index 1, in 'channels_last' mode it is at index 3.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
        validation_split: fraction of images reserved for validation (strictly
          between 0 and 1).
    """

    def __init__(self,
                 featurewise_center=False,
                 samplewise_center=False,
                 featurewise_std_normalization=False,
                 samplewise_std_normalization=False,
                 zca_whitening=False,
                 zca_epsilon=1e-6,
                 rotation_range=0.5,
                 width_shift_range=0.,
                 height_shift_range=0.,
                 brightness_range=None,
                 shear_range=0.,
                 zoom_range=0.8,
                 channel_shift_range=0.,
                 fill_mode='nearest',
                 cval=0.,
                 horizontal_flip=False,
                 vertical_flip=False,
                 rescale=1/255,
                 preprocessing_function=None,
                 data_format=None,
                 validation_split=0.1):
        super().__init__(featurewise_center,
                         samplewise_center,
                         featurewise_std_normalization,
                         samplewise_std_normalization,
                         zca_whitening,
                         zca_epsilon,
                         rotation_range,
                         width_shift_range,
                         height_shift_range,
                         brightness_range,
                         shear_range,
                         zoom_range,
                         channel_shift_range,
                         fill_mode,
                         cval,
                         horizontal_flip,
                         vertical_flip,
                         rescale,
                         preprocessing_function,
                         data_format,
                         validation_split)

    def flow_from_label(self,
                        label_data_path,
                        directory,
                        target_size=(256, 256),
                        color_mode='rgb',
                        batch_size=32,
                        shuffle=True,
                        seed=None,
                        subset=None,
                        interpolation='nearest'):
        return ImageIterator(label_data_path,
                             directory,
                             self,
                             target_size=target_size,
                             color_mode=color_mode,
                             data_format=self.data_format,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             seed=seed,
                             subset=subset,
                             interpolation=interpolation)


class ImageIterator(Iterator):
    """Iterator capable of reading images from a directory on disk and get label from csv file.

    Arguments:
        label_data_path: Path to csv label.
        directory: Path to the directory to read images from.
        image_data_generator: Instance of `ImageDataGenerator`
            to use for random transformations and normalization.
        target_size: tuple of integers, dimensions to resize input images to.
        color_mode: One of `"rgb"`, `"grayscale"`. Color mode to read images.
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seed for data shuffling.
        data_format: String, one of `channels_first`, `channels_last`.
        subset: Subset of data (`"training"` or `"validation"`) if
            validation_split is set in ImageDataGenerator.
        interpolation: Interpolation method used to resample the image if the
            target size is different from that of the loaded image.
            Supported methods are "nearest", "bilinear", and "bicubic".
            If PIL version 1.1.3 or newer is installed, "lanczos" is also
            supported. If PIL version 3.4.0 or newer is installed, "box" and
            "hamming" are also supported. By default, "nearest" is used.
    """

    def __init__(self,
                 label_data_path,
                 directory,
                 image_data_generator,
                 target_size=(256, 256),
                 color_mode='rgb',
                 batch_size=32,
                 shuffle=True,
                 seed=None,
                 data_format=None,
                 subset=None,
                 interpolation='nearest'):
        if data_format is None:
            data_format = K.image_data_format()

        self.directory = directory
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)

        if color_mode not in {'rgb', 'grayscale'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb" or "grayscale".')
        self.color_mode = color_mode
        self.data_format = data_format
        if self.color_mode == 'rgb':
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (3,)
            else:
                self.image_shape = (3,) + self.target_size
        else:
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (1,)
            else:
                self.image_shape = (1,) + self.target_size

        if subset:
            validation_split = self.image_data_generator.validation_split
            if subset == 'validation':
                split = (0, validation_split)
            elif subset == 'training':
                split = (validation_split, 1)
            else:
                raise ValueError('Invalid subset name: ', subset,
                                 '; expected "training" or "validation')
        else:
            split = None
        self.subset = subset

        self.data = pd.read_csv(label_data_path, converters={1: ast.literal_eval}).values
        num_files = len(self.data)

        for index in range(num_files):
            self.data[index, 1] = np.array(self.data[index, 1])

        self.samples = _get_train_or_validation_samples(split, num_files)
        self.interpolation = interpolation
        self.subset = subset

        super(ImageIterator, self).__init__(self.samples,
                                            batch_size,
                                            shuffle,
                                            seed)

    @staticmethod
    def __convert_array_of_array_to_matrix(array):
        """
        Convert numpy array of numpy array to numpy multidimensional array.

        :param array: numpy array
        :return: numpy array
        """
        temp = []
        for arr in array:
            temp.append(arr)
        return np.array(temp)

    def _get_batches_of_transformed_samples(self, index_array):
        """
        Gets a batch of transformed samples.

        :param index_array: numpy array
            array of sample indices to include in batch.
        :return: tuple
            Tuple batch of transformed example.
        """

        batch_x = np.zeros((len(index_array),) + self.image_shape, dtype=K.floatx())
        grayscale = self.color_mode == 'grayscale'
        filenames = self.data[:, 0]
        y = self.data[:, 1]

        for index, data_index in enumerate(index_array):
            filename = filenames[data_index]
            img = image.load_img(
                os.path.join(self.directory, filename),
                grayscale=grayscale,
                target_size=self.target_size,
                interpolation=self.interpolation)
            x = image.img_to_array(img, data_format=self.data_format)
            x = self.image_data_generator.random_transform(x)
            x = self.image_data_generator.standardize(x)
            batch_x[index] = x
        batch_y = self.__convert_array_of_array_to_matrix(y[index_array])

        return batch_x, batch_y

    def next(self):
        """For python 2.x.

        Returns:
            The next batch.
        """
        with self.lock:
            index_array = next(self.index_generator)
        return self._get_batches_of_transformed_samples(index_array)

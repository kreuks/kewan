import os

from keras.callbacks import CSVLogger, ModelCheckpoint, TensorBoard
from tensorflow.python.lib.io import file_io

from src.modeler.generator import KewanImageDataGenerator
from src.modeler.modeler import InceptionV3Modeler


def train(label_data_path, image_directory):
    os.makedirs('data/model', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('data/tensorboard', exist_ok=True)
    kewan_image_data_generator = KewanImageDataGenerator(rotation_range=0.5,
                                                         width_shift_range=0.1,
                                                         height_shift_range=0.1,
                                                         shear_range=0.1,
                                                         zoom_range=0.2,
                                                         horizontal_flip=True,
                                                         vertical_flip=True,
                                                         rescale=1/255,
                                                         validation_split=0.0)
    data_generator = kewan_image_data_generator.flow_from_label(label_data_path,
                                                                image_directory,
                                                                target_size=(224, 224),
                                                                batch_size=100)

    inception_v3_model = InceptionV3Modeler().get_model(input_shape=[224, 224, 3], classes=84)

    model_checkpoint = ModelCheckpoint('data/model/model_checkpoint.hdf5',
                                                 monitor='val_loss',
                                                 verbose=0,
                                                 save_best_only=True,
                                                 save_weights_only=False,
                                                 mode='auto',
                                                 period=1)

    csv_logger = CSVLogger('logs/keras_log.csv', append=True, separator=',')
    tensorboard = TensorBoard(log_dir='data/tensorboard/', write_graph=True, write_images=True)

    inception_v3_model.fit_generator(data_generator,
                                     steps_per_epoch=64,
                                     epochs=64,
                                     callbacks=[model_checkpoint, csv_logger, tensorboard])

    inception_v3_model.save('data/model/model.h5')
    with file_io.FileIO('data/model/model.h5', mode='r') as input_file:
        with file_io.FileIO('gs://kreuks/hackerearth/deep_learning_3/model.h5', mode='w+') as ouput_file:
            ouput_file.write(input_file.read())


if __name__ == '__main__':
    train('data/meta-data/annot.csv', 'data/image/')

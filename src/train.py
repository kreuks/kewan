import keras

from src.modeler.generator import KewanImageDataGenerator
from src.modeler.modeler import InceptionV3Modeler


def train(label_data_path, image_directory):
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
                                                                batch_size=32)

    inception_v3_model = InceptionV3Modeler().get_model(input_shape=[224, 224, 3],
                                                        classes=84)

    callbacks_ = keras.callbacks.ModelCheckpoint('data/model/model_checkpoint.hdf5',
                                                 monitor='val_loss',
                                                 verbose=0,
                                                 save_best_only=True,
                                                 save_weights_only=False,
                                                 mode='auto',
                                                 period=1)

    inception_v3_model.fit_generator(data_generator,
                                     steps_per_epoch=2,
                                     epochs=100,
                                     callbacks=[callbacks_])

    inception_v3_model.save('data/model/model/h5')


if __name__ == '__main__':
    train('data/meta-data/annot.csv', 'data/image/')

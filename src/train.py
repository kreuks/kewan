import os
from multiprocessing import cpu_count

from keras.callbacks import CSVLogger, ModelCheckpoint, TensorBoard

from src.modeler.generator import KewanImageDataGenerator
from src.modeler.modeler import InceptionV3Modeler, InceptionV4Modeler

CPU_COUNT = cpu_count()


class Trainer(object):
    def __init__(self, label_data_path, image_directory, target_size):
        self.label_data_path = label_data_path
        self.image_directory = image_directory
        self.target_size = target_size

    @staticmethod
    def get_or_create_folder(model_dir, logs_dir, tensorboard_dir):
        for directory in [model_dir, logs_dir, tensorboard_dir]:
            os.makedirs(directory, exist_ok=True)

    def create_train_validation_set(self):
        kewan_image_data_generator = KewanImageDataGenerator(rotation_range=0.5,
                                                             width_shift_range=0.1,
                                                             height_shift_range=0.1,
                                                             shear_range=0.1,
                                                             zoom_range=0.2,
                                                             horizontal_flip=True,
                                                             vertical_flip=True,
                                                             rescale=1 / 255,
                                                             validation_split=0.1)
        self.train_data_generator = kewan_image_data_generator.flow_from_label(self.label_data_path,
                                                                               self.image_directory,
                                                                               target_size=self.target_size,
                                                                               batch_size=100,
                                                                               subset='training')

        self.validation_data_generator = kewan_image_data_generator.flow_from_label(self.label_data_path,
                                                                                    self.image_directory,
                                                                                    target_size=self.target_size,
                                                                                    batch_size=100,
                                                                                    subset='validation')

    @staticmethod
    def get_callback(model_checkpoint_file, log_file, tensorboard_dir):
        model_checkpoint = ModelCheckpoint(model_checkpoint_file,
                                           monitor='val_loss',
                                           verbose=0,
                                           save_best_only=True,
                                           save_weights_only=False,
                                           mode='auto',
                                           period=1)

        csv_logger = CSVLogger(log_file, append=True, separator=',')
        tensorboard = TensorBoard(log_dir=tensorboard_dir, write_graph=True, write_images=True)
        return [model_checkpoint, csv_logger, tensorboard]

    def get_model(self):
        pass

    def train(self):
        pass


class InceptionV3Trainer(Trainer):
    def __init__(self, label_data_path, image_directory):
        super(InceptionV3Trainer, self).__init__(label_data_path, image_directory, (224, 224))

    def create_model(self):
        self.model = InceptionV3Modeler().get_model(input_shape=list(self.target_size + (3,)), classes=85)

    def train(self):
        callbacks = self.get_callback(model_checkpoint_file='data/model/inception_v3_model_checkpoint.hdf5',
                                      log_file='logs/inception_v3_keras_log.csv',
                                      tensorboard_dir='data/inception_v3_tensorboard/')
        self.create_train_validation_set()
        self.create_model()
        self.model.fit_generator(self.train_data_generator,
                                 steps_per_epoch=126,
                                 epochs=200,
                                 validation_data=self.validation_data_generator,
                                 validation_steps=1,
                                 workers=CPU_COUNT,
                                 use_multiprocessing=True,
                                 callbacks=callbacks)

        self.model.save('data/model/inception_v3_model.h5')


class InceptionV4Trainer(Trainer):
    def __init__(self, label_data_path, image_directory):
        super(InceptionV4Trainer, self).__init__(label_data_path, image_directory, (299, 299))

    def create_model(self):
        self.model = InceptionV4Modeler().get_model(input_shape=list(self.target_size + (3, )), classes=85)

    def train(self):
        callbacks = self.get_callback(model_checkpoint_file='data/model/inception_v4_model_checkpoint.hdf5',
                                      log_file='logs/inception_v4_keras_log.csv',
                                      tensorboard_dir='data/inception_v4_tensorboard/')
        self.create_train_validation_set()
        self.create_model()
        self.model.fit_generator(self.train_data_generator,
                                 steps_per_epoch=126,
                                 epochs=200,
                                 validation_data=self.validation_data_generator,
                                 validation_steps=1,
                                 workers=CPU_COUNT,
                                 use_multiprocessing=True,
                                 callbacks=callbacks)

        self.model.save('data/model/inception_v4_model.h5')


if __name__ == '__main__':
    trainer = InceptionV4Trainer('data/meta-data/annot.csv', 'data/image/')
    trainer.train()

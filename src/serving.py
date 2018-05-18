import os

import numpy as np
import pandas as pd

from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img


class ServingModel(object):
    def __init__(self, path, target_size):
        self.model = load_model(path)
        self.target_size = target_size

    def preprocesing(self, *args, **kwargs):
        pass

    def predict(self, *args, **kwargs):
        pass


class KerasImageServingModel(ServingModel):
    def __init__(self, path, target_size):
        super().__init__(path, target_size)

    def preprocesing(self, image):
        image = image.resize(self.target_size)
        image_np = img_to_array(image)
        image_np = np.expand_dims(image_np, axis=0)
        image_np = image_np / 255.
        return image_np

    def predict(self, image, *args, **kwargs):
        x = self.preprocesing(image)
        return self.model.predict(x)


if __name__ == '__main__':
    labels = []
    keras_image_serving_model = KerasImageServingModel('data/model/model/model.h5', (224, 224))
    for x in range(5400):
        image_path = 'data/test_img/Image-{}.jpg'.format(str(x + 1))
        print(image_path)
        pil_image = load_img(image_path)
        proba = keras_image_serving_model.predict(pil_image)
        labels.append([os.path.basename(image_path)] + list(np.where(np.squeeze(proba) > .5, 1, 0)))
    df = pd.DataFrame(labels)

    column_name = ['Image_name']
    for attrib_name in range(85):
        column_name.append('attrib_{}'.format(str(attrib_name + 1).zfill(2)))

    df.columns = column_name
    df.to_csv('data/label_final.csv', header=True, index=False)


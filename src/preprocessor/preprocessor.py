import argparse
import glob
import os

import numpy as np
import tensorflow as tf
from keras.preprocessing import image as keras_image
from PIL import Image

from src.utils import category_index, proper_index


class Preprocessor(object):
    def __init__(self, model_path):
        self.model_path = model_path

    @staticmethod
    def sort_by_score(boxes, classes, scores):
        index = np.argsort(scores[::-1])[:5]
        for score, class_ in zip(scores[index], classes[index]):
            if class_ not in proper_index:
                index = index[:-1]
        return boxes[index], classes[index], scores[index]

    @staticmethod
    def crop_object(pil_image, annotation):
        """

        :param pil_image: Pillow Image
        :param annotation: tuple
        :return:
        """
        im_width, im_height = pil_image.size
        try:
            crop_box = (annotation[1] * im_width,
                        annotation[0] * im_height,
                        annotation[3] * im_width,
                        annotation[2] * im_height)
            pil_image_crop = pil_image.crop(crop_box)
            return pil_image_crop
        except:
            return None

    @staticmethod
    def save_image(pil_image, output_path):
        pil_image.save(output_path, format='JPEG')

    def detect_objects(self, image_np):
        pass

    def run(self, *args, **kwargs):
        pass


class TensorflowPreprocessor(Preprocessor):
    def __init__(self, model_path):
        super().__init__(model_path=model_path)
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            self.sess = tf.Session(graph=self.detection_graph)

    def detect_objects(self, image_np):
        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')

        scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

        (boxes, scores, classes, num_detections) = self.sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded}
        )

        return self.sort_by_score(np.squeeze(boxes), np.squeeze(classes).astype(np.int32), np.squeeze(scores))

    def run(self, image_path, output_image_path):
        pil_image = Image.open(image_path)
        image_np = keras_image.img_to_array(pil_image)
        annotations = self.detect_objects(image_np)
        annotation = annotations[0][0]
        cropped_pil_image = self.crop_object(pil_image, annotation)
        if cropped_pil_image:
            self.save_image(cropped_pil_image, output_image_path)


if __name__ == '__main__':
    def main():
        parser = argparse.ArgumentParser()

        parser.add_argument(
            '-i', '--input-data-dir',
            type=str,
            default='data/train-img/',
            help='Path to input folder directory.'
        )

        parser.add_argument(
            '-o', '--train-data-dir',
            type=str,
            default='data/train-img-cropped/',
            help='Path to output folder directory.'
        )

        parser.add_argument(
            '-m', '--model=path',
            type=str,
            help='Path to model path.'
        )

        input_data_dir = parser.parse_args().input_data_dir
        output_data_dir = parser.parse_args().train_data_dir
        model_path = parser.parse_args().eval_data_dir

        files = glob.glob(os.path.join(input_data_dir, '*'))
        files_crop =

files = glob.glob('data/train_img/*')
files_crop = [file.replace('train_img', 'train_img_crop') for file in files]

for file, file_crop in zip(files, files_crop):
    predict(file, file_crop)
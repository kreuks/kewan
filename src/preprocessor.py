import glob

import numpy as np
import tensorflow as tf
from keras.preprocessing import image as keras_image
from PIL import Image

from src.utils import category_index, proper_index


detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile('data/model/faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb', 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
    sess = tf.Session(graph=detection_graph)


def sort_by_score(boxes, classes, scores):
    index = np.argsort(scores[::-1])[:5]
    for score, class_ in zip(scores[index], classes[index]):
        if class_ not in proper_index:
            index = index[:-1]
    return boxes[index], classes[index], scores[index]


def detect_objects(image_np, sess_, detection_graph_):
    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = detection_graph_.get_tensor_by_name('image_tensor:0')
    boxes = detection_graph_.get_tensor_by_name('detection_boxes:0')

    scores = detection_graph_.get_tensor_by_name('detection_scores:0')
    classes = detection_graph_.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph_.get_tensor_by_name('num_detections:0')

    (boxes, scores, classes, num_detections) = sess_.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded}
    )

    return None, sort_by_score(np.squeeze(boxes), np.squeeze(classes).astype(np.int32), np.squeeze(scores))


def predict(image_path, output):
    image = Image.open(image_path)
    im_width, im_height = image.size
    image_np = keras_image.img_to_array(image)
    image_np_copy = image_np.copy()
    _, annos = detect_objects(image_np_copy, sess, detection_graph)
    try:
        print(image_path + ' ' + category_index[annos[1][0]]['name'])
        box = annos[0][0]
        crop_box = (box[1] * im_width, box[0] * im_height, box[3] * im_width, box[2] * im_height)
        img_crop = image.crop(crop_box)
        img_crop.save(output, format='JPEG')
    except:
        return annos
    return annos


files = glob.glob('data/train_img/*')
files_crop = [file.replace('train_img', 'train_img_crop') for file in files]

for file, file_crop in zip(files, files_crop):
    predict(file, file_crop)
from object_detection.utils import label_map_util


NUM_CLASSES = 90

MODELS_FOLDER = 'models/'
MODEL_PATH_CAR_TYPE = MODELS_FOLDER + 'model_car_type.h5'
MODEL_PATH_CAR_COLOR = MODELS_FOLDER + 'model_car_color.h5'
PATH_TO_CKPT = MODELS_FOLDER + 'ssd_mobilenet_v1_coco_11_06_2017/frozen_inference_graph.pb'
PATH_TO_LABELS = 'object_detection/data/mscoco_label_map.pbtxt'

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)

categories = label_map_util.convert_label_map_to_categories(
    label_map,
    max_num_classes=NUM_CLASSES,
    use_display_name=True
)
category_index = label_map_util.create_category_index(categories)

proper_index = [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 88]

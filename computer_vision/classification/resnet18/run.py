import os
import time
import argparse
import utils.misc as utils
from utils.cv.coco import COCODataset
from utils.cv.imagenet import ImageNet
from utils.tf import TFSavedModelRunner
from utils.tf import TFSavedModelKERAS
from utils.benchmark import run_model
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.saved_model import tag_constants


def parse_args():
    parser = argparse.ArgumentParser(description="Run YOLO v4 tiny model.")
    parser.add_argument("-m", "--model_path",
                        type=str, required=True,
                        help="path to the model")
    parser.add_argument("-p", "--precision",
                        type=str, choices=["fp32"], required=True,
                        help="precision of the model provided")
    parser.add_argument("-b", "--batch_size",
                        type=int, default=1,
                        help="batch size to feed the model with")
    parser.add_argument("--timeout",
                        type=float, default=60.0,
                        help="timeout in seconds")
    parser.add_argument("--num_runs",
                        type=int,
                        help="number of passes through network to execute")
    parser.add_argument("--images_path",
                        type=str,
                        help="path to directory with COCO validation images")
    parser.add_argument("--anno_path",
                        type=str,
                        help="path to file with validation annotations")
    return parser.parse_args()


def run_tf_fp32(model_path, batch_size, num_of_runs, timeout, images_path, labels_path):

    # model = keras.models.load_model(model_path)

    def run_single_pass(tf_runner, imagenet):
        shape = (224, 224)
        output = tf_runner.run(tf.constant(imagenet.get_input_array(shape)))
        # tf_runner.set_input_tensor("input_tensor:0", imagenet.get_input_array(shape))
        # output = tf_runner.run()
        for i in range(batch_size):
            imagenet.submit_predictions(
                i,
                [1, 2, 3, 4],
                [1, 2, 3, 4]
            )

    dataset = ImageNet(batch_size, "RGB", images_path, labels_path,
                       pre_processing="VGG", is1001classes=True)

    runner = TFSavedModelKERAS()
    # saved_model_loaded = keras.models.load_model(model_path)
    saved_model_loaded = tf.saved_model.load(model_path)
    runner.model = saved_model_loaded

    # runner = TFSavedModelRunner()
    # saved_model_loaded = tf.saved_model.load(model_path, tags=[tag_constants.SERVING])
    # runner.model = saved_model_loaded.signatures['serving_default']

    return run_model(run_single_pass, runner, dataset, batch_size, num_of_runs, timeout)


# def run_tf_fp32(model_path, batch_size, num_of_runs, timeout, images_path, anno_path):
#     def run_single_pass(tf_runner, coco):
#         shape = (416, 416)
#         output = tf_runner.run(tf.constant(coco.get_input_array(shape)))
#         bboxes = output["tf.concat_12"][:, :, 0:4]
#         preds = output["tf.concat_12"][:, :, 4:]
#         detection_boxes, detection_scores, detection_classes, valid_detections = tf.image.combined_non_max_suppression(
#             boxes=tf.reshape(bboxes, (tf.shape(bboxes)[0], -1, 1, 4)),
#             scores=tf.reshape(
#                 preds, (tf.shape(preds)[0], -1, tf.shape(preds)[-1])),
#             max_output_size_per_class=50,
#             max_total_size=50,
#             iou_threshold=0.45,
#             score_threshold=0.25
#         )
#         for i in range(batch_size):
#             for d in range(int(valid_detections[i])):
#                 coco.submit_bbox_prediction(
#                     i,
#                     coco.convert_bbox_to_coco_order(detection_boxes[i][d] * shape[0], 1, 0, 3, 2),
#                     detection_scores[i][d],
#                     coco.translate_cat_id_to_coco(int(detection_classes[i][d]))
#                 )
#
#     dataset = COCODataset(batch_size, "RGB", "COCO_val2014_000000000000", images_path, anno_path,
#                           pre_processing="YOLO", sort_ascending=True)
#     runner = TFSavedModelRunner()
#     saved_model_loaded = tf.saved_model.load(model_path, tags=[tag_constants.SERVING])
#     runner.model = saved_model_loaded.signatures['serving_default']
#
#     return run_model(run_single_pass, runner, dataset, batch_size, num_of_runs, timeout)


def main():
    args = parse_args()
    if args.precision == "fp32":
        run_tf_fp32(
            args.model_path, args.batch_size, args.num_runs, args.timeout, args.images_path, args.anno_path
        )
    else:
        assert False


if __name__ == "__main__":
    main()

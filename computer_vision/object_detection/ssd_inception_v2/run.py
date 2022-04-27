import os
import time
import pathlib
import argparse
import subprocess

import utils.misc as utils
from utils.cv.coco import COCODataset
from utils.tflite import TFLiteRunner
from utils.benchmark import run_model
from utils.tf import TFFrozenModelRunner
from downloads.utils import get_downloads_path
from utils.misc import print_goodbye_message_and_die


def parse_args():
    parser = argparse.ArgumentParser(description="Run SSD Inception v2 model.")
    parser.add_argument("-m", "--model_path",
                        type=str,
                        help="path to the model")
    parser.add_argument("-p", "--precision",
                        type=str, choices=["fp32", "fp16"], required=True,
                        help="precision of the model provided")
    parser.add_argument("-b", "--batch_size",
                        type=int, default=1,
                        help="batch size to feed the model with")
    parser.add_argument("-f", "--framework",
                        type=str, default="tf",
                        choices=["tf"],
                        help="specify the framework in which a model should be run")
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


def run_tf_fp(model_path, batch_size, num_runs, timeout, images_path, anno_path):

    def run_single_pass(tf_runner, coco):
        shape = (300, 300)
        tf_runner.set_input_tensor("image_tensor:0", coco.get_input_array(shape))
        output = tf_runner.run()
        for i in range(batch_size):
            for d in range(int(output["num_detections:0"][i])):
                coco.submit_bbox_prediction(
                    i,
                    coco.convert_bbox_to_coco_order(output["detection_boxes:0"][i][d] * shape[0], 1, 0, 3, 2),
                    output["detection_scores:0"][i][d],
                    int(output["detection_classes:0"][i][d])
                )

    dataset = COCODataset(batch_size, "BGR", "COCO_val2014_000000000000", images_path, anno_path, sort_ascending=True)
    runner = TFFrozenModelRunner(
        model_path, ["detection_classes:0", "detection_boxes:0", "detection_scores:0", "num_detections:0"])

    return run_model(run_single_pass, runner, dataset, batch_size, num_runs, timeout)


def run_tf_fp32(model_path, batch_size, num_runs, timeout, images_path, anno_path, **kwargs):
    return run_tf_fp(model_path, batch_size, num_runs, timeout, images_path, anno_path)


def run_tf_fp16(model_path, batch_size, num_runs, timeout, images_path, anno_path, **kwargs):
    return run_tf_fp(model_path, batch_size, num_runs, timeout, images_path, anno_path)


def main():
    args = parse_args()

    labels = "https://ampereaimodelzoo.s3.amazonaws.com/COCO2014_anno_onspecta.json"

    images = "https://ampereaimodelzoo.s3.amazonaws.com/COCO2014_onspecta.tar.gz"
    coco_data = get_downloads_path(), "coco"

    subprocess.run(['wget https://ampereaimodelzoo.s3.eu-central-1.amazonaws.com/COCO2014_anno_onspecta.json'])

    quit()

    if not pathlib.Path(coco_data).is_dir():
        try:
            subprocess.run(["wget", labels])
            subprocess.run(["wget", images])
            subprocess.run(["mkdir", coco_data])
            subprocess.run(["mv", 'COCO2014_anno_onspecta.json', coco_data])
            subprocess.run(["tar", "-xf", 'COCO2014_onspecta.tar.gz', "-C", coco_data])
            subprocess.run(["rm", 'COCO2014_onspecta.tar.gz'])
        except KeyboardInterrupt:
            subprocess.run(["rm", 'COCO2014_onspecta.tar.gz'])
            subprocess.run(["rm", '-rf', coco_data])

    quit()

    # if not self.__annotations_file_path.is_file():
    #     try:
    #         subprocess.run(["wget", self.__annotations_link])
    #         subprocess.run(["mv", self.__annotations_file_name, str(self.__coco_dir_path)])
    #     except KeyboardInterrupt:
    #         subprocess.run(["rm", self.__annotations_file_name])

    if args.framework == "tf":
        if args.model_path is None:
            print_goodbye_message_and_die(
                "a path to model is unspecified!")

        if args.precision == "fp32":
            run_tf_fp32(**vars(args))
        elif args.precision == "fp16":
            run_tf_fp16(**vars(args))
        else:
            print_goodbye_message_and_die(
                "this model seems to be unsupported in a specified precision: " + args.precision)

    else:
        print_goodbye_message_and_die(
            "this model seems to be unsupported in a specified framework: " + args.framework)


if __name__ == "__main__":
    main()

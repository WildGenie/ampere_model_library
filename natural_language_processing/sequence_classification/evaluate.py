from transformers import TFBertForSequenceClassification, TFDistilBertForSequenceClassification
import tensorflow as tf
import time
import numpy as np
import argparse
import nlp_utils as nu

# https://huggingface.co/transformers/v3.0.2/training.html#tensorflow

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the Hugging Face models for GLUE tasks dataset")
    parser.add_argument("-m", "--model_name",
                        type=str, required=True,
                        help="name of the pre-trained model from which the \
                        trained model was trained[distilbert-base-uncased, bert-base-uncased]")
    # parser.add_argument("-p", "--model_path",
    #                     type=str, required=True,
    #                     help="path of the trained model")
    parser.add_argument("-b", "--batch_size",
                        type=int, default=8,
                        help="batch size to feed the model with")
    parser.add_argument("-t", "--task_name",
                        type=str, default="mrpc",
                        help="Task name of the GLUE dataset")
    return parser.parse_args()


def main():
	# tf.config.threading.set_intra_op_parallelism_threads(get_intra_op_parallelism_threads())
	# tf.config.threading.set_inter_op_parallelism_threads(1)

	# try:
	#     transformers.onspecta()
	# except AttributeError:
	#     print_goodbye_message_and_die("OnSpecta's fork of Transformers repo is not installed.\n"
	#                                   "\nPlease refer to the README.md in natural_language_processing/huggingface "
	#                                   "directory for instructions on how to set up the project.")

	args = parse_args()
	is_model_bert = True
	if("distil" in args.model_name):
		is_model_bert = False

	model_path = nu.get_model(is_model_bert)
	# Load pre-trained models
	if(is_model_bert):
		model = TFBertForSequenceClassification.from_pretrained(model_path)
	else:
		model = TFDistilBertForSequenceClassification.from_pretrained(model_path)

	eval_dataset, eval_label = nu.get_dataset(args.model_name, args.task_name, 
		args.batch_size, is_model_bert)
	# region Metric function

	start_time = time.time()
	eval_predictions = model.predict(eval_dataset)
	# eval_metric = model.evaluate(eval_dataset)
	end_time = time.time() - start_time
	eval_metrics = nu.compute_metrics(eval_predictions["logits"], eval_label, 
		args.task_name)
	print(f"Evaluation metrics ({args.task_name}):")
	print(eval_metrics, end_time, len(eval_label))

main()

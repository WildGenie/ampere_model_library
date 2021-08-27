from transformers import BertTokenizer, DistilBertTokenizer, glue_convert_examples_to_features
import tensorflow_datasets as tfds
from datasets import load_metric
import numpy as np
import subprocess
import pathlib
from pathlib import Path
import os


def get_downloads_path():
    """
    A function returning absolute path to downloads dir.

    :return: str, path to downloads dir
    """
    return os.path.dirname(os.path.realpath(__file__))

def get_dataset(model_name, task_name, batch_size, is_model_bert):
	if(is_model_bert):
		tokenizer = BertTokenizer.from_pretrained(model_name)
	else:
		tokenizer = DistilBertTokenizer.from_pretrained(model_name)
	data = tfds.load('glue/'+task_name)
	# train_dataset = glue_convert_examples_to_features(data['train'], tokenizer, max_length=128, task='mrpc')
	# train_dataset = train_dataset.shuffle(100).batch(32)#.repeat(2)
	eval_dataset = glue_convert_examples_to_features(data['validation'], tokenizer, max_length=128, task='mrpc')
	eval_dataset = eval_dataset.batch(batch_size)
	eval_label = np.concatenate([y for x, y in eval_dataset], axis=0)
	# print("Eval data ", eval_label)

	# for i, batch in enumerate(eval_dataset):
	# 	print("Batch values ", i, batch)
	return eval_dataset, eval_label

def compute_metrics(preds, label_ids, task_name):
	metric = load_metric("glue", task_name)

	# preds = preds["logits"]
	preds = np.argmax(preds, axis=1)
	result = metric.compute(predictions=preds, references=label_ids)
	if len(result) > 1:
	    result["combined_score"] = np.mean(list(result.values())).item()
	return result


def get_model(is_model_bert):
	downloads_dir_path = pathlib.Path(get_downloads_path())
	if is_model_bert:
		model_url = "https://www.dropbox.com/s/4kqjtxob7suxvra/bert_base_uncased.tar.gz?dl=0"
		model_filename = "bert_base_uncased.tar.gz"
	else:
		model_url = "https://www.dropbox.com/s/uswfbhhg4aigasu/distilbert-base-uncased.tar.gz?dl=0"
		model_filename = "distilbert-base-uncased.tar.gz"

	downloads_dir_path = pathlib.Path(get_downloads_path())

	path_to_model_base = Path(os.path.join(downloads_dir_path, 'tf_models'))
	path_to_model = Path(os.path.join(downloads_dir_path, 
		'tf_models', model_filename[:-7]))

	if path_to_model_base.is_dir():
		print("Path ", path_to_model_base, model_filename)
		pass
	else:
		os.makedirs(str(path_to_model_base))

	if path_to_model.is_dir():
	    pass
	else:
	    try:
	        subprocess.run(["wget", "-O", model_filename, model_url])
	        subprocess.run(["tar", "-xf", model_filename, "-C", str(path_to_model_base)])
	        subprocess.run(["rm", model_filename])
	    except KeyboardInterrupt:
	        subprocess.run(["rm", model_filename])

	return path_to_model

# dataset_filename = 'SMSSpamCollection'
# dataset_url = 'https://www.dropbox.com/s/ymd54rur6atkvqu/SMSSpamCollection'

# model_filename = 'model.tar.gz'
# model_url = 'https://www.dropbox.com/s/fp43347je178wo8/model.tar.gz'



# path_to_dataset = Path(os.path.join(downloads_dir_path, dataset_filename))
# path_to_model = Path(os.path.join(downloads_dir_path, 'senti_model'))

# if path_to_dataset.is_file():
#     pass
# else:
#     try:
#         subprocess.run(["wget", dataset_url])
#         subprocess.run(["mv", dataset_filename, str(downloads_dir_path)])
#     except KeyboardInterrupt:
#         subprocess.run(["rm", dataset_filename])

# if path_to_model.is_dir():
#     pass
# else:
#     try:
#         subprocess.run(["wget", model_url])
#         subprocess.run(["tar", "-xf", model_filename, "-C", str(downloads_dir_path)])
#         subprocess.run(["rm", model_filename])
#     except KeyboardInterrupt:
#         subprocess.run(["rm", model_filename])

if __name__ == "__main__":
	is_model_bert = False
	get_model(is_model_bert)

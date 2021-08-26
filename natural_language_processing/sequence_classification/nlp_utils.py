from transformers import BertTokenizer, DistilBertTokenizer, glue_convert_examples_to_features
import tensorflow_datasets as tfds
from datasets import load_metric
import numpy as np


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

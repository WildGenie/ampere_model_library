Requirements:
tensorflow_datasets
tf2onnx

Issue:
======
Running trained model on ARM with DLS enabled the accuracy does not match. Probably some optimizations are causing the error. With the DLS disabled the accuracy matches with that trained on a GPU.

Steps to Run:
=============
- To run tensorflow models (currently bert_base_uncased and distilbert_base_uncased) are supported
	- python evaluate.py -m bert-base-uncased

- To run onnx we need to convert the tensorflow models needs to be converted to onnx. 
	- python export_onnx.py -m bert-base-uncased -o tf_onnx_models/bert_base_uncased.onnx

- This dumps three files:
	- tf_onnx_models/bert_base_uncased.onnx - No Optimization and float32
	- tf_onnx_models/bert_base_uncased.opt.onnx - Optimized float32
	- tf_onnx_models/bert_base_uncased.opt.quant.onnx - int8 quantized and optimized

- After convertion, we can run the onnx models individually
	- python evaluate_onnx.py -m distilbert-base-uncased -p tf_onnx_models/distilbert_base_uncased.onnx
	- python evaluate_onnx.py -m distilbert-base-uncased -p tf_onnx_models/distilbert_base_uncased.opt.onnx
	- python evaluate_onnx.py -m distilbert-base-uncased -p tf_onnx_models/distilbert_base_uncased.opt.quant.onnx




Current performance metrics:
============================
python evaluate.py -m bert-base-uncased -p bert_base_uncased/
{'accuracy': 0.8063725490196079, 'f1': 0.8685524126455907, 'combined_score': 0.8374624808325992} 6.255127429962158 408

python evaluate.py -m distilbert-base-uncased -p train_outputs/
{'accuracy': 0.8284313725490197, 'f1': 0.8780487804878049, 'combined_score': 0.8532400765184123} 4.368492603302002 408


python export_onnx.py -m bert-base-uncased -p bert_base_uncased/ -o tf_onnx_models/bert_base_uncased.onnx


python evaluate_onnx.py -m distilbert-base-uncased -p tf_onnx_models/distilbert_base_uncased.onnx
Evaluation results  {'accuracy': 0.8284313725490197, 'f1': 0.8780487804878049, 'combined_score': 0.8532400765184123}
python train_and_accuracy/evaluate_onnx.py -m distilbert-base-uncased -p tf_onnx_models/distilbert_base_uncased.opt.onnx
Evaluation results  {'accuracy': 0.8284313725490197, 'f1': 0.8780487804878049, 'combined_score': 0.8532400765184123}
python evaluate_onnx.py -m distilbert-base-uncased -p tf_onnx_models/distilbert_base_uncased.opt.quant.onnx
Evaluation results  {'accuracy': 0.8382352941176471, 'f1': 0.8873720136518771, 'combined_score': 0.8628036538847621}

ONNX Runtime cpu inference time for sequence length 128: 128.15 ms
ONNX Runtime cpu inference time for sequence length 128: 84.67 ms

python evaluate_onnx.py -m bert-base-uncased -p tf_onnx_models/bert_base_uncased.onnx
Evaluation results  {'accuracy': 0.8063725490196079, 'f1': 0.8685524126455907, 'combined_score': 0.8374624808325992}
python evaluate_onnx.py -m bert-base-uncased -p tf_onnx_models/bert_base_uncased.opt.onnx
Evaluation results  {'accuracy': 0.8063725490196079, 'f1': 0.8685524126455907, 'combined_score': 0.8374624808325992}
python evaluate_onnx.py -m bert-base-uncased -p tf_onnx_models/bert_base_uncased.opt.quant.onnx
Evaluation results  {'accuracy': 0.8088235294117647, 'f1': 0.8717105263157895, 'combined_score': 0.8402670278637772}

ONNX Runtime cpu inference time for sequence length 128: 261.71 ms
ONNX Runtime cpu inference time for sequence length 128: 251.66 ms
ONNX Runtime cpu inference time for sequence length 128: 148.74 ms





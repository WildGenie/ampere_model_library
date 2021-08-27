Requirements:
tensorflow_datasets
tf2onnx

On Arm with DLS enabled the accuracy drops. Looks like there is a issue with some optimization. 




Current performance metrics:
python train_and_accuracy/evaluate.py -m bert-base-uncased -p bert_base_uncased/
{'accuracy': 0.8063725490196079, 'f1': 0.8685524126455907, 'combined_score': 0.8374624808325992} 6.255127429962158 408

python train_and_accuracy/evaluate.py -m distilbert-base-uncased -p train_outputs/
{'accuracy': 0.8284313725490197, 'f1': 0.8780487804878049, 'combined_score': 0.8532400765184123} 4.368492603302002 408


python train_and_accuracy/export_onnx.py -m bert-base-uncased -p bert_base_uncased/ -o tf_onnx_models/bert_base_uncased.onnx


python train_and_accuracy/evaluate_onnx.py -m distilbert-base-uncased -p tf_onnx_models/distilbert_base_uncased.onnx
Evaluation results  {'accuracy': 0.8284313725490197, 'f1': 0.8780487804878049, 'combined_score': 0.8532400765184123}
python train_and_accuracy/evaluate_onnx.py -m distilbert-base-uncased -p tf_onnx_models/distilbert_base_uncased.opt.onnx
Evaluation results  {'accuracy': 0.8284313725490197, 'f1': 0.8780487804878049, 'combined_score': 0.8532400765184123}
python train_and_accuracy/evaluate_onnx.py -m distilbert-base-uncased -p tf_onnx_models/distilbert_base_uncased.opt.quant.onnx
Evaluation results  {'accuracy': 0.8382352941176471, 'f1': 0.8873720136518771, 'combined_score': 0.8628036538847621}

ONNX Runtime cpu inference time for sequence length 128: 128.15 ms
ONNX Runtime cpu inference time for sequence length 128: 84.67 ms

python train_and_accuracy/evaluate_onnx.py -m bert-base-uncased -p tf_onnx_models/bert_base_uncased.onnx
Evaluation results  {'accuracy': 0.8063725490196079, 'f1': 0.8685524126455907, 'combined_score': 0.8374624808325992}
python train_and_accuracy/evaluate_onnx.py -m bert-base-uncased -p tf_onnx_models/bert_base_uncased.opt.onnx
Evaluation results  {'accuracy': 0.8063725490196079, 'f1': 0.8685524126455907, 'combined_score': 0.8374624808325992}
python train_and_accuracy/evaluate_onnx.py -m bert-base-uncased -p tf_onnx_models/bert_base_uncased.opt.quant.onnx
Evaluation results  {'accuracy': 0.8088235294117647, 'f1': 0.8717105263157895, 'combined_score': 0.8402670278637772}

ONNX Runtime cpu inference time for sequence length 128: 261.71 ms
ONNX Runtime cpu inference time for sequence length 128: 251.66 ms
ONNX Runtime cpu inference time for sequence length 128: 148.74 ms





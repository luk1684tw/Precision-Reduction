# Per-Layer Based Precision in PyTorch

One can assign different precision for different layers (as the same idea from [this paper](http://proceedings.mlr.press/v70/sakr17a/sakr17a.pdf)) to experiment the best settings for the trade-off between quantization bits and accuracy.

A small sample of experiment results can be seen from [here](https://docs.google.com/spreadsheets/d/13Vna79bhMChdDqkvx2IUVze24kbmV0jj-79lYkhJ-Gs/edit?usp=sharing)


## Requiremnts

- CUDA

pip packages:

- torch
- torchvision
- numpy
- pandas
- opencv-python


## Usage

### Run model evaluation for different precision settings

Use uniform precision as the same as original repo: (runs only one test)
```
python quantize.py --type cifar10 --quant_method linear --param_bits 8 --fwd_bits 8 --bn_bits 8 --gpu 0
```

Define precision of each layer in quantize_runner.py `param_bits`(--param_bits), `batch_norm_bits`(--bn_bits), `layer_output_bits`(--fwd_bits) and other settings, and then run:
(currently only support uniform precision for `layer_output_bits`)
```
python quantize_runner.py
```

Results will be written to both result.csv and result.pkl.

### Define and train your own model

1. Define your model class in <dataset>/model.py or change predefined arguments in <dataset>/train.py
2. Run `python <dataset>/train.py`
3. Copy trained model (default is <project_root>/log/default/<dataset>/latest.pth) to model_root/<dataset>.pth
4. Set custom model kwargs in quantize_runner.py
5. Run `python quantize_runner.py --model_root model_root` to evaluate self-trained model.


**NOTE**: For detailed project structure explanation please refer to README_orig.md


## Attribution

This code is modified from [this repository](https://github.com/aaron-xichen/pytorch-playground), which adopts uniform precision.

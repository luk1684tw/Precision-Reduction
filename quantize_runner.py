from collections import OrderedDict
from pprint import pprint

import argparse
import copy
import pandas as pd
import pickle
import random
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from utee import misc, selector, quant


def save_result(result, result_file='result'):
    result.to_csv(f'{result_file}.csv')

    with open(f'{result_file}.pkl', 'wb') as f:
        pickle.dump(result, f)


def bits_len_match(model, param_set, type):
    if not isinstance(param_set, list)\
    or not isinstance(param_set[0], list):
        raise ValueError("param_set should be a list")

    modules = model.named_modules()
    num = 0

    if type == 'param':
        for k in model.state_dict().keys():
            if not 'running' in k:
                num += 1

    if type == 'batch_norm':
        for k in model.state_dict().keys():
            if 'running' in k:
                num += 1

    if type == 'layer_output':
        for name, mod in modules:
            if name and isinstance(mod, \
                (nn.Conv2d, nn.Linear, nn.BatchNorm1d, nn.BatchNorm2d, nn.AvgPool2d)):
                    num += 1

    for s in param_set:
        if len(s) != num:
            raise ValueError(f"``{type}`` should have {num} elements in each list, but got {len(s)}")


def quantize_model(model_orig,
                   quant_method,
                   param_bits,
                   batch_norm_bits,
                   layer_output_bits,
                   overflow_rate=0.0,
                   n_sample=10):

    model = copy.deepcopy(model_orig)
    state_dict = model.state_dict()
    state_dict_quant = OrderedDict()

    p_idx = 0
    b_idx = 0

    for idx, (k, v) in enumerate(state_dict.items()):
        if 'running' in k:
            if batch_norm_bits[b_idx] >= 32:
                state_dict_quant[k] = v
                continue
            else:
                bits = batch_norm_bits[b_idx]
                b_idx += 1
        else:
            bits = param_bits[p_idx]
            p_idx += 1

        if quant_method == 'linear':
            sf = bits - 1. - quant.compute_integral_part(v, overflow_rate=overflow_rate)
            v_quant  = quant.linear_quantize(v, sf, bits=bits)
        elif quant_method == 'log':
            v_quant = quant.log_minmax_quantize(v, bits=bits)
        elif quant_method == 'minmax':
            v_quant = quant.min_max_quantize(v, bits=bits)
        else:
            v_quant = quant.tanh_quantize(v, bits=bits)

        state_dict_quant[k] = v_quant

    model.load_state_dict(state_dict_quant)

    model = quant.quantize_model_per_layer_output(model,
                        layer_output_bits=layer_output_bits,
                        overflow_rate=overflow_rate,
                        counter=n_sample,
                        type=quant_method)

    # To actually reduce param size by changing data type to float16
    # if layer_output_bits <= 16:
    #     model.half()

    return model


parser = argparse.ArgumentParser(description='PyTorch SVHN Example')
parser.add_argument('--type', default='cifar10', help='|'.join(selector.known_models))
parser.add_argument('--model_root', default='~/.torch/models/', help='folder to save the model')
parser.add_argument('--data_root', default='~/public_dataset/pytorch/', help='folder to save the model')
parser.add_argument('--logdir', default='log/default', help='folder to save to the log')
parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
parser.add_argument('--input_size', type=int, default=224, help='input size of image')
parser.add_argument('--overflow_rate', type=float, default=0.0, help='overflow rate')
args = parser.parse_args()

misc.ensure_dir(args.logdir)
args.model_root = misc.expand_user(args.model_root)
args.data_root = misc.expand_user(args.data_root)
args.input_size = 299 if 'inception' in args.type else args.input_size

# types to run
types = [
    'mnist',
    # 'cifar10',
    # 'cifar100',
]

# quantization methods to test
quant_methods = [
    'log',
    'minmax',
    'tanh'
]

model_bits = {
    'mnist': {
        'param_bits': [
            [32]*8,
            [16]*8,
            [8]*8,
            [4]*8,
            [7, 7, 5, 5, 4, 4, 4, 4]
        ],
        'batch_norm_bits': [
            []
        ],
        'layer_output_bits': [
            [32]*4,
            [16]*4,
            [8]*4,
            [4]*4,
            [3]*4,
            [2]*4,
            [4, 5, 4, 2]
        ],
    },

    'cifar10': {
        # precision of weights and biases of each layer
        # list length should be number of layers * 2
        # [W1, b1, W2, b2, W3, b3, ...] => [bits of W1, bits for b1, ...]
        'param_bits': [
            # cifar10, cifar100 => 16
            [32]*18,
            [16]*18,
            [12]*18,
            [8]*18,
            [6]*18,
            [4]*18,
            [16]*4 + [4]*8 + [16]*4,
            [16]*4 + [4]*8 + [8]*4,
            [16]*4 + [2]*8 + [16]*4,
            [16]*2 + [2]*12 + [16]*2,
            [8]*4 + [4]*10 + [8]*4,
            [16]*4 + [8]*14,
            [16]*4 + [4]*14,
            [16]*4 + [2]*14,
            [16]*2 + [8]*16,
            [16]*2 + [4]*16,
            [10, 10, 10, 10, 12, 12, 10, 10, 9, 9, 8, 8, 6, 6, 6, 6, 2, 2]
        ],
        'batch_norm_bits': [
            # cifar10, cifar100 => 14
            [32]*14,
            [16]*14,
        ],
        # # precision of output of each layer
        'layer_output_bits': [
            [32]*16,
            [16]*16,
            [12]*16,
            [8]*16,
            [4]*16,
            [10, 10, 12, 12, 10, 10, 9, 9, 8, 8, 6, 6, 6, 6, 2, 2]
        ]
    },
}

model_bits['cifar100'] = model_bits['cifar10']


# specify custom arguments for self-trained models
model_kwargs = {
    'mnist': {
        'n_hiddens': [512, 512, 512]
    },
    'cifar10': {
        'n_channel': 32
    },
    'cifar100': {
        'n_channel': 32
    }
}


# check types are valid
for typ in types:
    assert typ in selector.known_models

# check quant methods are valid
for method in quant_methods:
    assert method in ['linear', 'minmax', 'log', 'tanh']

assert torch.cuda.is_available(), 'no cuda'

cudnn.benchmark = True
torch.manual_seed(random.randint(1, 100))
torch.cuda.manual_seed(random.randint(1, 100))
count = 0

try:
    with open('result.pkl', 'rb') as f:
        result = pickle.load(f)
except (EOFError, FileNotFoundError):
    result = pd.DataFrame()


for typ in types:
    # If model_root is specified, assume user wants to use self-trained model
    if args.model_root:
        # Load custom keyward arguments for the model
        mk = model_kwargs[typ] if typ in model_kwargs else {}

    # Load model and dataset fetcher
    model_raw, ds_fetcher, is_imagenet = selector.select(typ, model_root=args.model_root, **mk)

    # Check number of elements in each settings are correct
    bits_len_match(model_raw, model_bits[typ]['param_bits'], 'param')
    bits_len_match(model_raw, model_bits[typ]['batch_norm_bits'], 'batch_norm')
    bits_len_match(model_raw, model_bits[typ]['layer_output_bits'], 'layer_output')

    # Load dataset
    val_ds = ds_fetcher(args.batch_size,
                        data_root=args.data_root,
                        train=False,
                        input_size=args.input_size)

    result_filename = f'result'

    for q_method in quant_methods:
        for p_bits in model_bits[typ]['param_bits']:
            for b_bits in model_bits[typ]['batch_norm_bits']:
                for l_bits in model_bits[typ]['layer_output_bits']:
                    model = quantize_model(
                        model_raw, q_method, p_bits, b_bits, l_bits,
                            overflow_rate=args.overflow_rate, n_sample=len(val_ds))

                    start = time.time()
                    acc1, acc5 = misc.eval_model(model, val_ds, is_imagenet=is_imagenet)
                    duration = time.time() - start
                    print(f"{typ}, {q_method}, {p_bits}, {b_bits}, {l_bits}")
                    print(f"Eval duration: {duration}, acc1: {acc1}, acc5: {acc5}")

                    rec = {
                        'type': typ,
                        'quant_method': q_method,
                        'param_bits': p_bits,
                        'batch_norm_bits': b_bits,
                        'layer_output_bits': l_bits,
                        'freq(test/s)': len(val_ds) / duration,
                        'top1': acc1,
                        'top5': acc5,
                    }

                    tmp_df = pd.DataFrame([rec], columns=rec.keys())
                    result = pd.concat([result, tmp_df], ignore_index=True)

                    count += 1
                    if count % 10 == 0:
                        save_result(result, result_filename)

save_result(result, result_filename)

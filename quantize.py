from collections import OrderedDict

import argparse
import time
import torch
import torch.backends.cudnn as cudnn

from utee import misc, quant, selector

default_model_root = "/home/NASICLab/nmsocug1/.torch/models"


def save_model(model, model_name, model_root='{default_model_root}'.format(default_model_root = default_model_root)):
    torch.save(model.state_dict(), '{model_root}{slash}{model_name}'.format(model_root = model_root, slash = '/', model_name = model_name))

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='PyTorch SVHN Example')
parser.add_argument('--type', default='cifar10', help='|'.join(selector.known_models))
parser.add_argument('--quant_method', default='linear', help='linear|minmax|log|tanh')
parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
parser.add_argument('--gpu', default=None, help='index of gpus to use')
parser.add_argument('--ngpu', type=int, default=8, help='number of gpus to use')
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument('--model_root', default='~/.torch/models/', help='folder to save the model')
parser.add_argument('--data_root', default='~/public_dataset/pytorch/', help='folder to save the model')
parser.add_argument('--logdir', default='log/default', help='folder to save to the log')

parser.add_argument('--input_size', type=int, default=224, help='input size of image')
parser.add_argument('--n_sample', type=int, default=99, help='number of samples to infer the scaling factor')
parser.add_argument('--param_bits', type=int, default=8, help='bit-width for parameters (weights and biases)')
parser.add_argument('--bn_bits', type=int, default=32, help='bit-width for batch normalization running mean and std')
parser.add_argument('--fwd_bits', type=int, default=8, help='bit-width for layer output')
parser.add_argument('--overflow_rate', type=float, default=0.0, help='overflow rate')
args = parser.parse_args()

args.gpu = misc.auto_select_gpu(utility_bound=0, num_gpu=args.ngpu, selected_gpus=args.gpu)
args.ngpu = len(args.gpu)
misc.ensure_dir(args.logdir)
args.model_root = misc.expand_user(args.model_root)
args.data_root = misc.expand_user(args.data_root)
args.input_size = 299 if 'inception' in args.type else args.input_size
assert args.quant_method in ['linear', 'minmax', 'log', 'tanh']
print("=================FLAGS==================")
for k, v in args.__dict__.items():
    print('{}: {}'.format(k, v))
print("========================================")

assert torch.cuda.is_available(), 'no cuda'
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# load model and dataset fetcher
model_raw, ds_fetcher, is_imagenet = selector.select(args.type, model_root=args.model_root)
model_quant, ds_fetcher, is_imagenet = selector.select(args.type, model_root=args.model_root)
args.ngpu = args.ngpu if is_imagenet else 1

# Load dataset
val_ds = ds_fetcher(args.batch_size, data_root=args.data_root, train=False, input_size=args.input_size)
val_ds_quant = ds_fetcher(args.batch_size, data_root=args.data_root, train=False, input_size=args.input_size)

# quantize parameters
if args.param_bits < 32:
    state_dict = model_raw.state_dict()
    state_dict_quant = OrderedDict()

    for idx, (k, v) in enumerate(state_dict.items()):
        if 'running' in k:
            if args.bn_bits >=32:
                print("Ignoring {}".format(k))
                state_dict_quant[k] = v
                continue
            else:
                bits = args.bn_bits
        else:
            bits = args.param_bits

        if args.quant_method == 'linear':
            sf = bits - 1. - quant.compute_integral_part(v, overflow_rate=args.overflow_rate)
            v_quant  = quant.linear_quantize(v, sf, bits=bits)
        elif args.quant_method == 'log':
            v_quant = quant.log_minmax_quantize(v, bits=bits)
        elif args.quant_method == 'minmax':
            v_quant = quant.min_max_quantize(v, bits=bits)
        else:
            v_quant = quant.tanh_quantize(v, bits=bits)
        state_dict_quant[k] = v_quant

    model_quant.load_state_dict(state_dict_quant)


# quantize forward activation
if args.fwd_bits < 32:
    model_quant = quant.quantize_model_layer_output(model_quant, bits=args.fwd_bits, overflow_rate=args.overflow_rate,
                                                   counter=args.n_sample, type=args.quant_method)
    if args.fwd_bits <= 16:
        model_quant.half()

    # print(model_quant)
    save_model(model_quant, model_name=args.type+'_quant')


# eval quant model
start = time.time()
acc1, acc5 = misc.eval_model(model_quant, val_ds_quant, ngpu=args.ngpu, is_imagenet=is_imagenet)
duration = time.time() - start
print ('Quant model eval duration: {}'.format(duration))


print(model_quant)
res_str = "type={}, quant_method={}, param_bits={}, bn_bits={}, fwd_bits={}, overflow_rate={}, acc1={:.4f}, acc5={:.4f}".format(
    args.type, args.quant_method, args.param_bits, args.bn_bits, args.fwd_bits, args.overflow_rate, acc1, acc5)
print(res_str)
with open('acc1_acc5.txt', 'a') as f:
    f.write('quant: ' + res_str + '\n')

'''
Copyright (C) 2010-2021 Alibaba Group Holding Limited.
'''


import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse, random, logging, time
import torch
from torch import nn
import numpy as np
import global_utils
# import Masternet
# import PlainNet
import matplotlib.pyplot as plt
# from tqdm import tqdm

# from ZeroShotProxy import compute_zen_score, compute_te_nas_score, compute_syncflow_score, compute_gradnorm_score, compute_NASWOT_score, compute_doc_score, compute_doc_score_hardware
# import benchmark_network_latency
from ZeroShotProxy import compute_TPC_score, compute_TPC_score_fast
from random_string_param import get_FLOPs, get_num_layers, get_model_size
# from timm.models.davit import DaViT_tiny, _create_transformer

working_dir = os.path.dirname(os.path.abspath(__file__))

def parse_cmd_options(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--zero_shot_score', type=str, default=None,
                        help='could be: Zen (for Zen-NAS), TE (for TE-NAS)')
    parser.add_argument('--search_space', type=str, default=None,
                        help='.py file to specify the search space.')
    parser.add_argument('--evolution_max_iter', type=int, default=int(48e4),
                        help='max iterations of evolution.')
    parser.add_argument('--budget_model_size', type=float, default=None, help='budget of model size ( number of parameters), e.g., 1e6 means 1M params')
    parser.add_argument('--budget_flops', type=float, default=None, help='budget of flops, e.g. , 1.8e6 means 1.8 GFLOPS')
    parser.add_argument('--budget_latency', type=float, default=None, help='latency of forward inference per mini-batch, e.g., 1e-3 means 1ms.')
    parser.add_argument('--max_layers', type=int, default=None, help='max number of layers of the network.')
    parser.add_argument('--batch_size', type=int, default=None, help='number of instances in one mini-batch.')
    parser.add_argument('--input_image_size', type=int, default=None,
                        help='resolution of input image, usually 32 for CIFAR and 224 for ImageNet.')
    parser.add_argument('--population_size', type=int, default=512, help='population size of evolution.')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='output directory')
    parser.add_argument('--gamma', type=float, default=1e-2,
                        help='noise perturbation coefficient')
    parser.add_argument('--num_classes', type=int, default=None,
                        help='number of classes')
    parser.add_argument('--dataset', type=str, default=None, help='name of the dataset')

    module_opt, _ = parser.parse_known_args(argv)
    return module_opt

def get_new_random_structure_dict(structure_dict, num_replaces=1, depth=12):
    # bert base
    # initial_structure_dict = dict(depth=12, input_channel=768, SelfAttention_out_channel=[768,768,768,768],
    #                                 BertIntermediate_out_channel=[3072,3072,3072,3072], num_of_heads=[12,12,12,12])

    replace_index = random.randint(0, depth-1) # depth = 4

    SelfAttention_out_channel   = list(structure_dict["SelfAttention_out_channel"])
    BertIntermediate_out_channel    = list(structure_dict["BertIntermediate_out_channel"])
    num_heads    = list(structure_dict["num_of_heads"])
    input_channel = structure_dict["input_channel"]

    # notice embed_dims must be a multiple of A
    num_heads[replace_index]                     = 2**random.randint(3,5) # always = 12
    SelfAttention_out_channel[replace_index]     = max(32, SelfAttention_out_channel[replace_index]+32*random.randint(-2,2))    # multiplies of 32
    # BertIntermediate_out_channel[replace_index]  = max(32, BertIntermediate_out_channel[replace_index]+32*random.randint(-2,2)) # multiplies of 32
    BertIntermediate_out_channel[replace_index]  = SelfAttention_out_channel[replace_index] * 4 # multiplies of 16
    # input_channel                                = max(12, input_channel +12*random.randint(-2,2))
    input_channel = 768 # fix to 768

    structure_dict["num_of_heads"]   = tuple(num_heads)
    structure_dict["BertIntermediate_out_channel"] = tuple(BertIntermediate_out_channel)
    structure_dict["SelfAttention_out_channel"] = tuple(SelfAttention_out_channel)
    structure_dict["input_channel"] = input_channel

    return structure_dict

# def get_latency(AnyPlainNet, random_structure_str, gpu, args):
#     the_model = AnyPlainNet(num_classes=args.num_classes, plainnet_struct=random_structure_str,
#                             no_create=False, no_reslink=False)
#     if gpu is not None:
#         the_model = the_model.cuda(gpu)
#     the_latency = benchmark_network_latency.get_model_latency(model=the_model, batch_size=args.batch_size,
#                                                               resolution=args.input_image_size,
#                                                               in_channels=3, gpu=gpu, repeat_times=1,
#                                                               fp16=True)
#     del the_model
#     torch.cuda.empty_cache()
#     return the_latency

def compute_nas_score(random_structure_str, gpu, args):
    # compute network zero-shot proxy score
    if not args.zero_shot_score == 'TPC_fast':
        the_model = _create_transformer('DaViT_224', pretrained=False, **random_structure_str)
        the_model = the_model.cuda(gpu)

    # try:
    if args.zero_shot_score == 'TPC':
        the_nas_core = compute_TPC_score.compute_nas_score(model=the_model, gpu=gpu, repeat=1)
    elif args.zero_shot_score == 'TPC_fast':
        the_nas_core = compute_TPC_score_fast.compute_nas_score(random_structure_str, args.num_classes)

    if not args.zero_shot_score == 'TPC_fast':
        del the_model
        torch.cuda.empty_cache()

    return the_nas_core

def main(args, argv):
    gpu = args.gpu
    if gpu is not None:
        torch.cuda.set_device('cuda:{}'.format(gpu))
        torch.backends.cudnn.benchmark = True

    best_structure_txt = os.path.join(args.save_dir, 'best_structure.txt')

    if os.path.isfile(best_structure_txt):
        print('skip ' + best_structure_txt)
        return None

    # build a initial dict (bert-based)
    depth = args.max_layers
    initial_structure_dict = dict(depth=depth, input_channel=64, SelfAttention_out_channel=[64]*depth,
                                    BertIntermediate_out_channel=[64]*depth, num_of_heads=[8]*depth)

    popu_structure_list = []
    popu_zero_shot_score_list = []
    popu_latency_list = []

    start_timer = time.time()
    for loop_count in range(args.evolution_max_iter):
        # too many networks in the population pool, remove one with the smallest score
        while len(popu_structure_list) > args.population_size:
            min_zero_shot_score = min(popu_zero_shot_score_list)
            tmp_idx = popu_zero_shot_score_list.index(min_zero_shot_score)
            popu_zero_shot_score_list.pop(tmp_idx)
            popu_structure_list.pop(tmp_idx)
            popu_latency_list.pop(tmp_idx)

        if loop_count >= 1 and loop_count % 1000 == 0:
            max_score = max(popu_zero_shot_score_list)
            min_score = min(popu_zero_shot_score_list)
            elasp_time = time.time() - start_timer
            logging.info(f'loop_count={loop_count}/{args.evolution_max_iter}, max_score={max_score:4g}, min_score={min_score:4g}, time={elasp_time/3600:4g}h')

        # start_timer = time.time()
        # ----- generate a random structure ----- #
        if len(popu_structure_list) <= 10:
            random_structure_str = get_new_random_structure_dict(structure_dict=initial_structure_dict.copy(), num_replaces=1, depth=depth)
        else:
            tmp_idx = random.randint(0, len(popu_structure_list) - 1)
            tmp_random_structure_dict = popu_structure_list[tmp_idx]
            random_structure_str = get_new_random_structure_dict(structure_dict=tmp_random_structure_dict.copy(), num_replaces=2, depth=depth)

        ## num of layer do not change in the case
        # if args.max_layers is not None:
        #     the_layers      = get_num_layers(random_structure_str)
        #     if args.max_layers < the_layers:
        #         continue

        if args.budget_model_size is not None: # only count encoder
            the_model_size  = get_model_size(random_structure_str, num_classes=args.num_classes)
            if args.budget_model_size < the_model_size:
                continue

        if args.budget_flops is not None: # only count encoder
            the_model_flops = get_FLOPs(random_structure_str, num_classes=args.num_classes)
            if args.budget_flops < the_model_flops:
                continue

        the_latency = np.inf
        if args.budget_latency is not None:
            the_latency = get_latency(AnyPlainNet, random_structure_str, gpu, args)
            if args.budget_latency < the_latency:
                continue

        the_nas_core = compute_nas_score(random_structure_str, gpu, args)

        popu_structure_list.append(random_structure_str)
        popu_zero_shot_score_list.append(the_nas_core)
        popu_latency_list.append(the_latency)

    return popu_structure_list, popu_zero_shot_score_list, popu_latency_list

if __name__ == '__main__':
    args = parse_cmd_options(sys.argv)
    log_fn = os.path.join(args.save_dir, 'evolution_search.log')
    global_utils.create_logging(log_fn)

    info = main(args, sys.argv)
    if info is None:
        exit()

    popu_structure_list, popu_zero_shot_score_list, popu_latency_list = info

    # export best structure
    best_score = max(popu_zero_shot_score_list)
    best_idx = popu_zero_shot_score_list.index(best_score)
    best_structure_str = popu_structure_list[best_idx]
    the_latency = popu_latency_list[best_idx]

    best_structure_txt = os.path.join(args.save_dir, 'best_structure.txt')
    global_utils.mkfilepath(best_structure_txt)
    # with open(best_structure_txt, 'w') as fid:
    #     fid.write(best_structure_str)

    import json
    with open(best_structure_txt, 'w') as convert_file:
        convert_file.write(json.dumps(best_structure_str))

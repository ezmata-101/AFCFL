#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse


#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse

def _pbool(x):
    """Parse flexible booleans: 1/0, true/false, yes/no, on/off (case-insensitive)."""
    s = str(x).strip().lower()
    if s in ('1','true','t','yes','y','on'):
        return True
    if s in ('0','false','f','no','n','off'):
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean, got: {x}")

def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--epochs', type=int, default=200, help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=100, help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.1, help='the fraction of clients: C')
    parser.add_argument('--local_ep', type=int, default=10, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=10, help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.5, help='SGD momentum (default: 0.5)')

    # model arguments
    parser.add_argument('--model', type=str, default='mlp', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9, help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5', help='comma-separated kernel sizes for conv')
    parser.add_argument('--num_channels', type=int, default=1, help="number of channels of imgs")
    parser.add_argument('--norm', type=str, default='batch_norm', help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32, help="number of filters for conv nets")
    parser.add_argument('--max_pool', type=str, default='True', help="Whether use max pooling rather than strided convs")

    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--gpu', type=int, default=0, help='GPU id (e.g., 0). Omit for CPU.')
    parser.add_argument('--optimizer', type=str, default='sgd', help="type of optimizer")
    parser.add_argument('--iid', type=int, default=1, help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--unequal', type=int, default=0, help='use unequal data splits for non-i.i.d (0 = equal)')
    parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
    parser.add_argument('--verbose', type=int, default=1, help='verbose')
    parser.add_argument('--seed', type=int, default=1, help='random seed')

    parser.add_argument('--fcfl', type=int, default=0, help='whether to use FCFL')
    parser.add_argument('--fcfl_alpha', type=float, default=0.5, help='alpha in FCFL')
    parser.add_argument('--fcfl_r', type=float, default=0.3, help='fraction r in FCFL client selection')

    parser.add_argument('--clients_per_round', type=int, default=None,
                        help='Override clients per round m; if unset, use frac*num_users')

    # adaptive alpha in FCFL
    parser.add_argument('--fcfl_adaptive_alpha', type=_pbool, default=False,
                        help='use adaptive alpha in FCFL (true/false, 1/0, yes/no, on/off)')
    parser.add_argument('--fcfl_alpha_min', type=float, default=0.1, help='minimum adaptive alpha')
    parser.add_argument('--fcfl_alpha_max', type=float, default=1.0, help='maximum adaptive alpha')
    parser.add_argument('--fcfl_alpha_beta', type=float, default=0.9, help='smoothing for adaptive alpha')
    parser.add_argument('--fcfl_alpha_warmup', type=float, default=0, help='warmup rounds for adaptive alpha')

    # adaptive r in FCFL
    parser.add_argument('--fcfl_adaptive_r', type=_pbool, default=False,
                        help='use adaptive r in FCFL (true/false, 1/0, yes/no, on/off)')
    parser.add_argument('--fcfl_r_min', type=float, default=0.1, help='minimum adaptive r')
    parser.add_argument('--fcfl_r_max', type=float, default=0.9, help='maximum adaptive r')

    # print selection flag
    parser.add_argument('--fcfl_print_selection', type=_pbool, default=False,
                        help="Print Q values and selected client IDs each round (true/false)")

    args = parser.parse_args()
    return args

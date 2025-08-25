#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from utils import get_dataset  # (we'll do our own weighted agg here)

# ---------------------------
# Weighted aggregation helper
# ---------------------------
def weighted_average_weights(local_weights, client_ids, weights_map):
    """
    local_weights: list[state_dict]
    client_ids:    list[int] aligned with local_weights
    weights_map:   dict[int -> float], must sum to 1 over selected client_ids
    """
    avg_w = copy.deepcopy(local_weights[0])
    # zero-out
    for k in avg_w.keys():
        avg_w[k].zero_()
    # accumulate
    for sd, cid in zip(local_weights, client_ids):
        w = float(weights_map[cid])
        for k in avg_w.keys():
            avg_w[k] += w * sd[k]
    return avg_w


if __name__ == '__main__':
    start_time = time.time()

    # paths & logger
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    # ---------------------------
    # Parse args & device
    # ---------------------------
    args = args_parser()

    # Optional FCFL flags (safe defaults if not present in options.py)
    if not hasattr(args, 'fcfl'):
        args.fcfl = 1
    if not hasattr(args, 'fcfl_alpha'):
        args.fcfl_alpha = 0.5   # α
    if not hasattr(args, 'fcfl_r'):
        args.fcfl_r = 0.2       # random fraction r in selection

    # Normalize r
    args.fcfl_r = max(0.0, min(0.99, float(args.fcfl_r)))

    # GPU selection
    if args.gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu))
    else:
        args.gpu = None
    device = torch.device(f"cuda:{0 if args.gpu is None else args.gpu}" if torch.cuda.is_available() else "cpu")
    print("DEVICE:", device)

    # ---------------------------
    # Data & model
    # ---------------------------
    train_dataset, test_dataset, user_groups = get_dataset(args)

    # Build model
    if args.model == 'cnn':
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args)
        else:
            raise ValueError("Unsupported dataset for cnn")
    elif args.model == 'mlp':
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
        global_model = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes)
    else:
        raise ValueError("Unrecognized model")

    global_model.to(device)
    global_model.train()
    print(global_model)
    print("Global model on:", next(global_model.parameters()).device)

    # ---------------------------
    # FCFL server state
    # ---------------------------
    N = args.num_users
    m = max(int(args.frac * N), 1)

    # data sizes per client (for t=0 weighting)
    n_i = np.array([len(user_groups[i]) for i in range(N)], dtype=float)

    # unfairness queues & bookkeeping
    Q      = np.zeros(N, dtype=float)  # Q_i(t)
    omega  = np.zeros(N, dtype=float)  # last-round weights for penalty
    x_prev = np.zeros(N, dtype=int)    # selection indicator at t-1
    Acc_hat_t = 0.0                    # estimated global accuracy at round t

    # Training logs
    train_loss, train_accuracy = [], []
    print_every = 2

    # ---------------------------
    # Training rounds
    # ---------------------------
    for epoch in tqdm(range(args.epochs)):
        print(f'\n | Global Training Round : {epoch+1} |')

        # 1) All clients evaluate CURRENT global model on their local test split: Acc_t[i]
        Acc_t = np.zeros(N, dtype=float)
        for i in range(N):
            lu = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[i], logger=None)
            acc_i, _ = lu.inference(model=global_model)  # inference moves model to correct device
            Acc_t[i] = acc_i

        # 2) Update unfairness queues (skip at t=0)
        if args.fcfl and epoch > 0:
            u_ft = np.maximum(Acc_hat_t - Acc_t, 0.0)  # unfairness level u^{f}_{t,i}
            Q = np.maximum(Q + args.fcfl_alpha * u_ft - omega * x_prev, 0.0)
        else:
            # keep Q as zeros at t=0 or when FCFL disabled
            pass

        # 3) Select m clients
        if (not args.fcfl) or (epoch == 0 and np.allclose(Q, 0)):
            idxs_users = np.random.choice(range(N), m, replace=False)
        else:
            k_rand = int(round(args.fcfl_r * m))
            rand_part = set(np.random.choice(range(N), k_rand, replace=False)) if k_rand > 0 else set()
            rest = [i for i in range(N) if i not in rand_part]
            top_part = sorted(rest, key=lambda i: Q[i], reverse=True)[: (m - k_rand)]
            idxs_users = np.array(list(rand_part.union(top_part)))

        # 4) Aggregation weights for THIS round
        if np.allclose(Q[idxs_users], 0):
            # Cold start → data-size weights
            w_sel = n_i[idxs_users] / (n_i[idxs_users].sum() if n_i[idxs_users].sum() > 0 else 1.0)
        else:
            w_sel = Q[idxs_users] / (Q[idxs_users].sum() if Q[idxs_users].sum() > 0 else 1.0)

        # keep omega and x_prev for penalty next round
        omega = np.zeros(N, dtype=float)
        omega[idxs_users] = w_sel
        x_prev = np.zeros(N, dtype=int)
        x_prev[idxs_users] = 1

        # 5) Local training on selected clients; also collect local training accuracy of updated locals
        local_weights, client_ids, local_train_acc = [], [], []
        local_losses = []

        for cid in idxs_users:
            lu = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[cid], logger=logger)
            w_i, loss_i = lu.update_weights(model=copy.deepcopy(global_model), global_round=epoch)
            local_losses.append(loss_i)

            # Evaluate the UPDATED local model on that client's test split
            tmp = copy.deepcopy(global_model)
            tmp.load_state_dict(w_i)
            acc_i, _ = lu.inference(model=tmp)

            local_weights.append(w_i)
            client_ids.append(cid)
            local_train_acc.append(acc_i)

        # 6) Aggregate with FCFL weights
        weights_map = {cid: float(w) for cid, w in zip(client_ids, w_sel)}
        global_weights = weighted_average_weights(local_weights, client_ids, weights_map)
        global_model.load_state_dict(global_weights)

        # 7) Estimate global accuracy for next round using selected clients
        Acc_hat_t = float(np.sum(w_sel * np.array(local_train_acc))) if len(local_train_acc) else 0.0

        # Logging (loss & accuracy proxy)
        loss_avg = float(np.mean(local_losses)) if len(local_losses) else 0.0
        train_loss.append(loss_avg)

        # average of current-round client accuracies (pre-update), just for display
        train_accuracy.append(float(np.mean(Acc_t)))

        if (epoch + 1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss)):.6f}')
            print('Proxy Train Accuracy (mean Acc_t): {:.2f}%'.format(100 * train_accuracy[-1]))

    # ---------------------------
    # Final test
    # ---------------------------
    test_acc, test_loss = test_inference(args, global_model, test_dataset)

    print(f'\n Results after {args.epochs} global rounds of training:')
    if len(train_accuracy):
        print("|---- Avg (proxy) Train Accuracy: {:.2f}%".format(100 * train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100 * test_acc))

    # ---------------------------
    # Save curves
    # ---------------------------
    os.makedirs('../save/objects', exist_ok=True)
    tag = 'fcfl' if args.fcfl else 'fedavg'
    file_name = '../save/objects/{}-{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_alpha[{}]_r[{}].pkl'.format(
        tag, args.dataset, args.model, args.epochs, args.frac, args.iid, args.local_ep, args.local_bs,
        args.fcfl_alpha, args.fcfl_r
    )
    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, train_accuracy], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time() - start_time))

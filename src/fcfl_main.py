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
from utils import get_dataset


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
    for k in avg_w.keys():
        avg_w[k].zero_()
    for sd, cid in zip(local_weights, client_ids):
        w = float(weights_map[cid])
        for k in avg_w.keys():
            avg_w[k] += w * sd[k]
    return avg_w


if __name__ == '__main__':
    np.random.seed(42)
    torch.manual_seed(42)
    start_time = time.time()

    # paths & logger
    os.makedirs('../logs', exist_ok=True)
    logger = SummaryWriter('../logs')

    # ---------------------------
    # Parse args & device
    # ---------------------------
    args = args_parser()

    # Optional flags (safe defaults if not present in options.py)
    if not hasattr(args, 'fcfl'):
        args.fcfl = 1
    if not hasattr(args, 'fcfl_alpha'):
        args.fcfl_alpha = 0.5
    if not hasattr(args, 'fcfl_r'):
        args.fcfl_r = 0.7     # fraction selected by Q
    if not hasattr(args, 'clients_per_round'):
        args.clients_per_round = None  # override m; else use frac*num_users

    # clamp to sane ranges
    args.fcfl_r = float(max(0.0, min(1.0, args.fcfl_r)))

    # Device
    if args.gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu))
    else:
        args.gpu = None
    device = torch.device(f"cuda:{0 if args.gpu is None else args.gpu}" if torch.cuda.is_available() else "cpu")
    tqdm.write(f"DEVICE: {device}")

    # ---------------------------
    # Data & model
    # ---------------------------
    train_dataset, test_dataset, user_groups = get_dataset(args)

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
    tqdm.write(str(global_model))
    tqdm.write(f"Global model on: {next(global_model.parameters()).device}")

    # ---------------------------
    # FCFL server state
    # ---------------------------
    N = args.num_users

    # m: number of clients per round
    if args.clients_per_round is not None and int(args.clients_per_round) > 0:
        m = min(int(args.clients_per_round), N)
    else:
        m = max(int(round(args.frac * N)), 1)

    n_i = np.array([len(user_groups[i]) for i in range(N)], dtype=float)

    Q      = np.zeros(N, dtype=float)  # queues
    omega  = np.zeros(N, dtype=float)  # last-round weights (penalty term)
    x_prev = np.zeros(N, dtype=int)    # last-round selection indicator
    Acc_hat_t = 0.0                    # estimated global acc from previous round

    train_loss, train_accuracy = [], []
    print_every = 2

    # clear log file
    with open('../logs/fcfl_round_log.tsv', 'w') as f:
        f.write("round\tQ_all\tselected_by_Q\tselected_random\tselected_all\tweights\tQ_selected\n")

    # ---------------------------
    # Training rounds
    # ---------------------------
    for epoch in tqdm(range(args.epochs)):
        # 1) Evaluate current global model on all clients (pre-update accuracies)
        Acc_t = np.zeros(N, dtype=float)
        for i in range(N):
            lu = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[i], logger=None)
            acc_i, _ = lu.inference(model=global_model)
            Acc_t[i] = acc_i

        # 2) Update Q (skip at cold start)
        if args.fcfl and epoch > 0:
            u_ft = np.maximum(Acc_hat_t - Acc_t, 0.0)
            Q = np.maximum(Q + args.fcfl_alpha * u_ft - omega * x_prev, 0.0)

        # ---- PRINT full Q values (id:value) ----
        q_pairs = [(int(i), float(Q[i])) for i in range(N)]
        # Print the Q values for each client in sorted order of q values(desc)
        q_pairs_sorted = sorted(q_pairs, key=lambda x: x[1], reverse=True)
        tqdm.write(f"[Round {epoch+1:03d}] Q per client: {q_pairs_sorted}")

        # 3) Select m clients
        if (not args.fcfl) or (epoch == 0 and np.allclose(Q, 0)):
            # random selection (cold start or FCFL disabled)
            idxs_users = np.random.choice(range(N), m, replace=False)
            selected_by_Q   = np.array([], dtype=int)
            selected_random = np.sort(np.array(idxs_users))
            selected_all    = selected_random
            tqdm.write(f"[Round {epoch+1:03d}] selected_by_Q   (0): []")
            tqdm.write(f"[Round {epoch+1:03d}] selected_random ({len(selected_random)}): {selected_random.tolist()}")
        else:
            # FCFL selection:
            #   k_q = floor(fcfl_r * m) by Q
            #   k_rand = m - k_q randomly from the remainder
            k_q = int(np.round(args.fcfl_r * m))
            k_q = max(0, min(k_q, m))
            k_rand = m - k_q

            all_ids = np.arange(N, dtype=int)
            # top-k by Q (desc)
            top_order = np.argsort(-Q)  # indices sorted by descending Q
            top_part = top_order[:k_q].astype(int)

            # random part from remaining pool
            remaining = np.setdiff1d(all_ids, top_part, assume_unique=False)
            rand_part = np.random.choice(remaining, size=k_rand, replace=False) if k_rand > 0 else np.array([], dtype=int)

            idxs_users = np.sort(np.concatenate([top_part, rand_part]).astype(int))
            selected_by_Q   = np.sort(top_part.astype(int))
            selected_random = np.sort(rand_part.astype(int))

            tqdm.write(f"[Round {epoch+1:03d}] m={m}, fcfl_r={args.fcfl_r} → by_Q={len(selected_by_Q)}, random={len(selected_random)}")
            tqdm.write(f"[Round {epoch+1:03d}] selected_by_Q   ({len(selected_by_Q)}): {selected_by_Q.tolist()}")
            tqdm.write(f"[Round {epoch+1:03d}] selected_random ({len(selected_random)}): {selected_random.tolist()}")

        selected_all = np.sort(idxs_users)
        tqdm.write(f"[Round {epoch+1:03d}] selected_all    ({len(selected_all)}): {selected_all.tolist()}")

        # 4) Aggregation weights
        if np.allclose(Q[idxs_users], 0):
            w_sel = n_i[idxs_users] / (n_i[idxs_users].sum() if n_i[idxs_users].sum() > 0 else 1.0)
        else:
            w_sel = Q[idxs_users] / (Q[idxs_users].sum() if Q[idxs_users].sum() > 0 else 1.0)

        order = np.argsort(idxs_users)
        weights_pairs = [(int(idxs_users[o]), float(w_sel[o])) for o in order]
        q_selected_pairs = [(int(idxs_users[o]), float(Q[int(idxs_users[o])])) for o in order]
        tqdm.write(f"[Round {epoch+1:03d}] agg weights (id, weight): {weights_pairs}")
        tqdm.write(f"[Round {epoch+1:03d}] Q(selected)           : {q_selected_pairs}")

        # log to TSV
        with open('../logs/fcfl_round_log.tsv', 'a') as f:
            sel_all_sorted = idxs_users[order]
            w_sorted = w_sel[order]
            q_sel_sorted = [Q[int(i)] for i in sel_all_sorted]
            f.write(
                f"{epoch+1}\t"
                f"{';'.join(f'{i}:{float(q):.6f}' for i, q in q_pairs)}\t"
                f"{','.join(map(str, selected_by_Q.tolist()))}\t"
                f"{','.join(map(str, selected_random.tolist()))}\t"
                f"{','.join(map(str, sel_all_sorted.tolist()))}\t"
                f"{','.join(f'{float(w):.6f}' for w in w_sorted)}\t"
                f"{','.join(f'{float(q):.6f}' for q in q_sel_sorted)}\n"
            )

        # update omega/x_prev for next round penalty
        omega = np.zeros(N, dtype=float)
        omega[idxs_users] = w_sel
        x_prev = np.zeros(N, dtype=int)
        x_prev[idxs_users] = 1

        # 5) Local training
        local_weights, client_ids, local_train_acc = [], [], []
        local_losses = []
        for cid in idxs_users:
            lu = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[cid], logger=logger)
            w_i, loss_i = lu.update_weights(model=copy.deepcopy(global_model), global_round=epoch)
            local_losses.append(loss_i)

            tmp = copy.deepcopy(global_model)
            tmp.load_state_dict(w_i)
            acc_i, _ = lu.inference(model=tmp)

            local_weights.append(w_i)
            client_ids.append(cid)
            local_train_acc.append(acc_i)

        # 6) Aggregate
        weights_map = {cid: float(w) for cid, w in zip(client_ids, w_sel)}
        global_weights = weighted_average_weights(local_weights, client_ids, weights_map)
        global_model.load_state_dict(global_weights)

        # 7) Update Acc_hat for next round
        Acc_hat_t = float(np.sum(w_sel * np.array(local_train_acc))) if len(local_train_acc) else 0.0

        # Logs
        loss_avg = float(np.mean(local_losses)) if len(local_losses) else 0.0
        train_loss.append(loss_avg)
        train_accuracy.append(float(np.mean(Acc_t)))

        if (epoch + 1) % print_every == 0:
            tqdm.write(f'Avg Training Stats after {epoch+1} global rounds:')
            tqdm.write(f'  Training Loss : {np.mean(np.array(train_loss)):.6f}')
            tqdm.write(f'  Proxy Train Accuracy (mean Acc_t): {100 * train_accuracy[-1]:.2f}%')

    # ---------------------------
    # Final test
    # ---------------------------
    test_acc, test_loss = test_inference(args, global_model, test_dataset)
    tqdm.write(f'\nResults after {args.epochs} global rounds of training:')
    if len(train_accuracy):
        tqdm.write("|---- Avg (proxy) Train Accuracy: {:.2f}%".format(100 * train_accuracy[-1]))
    tqdm.write("|---- Test Accuracy: {:.2f}%".format(100 * test_acc))

    # ---------------------------
    # Save curves
    # ---------------------------
    os.makedirs('../save/objects', exist_ok=True)
    tag = 'fcfl' if args.fcfl else 'fedavg'
    file_name = '../save/objects/{}-{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_alpha[{}]_r[{}]_m[{}].pkl'.format(
        tag, args.dataset, args.model, args.epochs, args.frac, args.iid,
        args.local_ep, args.local_bs, args.fcfl_alpha, args.fcfl_r, m
    )
    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, train_accuracy], f)

    tqdm.write('\nTotal Run Time: {0:0.4f}'.format(time.time() - start_time))

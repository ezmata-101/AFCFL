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
# Small utilities
# ---------------------------
def parse_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    return s in ("1", "true", "yes", "y", "on")


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


# ---------------------------
# Fairness signal (no Jain): accuracy gap
# g_t ∈ [0,1], higher => less fair
# ---------------------------
def fairness_signal_gap(acc_vec: np.ndarray, gap_norm: float = 0.15) -> float:
    """
    signal = mean(acc) - min(acc), normalized by gap_norm (e.g., 0.15 = 15 percentage points).
    """
    acc = np.asarray(acc_vec, dtype=float)
    if acc.size == 0:
        return 0.0
    gap = float(np.mean(acc) - np.min(acc))
    g = gap / max(1e-12, gap_norm)
    return float(np.clip(g, 0.0, 1.0))


# ---------------------------
# Adaptive alpha (modular)
# ---------------------------
# Internal shaping & smoothing constants (no flags)
_ALPHA_GAIN   = 1.0   # amplify unfairness
_ALPHA_GAMMA  = 1.0   # concave boost (<1) for small signals
_ALPHA_BETA   = 0.6   # EMA smoothing factor
_ALPHA_WARMUP = 1     # rounds with no EMA (take raw)

def _shape_signal(g: float) -> float:
    g = float(np.clip(g, 0.0, 1.0))
    g_pow = g ** _ALPHA_GAMMA
    g_amp = _ALPHA_GAIN * g_pow
    return float(np.clip(g_amp, 0.0, 1.0))

def compute_adaptive_alpha(prev_alpha: float,
                           acc_vec: np.ndarray,
                           epoch: int,
                           alpha_min: float,
                           alpha_max: float) -> tuple:
    """
    Returns (alpha_t, alpha_raw, g_t, g_shaped)
    - g_t uses the simple accuracy-gap signal
    - alpha_raw = alpha_min + (alpha_max - alpha_min) * g_shaped
    - alpha_t is EMA smoothed after warmup
    """
    g_t = fairness_signal_gap(acc_vec, gap_norm=0.15)  # 15 percentage points on [0,1] scale
    g_shaped = _shape_signal(g_t)
    alpha_raw = alpha_min + (alpha_max - alpha_min) * g_shaped

    if epoch < _ALPHA_WARMUP:
        alpha_t = alpha_raw
    else:
        alpha_t = (1.0 - _ALPHA_BETA) * float(prev_alpha) + _ALPHA_BETA * float(alpha_raw)

    alpha_t = float(np.clip(alpha_t, alpha_min, alpha_max))
    return alpha_t, float(alpha_raw), float(g_t), float(g_shaped)


# ---------------------------
# Client selection & weights
# ---------------------------
def select_clients(Q: np.ndarray, N: int, m: int, fcfl_enabled: bool, fcfl_r: float, cold_start: bool):
    """
    Return (idxs_users_sorted, selected_by_Q_sorted, selected_random_sorted).
    If cold_start or not fcfl, selection is purely random.
    Otherwise: top-k by Q plus random remainder.
    """
    if (not fcfl_enabled) or cold_start:
        idxs_users = np.random.choice(range(N), m, replace=False)
        sel_by_q = np.array([], dtype=int)
        sel_rand = np.sort(np.array(idxs_users, dtype=int))
        return np.sort(np.array(idxs_users, dtype=int)), sel_by_q, sel_rand

    k_q = int(np.round(float(fcfl_r) * m))
    k_q = max(0, min(k_q, m))
    k_rand = m - k_q

    all_ids = np.arange(N, dtype=int)
    top_order = np.argsort(-Q)  # descending Q
    top_part = top_order[:k_q].astype(int)

    remaining = np.setdiff1d(all_ids, top_part, assume_unique=False)
    rand_part = np.random.choice(remaining, size=k_rand, replace=False) if k_rand > 0 else np.array([], dtype=int)

    idxs_users = np.sort(np.concatenate([top_part, rand_part]).astype(int))
    sel_by_q = np.sort(top_part.astype(int))
    sel_rand = np.sort(rand_part.astype(int))
    return idxs_users, sel_by_q, sel_rand

def compute_round_weights(Q: np.ndarray, idxs_users: np.ndarray, n_i: np.ndarray) -> np.ndarray:
    """
    Aggregation weights: data-size if selected Q are all zero, else proportional to Q.
    Returns w_sel aligned to idxs_users order.
    """
    if np.allclose(Q[idxs_users], 0):
        denom = n_i[idxs_users].sum()
        return n_i[idxs_users] / (denom if denom > 0 else 1.0)
    denom = Q[idxs_users].sum()
    return Q[idxs_users] / (denom if denom > 0 else 1.0)


# ======================================================================
# Main (kept close to fcfl_main.py)
# ======================================================================
if __name__ == '__main__':
    # Reproducibility (optional)
    np.random.seed(42)
    torch.manual_seed(42)

    start_time = time.time()
    os.makedirs('../logs', exist_ok=True)
    logger = SummaryWriter('../logs')

    # ---------------------------
    # Parse args & device
    # ---------------------------
    args = args_parser()

    # Only the minimal extra flags you asked for:
    #   --fcfl_r
    #   --fcfl_adaptive_alpha  (true/false)
    #   --fcfl_alpha_min
    #   --fcfl_alpha_max
    if not hasattr(args, 'fcfl'):
        args.fcfl = 1
    if not hasattr(args, 'fcfl_r'):
        args.fcfl_r = 0.7
    if not hasattr(args, 'fcfl_adaptive_alpha'):
        args.fcfl_adaptive_alpha = True
    if not hasattr(args, 'fcfl_alpha_min'):
        args.fcfl_alpha_min = 0.10
    if not hasattr(args, 'fcfl_alpha_max'):
        args.fcfl_alpha_max = 0.80

    args.fcfl_r = float(np.clip(args.fcfl_r, 0.0, 1.0))
    args.fcfl_adaptive_alpha = parse_bool(args.fcfl_adaptive_alpha)
    args.fcfl_alpha_min = float(max(0.0, args.fcfl_alpha_min))
    args.fcfl_alpha_max = float(min(1.0, max(args.fcfl_alpha_min, args.fcfl_alpha_max)))

    # Device
    if args.gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu))
    else:
        args.gpu = None
    device = torch.device(f"cuda:{0 if args.gpu is None else args.gpu}" if torch.cuda.is_available() else "cpu")
    tqdm.write(f"DEVICE: {device}")

    args.device = device


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
    m = max(int(round(args.frac * N)), 1)  # keep like fcfl_main.py

    n_i = np.array([len(user_groups[i]) for i in range(N)], dtype=float)

    Q      = np.zeros(N, dtype=float)  # queues
    omega  = np.zeros(N, dtype=float)  # last-round weights (penalty term)
    x_prev = np.zeros(N, dtype=int)    # last-round selection indicator
    Acc_hat_t = 0.0                    # estimated global acc from previous round

    # Initialize α at midpoint
    alpha_t = 0.5 * (args.fcfl_alpha_min + args.fcfl_alpha_max)

    train_loss, train_accuracy = [], []
    print_every = 2

    # clear log file (includes alpha info)
    with open('../logs/fcfl_round_log.tsv', 'w') as f:
        f.write("round\talpha_raw\talpha_t\tsignal\tshaped\tQ_all\tselected_by_Q\tselected_random\tselected_all\tweights\tQ_selected\n")

    # ---------------------------
    # Training rounds
    # ---------------------------
    for epoch in tqdm(range(args.epochs)):
        # 1) Evaluate current global model on all clients (pre-update accuracies)
        Acc_t = np.zeros(N, dtype=float)
        for i in range(N):
            lu = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[i], logger=None)
            acc_i, _ = lu.inference(model=global_model)
            Acc_t[i] = acc_i  # expected in [0,1]

        # 2) Compute alpha (adaptive or fixed midpoint)
        if args.fcfl_adaptive_alpha:
            alpha_t, alpha_raw, g_out, g_shaped = compute_adaptive_alpha(
                prev_alpha=alpha_t, acc_vec=Acc_t, epoch=epoch,
                alpha_min=args.fcfl_alpha_min, alpha_max=args.fcfl_alpha_max
            )
        else:
            # Use fixed midpoint of the provided min/max
            g_out = 0.0
            g_shaped = 0.0
            alpha_raw = alpha_t  # keep reporting the constant
            # alpha_t remains constant (midpoint)

        tqdm.write(f"[Round {epoch+1:03d}] α(raw)={alpha_raw:.4f}, α(t)={alpha_t:.4f}, signal={g_out:.4f}, shaped={g_shaped:.4f}")

        # 3) Update Q (skip at cold start)
        if args.fcfl and epoch > 0:
            u_ft = np.maximum(Acc_hat_t - Acc_t, 0.0)             # unfairness level per client
            Q = np.maximum(Q + alpha_t * u_ft - omega * x_prev, 0.0)

        # ---- PRINT full Q values (sorted desc) ----
        q_pairs = [(int(i), float(Q[i])) for i in range(N)]
        q_pairs_sorted = sorted(q_pairs, key=lambda x: x[1], reverse=True)
        tqdm.write(f"[Round {epoch+1:03d}] Q per client (desc): {q_pairs_sorted}")

        # 4) Select m clients
        cold_start = (epoch == 0 and np.allclose(Q, 0))
        idxs_users, selected_by_Q, selected_random = select_clients(
            Q=Q, N=N, m=m, fcfl_enabled=bool(args.fcfl), fcfl_r=float(args.fcfl_r), cold_start=cold_start
        )
        if cold_start or (not args.fcfl):
            tqdm.write(f"[Round {epoch+1:03d}] selected_by_Q   (0): []")
        else:
            tqdm.write(f"[Round {epoch+1:03d}] m={m}, fcfl_r={args.fcfl_r} → by_Q={len(selected_by_Q)}, random={len(selected_random)}")
            tqdm.write(f"[Round {epoch+1:03d}] selected_by_Q   ({len(selected_by_Q)}): {selected_by_Q.tolist()}")
        tqdm.write(f"[Round {epoch+1:03d}] selected_random ({len(selected_random)}): {selected_random.tolist()}")
        tqdm.write(f"[Round {epoch+1:03d}] selected_all    ({len(idxs_users)}): {idxs_users.tolist()}")

        # 5) Aggregation weights
        w_sel = compute_round_weights(Q=Q, idxs_users=idxs_users, n_i=n_i)
        order = np.argsort(idxs_users)
        weights_pairs = [(int(idxs_users[o]), float(w_sel[o])) for o in order]
        q_selected_pairs = [(int(idxs_users[o]), float(Q[int(idxs_users[o])])) for o in order]
        tqdm.write(f"[Round {epoch+1:03d}] agg weights (id, weight): {weights_pairs}")
        tqdm.write(f"[Round {epoch+1:03d}] Q(selected)           : {q_selected_pairs}")

        # log to TSV (includes alpha info)
        with open('../logs/fcfl_round_log.tsv', 'a') as f:
            sel_all_sorted = idxs_users[order]
            w_sorted = w_sel[order]
            q_sel_sorted = [Q[int(i)] for i in sel_all_sorted]
            f.write(
                f"{epoch+1}\t"
                f"{alpha_raw:.6f}\t{alpha_t:.6f}\t{g_out:.6f}\t{g_shaped:.6f}\t"
                f"{';'.join(f'{i}:{float(q):.6f}' for i, q in q_pairs_sorted)}\t"
                f"{','.join(map(str, selected_by_Q.tolist()))}\t"
                f"{','.join(map(str, selected_random.tolist()))}\t"
                f"{','.join(map(str, sel_all_sorted.tolist()))}\t"
                f"{','.join(f'{float(w):.6f}' for w in w_sorted)}\t"
                f"{','.join(f'{float(q):.6f}' for q in q_sel_sorted)}\n"
            )

        # Update omega/x_prev for penalty next round
        omega = np.zeros(N, dtype=float)
        omega[idxs_users] = w_sel
        x_prev = np.zeros(N, dtype=int)
        x_prev[idxs_users] = 1

        # 6) Local training
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

        # 7) Aggregate
        weights_map = {cid: float(w) for cid, w in zip(client_ids, w_sel)}
        global_weights = weighted_average_weights(local_weights, client_ids, weights_map)
        global_model.load_state_dict(global_weights)

        # 8) Estimate global acc for next round
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
    tag = 'fcfl_aa_minimal'
    file_name = '../save/objects/{}-{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_r[{}]_alpha[{:.2f}-{:.2f}].pkl'.format(
        tag, args.dataset, args.model, args.epochs, args.frac, args.iid,
        args.local_ep, args.local_bs, args.fcfl_r, args.fcfl_alpha_min, args.fcfl_alpha_max
    )
    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, train_accuracy], f)

    tqdm.write('\nTotal Run Time: {0:0.4f}'.format(time.time() - start_time))

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


from torch.utils.data import Dataset
import torch

class DatasetSplit(Dataset):
    """Subset wrapper that accepts ints, lists (even nested), or tensors of indices."""
    def __init__(self, dataset, idxs):
        self.dataset = dataset

        # Normalize to a flat List[int]
        if torch.is_tensor(idxs):
            idxs = idxs.view(-1).tolist()
        else:
            try:
                idxs = list(idxs)
            except TypeError:
                idxs = [idxs]

            flat = []
            for i in idxs:
                if torch.is_tensor(i):
                    flat.extend(i.view(-1).tolist())
                elif isinstance(i, (list, tuple)):
                    flat.extend(int(j) for j in i)
                else:
                    flat.append(int(i))
            idxs = flat

        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        # return original (image, label) as provided by the base dataset
        return self.dataset[self.idxs[item]]

class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, logger):
        self.args = args
        self.logger = logger
        self.dataset = dataset
        self.idxs = list(idxs)

        # Choose device once per client
        self.device = torch.device(
            f"cuda:{args.gpu}" if (args.gpu is not None and torch.cuda.is_available()) else "cpu"
        )

        # Build loaders from the provided indices
        self.trainloader, self.validloader, self.testloader = self.train_val_test(self.dataset, self.idxs)

        # ---- choose ONE loss according to your model's final layer ----
        self.criterion = nn.CrossEntropyLoss().to(self.device)   # use if model returns logits
        # self.criterion = nn.NLLLoss().to(self.device)          # use if model returns log-probs (LogSoftmax)

    def train_val_test(self, dataset, idxs):
        """Create train/val/test splits from idxs and return DataLoaders."""
        # Flatten indices to plain Python ints
        if torch.is_tensor(idxs):
            idxs = idxs.view(-1).tolist()
        else:
            idxs = [int(i) for i in idxs]

        n = len(idxs)
        n_train = int(0.8 * n)
        n_val   = int(0.1 * n)
        idxs_train = idxs[:n_train]
        idxs_val   = idxs[n_train:n_train + n_val]
        idxs_test  = idxs[n_train + n_val:]

        use_cuda = (self.device.type == "cuda")
        trainloader = DataLoader(
            DatasetSplit(dataset, idxs_train),
            batch_size=self.args.local_bs,
            shuffle=True,
            pin_memory=use_cuda,
        )
        validloader = DataLoader(
            DatasetSplit(dataset, idxs_val),
            batch_size=max(1, int(len(idxs_val) / 10)) if len(idxs_val) else 1,
            shuffle=False,
            pin_memory=use_cuda,
        )
        testloader = DataLoader(
            DatasetSplit(dataset, idxs_test),
            batch_size=max(1, int(len(idxs_test) / 10)) if len(idxs_test) else 1,
            shuffle=False,
            pin_memory=use_cuda,
        )
        return trainloader, validloader, testloader

    def update_weights(self, model, global_round=None, **kwargs):
        """Local training; returns (updated_state_dict, avg_loss)."""
        model.to(self.device).train()

        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr, momentum=0.5)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=1e-4)

        epoch_losses = []
        non_block = (self.device.type == "cuda")

        for _ in range(self.args.local_ep):
            batch_losses = []
            for images, labels in self.trainloader:
                images = images.to(self.device, non_blocking=non_block)
                labels = labels.to(self.device, non_blocking=non_block)

                optimizer.zero_grad(set_to_none=True)
                outputs = model(images)                 # logits or log-probs (match criterion above)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                if self.logger is not None:
                    self.logger.add_scalar('loss', loss.item())
                batch_losses.append(loss.item())
            epoch_losses.append(sum(batch_losses) / max(1, len(batch_losses)))

        return model.state_dict(), sum(epoch_losses) / max(1, len(epoch_losses))

    def inference(self, model):
        """Evaluate on the client's test split; returns (acc, avg_loss)."""
        model.to(self.device).eval()
        loss_sum, total, correct = 0.0, 0, 0
        non_block = (self.device.type == "cuda")

        with torch.no_grad():
            for images, labels in self.testloader:
                images = images.to(self.device, non_blocking=non_block)
                labels = labels.to(self.device, non_blocking=non_block)

                outputs = model(images)
                loss_sum += self.criterion(outputs, labels).item()
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        acc = (correct / total) if total else 0.0
        avg_loss = loss_sum / max(1, len(self.testloader))
        return acc, avg_loss




# def test_inference(args, model, test_dataset):
#     """ Returns the test accuracy and loss.
#     """

#     model.eval()
#     loss, total, correct = 0.0, 0.0, 0.0

#     device = 'cuda' if args.gpu else 'cpu'
#     criterion = nn.NLLLoss().to(device)
#     testloader = DataLoader(test_dataset, batch_size=128,
#                             shuffle=False)

#     for batch_idx, (images, labels) in enumerate(testloader):
#         images, labels = images.to(device), labels.to(device)

#         # Inference
#         outputs = model(images)
#         batch_loss = criterion(outputs, labels)
#         loss += batch_loss.item()

#         # Prediction
#         _, pred_labels = torch.max(outputs, 1)
#         pred_labels = pred_labels.view(-1)
#         correct += torch.sum(torch.eq(pred_labels, labels)).item()
#         total += len(labels)

#     accuracy = correct/total
#     return accuracy, loss


def test_inference(args, model, test_dataset):
    import torch
    from torch.utils.data import DataLoader
    import torch.nn as nn

    model.eval()

    # pick the device the model is already on (cuda or cpu)
    device = next(model.parameters()).device
    model.to(device)

    testloader = DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
    )
    criterion = nn.CrossEntropyLoss().to(device)

    loss, total, correct = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total if total else 0.0
    return acc, loss / len(testloader)

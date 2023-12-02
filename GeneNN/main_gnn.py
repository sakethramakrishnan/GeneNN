import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


from dgl.dataloading import GraphDataLoader
from dgl.nn.pytorch.conv import GINConv
from dgl.nn.pytorch.glob import SumPooling
from sklearn.model_selection import StratifiedKFold
from torch.utils.data.sampler import SubsetRandomSampler

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.data

from typing import  Union
from pathlib import Path
PathLike = Union[str, Path]

from dataset import GeneticGraphDataset
from gnn_model import GIN


from genslm_model import GenSLM

from utils import read_fasta, parse_sequence_labels, preprocess_data




def split_fold10(labels, fold_idx=0, n_splits = 3):
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=0)
    idx_list = []
    for idx in skf.split(np.zeros(len(labels)), labels):
        idx_list.append(idx)
    train_idx, valid_idx = idx_list[fold_idx]
    return train_idx, valid_idx


def evaluate(dataloader, device, model):
    model.eval()
    total = 0
    total_correct = 0
    for batched_graph, labels in dataloader:
        batched_graph = batched_graph.to(device)
        labels = labels.to(device)
        feat = batched_graph.ndata.pop("feature")
        total += len(labels)
        logits = model(batched_graph, feat)
        _, predicted = torch.max(logits, 1)
        total_correct += (predicted == labels).sum().item()
    acc = 1.0 * total_correct / total
    return acc


def train(train_loader, val_loader, device, model):
    # loss function, optimizer and scheduler
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    # training loop
    for epoch in range(100):
        model.train()
        total_loss = 0

        for i, (batched_graph, labels) in enumerate(train_loader):
            batched_graph = batched_graph.to(device)
            labels = labels.to(device)
            feat = batched_graph.ndata.pop("feature")
            logits = model(batched_graph, feat)
            loss = loss_fcn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        train_acc = evaluate(train_loader, device, model)
        valid_acc = evaluate(val_loader, device, model)
        print(
            "Epoch {:05d} | Loss {:.4f} | Train Acc. {:.4f} | Validation Acc. {:.4f} ".format(
                epoch, total_loss / (i + 1), train_acc, valid_acc
            )
        )



# creating labels and format

if __name__ == "__main__":
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    path = "../data/datasets/dna/dna_classification_refined.fasta"
    sequences_raw = []
    #for p in Path(path).glob("*.fasta"):
    sequences_raw.extend(read_fasta(path))
    
    labels = parse_sequence_labels(sequences_raw)
    sequences, labels = preprocess_data(sequences_raw, labels)
    
    
    
    #sequences = [["ATG", "TGA", "TAA", "TAG"], ["ATG", "TGA", "TAA", "TAG", "TCA", "TAA", "TAG", "TGA", "TAA", "TAG"], ["ATG", "TGA", "TAA", "TAG", "TCA", "TAA", "TAG", "TGA", "TAA", "TAG"], ["ATG", "TGA", "TAA", "TAG"], ["ATG", "TGA", "TAA", "TAG"], ["ATG", "TGA", "TAA", "TAG"], ["ATG", "TGA", "TAA", "TAG"], ["ATG", "TGA", "TAA", "TAG"], ["ATG", "TGA", "TAA", "TAG"], ["ATG", "TGA", "TAA", "TAG"], ["ATG", "TGA", "TAA", "TAG"]]
    labels_dict = {'CDS': 0, 'ncRNA': 1, 'tRNA': 2, 'mRNA': 3, 'rRNA': 4}
    label_nums = [labels_dict[label_str] for label_str in labels]
    #label_nums = [0, 3, 0, 0, 0, 2, 3, 1, 4, 0, 2]

    model = GenSLM("genslm_25M_patric", model_cache_dir="genslm_weights/")
    model.eval()

    max_len_codon = 1024 # 2048 - gotten from GenSLM max len

    dataset = GeneticGraphDataset(sequences = sequences, labels =label_nums, num_feats=3, max_len_codon=max_len_codon, model=model)
    train_idx, val_idx = split_fold10(label_nums)

    # create dataloader
    train_loader = GraphDataLoader(
        dataset,
        sampler=SubsetRandomSampler(train_idx),
        batch_size=1,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = GraphDataLoader(
        dataset,
        sampler=SubsetRandomSampler(val_idx),
        batch_size=1,
        pin_memory=torch.cuda.is_available(),
    )

    # create GIN model
    in_size = 3
    out_size = 5
    model = GIN(in_size, 16, out_size).to(device)

    # model training/validating
    print("Training...")
    train(train_loader, val_loader, device, model)

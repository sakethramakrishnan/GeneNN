import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dgl.data import GINDataset
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
from torch.utils.data import Dataset
from pydantic import BaseModel
from Bio.SeqUtils import GC
from typing import Any, Dict, List, Optional, Set, Type, Union, Tuple
from pathlib import Path
PathLike = Union[str, Path]
import re
from collections import Counter, defaultdict
from operator import itemgetter
import random


class MLP(nn.Module):
    """Construct two-layer MLP-type aggreator for GIN model"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linears = nn.ModuleList()
        # two-layer MLP
        self.linears.append(nn.Linear(input_dim, hidden_dim, bias=False))
        self.linears.append(nn.Linear(hidden_dim, output_dim, bias=False))
        self.batch_norm = nn.BatchNorm1d((hidden_dim))

    def forward(self, x):
        h = x
        h = F.relu(self.batch_norm(self.linears[0](h)))
        return self.linears[1](h)


class GIN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.ginlayers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        num_layers = 5
        # five-layer GCN with two-layer MLP aggregator and sum-neighbor-pooling scheme
        for layer in range(num_layers - 1):  # excluding the input layer
            if layer == 0:
                mlp = MLP(input_dim, hidden_dim, hidden_dim)
            else:
                mlp = MLP(hidden_dim, hidden_dim, hidden_dim)
            self.ginlayers.append(
                GINConv(mlp, learn_eps=False)
            )  # set to True if learning epsilon
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        # linear functions for graph sum poolings of output of each layer
        self.linear_prediction = nn.ModuleList()
        for layer in range(num_layers):
            if layer == 0:
                self.linear_prediction.append(nn.Linear(input_dim, output_dim))
            else:
                self.linear_prediction.append(nn.Linear(hidden_dim, output_dim))
        self.drop = nn.Dropout(0.5)
        self.pool = (
            SumPooling()
        )  # change to mean readout (AvgPooling) on social network datasets

    def forward(self, g, h):
        # list of hidden representation at each layer (including the input layer)
        hidden_rep = [h]
        for i, layer in enumerate(self.ginlayers):
            h = layer(g, h)
            h = self.batch_norms[i](h)
            h = F.relu(h)
            hidden_rep.append(h)
        score_over_layer = 0
        # perform graph sum pooling over all nodes in each layer
        for i, h in enumerate(hidden_rep):
            pooled_h = self.pool(g, h)
            score_over_layer += self.drop(self.linear_prediction[i](pooled_h))
        return score_over_layer


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


class GeneticGraphDataset(Dataset):
    def __init__(self, sequences, labels, num_feats, max_len_codon):
        # where sequence is a list of codons
        self.labels = labels
        self.sequences = sequences
        self.num_feats = num_feats
        self.max_len_codon = max_len_codon

    def __len__(self):
        return len(self.codon_numbers)


    def codons_to_graph(self, sequence, num_feats, max_len_codon):
        codon_numbers = convert_codons_to_nums(sequence)
        codon_polarity = convert_codons_to_polarity(sequence)
        amino_acids = convert_codons_to_aa(sequence)
        
        num_codons = len(codon_numbers)
        src_nodes = []
        dst_nodes = []
        
        for i in range(num_codons+1):
            if i == 0:
                src_nodes.extend([i, i])
                dst_nodes.extend([i, i + 1])
            elif i == num_codons - 1:
                src_nodes.extend([i, i])
                dst_nodes.extend([i - 1, i])
            else:
                src_nodes.extend([i] * 3)
                dst_nodes.extend([i - 1, i, i + 1])
        
        g = dgl.graph((src_nodes, dst_nodes), num_nodes=20)
        
        node_features = torch.zeros(max_len_codon, num_feats, dtype=torch.float32)
        
        for i, (num, polar, aa) in enumerate(zip(codon_numbers, codon_polarity, amino_acids)):
           
            
            #node_features[i] = torch.squeeze(torch.tensor(list((num, polar, aa))))
            node_features[i][0] = num
            node_features[i][1] = polar
            node_features[i][2] = aa
           
        
      
        g.ndata["feature"] = node_features
        return g
        

    def __getitem__(self, idx):
            sequence = self.sequences[idx]
            #print(sequence)
            graph = self.codons_to_graph(sequence, self.num_feats, self.max_len_codon)
            label = self.labels[idx]
            #print(len(sequence))
           
            return graph, label
    

CODON_TO_AA = {
    'ATA': 'I', 'ATC': 'I', 'ATT': 'I', 'ATG': 'M',
    'ACA': 'T', 'ACC': 'T', 'ACG': 'T', 'ACT': 'T',
    'AAC': 'N', 'AAT': 'N', 'AAA': 'K', 'AAG': 'K',
    'AGC': 'S', 'AGT': 'S', 'AGA': 'R', 'AGG': 'R',
    'CTA': 'L', 'CTC': 'L', 'CTG': 'L', 'CTT': 'L',
    'CCA': 'P', 'CCC': 'P', 'CCG': 'P', 'CCT': 'P',
    'CAC': 'H', 'CAT': 'H', 'CAA': 'Q', 'CAG': 'Q',
    'CGA': 'R', 'CGC': 'R', 'CGG': 'R', 'CGT': 'R',
    'GTA': 'V', 'GTC': 'V', 'GTG': 'V', 'GTT': 'V',
    'GCA': 'A', 'GCC': 'A', 'GCG': 'A', 'GCT': 'A',
    'GAC': 'D', 'GAT': 'D', 'GAA': 'E', 'GAG': 'E',
    'GGA': 'G', 'GGC': 'G', 'GGG': 'G', 'GGT': 'G',
    'TCA': 'S', 'TCC': 'S', 'TCG': 'S', 'TCT': 'S',
    'TTC': 'F', 'TTT': 'F', 'TTA': 'L', 'TTG': 'L',
    'TAC': 'Y', 'TAT': 'Y', 'TAA': '_', 'TAG': '_',
    'TGC': 'C', 'TGT': 'C', 'TGA': '_', 'TGG': 'W',
    'XXX':'X'
}


AA_TO_POLARITY = {
    'T': 'polar', 'S': 'polar', 'N': 'polar', 'Q': 'polar',
    'A': 'nonpolar', 'V': 'nonpolar', 'I': 'nonpolar', 'L': 'nonpolar', 'M': 'nonpolar', 'F': 'nonpolar', 'Y': 'nonpolar', 'W': 'nonpolar',
    'R': 'positive', 'H': 'positive', 'K': 'positive', 
    'D': 'negative', 'E': 'negative',
    'C': 'special', 'G': 'special', 'P': 'special', 'X': 'special', '_': 'special'
}

# Define a mapping from codons to their polarity
CODON_POLARITY = {codon:AA_TO_POLARITY[CODON_TO_AA[codon]] for codon in CODON_TO_AA}
POLARITY_NUM = {
    "polar": 0,
    "nonpolar": 1,
    "positive": 2,
    "negative": 3,
    "special": 4
}

AMINO_ACIDS = ['R', 'H', 'K', 'D', 'E', 'S', 'T', 'N', 'Q', 'A', 'V', 'I', 'L', 'M', 'F', 'Y', 'W', 'C', 'G', 'P', '_', 'X']

CODON_LETTER_NUMBER_ASSIGN = dict(zip(CODON_TO_AA, range(len(CODON_TO_AA))))
AMINO_ACID_NUM_ASSIGN = dict(zip(AMINO_ACIDS, range(len(AMINO_ACIDS)))) 

def convert_codons_to_nums(codon_seq):
    return [CODON_LETTER_NUMBER_ASSIGN[codon] for codon in codon_seq]

def convert_codons_to_polarity(codon_seq):
    return [POLARITY_NUM[CODON_POLARITY[codon]] for codon in codon_seq]

def convert_codons_to_aa(codon_seq):
    return [AMINO_ACID_NUM_ASSIGN[CODON_TO_AA[codon]] for codon in codon_seq]



BASES = ["A", "T", "C", "G", "a", "t", "c", "g"]


AMINO_ACIDS = ['R', 'H', 'K', 'D', 'E', 'S', 'T', 'N', 'Q', 'A', 'V', 'I', 'L', 'M', 'F', 'Y', 'W', 'C', 'G', 'P', '_', 'X']

# Mapping codon letters to int
CODON_LETTER_NUMBER_ASSIGN = dict(zip(CODON_TO_AA, range(len(CODON_TO_AA))))

# Mapping AA label to int
AA_IDX_TOTAL = dict(zip(AMINO_ACIDS, range(len(AMINO_ACIDS))))


# creating labels and format

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="MUTAG",
        choices=["MUTAG", "PTC", "NCI1", "PROTEINS"],
        help="name of dataset (default: MUTAG)",
    )
    args = parser.parse_args()
    print(f"Training with DGL built-in GINConv module with a fixed epsilon = 0")
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    path = "/home/couchbucks/Downloads/all_fasta_files/training/GCF_000315915.1_ASM31591v1_genomic_extracted_sequences.fasta"
    sequences_raw = []
    #for p in Path(path).glob("*.fasta"):
    sequences_raw.extend(read_fasta(path))
    max_len_codon = 3000
    labels = parse_sequence_labels(sequences_raw)
    raw_sequences, labels = preprocess_data(sequences_raw, labels)
    sequences = [seq_to_codon_list(truncate_codon_sequence(seq[0].upper())) for seq in raw_sequences]
    
    print(f"max_len_codon: {max_len_codon}")
    
    
    #sequences = [["ATG", "TGA", "TAA", "TAG"], ["ATG", "TGA", "TAA", "TAG", "TCA", "TAA", "TAG", "TGA", "TAA", "TAG"], ["ATG", "TGA", "TAA", "TAG", "TCA", "TAA", "TAG", "TGA", "TAA", "TAG"], ["ATG", "TGA", "TAA", "TAG"], ["ATG", "TGA", "TAA", "TAG"], ["ATG", "TGA", "TAA", "TAG"], ["ATG", "TGA", "TAA", "TAG"], ["ATG", "TGA", "TAA", "TAG"], ["ATG", "TGA", "TAA", "TAG"], ["ATG", "TGA", "TAA", "TAG"], ["ATG", "TGA", "TAA", "TAG"]]
    labels_dict = {'CDS': 0, 'ncRNA': 1, 'tRNA': 2, 'mRNA': 3, 'rRNA': 4}
    label_nums = [labels_dict[label_str] for label_str in labels]
    #label_nums = [0, 3, 0, 0, 0, 2, 3, 1, 4, 0, 2]



    dataset = GeneticGraphDataset(sequences = sequences, labels =label_nums, num_feats=3, max_len_codon=max_len_codon)
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
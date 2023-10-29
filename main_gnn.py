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
from torch.utils.data import Dataset, DataLoader
from pydantic import BaseModel
from Bio.SeqUtils import GC
from typing import Any, Dict, List, Optional, Set, Type, Union, Tuple
from pathlib import Path
PathLike = Union[str, Path]
import re
from collections import Counter, defaultdict
from operator import itemgetter
import random
from torch_geometric.nn import GCNConv

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


def split_fold10(labels, fold_idx=0):
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
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
        for batch, (batched_graph, labels) in enumerate(train_loader):
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
                epoch, total_loss / (batch + 1), train_acc, valid_acc
            )
        )


class GeneticGraphDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)
    def codons_to_graph(self, codon_numbers):
        num_codons = len(codon_numbers)
        src_nodes = []
        dst_nodes = []

        for i in range(num_codons):
            src_nodes.extend([i] * 3)
            if i == 0:
                dst_nodes.extend([i, i + 1, i])
            elif i == num_codons - 1:
                dst_nodes.extend([i - 1, i, i])
            else:
                dst_nodes.extend([i - 1, i, i + 1])

        g = dgl.graph((src_nodes, dst_nodes), num_nodes=num_codons)

        # Initialize a tensor of shape (num_codons, 64) with zeros
        features = torch.zeros(num_codons, 65, dtype=torch.float32)
        for i, codon_num in enumerate(codon_numbers):
            # Set the corresponding index to 1 in the one-hot encoded vector
            features[i, codon_num] = 1

        # Assign the features tensor to the graph
        g.ndata['feature'] = features

        return g

    def __getitem__(self, idx):
            sequence = self.sequences[idx]
            #print(sequence)
            graph = self.codons_to_graph(sequence)
            label = self.labels[idx]
            #print(len(sequence))
            return graph, label

BASES = ["A", "T", "C", "G", "a", "t", "c", "g"]

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

AMINO_ACIDS = ['R', 'H', 'K', 'D', 'E', 'S', 'T', 'N', 'Q', 'A', 'V', 'I', 'L', 'M', 'F', 'Y', 'W', 'C', 'G', 'P', '_', 'X']

# Mapping codon letters to int
CODON_LETTER_NUMBER_ASSIGN = dict(zip(CODON_TO_AA, range(len(CODON_TO_AA))))

# Mapping AA label to int
AA_IDX_TOTAL = dict(zip(AMINO_ACIDS, range(len(AMINO_ACIDS))))


class Sequence(BaseModel):
    sequence: str
    """Biological sequence (Nucleotide sequence)."""
 
    tag: str
    """Sequence description tag."""


def read_fasta(fasta_file: PathLike) -> List[Sequence]:
    """Reads fasta file sequences and description tags into dataclass."""
    text = Path(fasta_file).read_text()
    pattern = re.compile("^>", re.MULTILINE)
    non_parsed_seqs = re.split(pattern, text)[1:]
    lines = [line.replace("\n", "") for seq in non_parsed_seqs for line in seq.split("\n", 1)]

    return [[seq, tag] for seq, tag in zip(lines[1::2], lines[::2])]

    # for seq, tag in zip(lines[1::2], lines[::2]):


def read_fasta_only_seq(fasta_file: PathLike) -> List[str]:
    """Reads fasta file sequences without description tag."""
    text = Path(fasta_file).read_text()
    pattern = re.compile("^>", re.MULTILINE)
    non_parsed_seqs = list(re.split(pattern, text)[1:])
    lines = [line.replace("\n", "") for seq in non_parsed_seqs for line in seq.split("\n", 1)]

    return lines[1::2]


def check_bases(seq):
    '''Check that each of of the letters in each sequence is of the set{'A', 'T', 'C', 'G'}''' 
    return not any(x not in BASES for x in seq)


def is_valid_coding_sequence(seq: str) -> bool:
    """Check that a codon sequence has valid start and stop codon"""
    return seq[0:3] == "ATG" and seq[-3:] in {"TGA", "TAA", "TAG"} and len(seq) % 3 == 0 and check_bases(seq)


def is_valid_non_coding_sequence(seq: str) -> bool:
    """Check that a non-coding sequence is div by 3 and has all letters in codon"""
    return len(seq) % 3 == 0 and check_bases(seq)


def seq_to_codon_list(seq: str) -> List[str]:
    '''split the sequence string into strings of len 3'''
    return [seq[i:i + 3] for i in range(0, len(seq), 3)]


def translate(seq: List[str]) -> List[str]:
    '''convert DNA letters to amino acids'''
    return [CODON_TO_AA[codon] for codon in seq]

def flatten(l):
    '''flatten a list'''
    return [item for sublist in l for item in sublist]


def truncate_codon_sequence(sequence):
    '''If the sequence is not evenly divisible by 3, then we take off %3 bases from the end'''
    remainder = len(sequence) % 3
    if remainder != 0:
        sequence = sequence[:-remainder]
    return sequence

def parse_sequence_labels(sequences: List[Sequence]) -> List[str]:
    ''' each sequence in the fasta files has an associated tag, and that tag has a 'gbkey' '''
    ''' the gbkey has the type of the sequence (e.g. tRNA, mRNA, etc.) '''
    ''' here we try to isolate the gbkey into labels for each sequence '''
    pattern = r'gbkey=([^;]+)'
    matches = [re.search(pattern, seq[1]) for seq in sequences]
    labels = [match.group(1) if match else "" for match in matches]
    return labels

def intersection(lst1, lst2):
    return list(set(lst1).intersection(lst2))

def filter_sequences_by_gc(dna_sequences):
    ''' all sequences that have a GC content of 0% or 100%, we eliminate from the list of sequences '''
    valid_inds = []
    for i, sequence in enumerate(dna_sequences):
        gc_content = GC(sequence[0])
        if gc_content > 0. and  gc_content < 100. and len(sequence[0])>=6:
            valid_inds.append(i)
    return valid_inds


def preprocess_data(sequences: List[Sequence], labels: List[str], per_of_each_class=1.0):
    # Note: This function modifies sequences and labels
    # Filter out any outlier labels
    valid_labels = set(['mRNA', 'tRNA', 'RNA', 'exon', 'misc_RNA', 'rRNA', 'CDS', 'ncRNA'])
    valid_inds_labels = [i for i, label in enumerate(labels) if label in valid_labels]
    valid_inds_sequences = filter_sequences_by_gc(sequences)
    valid_inds = intersection(valid_inds_labels, valid_inds_sequences)
    sequences = [sequences[ind] for ind in valid_inds]
    labels = [labels[ind] for ind in valid_inds]

    label_group = defaultdict(list)
    for i, label in enumerate(labels):
        label_group[label].append(i)

    class_lens = dict(Counter(labels))
    for key in class_lens:
        class_lens[key] = round(class_lens[key] * per_of_each_class)
    #print(class_lens)
    smallest_class, min_class_size = min(class_lens.items(), key=itemgetter(1))
    # min_class_size = class_lens[smallest_class]
    print(f"Smallest class: {smallest_class} with {min_class_size} examples")

    sampled_inds = []
    for label, inds in label_group.items():
        sampled_inds.extend(random.sample(inds, k=min_class_size))

    sequences = [sequences[ind] for ind in sampled_inds]
    labels = [labels[ind] for ind in sampled_inds]
    print(str(len(sequences)) + ": number of total sequences; even split between CDS, ncRNA, tRNA, mRNA, rRNA")

    return sequences, labels

def convert_codons_to_nums(codon_seq):
    for idx, codon in enumerate(codon_seq):
        if not check_bases(codon):
            codon_seq[idx] = 'XXX'
    return [CODON_LETTER_NUMBER_ASSIGN[codon] for codon in codon_seq]

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
    path = "/lus/eagle/projects/RL-fold/sakisubi/Data/NewDataJustModified"
    sequences_raw = []
    for p in Path(path).glob("*.fasta"):
        sequences_raw.extend(read_fasta(p))

    labels = parse_sequence_labels(sequences_raw)
    raw_sequences, labels = preprocess_data(sequences_raw, labels)
    sequences = [convert_codons_to_nums(seq_to_codon_list(truncate_codon_sequence(seq[0].upper()))) for seq in raw_sequences]
    labels_dict = {'CDS': 0, 'ncRNA': 1, 'tRNA': 2, 'mRNA': 3, 'rRNA': 4}
    label_nums = [labels_dict[label_str] for label_str in labels]

    dataset = GeneticGraphDataset(sequences, label_nums)
    train_idx, val_idx = split_fold10(labels)

    # create dataloader
    train_loader = GraphDataLoader(
        dataset,
        sampler=SubsetRandomSampler(train_idx),
        batch_size=128,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = GraphDataLoader(
        dataset,
        sampler=SubsetRandomSampler(val_idx),
        batch_size=128,
        pin_memory=torch.cuda.is_available(),
    )

    # create GIN model
    in_size = 65
    out_size = 5
    model = GIN(in_size, 16, out_size).to(device)

    # model training/validating
    print("Training...")
    train(train_loader, val_loader, device, model)
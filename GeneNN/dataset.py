import torch
from typing import Union

import dgl
import torch

import dgl.data
import itertools
from torch.utils.data import Dataset

from pathlib import Path
PathLike = Union[str, Path]


from genslm_model import SequenceDataset

from utils import seq_to_codon_list

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


class GeneticGraphDataset(Dataset):
    def __init__(self, sequences, labels, num_feats, max_len_codon, model):
        # where sequence is a list of codons
        self.labels = labels
        self.sequences = sequences
        self.num_feats = num_feats
        self.max_len_codon = max_len_codon
        self.model = model

    def __len__(self):
        return len(self.labels)

    def find_last_attention_weights(self, seq, model):
        tokenized_seq = model.tokenizer(
            SequenceDataset.group_by_kmer(seq, 3),
            max_length=model.seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
            )

        outputs = model(
            tokenized_seq["input_ids"],
            tokenized_seq["attention_mask"],
            #output_hidden_states=True,
            output_attentions=True
            )
        attention = outputs.attentions[0]
        final_layer_attn = attention[0, -1, :, :]
        return final_layer_attn
    
    def codons_to_graph(self, sequence, num_feats, max_len_codon, attention_weights):
   
        codon_numbers = convert_codons_to_nums(sequence)
        codon_polarity = convert_codons_to_polarity(sequence)
        amino_acids = convert_codons_to_aa(sequence)


        # Initialize source and destination nodes for all edges
        src_nodes = []
        dst_nodes = []
        edge_weights = []

        # Connect each node to every other node
        for i in range(max_len_codon):
            for j in range(max_len_codon):
                src_nodes.append(i)
                dst_nodes.append(j)

        for i, j in zip(src_nodes, dst_nodes):
            weight = attention_weights[i, j]
            edge_weights.append(weight.item())

        g = dgl.graph((src_nodes, dst_nodes), num_nodes=max_len_codon)
        g.edata['weight'] = torch.tensor(edge_weights, dtype=torch.float32).view(-1, 1)

        # Initialize node features tensor
        node_features = torch.zeros(max_len_codon, num_feats, dtype=torch.float32)

        # Assign features to nodes
        for i, (num, polar, aa) in enumerate(zip(codon_numbers, codon_polarity, amino_acids)):
            node_features[i][0] = num
            node_features[i][1] = polar
            node_features[i][2] = aa

        # Assign node features to the graph
        g.ndata["feature"] = node_features

        return g
        

    def __getitem__(self, idx):
            sequence = self.sequences[idx]

            attn_weights = self.find_last_attention_weights(sequence, self.model)
            
            graph = self.codons_to_graph(seq_to_codon_list(sequence), self.num_feats, self.max_len_codon, attn_weights)
            label = self.labels[idx]
            #print(len(sequence))
           
            return graph, label
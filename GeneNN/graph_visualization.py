import dgl
import torch
import networkx as nx
import matplotlib.pyplot as plt

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

POLARITY_TO_NUM = {
    'polar': 0, 'nonpolar': 1, 'positive': 2, 'negative': 3, 'special': 4
}

# Define a mapping from codons to their polarity
#CODON_POLARITY = {codon:AA_TO_POLARITY[CODON_TO_AA[codon]] for codon in CODON_TO_AA}
# Define a mapping from codons to their polarity
CODON_POLARITY = {codon:AA_TO_POLARITY[CODON_TO_AA[codon]] for codon in CODON_TO_AA}

AMINO_ACIDS = ['R', 'H', 'K', 'D', 'E', 'S', 'T', 'N', 'Q', 'A', 'V', 'I', 'L', 'M', 'F', 'Y', 'W', 'C', 'G', 'P', '_', 'X']

CODON_LETTER_NUMBER_ASSIGN = dict(zip(CODON_TO_AA, range(len(CODON_TO_AA)))) 
AMINO_ACID_NUMBER_ASSIGN = dict(zip(AMINO_ACIDS, range(len(AMINO_ACIDS)))) 

#print(CODON_LETTER_NUMBER_ASSIGN)

def convert_codons_to_nums(codon_seq):
    print(codon_seq)
    return [float(CODON_LETTER_NUMBER_ASSIGN[codon]) for codon in codon_seq]

def convert_codons_to_polarity(codon_seq):
    return [float(POLARITY_TO_NUM[CODON_POLARITY[codon]]) for codon in codon_seq]

def convert_codons_to_aa(codon_seq):
    return [float(AMINO_ACID_NUMBER_ASSIGN[CODON_TO_AA[codon]]) for codon in codon_seq]


def codons_to_graph(sequence, num_feats, max_len_codon):
    codon_numbers = convert_codons_to_nums(sequence)
    codon_polarity = convert_codons_to_polarity(sequence)
    amino_acids = convert_codons_to_aa(sequence)
    
    num_codons = len(codon_numbers)
    
    # Initialize source and destination nodes for all edges
    src_nodes = []
    dst_nodes = []
    
    # Connect each node to every other node
    for i in range(max_len_codon):
        for j in range(max_len_codon):
            src_nodes.append(i)
            dst_nodes.append(j)
    
    g = dgl.graph((src_nodes, dst_nodes), num_nodes=max_len_codon)
    
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

        

# Example usage
codon_list = ['GCT', 'GAC', 'TTA']

attention_weights = 
codon_num_list = convert_codons_to_nums(codon_list)
codon_polarity_list = convert_codons_to_polarity(codon_list)

graph = codons_to_graph(codon_list, num_feats=3, max_len_codon=2048)
print(graph)



def visualize_graph(graph, codon_numbers):
    nx_graph = graph.to_networkx()  # Convert DGL graph to NetworkX graph
    pos = nx.spring_layout(nx_graph)  # Choose a layout for visualization

    # Draw the graph with node labels as codon numbers
    labels = {i: str(codon_num) for i, codon_num in enumerate(codon_numbers)}
    nx.draw(nx_graph, pos, with_labels=True, labels=labels, node_color='skyblue', node_size=800)
    plt.show()
    

visualize_graph(graph, codon_num_list)
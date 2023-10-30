import dgl
import torch

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

AMINO_ACIDS = ['R', 'H', 'K', 'D', 'E', 'S', 'T', 'N', 'Q', 'A', 'V', 'I', 'L', 'M', 'F', 'Y', 'W', 'C', 'G', 'P', '_', 'X']

CODON_LETTER_NUMBER_ASSIGN = dict(zip(CODON_TO_AA, range(len(CODON_TO_AA)))) 

def convert_codons_to_nums(codon_seq):
    return [CODON_LETTER_NUMBER_ASSIGN[codon] for codon in codon_seq]

def convert_codons_to_polarity(codon_seq):
    return [CODON_POLARITY[codon] for codon in codon_seq]

def convert_codons_to_aa(codon_seq):
    return [CODON_TO_AA[codon] for codon in codon_seq]

def codons_to_graph(codon_numbers, codon_polarity, amino_acids):
    num_codons = len(codon_numbers)
    src_nodes = []
    dst_nodes = []
    
      
    if i == 0:
        src_nodes.extend([i, i])
        dst_nodes.extend([i, i + 1])
    elif i == num_codons - 1:
      src_nodes.extend([i, i])
      dst_nodes.extend([i - 1, i])
    else:
      src_nodes.extend([i] * 3)
      dst_nodes.extend([i - 1, i, i + 1])


    print(dst_nodes)
    print(src_nodes)
    g = dgl.graph((src_nodes, dst_nodes), num_nodes=num_codons)

    # Initialize a tensor of shape (num_codons, 64) with zeros for codon features
    codon_features = torch.zeros(num_codons, 64, dtype=torch.float32)
    for i, codon_num in enumerate(codon_numbers):
        codon_features[i, codon_num] = 1

    weights = torch.tensor([0.1, 0.2, 0.3, 0.4])

    

    # Assign the codon features to the graph
    g.ndata['codon_feature'] = codon_features
    g.edata['edge_weight'] = weights
    # Create a tensor for polarity feature
    polarity_features = torch.zeros(num_codons, len(CODON_POLARITY), dtype=torch.float32)
    for i, codon_polarity_label in enumerate(codon_polarity):
        polarity_features[i, list(CODON_POLARITY.values()).index(codon_polarity_label)] = 1

    # Assign the polarity features to the graph
    g.ndata['polarity_feature'] = polarity_features

    amino_acid_features = torch.zeros(num_codons, len(AMINO_ACIDS), dtype=torch.float32)
    for i, amino_acid in enumerate(amino_acids):
        amino_acid_features[i, list(AMINO_ACIDS).index(amino_acid)] = 1
    g.ndata['amino_acid_feature'] = amino_acid_features

    return g

# Example usage
codon_list = ['GCT', 'GAC']
codon_num_list = convert_codons_to_nums(codon_list)
codon_polarity_list = convert_codons_to_polarity(codon_list)

graph = codons_to_graph(codon_num_list, codon_polarity_list, convert_codons_to_aa(codon_list))
print(graph.ndata)

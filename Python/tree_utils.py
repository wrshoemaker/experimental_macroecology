from __future__ import division
import string
import os
import random, sys
import numpy
from ete3 import Tree
import ete3
from itertools import combinations


directory = os.path.expanduser("~/GitHub/experimental_macroecology")



random.seed(123456789)

sys.setrecursionlimit(15000)

n_fna_characters = 80
#tree = get_emp_tree()
#print(max(calculate_phylogenetic_distance_all(tree)))
# returns = 3.410779999999999


def subset_tree(species_list, tree):

    # deep copy. slowest copy option, can change later if this is too slow
    tree_copy = tree.copy()

    #tree_labels = [t_i.name for t_i in tree_copy.get_leaves()]

    tree_copy.prune(species_list, preserve_branch_length=True)


    return tree_copy




def cache_distances(tree):

    ''' precalculate distances of all nodes to the root'''

    node2rootdist = {tree:0}
    for node in tree.iter_descendants('preorder'):
        node2rootdist[node] = node.dist + node2rootdist[node.up]
    return node2rootdist


def collapse(tree_copy, node2tips, root_distance,  min_dist):

    # collapses all nodes within a given distance for the tree
    # cache the tip content of each node to reduce the number of times the tree_copy is traversed

    for node in tree_copy.get_descendants('preorder'):
        if not node.is_leaf():
            avg_distance_to_tips = numpy.mean([root_distance[tip]-root_distance[node]
                                         for tip in node2tips[node]])

            if avg_distance_to_tips < min_dist:
                # do whatever, ete support node annotation, deletion, labeling, etc.
                # rename
                # ' COLLAPSED'
                node.name += 'COLLAPSED-%g-%s' %(avg_distance_to_tips,
                                                 ','.join([tip.name for tip in node2tips[node]]))
                # label
                node.add_features(collapsed=True)
                # set drawing attribute so they look collapsed when displayed with tree_copy.show()
                #node.img_style['draw_descendants'] = False

    return tree_copy




def get_tree():
    tree = ete3.Tree('%s/data/migration_data_table_totabund_all_singleton_mapped_full_wT0_20210918.tre' % directory, quoted_node_names=True, format=1)
    return tree





def make_fasta_from_dada2_output():

    otu = open(directory + '/data/migration_data_table_totabund_all_singleton_mapped_full_wT0_20210918.csv')

    dada2_path = '%s/data/migration_data_table_totabund_all_singleton_mapped_full_wT0_20210918.csv' % directory
    fasta_path = '%s/data/migration_data_table_totabund_all_singleton_mapped_full_wT0_20210918.fna' % directory
    fasta_file = open(fasta_path, 'w')

    sequence_all = []

    for line_idx, line in enumerate(open(dada2_path, 'r')):

        if line_idx == 0:
            continue

        line = line.strip()

        sequence = line.split(',')[6].replace('"', '')
        sequence_all.append(sequence)

    sequence_all = list(set(sequence_all))
    for sequence in sequence_all:
        fasta_file.write('>%s\n' % sequence)

        for i in range(0, len(sequence), n_fna_characters):
            sequence_i = sequence[i : i + n_fna_characters]
            fasta_file.write('%s\n' % sequence_i)
        fasta_file.write('\n')

    fasta_file.close()




class classFASTA:

    # class to load FASTA file

    def __init__(self, fileFASTA):
        self.fileFASTA = fileFASTA

    def readFASTA(self):
        '''Checks for fasta by file extension'''
        file_lower = self.fileFASTA.lower()
        '''Check for three most common fasta file extensions'''
        if file_lower.endswith('.txt') or file_lower.endswith('.fa') or \
        file_lower.endswith('.fasta') or file_lower.endswith('.fna') or \
        file_lower.endswith('.fasta') or file_lower.endswith('.frn') or \
        file_lower.endswith('.faa') or file_lower.endswith('.ffn'):
            with open(self.fileFASTA, "r") as f:
                return self.ParseFASTA(f)
        else:
            print("Not in FASTA format.")

    def ParseFASTA(self, fileFASTA):
        '''Gets the sequence name and sequence from a FASTA formatted file'''
        fasta_list=[]
        for line in fileFASTA:
            if line[0] == '>':
                try:
                    fasta_list.append(current_dna)
            	#pass if an error comes up
                except UnboundLocalError:
                    #print "Inproper file format."
                    pass
                current_dna = [line.lstrip('>').rstrip('\n'),'']
            else:
                current_dna[1] += "".join(line.split())
        fasta_list.append(current_dna)
        '''Returns fasa as nested list, containing line identifier \
            and sequence'''
        return fasta_list




def clean_alignment(muscle_path, muscle_clean_path, min_n_sites=100, max_fraction_empty=0.8):

    # removes all sites where the fraction of empty bases across ASVs is greater than max_fraction_empty (putatively uninformative)
    # removes a sequencies with fewer than min_n_sites informative sites

    frn_aligned = classFASTA(muscle_path).readFASTA()

    n = len(frn_aligned)

    frn_aligned_seqs = [x[1] for x in frn_aligned]
    frn_aligned_seqs_names = [x[0] for x in frn_aligned]

    frns = []
    for site in zip(*frn_aligned_seqs):

        fraction_empty = site.count('-')/n

        if fraction_empty > max_fraction_empty:
            continue

        # skip site if it is uninformative
        if len(set([s for s in site if s != '-'])) == 1:
            continue

        frns.append(site)

    if len(frns) < min_n_sites:
        exit()

    clean_sites_list = zip(*frns)

    # skip if there are too few sites
    #if len(list(clean_sites_list)[0]) < min_n_sites:
    #    continue

    frn_aligned_clean = open(muscle_clean_path, 'w')

    for clean_sites_idx, clean_sites in enumerate(clean_sites_list):
        clean_sites_species = frn_aligned_seqs_names[clean_sites_idx]
        clean_sites_seq = "".join(clean_sites)

        frn_aligned_clean.write('>%s\n' % clean_sites_species)

        clean_sites_seq_split = [clean_sites_seq[i:i+n_fna_characters] for i in range(0, len(clean_sites_seq), n_fna_characters)]

        for seq in clean_sites_seq_split:

            frn_aligned_clean.write('%s\n' % seq)

        frn_aligned_clean.write('\n')


    frn_aligned_clean.close()



make_fasta_from_dada2_output()
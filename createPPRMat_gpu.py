#!/usr/bin/python

# Load required modules
import sys, os
import networkx as nx
import scipy as sp, scipy.io

# numpy for CUDA
import cupy as cp

import numpy as np
import os.path
sys.path.append(os.path.split(os.path.split(sys.argv[0])[0])[0])
from hotnet2 import hnap

# garbage collector to manage memory manually
import gc

# Parse arguments
def get_parser():
	description = 'Create the personalized pagerank matrix for the given '\
                  'network and restart probability beta.'
	parser = hnap.HotNetArgParser(description=description, fromfile_prefix_chars='@')
	parser.add_argument('-e', '--edgelist_file', required=True,
                        help='Location of edgelist file.')
	parser.add_argument('-i', '--gene_index_file', required=True,
                        help='Location of gene-index file.')
	parser.add_argument('-o', '--output_dir', required=True,
	                help="Output dir.")
	parser.add_argument('-p', '--prefix', required=True,
	                help="Output prefix.")
	parser.add_argument('-a', '--alpha', required=True, type=float,
	                help="Page Rank dampening factor.")
	return parser

def run(args):
	# Load gene-index map
	print("* Loading gene index:\t%s" % args.gene_index_file)
	arrs = [l.rstrip().split() for l in open(args.gene_index_file)]
	
	# create two way mapping for genes to gene ids
	index2gene = dict((int(arr[0]), arr[1]) for arr in arrs)
	gene2index = dict((arr[1], int(arr[0])) for arr in arrs)
	
	# Generate graph
	# add nodes
	G = nx.Graph()
	G.add_nodes_from( index2gene.values() ) # in case any nodes have degree zero

	# Ladd edges
	print("* Loading graph:\t%s" % args.edgelist_file)
	edges = [map(int, l.rstrip().split()[:2]) for l in open(args.edgelist_file)]
	G.add_edges_from( [(index2gene[u], index2gene[v]) for u,v in edges] )
	
	print("\t- Edges:\t%d" % len(G.edges()))
	print("\t- Nodes:\t%d" % len(G.nodes()))

	# Remove self-loops and restrict to largest connected component
	print("* Removing self-loops, multi-edges, and restricting to largest connected component...")
	self_loops = [(u, v) for u, v in G.edges() if u == v]
	G.remove_edges_from( self_loops )
	G = G.subgraph( sorted(nx.connected_components( G ), key=lambda cc: len(cc), reverse=True)[0] )

	nodes = sorted(G.nodes())
	n = len(nodes)
	print("\t- Largest CC Edges:\t%d" % len( G.edges() ))
	print("\t- Largest CC Nodes:\t%d"% len( G.nodes() ))

	# Set up output directory
	print("* Saving updated graph to file...")
	os.system( 'mkdir -p ' + args.output_dir )
	output_prefix = "{}/{}".format(args.output_dir, args.prefix)
	pprfile = "{}_ppr_{:g}.mat".format(output_prefix, args.alpha)

	# Output new files
	# Index mapping for genes
	index_map = [ "{} {}".format(gene2index[nodes[i]], nodes[i]) for i in range(n) ]
	open("{}_index_genes".format(output_prefix), 'w').write( "\n".join(index_map) )

	# Edge list
	edges = [sorted([gene2index[u], gene2index[v]]) for u, v in G.edges()]
	edgelist = [ "{} {} 1".format(u, v) for u, v in edges ]		
	open("{}_edge_list".format(output_prefix), 'w').write( "\n".join(edgelist) )

	# Create the PPR matrix either using Scipy or MATLAB
	# Create "walk" matrix (normalized adjacency matrix)
	print "* Creating PPR  matrix..."

	# save memory and space w/ float32
	# generate numpy matrix from graph
	W = nx.to_numpy_matrix( G , nodelist=nodes, dtype=np.float32 )
	
	print("\t- Moving data to GPU")
	# save memory and space w/ float32
	W_g = cp.array(W / W.sum(axis=1),dtype=np.float32) # normalization step
	I_g = cp.identity(W_g.shape[0],dtype=np.float32)

	# free memory
	del W
	gc.collect()

	## Create PPR matrix using Python
	print("\t- Computing PPR on GPU")
	PPR_g = (1.0-args.alpha)*cp.linalg.inv(I_g-args.alpha*cp.transpose(W_g))
	print("\t- Moving data from GPU")
	PPR = cp.asnumpy(PPR_g)
	print("\t- Saving PPR:\t%s" % pprfile)
	scipy.io.savemat( pprfile, dict(PPR=PPR), oned_as='column')

if __name__ == "__main__":
	run(get_parser().parse_args(sys.argv[1:]))

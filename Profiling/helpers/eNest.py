from torch import nn
import numpy as np
import torch
import torch.nn as nn
import networkx as nx
import networkx.algorithms.components.connected as nxacc
import networkx.algorithms.dag as nxadag
from collections import OrderedDict
import sys

from .nest_data_paths import nest_data_file_paths, resolve_nest_data_dir


class eNest(nn.Module):
    def __init__(self, nodes_per_assembly, dropout=0.0, activation=nn.Tanh, verbosity=-1):
        super().__init__()
        datadir = resolve_nest_data_dir()
        ontfile, gene2idfile = nest_data_file_paths(datadir)
        self.verbosity = verbosity
        self.activation = activation
        # Assert that activation function returns 0 for input 0 (required for proper masking behavior)
        test_input = torch.tensor(0.0)
        activation_instance = self.activation()
        test_output = activation_instance(test_input)
        assert torch.allclose(test_output, torch.tensor(0.0), atol=1e-6), f"Activation function {activation.__name__} must return 0 for input 0, but got {test_output.item()}"
        self.np_lm, self.np_flm, self.np_fasmlm, self.node_idx, self.assembly_count = setup_Nest_Masks.get_layer_masks(ontfile, gene2idfile, nodes_per_assembly) #numpy layer masks
        self.nodes_per_assembly = nodes_per_assembly
        #Mask for when things are forwarded only
        self.forward_layer_masks = list()
        #Mask for the final destination of things
        self.layer_masks = list()
        #Mask for assemblies only when they accumulate all inputs
        self.final_asm_layer_masks = list()
        #Should only be 1 root
        self.hidden_assemblies = self.assembly_count - 1 # * self.nodes_per_assembly #Number of total assemblies - 1 assembly (1 assembly is final output)
        for i in range(0, len(self.np_lm)):
            torch_layer_mask = torch.from_numpy(self.np_lm[i])
            self.register_buffer(f'layer_masks_{i}', torch_layer_mask)
            self.layer_masks.append(getattr(self, f'layer_masks_{i}'))

            torch_forward_layer_mask = torch.from_numpy(self.np_flm[i].astype(bool))
            self.register_buffer(f'forward_layer_masks_{i}', torch_forward_layer_mask)
            self.forward_layer_masks.append(getattr(self, f'forward_layer_masks_{i}'))

            torch_final_asm_layer_mask = torch.from_numpy(self.np_fasmlm[i].astype(bool))
            self.register_buffer(f'final_asm_layer_masks_{i}', torch_final_asm_layer_mask)
            self.final_asm_layer_masks.append(getattr(self, f'final_asm_layer_masks_{i}'))

            #self.print(f"Final asm Layer mask sum {i}:", self.np_fasmlm[i].sum())
            #self.layer_masks.append(torch_layer_mask)
            #self.print(self.layer_masks[i].shape)
        #assert(self.hidden_assemblies == 130 * nodes_per_assembly)

        self.layer_bias_masks = list()
        for i in range(0, len(self.layer_masks)):
            active_inputs_per_output = self.layer_masks[i].sum(dim=0)
            dead_outputs = (active_inputs_per_output == 0)
            bias_mask = torch.ones(active_inputs_per_output.shape)
            bias_mask[dead_outputs] = 0.0
            #Neater way?: bias_mask = torch.sum(self.layer_masks[i], dim=0) != 0; maybe change dtype?
            self.register_buffer(f'layer_bias_masks_{i}', bias_mask)
            self.layer_bias_masks.append(getattr(self, f'layer_bias_masks_{i}'))
            #self.print("getattr bias version:", getattr(self, f'layer_bias_masks_{i}').shape)
            #self.layer_bias_masks.append(bias_mask)

        #self.layer_list = OrderedDict()
        #self.layer_list = list()
        #Need to change forward to correspondingly match
        self.linear_layer_list = nn.ModuleList()
        self.batchnorm_layer_list = nn.ModuleList()
        for i in range(0, len(self.layer_masks)):
            layer_shape = self.layer_masks[i].shape
            self.linear_layer_list.append(nn.Linear(layer_shape[0], layer_shape[1]))
            self.batchnorm_layer_list.append(nn.BatchNorm1d(layer_shape[1]))
            # self.layer_list[f"linear_layer_{i}"] = nn.Linear(layer_shape[0], layer_shape[1])
            # self.layer_list[f"tanh_layer_{i}"] = nn.Tanh()
            # self.layer_list[f"batchnorm_layer_{i}"] = nn.BatchNorm1d(layer_shape[1])
            #self.layer_list.append(nn.Linear(layer_shape[0], layer_shape[1]))
        #self.model = nn.Sequential(self.layer_list)
        self.final_output_linear_layer = nn.Linear(nodes_per_assembly, 1)
        self.final_output_activation = nn.Sigmoid()

        #There needs to be masking so input only takes in desired nodes_per_assembly nodes, make sure to get columns and rows correct
        #Maybe try using notebook with weights/biases of 1/0 to make sure sum is expected
        self.asm_out_linear_layer_1 = nn.Linear(self.hidden_assemblies * self.nodes_per_assembly, self.hidden_assemblies)
        
        
        asm_out_linear_layer_1_mask = torch.zeros(self.hidden_assemblies * self.nodes_per_assembly, self.hidden_assemblies) #Assumes one output assembly
        for asm_idx in range(0, self.hidden_assemblies):
            for a in range(0, self.nodes_per_assembly):
                idx = self.nodes_per_assembly * asm_idx + a
                asm_out_linear_layer_1_mask[idx, asm_idx] = 1
        
        self.register_buffer(f'asm_out_linear_layer_1_mask', asm_out_linear_layer_1_mask)
        self.asm_out_linear_layer_1.weight.data *= self.asm_out_linear_layer_1_mask.T
        
        #Here masking is just identity function
        self.asm_out_linear_layer_2 = nn.Linear(self.hidden_assemblies, self.hidden_assemblies)

        asm_out_linear_layer_2_mask = torch.zeros(self.hidden_assemblies, self.hidden_assemblies) #Assumes one output assembly
        for asm_idx in range(0, self.hidden_assemblies):
            asm_out_linear_layer_2_mask[asm_idx, asm_idx] = 1
        
        self.register_buffer(f'asm_out_linear_layer_2_mask', asm_out_linear_layer_2_mask)
        self.asm_out_linear_layer_2.weight.data *= self.asm_out_linear_layer_2_mask.T
        #Passed on value for all assemblies is input -> Linear -> Tanh -> BatchNorm1d
        #Value used for loss on all assemblies is input -> Linear -> Tanh -> BatchNorm1d -> Linear -> Tanh -> Linear
        self.dropout_layers = nn.ModuleList()
        
        for i in range(0, len(self.layer_masks)):
            #self.print("X shape:", X.shape)
            #Masking layer connections and biases
            linear_layer = self.linear_layer_list[i]
            #Initial masking, later will rely on masking in backprop to ensure these remain zeroed
            #Registering hooks to ensure in backprop gradients get zeroed
            linear_layer.weight.data *= self.layer_masks[i].T
            ##self.print(f"Post layer {i} mask sum: {len(torch.nonzero(linear_layer.weight.data).squeeze())}")
            
            linear_layer.weight.data[self.forward_layer_masks[i].T] = 1.0
            
            ##self.print("Post flm sum:", len(torch.nonzero(linear_layer.weight.data).squeeze()))
            #linear_layer.weight.register_hook(grad_mask_hook(self.layer_masks[i].T))
            #linear_layer.weight.register_hook(grad_mask_hook(getattr(self, f'layer_masks_{i}').T))
            
            #NEED TO ZERO BIAS OR NOT???
            #Bias needs to be 0 on forwarded things, should be currently correct
            linear_layer.bias.data *= self.layer_bias_masks[i]
            #linear_layer.bias.register_hook(grad_mask_hook(getattr(self, f'layer_bias_masks_{i}')))
            #linear_layer.bias.register_hook(grad_mask_hook(self.layer_bias_masks[i]))
            self.dropout_layers.append(nn.Dropout(p=dropout))
            #self.register_buffer(f'layer_masks_{i}', self.layer_masks[i])
            #self.register_buffer(f'layer_bias_masks_{i}', self.layer_bias_masks[i])
            ##self.print(f"Layer {i} weight mask sum: {self.layer_bias_masks[i].sum()}, bias mask sum: {self.layer_masks[i].sum()}")
            
            #assert((linear_layer.bias.data != 0).sum() == self.layer_bias_masks[i].sum())
            #Could this assert fail due to being initialized to zero in rare instance?
            #assert((linear_layer.weight.data != 0).sum() == self.layer_masks[i].sum() + self.forward_layer_masks[i].sum())
        
        
        self.print("Efficient Nest VNN Initialized")
              
            
    def forward(self, X):
        hidden_asm_states = list()
        for i in range(0, len(self.linear_layer_list)):
            X = self.dropout_layers[i](X)
            X = self.linear_layer_list[i](X)# * self.layer_masks[i] #Note: activation returns 0 for inputs of 0 (if using Tanh or ReLU)
            Z = self.batchnorm_layer_list[i](X)# * self.layer_masks[i]
            #Z = X
            Z = self.activation()(Z)
            #Note torch.equal(torch.sum(self.layer_masks[1], dim=0) != 0, self.layer_bias_masks[1] != 0) == True
            #In other words, bias mask is exactly a 1 in all non-zero things for layer_mask
            X = X * torch.sum(getattr(self, f'forward_layer_masks_{i}'), dim=0) + Z * getattr(self, f'layer_bias_masks_{i}')
            if (i < len(self.linear_layer_list) - 1):
                hidden_asm_states.append(X[:, self.final_asm_layer_masks[i]])

            #Don't append final output

            #self.print("X shape:", X.shape)
            #Masking layer connections and biases
            #self.layer_list[i].weight.data *= self.layer_masks[i].T
            #self.layer_list[i].bias.data[self.dead_outputs] = 0.0

        hidden_asm_X = torch.cat(hidden_asm_states, dim=1)
        hidden_asm_X = self.activation()(self.asm_out_linear_layer_1(hidden_asm_X))
        hidden_asm_Y = self.asm_out_linear_layer_2(hidden_asm_X)

        #For continuity could just make is a mask instead w/ fully connected layer
        X = X[:, self.node_idx["NEST"] : self.node_idx["NEST"] + self.nodes_per_assembly]
        Y = self.final_output_activation(self.final_output_linear_layer(X))
        return Y, hidden_asm_Y
        #     X = self.layer_list[i](X)
        #self.print("Returning shape:", X[self.node_idx["NEST"]].shape)
        # X = self.model(X)[:, self.node_idx["NEST"] : self.node_idx["NEST"] + self.nodes_per_assembly]
        # return self.final_output_activation(self.final_output_linear_layer(X))
    
    #Call once device is on its desired device
    def register_grad_hooks(self):
        def grad_mask_hook(mask):
            def hook(grad):
                #self.print("Mask device:", mask.device)
                return grad * mask.type(grad.dtype)  # apply mask to gradient
            return hook
        for i in range(0, len(self.layer_masks)):
            linear_layer = self.linear_layer_list[i]
            #self.print("registering mask on device:", getattr(self, f'layer_masks_{i}').T.device)
            linear_layer.weight.register_hook(grad_mask_hook(getattr(self, f'layer_masks_{i}').T))
            linear_layer.bias.register_hook(grad_mask_hook(getattr(self, f'layer_bias_masks_{i}')))
        self.asm_out_linear_layer_1.weight.register_hook(grad_mask_hook(getattr(self, f'asm_out_linear_layer_1_mask').T))
        self.asm_out_linear_layer_2.weight.register_hook(grad_mask_hook(getattr(self, f'asm_out_linear_layer_2_mask').T))
        assert(getattr(self, f'layer_bias_masks_{i}').device == getattr(self, f'layer_masks_{i}').T.device)
        #self.print(f"Hooks (Grad Masks) registered to {getattr(self, f'layer_bias_masks_{i}').device}")

    def print(self, msg):
        if (self.verbosity < 0):
            return
        else:
            self.print(msg)

class setup_Nest_Masks():
    # def __init__(self, file_name):
    #     G = self.load_ontology(file_name)
    #     self.n2i = self.node_to_idx_dict(G)
    #     self.nd = self.calculate_depths(G)
    #     self.lm = make_layer_masks(G, node_depths)

    @staticmethod
    def get_layer_masks(file_name, g2id_file, nodes_per_assembly):
        '''
        Wrapper to do everything calling below functions to get layer masks and node to idx dictionary
        file_name is file for NN connection graph, networkx graph
        g2id_file is file specifying gene2id (artifact of NeST code)
        '''
        G, term_direct_gene_map, num_assemblies = setup_Nest_Masks.load_ontology(file_name, g2id_file)
        G = setup_Nest_Masks.add_genes(G, term_direct_gene_map)
        n2i = setup_Nest_Masks.node_to_idx_dict(G)
        nd = setup_Nest_Masks.calculate_depths(G)
        lm, flm, fasmlm = setup_Nest_Masks.make_layer_masks(G, nd, n2i, nodes_per_assembly)
        #fasmlm, final assembly layer masks, contains 1's by layer if it is where assembly has accumulated all inputs
        return lm, flm, fasmlm, n2i, num_assemblies

    @staticmethod
    def load_mapping(mapping_file, mapping_type):
        mapping = {}
        file_handle = open(mapping_file)
        for line in file_handle:
            line = line.rstrip().split()
            mapping[line[1]] = int(line[0])

        file_handle.close()
        # print('Total number of {} = {}'.format(mapping_type, len(mapping)))
        return mapping

    @staticmethod
    def load_ontology(file_name, g2id_file):
        dG = nx.DiGraph()
        term_direct_gene_map = {}
        term_size_map = {}
        gene_set = set()
        gene_id_mapping = setup_Nest_Masks.load_mapping(g2id_file, 'genes')

        file_handle = open(file_name)
        for line in file_handle:
            line = line.rstrip().split()
            if line[2] == 'default':
                dG.add_edge(line[0], line[1])
            else:
                if line[1] not in gene_id_mapping:
                    continue
                if line[0] not in term_direct_gene_map:
                    term_direct_gene_map[line[0]] = set()
                term_direct_gene_map[line[0]].add(gene_id_mapping[line[1]])
                gene_set.add(line[1])
        file_handle.close()

        for term in dG.nodes():
            term_gene_set = set()
            if term in term_direct_gene_map:
                term_gene_set = term_direct_gene_map[term]
            deslist = nxadag.descendants(dG, term)
            for child in deslist:
                if child in term_direct_gene_map:
                    term_gene_set = term_gene_set | term_direct_gene_map[child]
            # jisoo
            if len(term_gene_set) == 0:
                self.print('There is empty terms, please delete term:', term)
                sys.exit(1)
            else:
                term_size_map[term] = len(term_gene_set)

        roots = [n for n in dG.nodes if dG.in_degree(n) == 0]

        uG = dG.to_undirected()
        connected_subG_list = list(nxacc.connected_components(uG))

        # print('There are', len(roots), 'roots:', roots[0])
        #The number of terms is exactly the number of assemblies!!! May be useful to non-manually set assembly count
        # print('There are', len(dG.nodes()), 'terms')
        # print('There are', len(connected_subG_list), 'connected componenets')

        if len(roots) > 1:
            print('There are more than 1 root of ontology. Please use only one root.')
            sys.exit(1)
        if len(connected_subG_list) > 1:
            print('There are more than 1 connected components. Please connect them.')
            sys.exit(1)

        #self.dG = dG
        #self.root = roots[0]
        #self.term_size_map = term_size_map
        #self.term_direct_gene_map = term_direct_gene_map
        return dG, term_direct_gene_map, len(dG.nodes) #Last one is number of assemblies
    
    @staticmethod
    def add_genes(nestG, term_direct_gene_map):
        '''
        Adds connections in graph for genes
        '''
        for node,inputs in term_direct_gene_map.items():
            for gene in inputs:
                if (nestG.has_node(f"gene_{gene}") != True):
                    nestG.add_node(f"gene_{gene}")
                nestG.add_edge(node, f"gene_{gene}")
        return nestG


    @staticmethod
    def node_to_idx_dict(G):
        '''
        Takes in NeST graph and assigns nodes an idx in each layer
        '''
        #Setting up association between networkx graph node and NN node idx
        #Puts all genes in front for convenience
        node_idx = dict()
        i = 0
        for node in G.nodes():
            if (node.startswith('gene_')):
                node_idx[node] = i
                i += 1
        for node in G.nodes():
            if (node.startswith('NEST')):
                node_idx[node] = i
                i += 1
        assert(i == G.number_of_nodes())
        return node_idx

    @staticmethod
    def calculate_depths(G):
        # print("Calculating node depths")
        # Perform a topological sort to get the valid order
        G = G.reverse()
        sorted_nodes = list(nx.topological_sort(G))

        # Initialize a dictionary to store the depth of each node
        node_depths = {node: 0 for node in G.nodes}

        # Calculate the depth of each node based on the topological order
        for node in sorted_nodes:
            # Check all predecessors (nodes that point to the current node)
            for pred in G.predecessors(node):
                # Update the depth of the current node based on the maximum depth of the predecessors
                node_depths[node] = max(node_depths[node], node_depths[pred] + 1)
        
        #Optional debugging sanity check
        '''for node in G.nodes():
            if (node.startswith('gene_')):
                assert(node_depths[node] == 0)
            else:
                assert(node_depths[node] > 0)'''
        # print("Done calculating node depths")
        return node_depths

    @staticmethod
    def make_layer_masks(G, node_depths, node_idx, nodes_per_assembly=1):
        '''Takes 3 arguments, G and node_depths and returns the 1, 0 masks for each layer
        G -- networkx graph with the full NeST connections (including gene to assembly)
        node_depths -- dictionary where key is node name in G, and value is depth (topological depth of a DAG where depth of genes is 0)
        node_idx -- dictionary designating each node in each layer's index
        nodes_per_assembly -- number of nodes per assembly
        '''
        # print("Making Layer Masks")
        assert(len(G.nodes()) == 820)
        assert(len(node_depths.values()) == 820)
        assert(G.number_of_nodes() == 820)
        max_depth = max(node_depths.values()) #min depth is 0
        # print("Max Depth:", max_depth)
        assert(nodes_per_assembly > 0)
        layer_size = 689 + nodes_per_assembly * 131 #Hardcoded to 689 input genes and 131 assemblies right now
        layer_masks = list() #Normal list of all connections needed
        forward_layer_masks = list() #List of all connections which are just forwarding connections
        final_asm_layer_masks = list()
        for l in range(0, max_depth+1): #Used to be 0, max_depth + 1
            if (l == 0):
                layer_mask = np.zeros((689, layer_size))
                forward_layer_mask = np.zeros((689, layer_size))
                final_asm_layer_mask = np.zeros((layer_size))
                for i in range(0, 689):
                    layer_mask[i, i] = 1
                layer_masks.append(layer_mask)
                forward_layer_masks.append(forward_layer_mask)
                final_asm_layer_masks.append(final_asm_layer_mask)
                continue
            else:
                layer_mask = np.zeros((layer_size, layer_size))
                forward_layer_mask = np.zeros((layer_size, layer_size))
                final_asm_layer_mask = np.zeros((layer_size))
            for node in node_depths:
                #source of G.in_edges(node) will be next layer, ex. G.in_edges("NEST:1") yields "NEST"
                for dest, source in G.in_edges(node):
                    assert(node == source)
                    if (node_depths[dest] == l):
                        # if (l == 1):
                            # self.print("Source:", source, "Dest:", dest)
                        # assert(layer_masks[l - 1][:, node_idx[node]].sum() >= 1)
                        assert('NEST' in dest)
                        if ('NEST' in node):
                            assert('gene' not in node)
                            assert('NEST' in dest)
                            for a in range(0, nodes_per_assembly):
                                for b in range(0, nodes_per_assembly):
                                    assert(node_idx[node] >= 689)
                                    assert(node_idx[dest] >= 689)
                                    source_idx = 689 + ((node_idx[node] - 689) * nodes_per_assembly) + a
                                    dest_idx = 689 + ((node_idx[dest] - 689) * nodes_per_assembly) + b
                                    layer_mask[source_idx, dest_idx] = 1
                                    #Mask by layer of the final states of assemblies
                                    final_asm_layer_mask[dest_idx] = 1
                        else:
                            assert('gene' in node)
                            assert('NEST' in dest)
                            for b in range(0, nodes_per_assembly):
                                assert(node_idx[dest] >= 689)
                                dest_idx = 689 + ((node_idx[dest] - 689) * nodes_per_assembly) + b
                                layer_mask[node_idx[node], dest_idx] = 1
                                final_asm_layer_mask[dest_idx] = 1
                        #layer_mask[node_idx[node], node_idx[dest]] = 1
                    elif ((l > node_depths[source]) and (node_depths[dest] > l)):
                        #depth of source < l < depth of destination
                        # self.print("node", node, source, dest)
                        # self.print("layer", l)
                        #THIS IS WHERE TO TRACK FORWARDING IF WANTED
                        
                        if ('NEST' in node):
                            assert('gene' not in node)
                            assert('gene' not in dest)
                            for a in range(0, nodes_per_assembly):
                                assert(node_idx[node] >= 689)
                                source_idx = 689 + ((node_idx[node] - 689) * nodes_per_assembly) + a
                                #for b in range(0, nodes_per_assembly):
                                    
                                    #assert(node_idx[dest] >= 689)
                                    
                                    #dest_idx = 689 + ((node_idx[dest] - 689) * nodes_per_assembly) + b
                                    #layer_mask[source_idx, dest_idx] = 1
                                    # if (l == 1):
                                    #     self.print(f"Source: {source} {node_depths[source]}, node: {node}, dest: {dest} {node_depths[dest]}")
                                forward_layer_mask[source_idx, source_idx] = 1
                                #self.print(l, a, source_idx)
                        else:
                            assert('gene' in node)
                            # assert(node_idx[node] < 689)
                            # for a in range(0, nodes_per_assembly):
                            #     source_idx = 689 + ((node_idx[dest] - 689) * nodes_per_assembly) + a
                                #layer_mask[source_idx, source_idx] = 1
                            #Just forwarding gene to itself in next layer
                            source_idx = node_idx[source]
                            forward_layer_mask[source_idx, source_idx] = 1
                        #layer_mask[node_idx[node], node_idx[node]] = 1
            layer_masks.append(layer_mask)
            forward_layer_masks.append(forward_layer_mask)
            final_asm_layer_masks.append(final_asm_layer_mask)
        # print("Done Making Layer Masks")
        return layer_masks, forward_layer_masks, final_asm_layer_masks


# import os

# if __name__ == "__main__":
#     os.environ["CUDA_VISIBLE_DEVICES"] = "10,11,12,13"
#     device = f'cuda:{0}'
#     torch.cuda.set_device(0)
#     nn = eNest(6).to(device)
#     nn.register_grad_hooks()
#     X = torch.rand(10, 689).to(device)
#     out = nn(X)
#     self.print("All done")
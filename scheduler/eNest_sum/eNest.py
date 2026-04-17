from torch import nn
import numpy as np
import torch
import torch.nn as nn
import networkx as nx
import networkx.algorithms.components.connected as nxacc
import networkx.algorithms.dag as nxadag
from collections import OrderedDict


class eNest(nn.Module):
    def __init__(self, nodes_per_assembly, dropout=0.0, activation=nn.Tanh, 
                 output_method='final', batchnorm_position='before_activation', verbosity=-1):
        """
        eNest model with configurable output method and batchnorm position.
        
        Args:
            nodes_per_assembly: Number of nodes per assembly
            dropout: Dropout probability
            activation: Activation function class (nn.Tanh or nn.ReLU)
            output_method: Output method for final prediction:
                - 'final': Use only final assembly prediction (original behavior)
                - 'sum': Sum of all assembly predictions including final
                - 'linear': Pass all assembly predictions through linear layer + sigmoid
            batchnorm_position: Position of batchnorm relative to activation:
                - 'before_activation': Linear -> BatchNorm -> Activation (original)
                - 'after_activation': Linear -> Activation -> BatchNorm
            verbosity: Verbosity level (-1 for silent)
        """
        super().__init__()
        datadir = "/nfs/ml_lab/projects/Pilot1_PreclinicalHPC/jchia_backup/ANL_Drug_CData"
        ontfile = f"{datadir}/red_ontology.txt"
        gene2idfile = f"{datadir}/red_gene2ind.txt"
        self.verbosity = verbosity
        self.activation = activation
        self.output_method = output_method
        self.batchnorm_position = batchnorm_position
        
        # Assert that activation function returns 0 for input 0 (required for proper masking behavior)
        test_input = torch.tensor(0.0)
        activation_instance = self.activation()
        test_output = activation_instance(test_input)
        assert torch.allclose(test_output, torch.tensor(0.0), atol=1e-6), f"Activation function {activation.__name__} must return 0 for input 0, but got {test_output.item()}"
        
        self.np_lm, self.np_flm, self.np_fasmlm, self.node_idx, self.assembly_count = setup_Nest_Masks.get_layer_masks(ontfile, gene2idfile, nodes_per_assembly)
        self.nodes_per_assembly = nodes_per_assembly
        
        # Mask for when things are forwarded only
        self.forward_layer_masks = list()
        # Mask for the final destination of things
        self.layer_masks = list()
        # Mask for assemblies only when they accumulate all inputs
        self.final_asm_layer_masks = list()
        # Should only be 1 root
        self.hidden_assemblies = self.assembly_count - 1
        
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

        self.layer_bias_masks = list()
        for i in range(0, len(self.layer_masks)):
            active_inputs_per_output = self.layer_masks[i].sum(dim=0)
            dead_outputs = (active_inputs_per_output == 0)
            bias_mask = torch.ones(active_inputs_per_output.shape)
            bias_mask[dead_outputs] = 0.0
            self.register_buffer(f'layer_bias_masks_{i}', bias_mask)
            self.layer_bias_masks.append(getattr(self, f'layer_bias_masks_{i}'))

        self.linear_layer_list = nn.ModuleList()
        self.batchnorm_layer_list = nn.ModuleList()
        for i in range(0, len(self.layer_masks)):
            layer_shape = self.layer_masks[i].shape
            linear_layer = nn.Linear(layer_shape[0], layer_shape[1])
            # Initialize weights before masking
            self._init_weights(linear_layer)
            self.linear_layer_list.append(linear_layer)
            self.batchnorm_layer_list.append(nn.BatchNorm1d(layer_shape[1]))
        
        self.final_output_linear_layer = nn.Linear(nodes_per_assembly, 1)
        # Initialize weights before any other operations
        self._init_weights(self.final_output_linear_layer)
        self.final_output_activation = nn.Sigmoid()

        # Assembly output layers (for hidden assembly predictions)
        self.asm_out_linear_layer_1 = nn.Linear(self.hidden_assemblies * self.nodes_per_assembly, self.hidden_assemblies)
        # Initialize weights before masking
        self._init_weights(self.asm_out_linear_layer_1)
        
        asm_out_linear_layer_1_mask = torch.zeros(self.hidden_assemblies * self.nodes_per_assembly, self.hidden_assemblies)
        for asm_idx in range(0, self.hidden_assemblies):
            for a in range(0, self.nodes_per_assembly):
                idx = self.nodes_per_assembly * asm_idx + a
                asm_out_linear_layer_1_mask[idx, asm_idx] = 1
        
        self.register_buffer(f'asm_out_linear_layer_1_mask', asm_out_linear_layer_1_mask)
        self.asm_out_linear_layer_1.weight.data *= self.asm_out_linear_layer_1_mask.T
        
        self.asm_out_linear_layer_2 = nn.Linear(self.hidden_assemblies, self.hidden_assemblies)
        # Initialize weights before masking
        self._init_weights(self.asm_out_linear_layer_2)

        asm_out_linear_layer_2_mask = torch.zeros(self.hidden_assemblies, self.hidden_assemblies)
        for asm_idx in range(0, self.hidden_assemblies):
            asm_out_linear_layer_2_mask[asm_idx, asm_idx] = 1
        
        self.register_buffer(f'asm_out_linear_layer_2_mask', asm_out_linear_layer_2_mask)
        self.asm_out_linear_layer_2.weight.data *= self.asm_out_linear_layer_2_mask.T
        
        self.dropout_layers = nn.ModuleList()
        
        for i in range(0, len(self.layer_masks)):
            linear_layer = self.linear_layer_list[i]
            linear_layer.weight.data *= self.layer_masks[i].T
            linear_layer.weight.data[self.forward_layer_masks[i].T] = 1.0
            linear_layer.bias.data *= self.layer_bias_masks[i]
            self.dropout_layers.append(nn.Dropout(p=dropout))
        
        # Output method specific layers
        if self.output_method == 'linear':
            # Linear layer to combine all assembly predictions (hidden + final)
            # Total assemblies = hidden_assemblies + 1 (for final)
            self.combined_output_linear = nn.Linear(self.hidden_assemblies + 1, 1)
            # Initialize weights
            self._init_weights(self.combined_output_linear)
            self.combined_output_activation = nn.Sigmoid()
        
        self.print("Efficient Nest VNN Initialized")
        self.print(f"Output method: {self.output_method}")
        self.print(f"BatchNorm position: {self.batchnorm_position}")
              
    def _init_weights(self, layer):
        """
        Initialize weights of a linear layer using Xavier uniform initialization.
        
        Args:
            layer: nn.Linear layer to initialize
        """
        # Initialize weights using Xavier uniform initialization
        nn.init.xavier_uniform_(layer.weight)
        
        # Initialize bias to zeros (standard practice)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)
            
    def forward(self, X):
        hidden_asm_states = list()
        for i in range(0, len(self.linear_layer_list)):
            X = self.dropout_layers[i](X)
            X = self.linear_layer_list[i](X)
            
            if self.batchnorm_position == 'before_activation':
                # Original: Linear -> BatchNorm -> Activation
                Z = self.activation()(X)
                Z = self.batchnorm_layer_list[i](Z)
            else:
                # Alternative: Linear -> Activation -> BatchNorm
                Z = self.batchnorm_layer_list[i](X)
                Z = self.activation()(Z)
            
            X = X * torch.sum(getattr(self, f'forward_layer_masks_{i}'), dim=0) + Z * getattr(self, f'layer_bias_masks_{i}')
            if (i < len(self.linear_layer_list) - 1):
                hidden_asm_states.append(X[:, self.final_asm_layer_masks[i]])

        # Process hidden assembly states
        hidden_asm_X = torch.cat(hidden_asm_states, dim=1)
        hidden_asm_X = self.activation()(self.asm_out_linear_layer_1(hidden_asm_X))
        hidden_asm_Y = self.asm_out_linear_layer_2(hidden_asm_X)  # [batch, hidden_assemblies]

        # Get final assembly output
        X = X[:, self.node_idx["NEST"] : self.node_idx["NEST"] + self.nodes_per_assembly]
        final_Y = self.final_output_activation(self.final_output_linear_layer(X))  # [batch, 1]
        
        # Compute output based on output_method
        if self.output_method == 'final':
            # Original behavior: return final prediction and hidden predictions separately
            return final_Y, hidden_asm_Y
        
        elif self.output_method == 'sum':
            # Sum of all assembly predictions (final + hidden)
            # final_Y is [batch, 1], hidden_asm_Y is [batch, hidden_assemblies]
            # Sum all predictions
            combined_sum = final_Y + hidden_asm_Y.sum(dim=1, keepdim=True)  # [batch, 1]
            # Apply sigmoid to keep in [0, 1] range
            combined_output = torch.sigmoid(combined_sum)
            return combined_output, hidden_asm_Y
        
        elif self.output_method == 'linear':
            # Pass all assembly predictions through a linear layer + sigmoid
            # Concatenate final and hidden predictions: [batch, 1 + hidden_assemblies]
            all_predictions = torch.cat([final_Y, hidden_asm_Y], dim=1)
            # Linear layer + sigmoid
            combined_output = self.combined_output_activation(self.combined_output_linear(all_predictions))
            return combined_output, hidden_asm_Y
        
        else:
            raise ValueError(f"Unknown output_method: {self.output_method}")
    
    # Call once device is on its desired device
    def register_grad_hooks(self):
        def grad_mask_hook(mask):
            def hook(grad):
                return grad * mask.type(grad.dtype)
            return hook
        for i in range(0, len(self.layer_masks)):
            linear_layer = self.linear_layer_list[i]
            linear_layer.weight.register_hook(grad_mask_hook(getattr(self, f'layer_masks_{i}').T))
            linear_layer.bias.register_hook(grad_mask_hook(getattr(self, f'layer_bias_masks_{i}')))
        self.asm_out_linear_layer_1.weight.register_hook(grad_mask_hook(getattr(self, f'asm_out_linear_layer_1_mask').T))
        self.asm_out_linear_layer_2.weight.register_hook(grad_mask_hook(getattr(self, f'asm_out_linear_layer_2_mask').T))
        assert(getattr(self, f'layer_bias_masks_{i}').device == getattr(self, f'layer_masks_{i}').T.device)

    def print(self, msg):
        if (self.verbosity < 0):
            return
        else:
            print(msg)


class setup_Nest_Masks():
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
        return lm, flm, fasmlm, n2i, num_assemblies

    @staticmethod
    def load_mapping(mapping_file, mapping_type):
        mapping = {}
        file_handle = open(mapping_file)
        for line in file_handle:
            line = line.rstrip().split()
            mapping[line[1]] = int(line[0])
        file_handle.close()
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
            if len(term_gene_set) == 0:
                print('There is empty terms, please delete term:', term)
                import sys
                sys.exit(1)
            else:
                term_size_map[term] = len(term_gene_set)

        roots = [n for n in dG.nodes if dG.in_degree(n) == 0]

        uG = dG.to_undirected()
        connected_subG_list = list(nxacc.connected_components(uG))

        if len(roots) > 1:
            print('There are more than 1 root of ontology. Please use only one root.')
            import sys
            sys.exit(1)
        if len(connected_subG_list) > 1:
            print('There are more than 1 connected components. Please connect them.')
            import sys
            sys.exit(1)

        return dG, term_direct_gene_map, len(dG.nodes)
    
    @staticmethod
    def add_genes(nestG, term_direct_gene_map):
        '''Adds connections in graph for genes'''
        for node, inputs in term_direct_gene_map.items():
            for gene in inputs:
                if (nestG.has_node(f"gene_{gene}") != True):
                    nestG.add_node(f"gene_{gene}")
                nestG.add_edge(node, f"gene_{gene}")
        return nestG

    @staticmethod
    def node_to_idx_dict(G):
        '''Takes in NeST graph and assigns nodes an idx in each layer'''
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
        G = G.reverse()
        sorted_nodes = list(nx.topological_sort(G))
        node_depths = {node: 0 for node in G.nodes}

        for node in sorted_nodes:
            for pred in G.predecessors(node):
                node_depths[node] = max(node_depths[node], node_depths[pred] + 1)
        
        return node_depths

    @staticmethod
    def make_layer_masks(G, node_depths, node_idx, nodes_per_assembly=1):
        '''Takes 3 arguments, G and node_depths and returns the 1, 0 masks for each layer'''
        assert(len(G.nodes()) == 820)
        assert(len(node_depths.values()) == 820)
        assert(G.number_of_nodes() == 820)
        max_depth = max(node_depths.values())
        assert(nodes_per_assembly > 0)
        layer_size = 689 + nodes_per_assembly * 131
        layer_masks = list()
        forward_layer_masks = list()
        final_asm_layer_masks = list()
        
        for l in range(0, max_depth+1):
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
                for dest, source in G.in_edges(node):
                    assert(node == source)
                    if (node_depths[dest] == l):
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
                                    final_asm_layer_mask[dest_idx] = 1
                        else:
                            assert('gene' in node)
                            assert('NEST' in dest)
                            for b in range(0, nodes_per_assembly):
                                assert(node_idx[dest] >= 689)
                                dest_idx = 689 + ((node_idx[dest] - 689) * nodes_per_assembly) + b
                                layer_mask[node_idx[node], dest_idx] = 1
                                final_asm_layer_mask[dest_idx] = 1
                    elif ((l > node_depths[source]) and (node_depths[dest] > l)):
                        if ('NEST' in node):
                            assert('gene' not in node)
                            assert('gene' not in dest)
                            for a in range(0, nodes_per_assembly):
                                assert(node_idx[node] >= 689)
                                source_idx = 689 + ((node_idx[node] - 689) * nodes_per_assembly) + a
                                forward_layer_mask[source_idx, source_idx] = 1
                        else:
                            assert('gene' in node)
                            source_idx = node_idx[source]
                            forward_layer_mask[source_idx, source_idx] = 1
            
            layer_masks.append(layer_mask)
            forward_layer_masks.append(forward_layer_mask)
            final_asm_layer_masks.append(final_asm_layer_mask)
        
        return layer_masks, forward_layer_masks, final_asm_layer_masks

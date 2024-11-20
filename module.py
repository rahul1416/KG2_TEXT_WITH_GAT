# coding=utf-8
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn.conv.gcn_conv import GCNConv
from torch_geometric.nn.conv.rgcn_conv import RGCNConv 
from torch_geometric.nn import GATConv
from transformers import RobertaTokenizer, RobertaForMaskedLM
import math
import numpy as np
import torch.backends.cudnn as cudnn


class ListModule(nn.Module):
    """
    Abstract list layer class.
    """

    def __init__(self, *args):
        """
        Model initializing.
        """
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        """
        Getting the indexed layer.
        """
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        """
        Iterating on the layers.
        """
        return iter(self._modules.values())

    def __len__(self):
        """
        Number of layers.
        """
        return len(self._modules)


class GraphEncoder(nn.Module):
    def __init__(self, num_nodes, num_relations, gnn_layers, embedding_size, initialized_embedding, dropout_ratio=0.3, use_gat=False):
        super(GraphEncoder, self).__init__()
        self.num_nodes = num_nodes
        self.num_relations = num_relations
        self.gnn_layers = gnn_layers
        self.embedding_size = embedding_size
        self.dropout_ratio = dropout_ratio
        self.use_gat = use_gat  # Flag to switch between RGCN and GAT
        self.node_embedding = nn.Embedding(num_nodes, embedding_size)
        self.node_embedding.from_pretrained(torch.from_numpy(np.load(initialized_embedding)), freeze=False)
        
        self.dropout = nn.Dropout(dropout_ratio)
        
        self.layer_norm = nn.LayerNorm(embedding_size)

        self.gnn = nn.ModuleList()

        if use_gat:
            for _ in range(gnn_layers):
                self.gnn.append(GATConv(embedding_size, embedding_size, heads=8, dropout=dropout_ratio))
        else:
            for _ in range(gnn_layers):
                self.gnn.append(RGCNConv(embedding_size, embedding_size, num_relations=num_relations))

    def forward(self, nodes, edges, types):
        """
        :param nodes: Tensor, shape [batch_size, num_nodes]
        :param edges: List[List[edge_idx]], where each edge_idx is [2, num_edges]
        :param types: List[List[edge_types]], where each edge_types is [num_edges]
        """
        batch_size = nodes.size(0)
        device = nodes.device
        node_embeddings = []
        for bid in range(batch_size):
            edge_index = torch.tensor(edges[bid], dtype=torch.long, device=device)  # Shape: [2, num_edges]
            edge_type = torch.tensor(types[bid], dtype=torch.long, device=device)  # Shape: [num_edges]

            embed = self.node_embedding(nodes[bid, :])

            for lidx, gnn_layer in enumerate(self.gnn):
                residual = embed  
                
                if self.use_gat:
                    embed = gnn_layer(embed, edge_index)
                else:
                    embed = gnn_layer(embed, edge_index, edge_type)
                embed = self.layer_norm(embed)
                embed = self.dropout(F.relu(embed + residual))  
            node_embeddings.append(embed)
        node_embeddings = torch.stack(node_embeddings, 0) 
        return node_embeddings

class GATEncoder(nn.Module):
    def __init__(self, embedding_size, gnn_layers, dropout_ratio=0.3):
        super(GATEncoder, self).__init__()
        self.embedding_size = embedding_size
        self.gnn_layers = gnn_layers
        self.dropout_ratio = dropout_ratio
        self.dropout = nn.Dropout(dropout_ratio)
        self.layer_norm = nn.LayerNorm(embedding_size)
        self.gnn = nn.ModuleList([GATConv(embedding_size, embedding_size, heads=8, dropout=dropout_ratio) for _ in range(gnn_layers)])
        
        # Linear layers to adjust output size to match embedding size
        # After GAT layer with 8 heads (embedding_size * 8), use a linear layer to match the original embedding size
        self.linear_layers = nn.ModuleList([nn.Linear(embedding_size * 8, embedding_size) for _ in range(gnn_layers)])

    def forward(self, node_embeddings, edges):
        """
        :param node_embeddings: Tensor, shape [batch_size, num_nodes, embedding_size] (from GraphEncoder)
        :param edges: List[List[edge_idx]], where each edge_idx is [2, num_edges] for each batch
        """
        batch_size = node_embeddings.size(0)
        device = node_embeddings.device

        refined_embeddings = []
        for bid in range(batch_size):
            edge_index = torch.tensor(edges[bid], dtype=torch.long, device=device)  # Shape: [2, num_edges]
            embed = node_embeddings[bid]  
            for lidx, gnn_layer in enumerate(self.gnn):
                residual = embed  
                embed = gnn_layer(embed, edge_index)  
                embed = self.linear_layers[lidx](embed) 
                embed = self.layer_norm(embed)
                embed = self.dropout(F.relu(embed + residual))  
            refined_embeddings.append(embed)

        refined_embeddings = torch.stack(refined_embeddings, 0) 
        return refined_embeddings


class GraphReconstructor(nn.Module):
    def __init__(self, num_relations, hidden_size):
        super(GraphReconstructor, self).__init__()
        self.num_relations = num_relations
        self.hidden_size = hidden_size

        self.proj_linear = nn.Linear(3 * hidden_size, num_relations)

    def forward(self, pairs, hidden_states):
        """
        :param pairs: tensor [batch_size, num_pairs, 2, 2]
        :param hidden_states: tensor [batch_size, seq_len, hidden_size]
        """
        batch_size, num_pairs = pairs.size(0), pairs.size(1)
        hidden_size = hidden_states.size(-1)

        head, tail = torch.chunk(pairs, chunks=2, dim=2)

        h_start, h_end = torch.chunk(head, chunks=2, dim=3)
        t_start, t_end = torch.chunk(tail, chunks=2, dim=3)

        hs_expand = h_start.contiguous().view(batch_size, num_pairs).unsqueeze(-1).expand(-1, -1, hidden_size)
        hs_embed = torch.gather(hidden_states, dim=1, index=hs_expand)

        he_expand = h_end.contiguous().view(batch_size, num_pairs).unsqueeze(-1).expand(-1, -1, hidden_size)
        he_embed = torch.gather(hidden_states, dim=1, index=he_expand)

        head_embed = (hs_embed + he_embed) / 2.0

        ts_expand = t_start.contiguous().view(batch_size, num_pairs).unsqueeze(-1).expand(-1, -1, hidden_size)
        ts_embed = torch.gather(hidden_states, dim=1, index=ts_expand)

        te_expand = t_end.contiguous().view(batch_size, num_pairs).unsqueeze(-1).expand(-1, -1, hidden_size)
        te_embed = torch.gather(hidden_states, dim=1, index=te_expand)

        tail_embed = (ts_embed + te_embed) / 2.0

        logits = self.proj_linear(torch.cat([head_embed, tail_embed, head_embed * tail_embed], dim=-1))

        return logits


class GraphPointer(nn.Module):
    def __init__(self, embedding_size, hidden_size):
        super(GraphPointer, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.proj_linear = nn.Linear(embedding_size + hidden_size, 1)

    def forward(self, embeddings, hidden_states, pointer):
        """
        :param embeddings: tensor [batch_size, seq_len, embedding_size]
        :param hidden_states: tensor [batch_size, seq_len, hidden_size]
        :param pointer: tensor [batch_size, seq_len]
        """
        copy_prob = torch.sigmoid(self.proj_linear(torch.cat([embeddings, hidden_states], dim=-1))).squeeze(-1)
        copy_prob = torch.where(pointer.bool(), 1 - copy_prob, copy_prob)

        return copy_prob
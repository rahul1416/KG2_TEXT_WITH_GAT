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
                embed = self.dropout(F.relu(embed + residual))  # No normalization here
            node_embeddings.append(embed)
        node_embeddings = torch.stack(node_embeddings, 0) 
        return node_embeddings
class GATEncoder(nn.Module):
    def __init__(self, embedding_size, gnn_layers, dropout_ratio=0.3):
        super(GATEncoder, self).__init__()
        self.embedding_size = embedding_size
        self.gnn_layers = gnn_layers
        self.dropout_ratio = dropout_ratio
        
        # Dropout
        self.dropout = nn.Dropout(dropout_ratio)
        
        # Define GAT layers
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
                embed = self.linear_layers[lidx](embed)  # Apply linear layer after GAT
                embed = self.dropout(F.relu(embed + residual))  # No normalization here
            refined_embeddings.append(embed)

        refined_embeddings = torch.stack(refined_embeddings, 0) 
        return refined_embeddings

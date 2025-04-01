import torch  # Missing import

from typing import Optional
from torch_geometric.typing import Adj, OptTensor

from torch import Tensor
from torch.nn import Linear, Parameter
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing
from typing import Optional
from torch_geometric.typing import Adj, OptTensor

from torch import Tensor
from torch.nn import Linear
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm

class SGATConv(MessagePassing):
    _cached_x: Optional[Tensor]

    def __init__(self, in_channels: int, out_channels: int, K: int = 1,
                 heads: int = 1, concat: bool = True, dropout: float = 0.6,
                 cached: bool = False, bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(SGATConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.heads = heads
        self.concat = concat
        self.cached = cached
        self.dropout = dropout

        self._cached_x = None

        self.lin = Linear(in_channels, out_channels * heads, bias=False)
        self.att = Parameter(Tensor(1, heads, out_channels))
        
        if bias and concat:
            self.bias = Parameter(Tensor(out_channels * heads))
        elif bias:
            self.bias = Parameter(Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        torch.nn.init.xavier_uniform_(self.att)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)
        self._cached_x = None

    def forward(self, x: Tensor, edge_index: Adj, edge_weight: OptTensor = None) -> Tensor:
        cache = self._cached_x
        if cache is None:
            x = self.lin(x).view(-1, self.heads, self.out_channels)
            for k in range(self.K):
                x = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)
                if self.cached:
                    self._cached_x = x
        else:
            x = cache
        
        if self.concat:
            x = x.view(-1, self.heads * self.out_channels)
        else:
            x = x.mean(dim=1)
        
        if self.bias is not None:
            x = x + self.bias
        
        return x

    def message(self, x_j: Tensor, x_i: Tensor, edge_weight: OptTensor) -> Tensor:
        alpha = (x_i * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, negative_slope=0.2)
        alpha = F.softmax(alpha, dim=1)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha.unsqueeze(-1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {}, heads={}, K={})'.format(self.__class__.__name__,
                                                   self.in_channels, self.out_channels,
                                                   self.heads, self.K)

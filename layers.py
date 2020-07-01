# Author: Piyush Vyas
import math
import torch
from torch.nn.parameter import Parameter


class GraphConv(torch.nn.Module):
    """ Applies the Graph Convolution operation to the incoming data: math: `X' = \hat{A}XW`

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        dropout: dropout probability
            Default: 0.2
        bias: If set to ``True ``, the layer will learn an additive bias.
            Default: ``False``
        normalize: If set to ``False``, the layer will not apply Layer Normalization to the features.
            Default: ``True``
        last: If set to ``True``, the layer will act as the final/classification layer and return logits.
            Default: ``False`` 
    Shape:
        - Input: :math:`(N, H_{in})` where :math:`H_{in} = \text{in\_features}`
        - Output: :math:`(N, H_{out})` where :math:`H_{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
                :math:`(\text{out\_features}, \text{in\_features})`. The values are
                initialized from :math:`\mathcal{U}(-\text{bound}, \text{bound})` where
                :math: `\text{bound} = \sqrt{\frac{6}{\text{in\_features + out\_features}}}`
        bias: the learnable bias of the module of shape
              :math:`(\text{out\_features})`.
              If :attr:`bias` is ``True``, the values are initialized with the scalar value `0`.

    """

    __constants__ = ['in_features, out_features']

    def __init__(self, in_features, out_features, dropout=0.2, bias=False, normalize=True, last=False, precalc=False):
        super(GraphConv, self).__init__()
        self.in_features = in_features*2
        self.out_features = out_features
        self.normalize = normalize
        self.p = dropout
        self.last = last
        self.precalc = precalc
        if not last:
            self.layer_norm = torch.nn.LayerNorm(normalized_shape=out_features, elementwise_affine=True)
        else:
            self.layer_norm = lambda x: x
        self.weight = Parameter(torch.Tensor(self.out_features, self.in_features), requires_grad=True)
        if bias:
            self.bias = Parameter(torch.Tensor(self.out_features), requires_grad=True)
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # Xavier Glorot Uniform
        bound = math.sqrt(6.0/float(self.out_features + self.in_features))
        self.weight.data.uniform_(-bound, bound)

        # Kaiming He Uniform
        # torch.nn.init.kaiming_uniform_(self.weight, a=0.01, mode='fan_in', nonlinearity='leaky_relu')

        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, A, x):
        if self.precalc:
            support = x
        else:
            support = torch.sparse.mm(A, x)  # (N, N) x (N, F) -> (N, F)
            support = torch.cat((support, x), dim=1)
        support = torch.nn.functional.dropout(support, p=self.p, training=self.training)
        output = torch.nn.functional.linear(support, self.weight, self.bias)

        if self.last:
            return output

        if self.normalize:
            output = self.layer_norm(output)
        return torch.nn.functional.relu(output)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None)

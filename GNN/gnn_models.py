#############
##  MLP    ##
#############
#mlp.py

from typing import Dict, Union
from itertools import product

from typing import List, Any, Iterable

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_dim: int, fc_dims: Iterable[int], nonlinearity: nn.Module,
                 dropout_p: float = 0, use_batchnorm: bool = False, last_output_free: bool = False):
        super().__init__()
        assert isinstance(fc_dims, (list, tuple)
                          ), f"fc_dims must be a list or a tuple, but got {type(fc_dims)}"

        self.input_dim = input_dim
        self.fc_dims = fc_dims
        self.nonlinearity = nonlinearity
        # if dropout_p is None:
        #     dropout_p = 0
        # if use_batchnorm is None:
        # use_batchnorm = False
        self.dropout_p = dropout_p
        self.use_batchnorm = use_batchnorm

        layers: List[nn.Module] = []
        for layer_i, dim in enumerate(fc_dims):
            layers.append(nn.Linear(input_dim, dim))
            if last_output_free and layer_i == len(fc_dims) - 1:
                continue

            layers.append(nonlinearity)
            if dim != 1:
                if use_batchnorm:
                    layers.append(nn.BatchNorm1d(dim))
                if dropout_p > 0:
                    layers.append(nn.Dropout(p=dropout_p))
            input_dim = dim
        self.fc_layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc_layers(x)

    @property
    def output_dim(self) -> int:
        return self.fc_dims[-1]
    
#############
## Utils   ##
#############

from typing import Iterable, Tuple, Mapping, Dict

import torch
from torch_scatter import scatter
from torch_geometric.utils import softmax as edge_softmax

#from models.mlp import MLP


def aggregate_features(features: torch.Tensor, agg_index: torch.Tensor, num_nodes: int, agg_mode: str, attention_model=None, edge_attr=None, **kwargs):
    if "attention" not in agg_mode:
        return scatter(src=features, index=agg_index, reduce=agg_mode, dim=0, dim_size=num_nodes)

    if "classifier" in agg_mode:
        assert edge_attr is not None
        coeffs = attention_model(edge_attr)
    else:
        coeffs = attention_model(features)
    assert coeffs.shape == (len(features), 1), f"features {features.shape}, coeffs {coeffs.shape}"
    # normalize coefficients for each node to sum to 1 (forward edges only)
    if "softmax" in agg_mode:
        coeffs = edge_softmax(coeffs, index=agg_index, num_nodes=num_nodes)
    elif "normalized" in agg_mode:
        coeffs = normalize_edge_coefficients(coeffs, index=agg_index, num_nodes=num_nodes)
    weighted_features = features * coeffs  # multiply by their attention coefficients and sum up
    return scatter(src=weighted_features, index=agg_index, reduce="sum", dim=0, dim_size=num_nodes)


def normalize_edge_coefficients(coeffs: torch.Tensor, index: torch.Tensor, num_nodes: int):
    coeffs_sum = scatter(coeffs, index=index, reduce="sum", dim=0, dim_size=num_nodes)
    coeffs_sum_selected = coeffs_sum.index_select(index=index, dim=0)
    return coeffs / (coeffs_sum_selected + 1e-16)


def dims_from_multipliers(output_dim: int, multipliers: Iterable[int]) -> Tuple[int, ...]:
    return tuple(int(output_dim * mult) for mult in multipliers)


def model_params_from_params(params: Mapping, model_prefix: str, param_names: Iterable[str]):
    return {param_name: params[f"{model_prefix}_{param_name}"] for param_name in param_names}

###########
## MPN ####
############
#mpn.py

from typing import List, Optional

import torch
from torch import nn
from torch_geometric.data import Data

class MessagePassingNetworkNonRecurrent(nn.Module):
    def __init__(self, edge_models: List[nn.Module], node_models: List[nn.Module], steps: int, use_same_frame: bool=False):
        """
        Args:
            edge_models: a list/tuple of callable Edge Update models
            node_models: a list/tuple of callable Node Update models
        """
        super().__init__()
        assert len(edge_models) == steps, f"steps={steps} not equal edge models {len(edge_models)}"
        assert len(node_models) == steps - 1, f"steps={steps} -1 not equal node models {len(node_models)}"
        self.edge_models = nn.ModuleList(edge_models)
        self.node_models = nn.ModuleList(node_models)
        self.steps = steps
        self.use_same_frame = use_same_frame

    def forward(self, x, edge_index, edge_attr, num_nodes: int):
        """
        Does a single node and edge feature vectors update.
        Args:
            x: node features matrix
            edge_index: tensor with shape [2, M], with M being the number of edges, indicating nonzero entries in the
            graph adjacency (i.e. edges)
            edge_attr: edge features matrix (ordered by edge_index)
        Returns: Updated Node and Edge Feature matrices
        """
        edge_embeddings = []
        for step, (edge_model, node_model) in enumerate(zip(self.edge_models, self.node_models.append(None))):
            # Edge Update
            edge_attr_mpn = edge_model(x, edge_index, edge_attr)
            edge_embeddings.append(edge_attr_mpn)

            if step == self.steps - 1:
                continue  # do not process nodes in the last step - only edge features are used for classification
            # Node Update
            x = node_model(x, edge_index, edge_attr_mpn)
        assert len(
            edge_embeddings) == self.steps, f"Collected {len(edge_embeddings)} edge embeddings for {self.steps} steps"
        return x, edge_embeddings


class MessagePassingNetworkRecurrent(nn.Module):
    def __init__(self, edge_model: nn.Module, node_model: nn.Module, steps: int,
                 use_same_frame: bool = False, same_frame_edge_model: Optional[nn.Module] = None):
        """
        Args:
            edge_model: an Edge Update model
            node_model: an Node Update model
        """
        super().__init__()
        self.edge_model = edge_model
        self.same_frame_edge_model = same_frame_edge_model
        self.node_model = node_model
        self.steps = steps
        self.use_same_frame = use_same_frame

    def forward(self, x, edge_index, edge_attr, num_nodes: int, same_frame_edge_index=None, same_frame_edge_attr=None):
        """
        Does a single node and edge feature vectors update.
        Args:
            x: node features matrix
            edge_index: tensor with shape [2, M], with M being the number of edges, indicating nonzero entries in the
            graph adjacency (i.e. edges)
            edge_attr: edge features matrix (ordered by edge_index)
        Returns: Updated Node and Edge Feature matrices
        """
        for step in range(self.steps):
            # Edge Update
            edge_attr_mpn = self.edge_model(x, edge_index, edge_attr)
            if self.use_same_frame:
                if self.same_frame_edge_model is not None:
                    same_frame_edge_attr_mpn = self.same_frame_edge_model(x, same_frame_edge_index, same_frame_edge_attr)
                else:
                    same_frame_edge_attr_mpn = self.edge_model(x, same_frame_edge_index, same_frame_edge_attr)
            else:
                same_frame_edge_attr_mpn = None

            if step == self.steps - 1:
                continue  # do not process nodes in the last step - only edge features are used for classification
            # Node Update
            x = self.node_model(x, edge_index, edge_attr_mpn,
                                same_frame_edge_index=same_frame_edge_index,
                                same_frame_edge_attr=same_frame_edge_attr_mpn)
        return x, edge_attr_mpn


class MessagePassingNetworkRecurrentNodeEdge(nn.Module):
    def __init__(self, edge_model: nn.Module, node_model: nn.Module, steps: int,
                 use_same_frame: bool = False, same_frame_edge_model: Optional[nn.Module] = None):
        """
        Args:
            edge_model: an Edge Update model
            node_model: an Node Update model
        """
        super().__init__()
        self.edge_model = edge_model
        self.same_frame_edge_model = same_frame_edge_model
        self.node_model = node_model
        self.steps = steps
        self.use_same_frame = use_same_frame

    def forward(self, x, edge_index, edge_attr, num_nodes: int, same_frame_edge_index=None, same_frame_edge_attr=None):
        """
        Does a single node and edge feature vectors update.
        Args:
            x: node features matrix
            edge_index: tensor with shape [2, M], with M being the number of edges, indicating nonzero entries in the
            graph adjacency (i.e. edges)
            edge_attr: edge features matrix (ordered by edge_index)
        Returns: Updated Node and Edge Feature matrices
        """
        edge_attr_mpn = edge_attr
        same_frame_edge_attr_mpn = same_frame_edge_attr
        for step in range(self.steps):
            # Node Update
            x = self.node_model(x, edge_index, edge_attr_mpn,
                                same_frame_edge_index=same_frame_edge_index,
                                same_frame_edge_attr=same_frame_edge_attr_mpn)

            # Edge Update
            edge_attr_mpn = self.edge_model(x, edge_index, edge_attr)
            if self.use_same_frame:
                if self.same_frame_edge_model is not None:
                    same_frame_edge_attr_mpn = self.same_frame_edge_model(x, same_frame_edge_index, same_frame_edge_attr)
                else:
                    same_frame_edge_attr_mpn = self.edge_model(x, same_frame_edge_index, same_frame_edge_attr)
            else:
                same_frame_edge_attr_mpn = None

        return x, edge_attr_mpn
    
####################
## Edge Models #####
####################
#edge_models.py

from typing import List, Any

import torch
from torch import nn


class BasicEdgeModel(nn.Module):
    """ Class used to peform an edge update during neural message passing """

    def __init__(self, edge_mlp):
        super(BasicEdgeModel, self).__init__()
        self.edge_mlp = edge_mlp

    def forward(self, x, edge_index, edge_attr):
        source_nodes, target_nodes = edge_index
        # assert len(source_nodes) == len(target_nodes) == len(
            # edge_attr), f"Different lengths {len(source_nodes)}, {len(target_nodes)}, {len(edge_attr)} "
        merged_features = torch.cat([x[source_nodes], x[target_nodes], edge_attr], dim=1)
        # print(f"merged_features, {merged_features.shape}")
        assert len(merged_features) == len(source_nodes), f"Merged input has wrong length {merged_features.shape} != {edge_attr.shape}"
        return self.edge_mlp(merged_features)
    
####################
## Node Models #####
####################
#node_models.py

from tkinter import E
from typing import List, Any

import torch
from torch import nn
from torch_scatter import scatter
from torch_geometric.utils import softmax as edge_softmax


class TimeAwareNodeModel(nn.Module):
    """ Class used to peform a node update during neural message passing """

    def __init__(self, flow_forward_model: MLP, flow_backward_model: MLP, node_mlp: MLP, node_agg_mode: str):
        super().__init__()
        assert (flow_forward_model.output_dim + flow_backward_model.output_dim ==
                node_mlp.input_dim), f"Flow models have incompatible output/input dims"
        self.flow_forward_model = flow_forward_model
        self.flow_backward_model = flow_backward_model
        self.node_mlp = node_mlp
        self.node_agg_mode = node_agg_mode

    def forward(self, x, edge_index, edge_attr, **kwargs):
        past_nodes, future_nodes = edge_index

        """
        x[past_nodes]  # features of nodes emitting messages
        edge_attr       # emitted messages
        x[future_nodes] # features of nodes receiving messages
        """
        flow_forward_input = torch.cat([x[future_nodes], x[past_nodes], edge_attr], dim=1)  # [n_edges x edge_feature_count]
        assert len(flow_forward_input) == len(edge_attr)

        flow_forward = self.flow_forward_model(flow_forward_input)

        flow_forward_aggregated = scatter(src=flow_forward, index=future_nodes,
                                          reduce=self.node_agg_mode, dim=0, dim_size=len(x))

        # Collect and process backward-directed edge messages (present->past)
        flow_backward_input = torch.cat([x[past_nodes], x[future_nodes], edge_attr], dim=1)  # [n_edges x edge_feature_count]
        flow_backward = self.flow_backward_model(flow_backward_input)
        flow_backward_aggregated = scatter(src=flow_backward, index=past_nodes,
                                           reduce=self.node_agg_mode, dim=0, dim_size=len(x))

        # Concat both flows for each node
        flow_total = torch.cat((flow_forward_aggregated, flow_backward_aggregated), dim=1)
        return self.node_mlp(flow_total)


class ContextualNodeModel(nn.Module):
    """ Class used to peform a node update during neural message passing """

    def __init__(self, flow_forward_model: MLP, flow_frame_model: MLP, flow_backward_model: MLP,
                 total_flow_model: MLP, node_agg_mode: str, attention_model: nn.Module = None, node_aggr_sections: int = 3):
        super().__init__()
        assert (flow_forward_model.output_dim + flow_frame_model.output_dim + flow_backward_model.output_dim ==
                total_flow_model.input_dim), f"Flow models have incompatible output/input dims"
        self.flow_forward_model = flow_forward_model
        self.flow_frame_model = flow_frame_model
        self.flow_backward_model = flow_backward_model
        self.node_agg_mode = node_agg_mode
        self.attention_model = attention_model
        self.total_flow_model = total_flow_model
        self.node_aggr_sections = node_aggr_sections

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor,
                same_frame_edge_index: torch.Tensor, same_frame_edge_attr: torch.Tensor, **kwargs):
        past_nodes, future_nodes = edge_index
        """
        x[past_nodes]  # features of nodes emitting messages
        edge_attr       # emitted messages
        x[future_nodes] # features of nodes receiving messages
        """
        early_frame_nodes, later_frame_nodes = same_frame_edge_index

        # Collect and process forward-directed edge messages (past->present)
        flow_forward_input = torch.cat([x[future_nodes], x[past_nodes], edge_attr], dim=1)  # [n_edges x edge_feature_count]
        flow_forward = self.flow_forward_model(flow_forward_input)

        # Collect and process backward-directed edge messages (present->past)
        # [n_edges x edge_feature_count]
        flow_backward_input = torch.cat([x[past_nodes], x[future_nodes], edge_attr], dim=1)
        flow_backward = self.flow_backward_model(flow_backward_input)

        # Collect and process same frame edge messages
        flow_frame_input = torch.cat([x[early_frame_nodes],
                                      x[later_frame_nodes],
                                      same_frame_edge_attr], dim=1)  # [n_edges x edge_feature_count]
        flow_frame = self.flow_frame_model(flow_frame_input)

        if self.node_aggr_sections == 3:
            flow_forward_aggregated = aggregate_features(flow_forward, future_nodes,
                                                        len(x), self.node_agg_mode, self.attention_model, edge_attr=edge_attr)
            flow_backward_aggregated = aggregate_features(flow_backward, past_nodes,
                                                        len(x), self.node_agg_mode, self.attention_model, edge_attr=edge_attr)
            flow_frame_aggregated = aggregate_features(torch.vstack((flow_frame, flow_frame)),
                                                    torch.cat((early_frame_nodes, later_frame_nodes)),
                                                    len(x), self.node_agg_mode, self.attention_model, 
                                                    edge_attr=torch.vstack((same_frame_edge_attr, same_frame_edge_attr)))
        elif self.node_aggr_sections == 2:
            flow_temporal_aggregated = aggregate_features(torch.vstack((flow_forward, flow_backward)), 
                                                        torch.cat((future_nodes, past_nodes)),
                                                        len(x), self.node_agg_mode, self.attention_model, 
                                                        edge_attr=edge_attr)
            flow_temporal_aggregated /= 2.0
            flow_forward_aggregated = flow_temporal_aggregated
            flow_backward_aggregated = flow_temporal_aggregated
            flow_frame_aggregated = aggregate_features(torch.vstack((flow_frame, flow_frame)),
                                                    torch.cat((early_frame_nodes, later_frame_nodes)),
                                                    len(x), self.node_agg_mode, self.attention_model, 
                                                    edge_attr=torch.vstack((same_frame_edge_attr, same_frame_edge_attr)))
        elif self.node_aggr_sections == 1:
            flow_total_aggregated = aggregate_features(torch.vstack((flow_forward, flow_frame, flow_frame, flow_backward)),
                                                    torch.cat((future_nodes, early_frame_nodes, later_frame_nodes, past_nodes)),
                                                    len(x), self.node_agg_mode, self.attention_model, 
                                                    edge_attr=torch.vstack((edge_attr, same_frame_edge_attr, same_frame_edge_attr, edge_attr)))
            flow_total_aggregated /= 3.0
            flow_forward_aggregated = flow_total_aggregated
            flow_backward_aggregated = flow_total_aggregated
            flow_frame_aggregated = flow_total_aggregated

        # stack and aggregate everything, then triple duplicate for the total_flow_model

        # Concat both flows for each node
        flow_total = torch.cat((flow_forward_aggregated, flow_frame_aggregated, flow_backward_aggregated), dim=1)
        return self.total_flow_model(flow_total)


class UniformAggNodeModel(nn.Module):
    """ Class used to peform a node update during neural message passing """

    def __init__(self, flow_model, node_mlp, node_agg_mode: str):
        super().__init__()
        assert (flow_model.output_dim == node_mlp.input_dim), f"Flow models have incompatible output/input dims"
        self.flow_model = flow_model
        self.node_mlp = node_mlp
        self.node_agg_mode = node_agg_mode

    def forward(self, x, edge_index, edge_attr, **kwargs):
        past_nodes, future_nodes = edge_index

        """
        x[past_nodes]  # features of nodes emitting messages, past -> future
        edge_attr       # emitted messages
        x[future_nodes] # features of nodes receiving messages, future nodes receiving from earlier ones
        """
        # input features order does not matter as long as it is symmetric between two flow inputs
        #                               nodes receiving, nodes sending, edges
        flow_forward_input = torch.hstack((x[future_nodes], x[past_nodes], edge_attr))
        assert len(flow_forward_input) == len(edge_attr)
        # [n_edges x edge_feature_count]
        flow_backward_input = torch.hstack((x[past_nodes], x[future_nodes], edge_attr))
        assert flow_forward_input.shape == flow_backward_input.shape, f"{flow_forward_input.shape} != {flow_backward_input.shape}"

        # [2*n_edges x edge_feature_count]
        flow_total_input = torch.vstack((flow_forward_input, flow_backward_input))
        flow_processed = self.flow_model(flow_total_input)

        # aggregate features for each node based on features taken over each node
        # the index has to account for both incoming and outgoing edges - so that each edge is considered by both of its nodes
        flow_total = scatter(src=flow_processed, index=torch.cat((future_nodes, past_nodes)),
                             reduce=self.node_agg_mode, dim=0, dim_size=len(x))
        return self.node_mlp(flow_total)


class InitialTimeAwareNodeModel(nn.Module):
    """ Class used to peform an initial update for empty nodes at the beginning of message passing
    The initial features are simply a transformation of edge features and therefore depend largely on the graph structure
    The aggregation logic is the same as for `TimeAwareNodeModel` but without node features.

    The abscense of node features means that transforming features before aggregation is the same as transforming the input edge features before the node update. Therefore, the transform is only applied after aggregation and instead the initial edge model is more powerful in comparison.
    """

    def __init__(self, node_mlp, node_agg_mode: str):
        super().__init__()
        self.node_mlp = node_mlp
        self.node_agg_mode = node_agg_mode

    def forward(self, edge_index, edge_attr, num_nodes: int, **kwargs):
        past_nodes, future_nodes = edge_index
        # print(f"edge_attr, {edge_attr.shape}")

        flow_forward_aggregated = scatter(src=edge_attr, index=future_nodes,
                                          reduce=self.node_agg_mode, dim=0, dim_size=num_nodes)
        flow_backward_aggregated = scatter(src=edge_attr, index=past_nodes,
                                           reduce=self.node_agg_mode, dim=0, dim_size=num_nodes)

        # Concat both flows for each node
        flow_total = torch.cat((flow_forward_aggregated, flow_backward_aggregated), dim=1)
        # print(f"flow_total {flow_total.shape}")
        # print(f"initial flow_total {flow_total.grad_fn}")
        assert len(flow_total) == num_nodes
        return self.node_mlp(flow_total)


class InitialContextualNodeModel(nn.Module):
    """ Class used to peform an initial update for empty nodes at the beginning of message passing
    The initial features are simply a transformation of edge features and therefore depend largely on the graph structure
    The aggregation logic is the same as for `ContextualNodeModel` but without node features.

    The abscense of node features means that transforming features before aggregation is the same as transforming the input edge features before the node update. Therefore, the transform is only applied after aggregation and instead the initial edge model is more powerful in comparison.
    """

    def __init__(self, node_mlp, node_agg_mode: str, attention_model: nn.Module = None):
        super().__init__()
        self.node_mlp = node_mlp
        self.node_agg_mode = node_agg_mode
        self.attention_model = attention_model

    def forward(self, edge_index: torch.Tensor, edge_attr: torch.Tensor, num_nodes: int,
                same_frame_edge_index: torch.Tensor, same_frame_edge_attr: torch.Tensor, **kwargs):
        past_nodes, future_nodes = edge_index
        early_frame_nodes, later_frame_nodes = same_frame_edge_index

        flow_forward_aggregated = aggregate_features(edge_attr, future_nodes,
                                                     num_nodes, self.node_agg_mode, self.attention_model)
        flow_backward_aggregated = aggregate_features(edge_attr, past_nodes,
                                                      num_nodes, self.node_agg_mode, self.attention_model)
        flow_frame_aggregated = aggregate_features(torch.vstack((same_frame_edge_attr, same_frame_edge_attr)),
                                                   torch.cat((early_frame_nodes, later_frame_nodes)),
                                                   num_nodes, self.node_agg_mode, self.attention_model)
        # Concat all flows for each node
        flow_total = torch.cat((flow_forward_aggregated, flow_frame_aggregated, flow_backward_aggregated), dim=1)
        assert len(flow_total) == num_nodes
        return self.node_mlp(flow_total)


class InitialUniformAggNodeModel(nn.Module):
    """ Class used to peform an initial update for empty nodes at the beginning of message passing
    The initial features are simply a transformation of edge features and therefore depend largely on the graph structure
    The aggregation logic is the same as for `UniformAggNodeModel` but without node features.

    The abscense of node features means that transforming features before aggregation is the same as transforming the input edge features before the node update. Therefore, the transform is only applied after aggregation and instead the initial edge model is more powerful in comparison.
    """

    def __init__(self, node_mlp, node_agg_mode: str):
        super().__init__()
        self.node_mlp = node_mlp
        self.node_agg_mode = node_agg_mode

    def forward(self, edge_index, edge_attr, num_nodes: int, **kwargs):
        past_nodes, future_nodes = edge_index
        flow_total = scatter(src=torch.vstack((edge_attr, edge_attr)),
                             index=torch.cat((future_nodes, past_nodes)),
                             reduce=self.node_agg_mode, dim=0, dim_size=num_nodes)
        return self.node_mlp(flow_total)


class InitialZeroNodeModel(nn.Module):
    """ Class used to peform an initial update for empty nodes at the beginning of message passing
    The initial features are simply a transformation of edge features and therefore depend largely on the graph structure
    The aggregation logic is the same as for `ContextualNodeModel` but without node features.

    The abscense of node features means that transforming features before aggregation is the same as transforming the input edge features before the node update. Therefore, the transform is only applied after aggregation and instead the initial edge model is more powerful in comparison.
    """

    def __init__(self, node_dim: int):
        super().__init__()
        self.node_dim = node_dim

    def forward(self, edge_index: torch.Tensor, edge_attr: torch.Tensor, num_nodes: int,
                same_frame_edge_index: torch.Tensor, same_frame_edge_attr: torch.Tensor, device, **kwargs):
        return torch.zeros((num_nodes, self.node_dim), dtype=edge_attr.dtype, device=device)
    
###########################
## Utils Models ###########
##########################
#utils_models.py
def dims_from_multipliers(output_dim: int, multipliers: Iterable[int]) -> Tuple[int, ...]:
    return tuple(int(output_dim * mult) for mult in multipliers)
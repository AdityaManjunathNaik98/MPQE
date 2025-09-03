import torch
import torch.nn as nn

import random

from torch_scatter import scatter_add, scatter_max, scatter_mean, scatter_min
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch.nn import Parameter as Param
from torch_geometric.nn import inits



class MLPReadout(nn.Module):
    def __init__(self, input_dim, output_dim, scatter_fn):
        super(MLPReadout, self).__init__()
        self.layers = nn.Sequential(nn.Linear(in_features=input_dim,
                                              out_features=output_dim),
                                    nn.ReLU(),
                                    nn.Linear(in_features=output_dim,
                                              out_features=output_dim))
        self.scatter_fn = scatter_fn

    def forward(self, embs, batch_idx, **kwargs):
        x = self.layers(embs)
        x = self.scatter_fn(x, batch_idx, dim=0)

        # If scatter_fn is max or min, values and indices are returned
        if isinstance(x, tuple):
            x = x[0]

        return x


class TargetMLPReadout(nn.Module):
    def __init__(self, dim, scatter_fn):
        super(TargetMLPReadout, self).__init__()
        self.layers = nn.Sequential(nn.Linear(in_features=2*dim,
                                              out_features=dim),
                                    nn.ReLU(),
                                    nn.Linear(in_features=dim,
                                              out_features=dim))
        self.scatter_fn = scatter_fn

    def forward(self, embs, batch_idx, batch_size, num_nodes, num_anchors,
                **kwargs):
        device = embs.device

        non_target_idx = torch.ones(num_nodes, dtype=torch.bool)
        non_target_idx[num_anchors] = 0
        non_target_idx.to(device)

        batch_idx = batch_idx.reshape(batch_size, -1)
        batch_idx = batch_idx[:, non_target_idx].reshape(-1)

        embs = embs.reshape(batch_size, num_nodes, -1)
        non_targets = embs[:, non_target_idx]
        targets = embs[:, ~non_target_idx].expand_as(non_targets)

        x = torch.cat((targets, non_targets), dim=-1)
        x = x.reshape(batch_size * (num_nodes - 1), -1).contiguous()

        x = self.layers(x)
        x = self.scatter_fn(x, batch_idx, dim=0)

        # If scatter_fn is max or min, values and indices are returned
        if isinstance(x, tuple):
            x = x[0]

        return x



class RGCNConv(MessagePassing):
    r"""The relational graph convolutional operator from the `"Modeling
    Relational Data with Graph Convolutional Networks"
    <https://arxiv.org/abs/1703.06103>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{\Theta}_0 \cdot \mathbf{x}_i +
        \sum_{r \in \mathcal{R}} \sum_{j \in \mathcal{N}_r(i)}
        \frac{1}{|\mathcal{N}_r(i)|} \mathbf{\Theta}_r \cdot \mathbf{x}_j,

    where :math:`\mathcal{R}` denotes the set of relations, *i.e.* edge types.
    Edge type needs to be a one-dimensional :obj:`torch.long` tensor which
    stores a relation identifier
    :math:`\in \{ 0, \ldots, |\mathcal{R}| - 1\}` for each edge.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        num_relations (int): Number of relations.
        num_bases (int): Number of bases used for basis-decomposition.
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_relations,
                 num_bases,
                 bias=True):
        super(RGCNConv, self).__init__('add')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = num_bases

        if num_bases == 0:
            self.basis = Param(torch.Tensor(num_relations, in_channels, out_channels))
            self.att = None
        else:
            self.basis = Param(torch.Tensor(num_bases, in_channels, out_channels))
            self.att = Param(torch.Tensor(num_relations, num_bases))
        self.root = Param(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Param(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.att is None:
            size = self.num_relations * self.in_channels
        else:
            size = self.num_bases * self.in_channels
            inits.uniform(size, self.att)

        inits.uniform(size, self.basis)
        inits.uniform(size, self.root)
        inits.uniform(size, self.bias)

    def forward(self, x, edge_index, edge_type, edge_norm=None):
        """"""
        if x is None:
            x = torch.arange(
                edge_index.max().item() + 1,
                dtype=torch.long,
                device=edge_index.device)

        return self.propagate(
            edge_index, x=x, edge_type=edge_type, edge_norm=edge_norm)

    def message(self, x_j, edge_type, edge_norm):
        if self.att is None:
            w = self.basis.view(self.num_relations, -1)
        else:
            w = torch.matmul(self.att, self.basis.view(self.num_bases, -1))

        if x_j.dtype == torch.long:
            w = w.view(-1, self.out_channels)
            index = edge_type * self.in_channels + x_j
            out = torch.index_select(w, 0, index)
            return out if edge_norm is None else out * edge_norm.view(-1, 1)
        else:
            w = w.view(self.num_relations, self.in_channels, self.out_channels)
            w = torch.index_select(w, 0, edge_type)
            out = torch.bmm(x_j.unsqueeze(1), w).squeeze(-2)
            return out if edge_norm is None else out * edge_norm.view(-1, 1)

    def update(self, aggr_out, x):
        if x.dtype == torch.long:
            out = aggr_out + self.root
        else:
            out = aggr_out + torch.matmul(x, self.root)

        if self.bias is not None:
            out = out + self.bias
        return out

    def __repr__(self):
        return '{}({}, {}, num_relations={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.num_relations)



from mpqe.data_utils import RGCNQueryDataset

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from torch_scatter import scatter_add, scatter_max, scatter_mean

from mpqe.train_helpers import run_batch, run_batch_v2, check_conv, update_loss, get_queries_iterator
from mpqe.utils import eval_auc_queries, eval_perc_queries

class RGCNEncoderDecoder(nn.Module):
    def __init__(self, graph, enc, readout='mp',
                 scatter_op='add', dropout=0, weight_decay=1e-3,
                 num_layers=3, shared_layers=True, adaptive=True):
        super(RGCNEncoderDecoder, self).__init__()
        self.enc = enc
        self.graph = graph
        self.emb_dim = graph.feature_dims[next(iter(graph.feature_dims))]
        self.mode_embeddings = nn.Embedding(len(graph.mode_weights),
                                            self.emb_dim)
        self.num_layers = num_layers
        self.adaptive = adaptive

        self.mode_ids = {}
        mode_id = 0
        for mode in graph.mode_weights:
            self.mode_ids[mode] = mode_id
            mode_id += 1

        self.rel_ids = {}
        id_rel = 0
        for r1 in graph.relations:
            for r2 in graph.relations[r1]:
                rel = (r1, r2[1], r2[0])
                self.rel_ids[rel] = id_rel
                id_rel += 1

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if len(self.layers) == 0 or not shared_layers:
                rgcn = RGCNConv(in_channels=self.emb_dim,
                                out_channels=self.emb_dim,
                                num_relations=len(graph.rel_edges),
                                num_bases=0)

            self.layers.append(rgcn)

        if scatter_op == 'add':
            scatter_fn = scatter_add
        elif scatter_op == 'max':
            scatter_fn = scatter_max
        elif scatter_op == 'mean':
            scatter_fn = scatter_mean
        else:
            raise ValueError(f'Unknown scatter op {scatter_op}')

        self.readout_str = readout

        if readout == 'sum':
            self.readout = self.sum_readout
        elif readout == 'max':
            self.readout = self.max_readout
        elif readout == 'mlp':
            self.readout = MLPReadout(self.emb_dim, self.emb_dim, scatter_fn)
        elif readout == 'targetmlp':
            self.readout = TargetMLPReadout(self.emb_dim, scatter_fn)
        elif readout == 'concat':
            self.readout = MLPReadout(self.emb_dim * num_layers, self.emb_dim,
                                      scatter_fn)
        elif readout == 'mp':
            self.readout = self.target_message_readout
        else:
            raise ValueError(f'Unknown readout function {readout}')

        self.dropout = nn.Dropout(dropout)
        self.weight_decay = weight_decay
        
        # Training state variables (like run_train function)
        self.optimizer = None
        self.criterion = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Training state from run_train
        self.edge_conv = False
        self.ema_loss = None
        self.vals = []
        self.losses = []
        self.conv_test = None
        self.train_iterators = {}
        
        # Training parameters
        self.max_burn_in = 100000
        self.batch_size = 512
        self.log_every = 500
        self.val_every = 1000
        self.tol = 1e-6
        self.inter_weight = 0.005
        self.path_weight = 0.01
        self.current_iter = 0

    def sum_readout(self, embs, batch_idx, **kwargs):
        return scatter_add(embs, batch_idx, dim=0)

    def max_readout(self, embs, batch_idx, **kwargs):
        out, argmax = scatter_max(embs, batch_idx, dim=0)
        return out

    def target_message_readout(self, embs, batch_size, num_nodes, num_anchors,
                               **kwargs):
        device = embs.device

        non_target_idx = torch.ones(num_nodes, dtype=torch.bool)
        non_target_idx[num_anchors] = 0
        non_target_idx.to(device)

        embs = embs.reshape(batch_size, num_nodes, -1)
        targets = embs[:, ~non_target_idx].reshape(batch_size, -1)

        return targets

    def forward(self, formula, queries, target_nodes,
                anchor_ids=None, var_ids=None, q_graphs=None,
                neg_nodes=None, neg_lengths=None):

        if anchor_ids is None or var_ids is None or q_graphs is None:
            query_data = RGCNQueryDataset.get_query_graph(formula, queries,
                                                          self.rel_ids,
                                                          self.mode_ids)
            anchor_ids, var_ids, q_graphs = query_data

        device = next(self.parameters()).device
        var_ids = var_ids.to(device)
        q_graphs = q_graphs.to(device)

        batch_size, num_anchors = anchor_ids.shape
        n_vars = var_ids.shape[0]
        n_nodes = num_anchors + n_vars

        x = torch.empty(batch_size, n_nodes, self.emb_dim).to(var_ids.device)
        for i, anchor_mode in enumerate(formula.anchor_modes):
            x[:, i] = self.enc(anchor_ids[:, i], anchor_mode).t()
        x[:, num_anchors:] = self.mode_embeddings(var_ids)
        x = x.reshape(-1, self.emb_dim)
        q_graphs.x = x

        if self.adaptive:
            num_passes = RGCNQueryDataset.query_diameters[formula.query_type]
            if num_passes > len(self.layers):
                raise ValueError(f'RGCN is adaptive with {len(self.layers)}'
                                 f' layers, but query requires {num_passes}.')
        else:
            num_passes = self.num_layers

        h1 = q_graphs.x
        h_layers = []
        for i in range(num_passes - 1):
            h1 = self.layers[i](h1, q_graphs.edge_index, q_graphs.edge_type)
            h1 = F.relu(h1)
            if self.readout_str == 'concat':
                h_layers.append(h1)

        h1 = self.layers[-1](h1, q_graphs.edge_index, q_graphs.edge_type)

        if self.readout_str == 'concat':
            h_layers.append(h1)
            h1 = torch.cat(h_layers, dim=1)

        out = self.readout(embs=h1, batch_idx=q_graphs.batch,
                           batch_size=batch_size, num_nodes=n_nodes,
                           num_anchors=num_anchors)

        target_embeds = self.enc(target_nodes, formula.target_mode).t()
        scores = F.cosine_similarity(out, target_embeds, dim=1)

        if neg_nodes is not None:
            neg_embeds = self.enc(neg_nodes, formula.target_mode).t()
            out = out.repeat_interleave(torch.tensor(neg_lengths).to(device),
                                        dim=0)
            neg_scores = F.cosine_similarity(out, neg_embeds)

            scores = torch.cat((scores, neg_scores), dim=0)

        return scores

    def margin_loss(self, formula, queries, anchor_ids=None, var_ids=None,
                    q_graphs=None, hard_negatives=False, margin=1):
        if not "inter" in formula.query_type and hard_negatives:
            raise Exception("Hard negative examples can only be used with "
                            "intersection queries")
        elif hard_negatives:
            neg_nodes = [random.choice(query.hard_neg_samples)
                         for query in queries]

        elif formula.query_type == "1-chain":
            neg_nodes = [random.choice(self.graph.full_lists[formula.target_mode]) for _ in queries]
        else:
            neg_nodes = [random.choice(query.neg_samples) for query in queries]

        affs = self.forward(formula, queries,
                            [query.target_node for query in queries],
                            anchor_ids, var_ids, q_graphs)
        neg_affs = self.forward(formula, queries, neg_nodes,
                                anchor_ids, var_ids, q_graphs)
        loss = margin - (affs - neg_affs)
        loss = torch.clamp(loss, min=0)
        loss = loss.mean()

        if isinstance(self.readout, nn.Module) and self.weight_decay > 0:
            l2_reg = 0
            for param in self.readout.parameters():
                l2_reg += torch.norm(param)

            loss += self.weight_decay * l2_reg

        return loss
    
    def run_train(self, optimizer, train_queries, val_queries, test_queries,
                  logger, max_burn_in=100000, batch_size=512, log_every=500,
                  val_every=1000, tol=1e-6, max_iter=int(10e7), inter_weight=0.005,
                  path_weight=0.01, model_file=None, _run=None):
        """
        Train the RGCN model.
        
        Args:
            optimizer: PyTorch optimizer
            train_queries: Dictionary of training queries by type
            val_queries: Dictionary of validation queries by type  
            test_queries: Dictionary of test queries by type
            logger: Logger for output
            max_burn_in: Maximum burn-in iterations
            batch_size: Batch size for training
            log_every: Log frequency
            val_every: Validation frequency
            tol: Convergence tolerance
            max_iter: Maximum iterations
            inter_weight: Weight for intersection queries
            path_weight: Weight for path queries
            model_file: File to save model
            _run: Sacred run object for logging
        """
        edge_conv = False
        ema_loss = None
        vals = []
        losses = []
        conv_test = None

        print('Training RGCN-Enc-Dec')
        train_iterators = {}
        for query_type in train_queries:
            queries = train_queries[query_type]
            train_iterators[query_type] = get_queries_iterator(queries,
                                                             batch_size,
                                                             self)

        for i in range(max_iter):
            
            optimizer.zero_grad()

            loss = run_batch_v2(train_iterators['1-chain'], self)

            if not edge_conv and (check_conv(vals) or len(losses) >= max_burn_in):
                logger.info("Edge converged at iteration {:d}".format(i-1))
                logger.info("Testing at edge conv...")
                conv_test = self.run_eval(test_queries, i, logger)
                conv_test = np.mean(list(conv_test.values()))
                edge_conv = True
                losses = []
                ema_loss = None
                vals = []
                if model_file is not None:
                    torch.save(self.state_dict(), model_file+"-edge_conv")
            
            if edge_conv:
                for query_type in train_queries:
                    if query_type == "1-chain" and max_burn_in > 0:
                        continue
                    if "inter" in query_type:
                        loss += inter_weight * run_batch_v2(train_iterators[query_type], self)
                        loss += inter_weight * run_batch_v2(train_iterators[query_type], self, hard_negatives=True)
                    else:
                        loss += path_weight * run_batch_v2(train_iterators[query_type], self)

                if check_conv(vals):
                        logger.info("Fully converged at iteration {:d}".format(i))
                        break

            losses, ema_loss = update_loss(loss.item(), losses, ema_loss)
            loss.backward()
            optimizer.step()
                
            if i % log_every == 0:
                logger.info("Iter: {:d}; ema_loss: {:f}".format(i, ema_loss))
                if _run is not None:
                    _run.log_scalar('ema_loss', ema_loss, i)
                
            if i >= val_every and i % val_every == 0:
                v = self.run_eval(val_queries, i, logger)
                if edge_conv:
                    vals.append(np.mean(list(v.values())))
                else:
                    vals.append(v["1-chain"])
        
        v = self.run_eval(test_queries, i, logger, by_type=False)
        test_avg_auc = np.mean(list(v.values()))
        logger.info("Test macro-averaged val: {:f}".format(test_avg_auc))
        if _run is not None:
            _run.log_scalar('test_auc', test_avg_auc)
        
        if conv_test is not None and conv_test != 0:
            improvement = (np.mean(list(v.values()))-conv_test)/conv_test
            logger.info("Improvement from edge conv: {:f}".format(improvement))
        else:
            logger.info("Improvement from edge conv: N/A (baseline test failed)")

        if model_file is not None:
            torch.save(self.state_dict(), model_file)
        
        return test_avg_auc
    
    def run_eval(self, queries, iteration, logger, batch_size=128, by_type=False,
                 _run=None):
        """
        Evaluate the model on given queries.
        
        Args:
            queries: Dictionary of queries by type
            iteration: Current iteration number
            logger: Logger for output
            batch_size: Batch size for evaluation
            by_type: Whether to print results by relation type
            _run: Sacred run object for logging
            
        Returns:
            Dictionary of validation results by query type
        """
        self.eval()
        vals = {}
        
        def _print_by_rel(rel_aucs, logger):
            for rels, auc in rel_aucs.items():
                logger.info(str(rels) + "\t" + str(auc))
                
        for query_type in queries["one_neg"]:
            auc, rel_aucs = eval_auc_queries(queries["one_neg"][query_type], self)
            perc = eval_perc_queries(queries["full_neg"][query_type], self, batch_size)
            vals[query_type] = auc
            logger.info("{:s} val AUC: {:f} val perc {:f}; iteration: {:d}".format(query_type, auc, perc, iteration))
            if _run is not None:
                _run.log_scalar(f'{query_type}_val_auc', auc, iteration)
                _run.log_scalar(f'{query_type}_val_perc', perc, iteration)
            if by_type:
                _print_by_rel(rel_aucs, logger)
            if "inter" in query_type:
                auc, rel_aucs = eval_auc_queries(queries["one_neg"][query_type], self, hard_negatives=True)
                perc = eval_perc_queries(queries["full_neg"][query_type], self, hard_negatives=True)
                logger.info("Hard-{:s} val AUC: {:f} val perc {:f}; iteration: {:d}".format(query_type, auc, perc, iteration))
                if _run is not None:
                    _run.log_scalar(f'hard_{query_type}_val_auc', auc, iteration)
                    _run.log_scalar(f'hard_{query_type}_val_perc', perc, iteration)
                if by_type:
                    _print_by_rel(rel_aucs, logger)
                vals[query_type + "hard"] = auc
        return vals

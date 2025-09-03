import numpy as np
from .utils import eval_auc_queries, eval_perc_queries
from .data_utils import get_queries_iterator
# from .models.rgcn import RGCNEncoderDecoder
import torch
from sacred import Ingredient

train_ingredient = Ingredient('train')


def check_conv(vals, window=2, tol=1e-6):
    if len(vals) < 2 * window:
        return False
    conv = np.mean(vals[-window:]) - np.mean(vals[-2*window:-window]) 
    return conv < tol


def update_loss(loss, losses, ema_loss, ema_alpha=0.01):
    losses.append(loss)
    if ema_loss is None:
        ema_loss = loss
    else:
        ema_loss = (1-ema_alpha)*ema_loss + ema_alpha*loss
    return losses, ema_loss


def run_batch(train_queries, enc_dec, iter_count, batch_size, hard_negatives=False):
    num_queries = [float(len(queries)) for queries in list(train_queries.values())]
    denom = float(sum(num_queries))
    formula_index = np.argmax(np.random.multinomial(1, 
            np.array(num_queries)/denom))
    formula = list(train_queries.keys())[formula_index]
    n = len(train_queries[formula])
    start = (iter_count * batch_size) % n
    end = min(((iter_count+1) * batch_size) % n, n)
    end = n if end <= start else end
    queries = train_queries[formula][start:end]
    loss = enc_dec.margin_loss(formula, queries, hard_negatives=hard_negatives)
    return loss


def run_batch_v2(queries_iterator, enc_dec, hard_negatives=False):
    enc_dec.train()

    batch = next(queries_iterator)
    loss = enc_dec.margin_loss(*batch, hard_negatives=hard_negatives)
    return loss

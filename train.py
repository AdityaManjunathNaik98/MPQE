#------------------------------
# IMPORTS AND CONFIGURATION
#------------------------------
import os
import os.path as osp
import pickle as pkl
import numpy as np
from argparse import ArgumentParser
import torch 
from torch import optim

import mpqe.utils as utils
from mpqe.data_utils import load_queries_by_formula, load_test_queries_by_formula, load_graph
from mpqe.rgcn import RGCNEncoderDecoder
from sacred import Experiment
from sacred.observers import MongoObserver

# Configuration - only essential arguments
parser = ArgumentParser()
parser.add_argument("--max_iter", type=int, default=10000000, help="Maximum training iterations")
parser.add_argument("--max_burn_in", type=int, default=100000, help="Maximum burn-in iterations")
parser.add_argument("--num_layers", type=int, default=2, help="Number of model layers")
args = parser.parse_args()

# Fixed hyperparameters
MODEL = "qrgcn"
EMBED_DIM = 128
DATA_DIR = "F:/RGCN/data"
LEARNING_RATE = 0.01
BATCH_SIZE = 512
LOG_EVERY = 500
VAL_EVERY = 1000
TOLERANCE = 1e-6
USE_CUDA = True
LOG_DIR = "./"
DECODER = "bilinear"
READOUT = "sum"
INTER_WEIGHT = 0.005
OPTIMIZER = "adam"
DROPOUT = 0.0
WEIGHT_DECAY = 0.0
SCATTER_OP = 'add'
PATH_WEIGHT = 0.01
DEPTH = 0
SHARED_LAYERS = False
ADAPTIVE = False

#------------------------------
# DATA LOADING
#------------------------------
print("Loading graph data...")
graph, feature_modules, node_maps = load_graph(DATA_DIR, EMBED_DIM)
if USE_CUDA:
    graph.features = utils.cudify(feature_modules, node_maps)
out_dims = {mode: EMBED_DIM for mode in graph.relations}

print("Loading queries...")
train_queries = load_queries_by_formula(DATA_DIR + "/train_edges.pkl")
val_queries = load_test_queries_by_formula(DATA_DIR + "/val_edges.pkl")
test_queries = load_test_queries_by_formula(DATA_DIR + "/test_edges.pkl")

for i in range(2, 4):
    train_queries.update(load_queries_by_formula(DATA_DIR + "/train_queries_{:d}.pkl".format(i)))
    i_val_queries = load_test_queries_by_formula(DATA_DIR + "/val_queries_{:d}.pkl".format(i))
    val_queries["one_neg"].update(i_val_queries["one_neg"])
    val_queries["full_neg"].update(i_val_queries["full_neg"])
    i_test_queries = load_test_queries_by_formula(DATA_DIR + "/test_queries_{:d}.pkl".format(i))
    test_queries["one_neg"].update(i_test_queries["one_neg"])
    test_queries["full_neg"].update(i_test_queries["full_neg"])

print("Data loaded successfully.")

#------------------------------
# MODEL SETUP
#------------------------------
enc = utils.get_encoder(DEPTH, graph, out_dims, feature_modules, USE_CUDA)
enc_dec = RGCNEncoderDecoder(
    graph, enc, READOUT, SCATTER_OP,
    DROPOUT, WEIGHT_DECAY,
    args.num_layers, SHARED_LAYERS, ADAPTIVE
)

if USE_CUDA:
    enc_dec.cuda()

optimizer = optim.Adam(enc_dec.parameters(), lr=LEARNING_RATE)
logger = utils.setup_logging(LOG_DIR + "training.log")

# Sacred setup
ex = Experiment()
uri = os.environ.get('MLAB_URI')
database = os.environ.get('MLAB_DB')
if all([uri, database]):
    ex.observers.append(MongoObserver.create(uri, database))

@ex.config
def config():
    model = MODEL
    lr = LEARNING_RATE
    num_layers = args.num_layers
    max_burn_in = args.max_burn_in
    max_iter = args.max_iter

@ex.main
def main(_run):
    print("Starting training...")
    
    # Setup output directory
    exp_id = '-' + str(_run._id) if _run._id is not None else ''
    db_name = database if database is not None else ''
    folder_path = osp.join(LOG_DIR, db_name, 'output' + exp_id)
    if not osp.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
    model_path = osp.join(folder_path, "model.pt")

    # Run training
    test_auc = enc_dec.run_train(
        optimizer=optimizer,
        train_queries=train_queries,
        val_queries=val_queries,
        test_queries=test_queries,
        logger=logger,
        max_burn_in=args.max_burn_in,
        batch_size=BATCH_SIZE,
        log_every=LOG_EVERY,
        val_every=VAL_EVERY,
        tol=TOLERANCE,
        max_iter=args.max_iter,
        inter_weight=INTER_WEIGHT,
        path_weight=PATH_WEIGHT,
        model_file=model_path,
        _run=_run
    )

    # Save vocabulary
    vocab_path = osp.join(folder_path, 'training_vocabulary.pkl')
    with open(vocab_path, 'wb') as f:
        pkl.dump(enc_dec.graph.full_sets, f)
    
    print(f"Training completed. Final test AUC: {test_auc:.4f}")
    return test_auc

if __name__ == "__main__":
    ex.run()
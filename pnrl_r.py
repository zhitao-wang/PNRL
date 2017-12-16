import numpy as np
import random
np.random.seed(13)
from keras import backend as K
from keras.models import Sequential, Graph, Model
from keras.layers import Input, Embedding, LSTM, Dense, merge, Reshape, Activation
from keras.utils import np_utils
from keras.utils.data_utils import get_file
from keras.preprocessing.text import Tokenizer, base_filter
from keras.preprocessing.sequence import skipgrams,make_sampling_table
from keras.optimizers import rmsprop, SGD, Adam
from keras.regularizers import l2
from sklearn.metrics import classification_report, roc_auc_score
import copy
from util import *

class Config(object):
    """Model setting and data setting"""
    train_graph_file = 'data/example-train.net' # training file
    val_pos_file = 'data/example-val-pos.net' # validation positive file
    val_neg_file = 'data/example-val-neg.net' # validation negative file

    batch_size = 256
    neg_sample_num = 5 # max negative sampling
    dim_embedddings = 64

    hi_sample_beta = 0.2
    max_sample_iter = int(1/hi_sample_beta) # max iterations for hidden edge sample
    max_train_iter = 250


def ranking_loss_bpr(y_true, y_pred):
    pos = y_pred[:,0]
    neg = y_pred[:,1]
    loss = -K.sigmoid(pos-neg) # use loss = K.maximum(1.0 + neg - pos, 0.0) if you want to use margin ranking loss
    return K.mean(loss) + 0 * y_true

def ranking_loss_mm(y_true, y_pred):
    pos = y_pred[:,0]
    neg = y_pred[:,1]
    loss = K.maximum(1.0 + neg - pos, 0.0)
    return K.mean(loss) + 0 * y_true

def ranking_loss_mse(y_true, y_pred):
    pos = y_pred[:,0]
    neg = y_pred[:,1]
    loss = (1/2)*K.square(pos-neg-1)
    return K.mean(loss) + 0 * y_true

def link_prediction(model, user_pair):
    u1 = user_pair[0]
    u2 = user_pair[1]
    W = model.layers[5].get_weights()[0]
    X = model.layers[3].get_weights()[0]
    x = X[u1]
    y = X[u2]
    xW = np.dot(x,W)
    xWy = np.sum(xW*y, axis=1)
    return xWy

def PNRLR_Model(V, dim_embedddings=128):
    input_source = Input(shape=(1,), name='input_source')
    input_pos = Input(shape=(1,), name='input_pos')
    input_neg = Input(shape=(1,), name='input_neg')
    context_input = Input(shape=(1,), name='context_input')
    shared_embedding = Embedding(V, output_dim=dim_embedddings, input_length=1)
    X_source = shared_embedding(input_source)
    X_pos = shared_embedding(input_pos)
    X_neg = shared_embedding(input_neg)
    Y = Embedding(V, output_dim=dim_embedddings, input_length=1)(context_input)

    XY = merge([X_source, Y], mode='dot', dot_axes=2)
    res = Reshape((1,), input_shape=(1,1))(XY)
    nrl_output = Activation( activation='sigmoid', name='nrl_output')(res)
    model1 = Model(input=[input_source, context_input], output=[nrl_output])
    optimizer1 = rmsprop()
    model1.compile(loss="binary_crossentropy", optimizer=optimizer1)

    reshape_layer = Reshape((dim_embedddings,), name='reshape_layer')
    X_S = reshape_layer(X_source)
    X_P = reshape_layer(X_pos)
    X_N = reshape_layer(X_neg)
    XW =  Dense(dim_embedddings, activation='linear', W_regularizer=l2(0.01), bias=False,name='BiW')(X_S)
    pos_ouput = merge([XW, X_P], mode='dot')
    neg_output = merge([XW, X_N], mode='dot')
    lp_output = merge([pos_ouput, neg_output], mode='concat', concat_axis=-1, name='lp_output')
    model2 = Model(input=[input_source, input_pos, input_neg], output=[lp_output])
    optimizer2 = Adam()
    model2.compile(loss={'lp_output': ranking_loss_mm}, optimizer=optimizer2)
    return model1, model2

def hidden_edge_sampling(sample_iter, sample_size, edges, neighbor_set):
    hi_edges = edges[sample_iter*sample_size : (sample_iter+1)*sample_size]
    ob_edges = edges[:sample_iter*sample_size]
    ob_edges.extend(edges[(sample_iter+1)*sample_size:])
    ns = copy.deepcopy(neighbor_set)
    hi_ns = {}
    for e in hi_edges:
        if hi_ns.has_key(e[0]):
            hi_ns[e[0]].append(e[1])
        else:
            hi_ns[e[0]] = [e[1]]
        ns[e[0]].remove(e[1])
        if hi_ns.has_key(e[1]):
            hi_ns[e[1]].append(e[0])
        else:
            hi_ns[e[1]] = [e[0]]
        ns[e[1]].remove(e[0])
    return ob_edges, hi_edges, ns, hi_ns

def train_batch_generation(graph_nodes, ob_ns, hi_ns, neg_ns, gindex, ntable, num_neg, batch_size):
    counter = 0
    X1_input1 = []
    X1_input2 = []
    Y1 = []
    X2_input1 = []
    X2_input2 = []
    X2_input3 = []
    Y2 = []
    batch_data = []
    for node in graph_nodes:
        if ob_ns[node]:
            X1_input1.append(gindex[node])
            linked = gindex[random.choice(ob_ns[node])]
            X1_input2.append(linked)
            Y1.append(1.)
            neg = neg_sample(ntable, num_neg)
            for i in range(num_neg):
                X1_input1.append(gindex[node])
                X1_input2.append(neg[i])
                Y1.append(0.)
            if hi_ns.has_key(node):
                hi_linked = gindex[random.choice(hi_ns[node])]
                hi_nolinked = gindex[random.choice(neg_ns[node])]
                X2_input1.append(gindex[node])
                X2_input2.append(hi_linked)
                X2_input3.append(hi_nolinked)
                Y2.append(1.)
                counter += 1
        if counter == batch_size:
            X1 = [np.array(X1_input1), np.array(X1_input2)]
            Y1 = np.array(Y1,dtype=np.int32)
            X2 = [np.array(X2_input1), np.array(X2_input2), np.array(X2_input3)]
            Y2 = np.array(Y2,dtype=np.int32)
            batch_data.append((X1, Y1, X2, Y2))
            counter = 0
            X1_input1 = []
            X1_input2 = []
            Y1 = []
            X2_input1 = []
            X2_input2 = []
            X2_input3 = []
            Y2 = []
    X1 = [np.array(X1_input1), np.array(X1_input2)]
    Y1 = np.array(Y1,dtype=np.int32)
    X2 = [np.array(X2_input1), np.array(X2_input2), np.array(X2_input3)]
    Y2 = np.array(Y2,dtype=np.int32)
    batch_data.append((X1, Y1, X2, Y2))
    return batch_data

def train(config):
    # Load settings
    batch_size = config.batch_size
    num_neg = config.neg_sample_num
    dim_embedddings = config.dim_embedddings
    max_sample_iter = config.max_sample_iter
    max_train_iter = config.max_train_iter
    train_graph_file = config.train_graph_file
    val_pos_file = config.val_pos_file
    val_neg_file = config.val_neg_file

    # Read Data
    G = load_training_graph(train_graph_file, False)
    gindex = creat_index(G)
    edges = G.edges()
    ns_complete, neg_ns = neighbor_set(G)
    ntable = NegTable(G, gindex)
    V = len(gindex) + 1
    X_val, Y_val = read_valset(val_pos_file, val_neg_file, gindex, False)

    # Model initialization
    model1, model2 = PNRLR_Model(V, dim_embedddings)

    # Randomly shuffle data for training
    seed = random.randint(0, 10e6)
    random.seed(seed)
    random.shuffle(edges)

    sample_size = len(edges)/max_sample_iter
    for sample_iter in range(max_sample_iter):
        # Hidden edge sampling
        ob_edges, hi_edges, ob_ns, hi_ns = hidden_edge_sampling(sample_iter, sample_size, edges, ns_complete)
        print 'Observed edges: %d, Hidden edges: %d' %(len(ob_edges), len(hi_edges))

        # Training for the sampled hidden edges set and observed edges set
        for train_iter in range(max_train_iter):
            loss1 = 0.
            loss2 = 0.
            batch_data = train_batch_generation(G.nodes(), ob_ns, hi_ns, neg_ns, gindex, ntable, num_neg, batch_size)
            for batch in batch_data:
                (X1, Y1, X2, Y2) = batch
                loss1 += model1.train_on_batch(X1, Y1)
                loss2 += model2.train_on_batch(X2, Y2)
            Y_pred = link_prediction(model2, X_val)
            auc = roc_auc_score(Y_val, Y_pred)

            print 'Training Iteration: %d/%d, Sample Iteration: %d/%d'%(train_iter+1, max_train_iter,sample_iter+1, max_sample_iter)
            print 'Loss1: %f, Loss2: %f'%(loss1, loss2)
            print 'Link Prediction AUC on Validation: %f\n' %(auc)


if __name__ == '__main__':
    config = Config()
    train(config)



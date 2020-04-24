import networkx as nx
import math
import numpy as np

def load_training_graph(file, directed):
    if directed:
        return nx.read_edgelist(file, data=False, create_using = nx.DiGraph())
    else:
        return nx.read_edgelist(file, data=False)

def read_valset(pos_file, neg_file, graph_index, directed=False):
    print('Creating Validation edges set...')
    if directed:
        pos_val = nx.read_edgelist(pos_file, data=False, create_using = nx.DiGraph())
        neg_val = nx.read_edgelist(neg_file, data=False, create_using = nx.DiGraph())
    else:
        pos_val = nx.read_edgelist(pos_file, data=False)
        neg_val = nx.read_edgelist(neg_file, data=False)
    pos_val_set = pos_val.edges()
    neg_val_set = neg_val.edges()
    X_val_source = []
    X_val_target = []
    Y_val = []
    for e in pos_val_set:
        if e[0] in graph_index and e[1] in graph_index:
            X_val_source.append(graph_index[e[0]])
            X_val_target.append(graph_index[e[1]])
            Y_val.append(1.)
    for e in neg_val_set:
        if e[0] in graph_index and e[1] in graph_index:
            X_val_source.append(graph_index[e[0]])
            X_val_target.append(graph_index[e[1]])
            Y_val.append(0.)
    X_val = [np.array(X_val_source), np.array(X_val_target)]
    Y_val = np.array(Y_val)
    return X_val, Y_val

def creat_index(G):
    graph_index = {}
    i = 0
    for key in G.nodes():
        graph_index[key] = i
        i += 1
    return graph_index

def NegTable(G, graph_index):
    print('Creating negative sampling table...')
    nodes_size = len(G.nodes())
    power = 0.75
    norm = sum([math.pow(G.degree(node), power) for node in G.nodes()]) # Normalizing constant
    table_size = int(1e8) # Length of the unigram table
    table = np.zeros(table_size, dtype=np.uint32)
    # print 'Filling unigram table'
    p = 0 # Cumulative probability
    i = 0
    for node in G.nodes():
        p += float(math.pow(G.degree(node), power))/norm
        while i < table_size and float(i) / table_size < p:
            table[i] = graph_index[node]
            i += 1
    print('Finish')
    return table

def neg_sample(neg_table, num_neg):
    indices = np.random.randint(low=0, high=len(neg_table), size=num_neg)
    return [neg_table[i] for i in indices]

def neighbor_set(G):
    neighbor_set = {}
    non_neighbor_set = {}
    whole_set = set(G.nodes())
    print('Creating neighbor set...')
    for node in G.nodes():
        neighbours = nx.neighbors(G,node)
        neighbor_set[node] = list(neighbours)
        non_neighbor_set[node] = list(whole_set-set(neighbours))
    print('Finish creating neighbor set')
    return neighbor_set, non_neighbor_set

# def edge_sampling(neighbor_set, non_neighbor_set, graph_index, node, neg_table, num_neg):
#     if len(neighbor_set[node])==0:
#         return False
#     else:
#         linked = graph_index[random.choice(neighbor_set[node])]
#         nolinked = graph_index[random.choice(non_neighbor_set[node])]
#         source = graph_index[node]
#         neg = neg_sample(neg_table, num_neg)
#         return source, linked, nolinked, neg

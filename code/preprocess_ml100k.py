import json
import numpy as np
import pandas as pd
import gzip
from collections import defaultdict
import argparse


def preprocess(data_name, meta_path):
    u_list, i_list, ts_list, label_list = [], [], [], []
    feat_l = []
    idx_list = []
    u_map = {}
    i_map = {}

    u_ind = 0
    i_ind = 0

    user_count = defaultdict(int)
    item_count = defaultdict(int)
    with open(data_name, 'r') as f:
        for idx, line in enumerate(f):
            one_interaction = line.strip().split("\t")
            u, i = one_interaction[0], one_interaction[1]
            user_count[u] += 1
            item_count[i] += 1 

    with open(data_name, 'r') as f:
        for line in f:
            one_interaction = line.strip().split("\t")
            # if user_count[one_interaction[0]] < 5 or user_count[one_interaction[1]] < 5:
            #     continue
            u = int(one_interaction[0])
            i = int(one_interaction[1])

            if u not in u_map:
                u_map[u] = u_ind
                u_ind += 1
            if i not in i_map:
                i_map[i] = i_ind
                i_ind += 1

    i_meta_map = {}
    with open(meta_path, 'r', encoding='latin-1') as f:
        for line in f:
            one_item_meta = line.strip().split("|")
            item_id = int(one_item_meta[0])
            if item_id not in i_meta_map:
                i_meta_map[item_id] = one_item_meta[1]

    u2i_map_list = defaultdict(list)
    i2u_map_list = defaultdict(list)

    u_index_list = []
    i_index_list = []
    
    with open(data_name,'r') as f:
        for idx, line in enumerate(f):
            one_interaction = line.strip().split("\t")
            u = u_map[int(one_interaction[0])]
            i = i_map[int(one_interaction[1])]
            
            ts = float(one_interaction[3])
            label = 1
            
            feat = np.array([0 for _ in range(8)])

            # u2i_map_list[u].append(i)
            # i2u_map_list[i].append(u)

            # u_index_list.append(len(u2i_map_list[u]) - 1)
            # i_index_list.append(len(i2u_map_list[i]) - 1)
            
            u_list.append(u)
            i_list.append(i)
            ts_list.append(ts)
            label_list.append(label)
            idx_list.append(idx)
            
            feat_l.append(feat)

    user_ind_id_map = {v:k for k, v in u_map.items()}
    item_ind_id_map = {v:{'item_id': k, 'title': i_meta_map.get(k, '')} for k, v in i_map.items()}

    df = pd.DataFrame({'u': u_list, 
                        'i':i_list, 
                        'ts':ts_list, 
                        'label':label_list, 
                        'idx':idx_list})
    
    df.sort_values("ts", inplace=True)

    re_u_map = {}
    re_i_map = {}

    users = df.u.values
    items = df.i.values

    #     u_index_list.append(len(u2i_map_list[u]) - 1)
    #     i_index_list.append(len(i2u_map_list[i]) - 1)
    
    # df['u_idx'] = u_index_list
    # df['i_idx'] = i_index_list

    return df, np.array(feat_l), user_ind_id_map, item_ind_id_map



# def reindex(df):
#     assert(df.u.max() - df.u.min() + 1 == len(df.u.unique()))
#     assert(df.i.max() - df.i.min() + 1 == len(df.i.unique()))
    
#     upper_u = df.u.max() + 1
#     new_i = df.i + upper_u
    
#     new_df = df.copy()
#     print(new_df.u.max())
#     print(new_df.i.max())
    
#     new_df.i = new_i
#     new_df.u += 1
#     new_df.i += 1
#     new_df.idx += 1
    
#     print(new_df.u.max())
#     print(new_df.i.max())
    
#     return new_df

def reindex(df, bipartite=False):
    new_df = df.copy()
    if bipartite:
        assert (df.u.max() - df.u.min() + 1 == len(df.u.unique()))
        assert (df.i.max() - df.i.min() + 1 == len(df.i.unique()))

        upper_u = df.u.max() + 1
        new_i = df.i + upper_u

        new_df.i = new_i
        new_df.u += 1
        new_df.i += 1
        new_df.idx += 1
        print(new_df.u.max())
        print(new_df.i.max())
    else:
        new_df.u += 1
        new_df.i += 1
        new_df.idx += 1
        print(new_df.u.max())
        print(new_df.i.max())

    return new_df



def run(data_name, bipartite=True):
    PATH = '../data/ml100k/u.data'
    meta_path = '../data/ml100k/u.item'
    OUT_DF = '../data/ml100k/ml_{}.csv'.format(data_name)
    train_path = '../data/ml100k/train.txt'
    test_path = '../data/ml100k/test.txt'
    #OUT_FEAT = './processed/ml_{}.npy'.format(data_name)
    #OUT_NODE_FEAT = './processed/ml_{}_node.npy'.format(data_name)
    u_map_file = '../data/ml100k/{}_u_map.json'.format(data_name)
    i_map_file = '../data/ml100k/{}_i_map.json'.format(data_name)
    
    df, feat, u_ind_id_map, i_ind_id_map = preprocess(PATH, meta_path)
    with open(u_map_file, 'w') as f:
        f.write(json.dumps(u_ind_id_map, sort_keys=True, indent=4))
    with open(i_map_file, 'w') as f:
        f.write(json.dumps(i_ind_id_map, sort_keys=True, indent=4))

    # df.sort_values("ts", inplace=True)
    new_df = reindex(df)
    
    #print(feat.shape)
    #empty = np.zeros(feat.shape[1])[np.newaxis, :]
    #feat = np.vstack([empty, feat])
    #
    #max_idx = max(new_df.u.max(), new_df.i.max())
    #rand_feat = np.zeros((max_idx + 1, feat.shape[1]))
    #
    #print(feat.shape)
    new_df.to_csv(OUT_DF, index=False)
    #np.save(OUT_FEAT, feat)
    #np.save(OUT_NODE_FEAT, rand_feat)

    df_path = OUT_DF
    ### Load data and train val test split
    graph_df = pd.read_csv(df_path)

    val_time, test_time = list(np.quantile(graph_df.ts, [0.80, 0.90]))

    sources = graph_df.u.values
    destinations = graph_df.i.values
    edge_idxs = graph_df.idx.values
    labels = graph_df.label.values
    timestamps = graph_df.ts.values

    u_map = {}
    i_map = {}
    u_ind = 0
    i_ind = 0
    for (u, i) in zip(sources, destinations):
        if u not in u_map:
            u_map[u] = u_ind
            u_ind += 1
        if i not in i_map:
            i_map[i] = i_ind
            i_ind += 1

    user_item_src = []
    user_item_dst = []

    valid_train_flag = (timestamps <= val_time)
    
    # train
    user_item_src = sources[valid_train_flag]
    user_item_dst = destinations[valid_train_flag]
    train_e_idx_l = edge_idxs[valid_train_flag]

    valid_train_userset = set(np.unique(user_item_src))
    valid_train_itemset = set(np.unique(user_item_dst))
    
    # select validation and test dataset
    valid_val_flag = (timestamps <= test_time) * (timestamps > val_time)
    valid_test_flag = timestamps > test_time

    # validation and test with all edges
    val_src_ln = sources[valid_val_flag]
    val_dst_ln = destinations[valid_val_flag]
    
    # valid_is_old_node_edge = np.array([(a in valid_train_userset and b in valid_train_itemset) for a, b in zip(val_src_l, val_dst_l)])
    # val_src_ln = val_src_l[valid_is_old_node_edge]
    # val_dst_ln = val_dst_l[valid_is_old_node_edge]

    # valid data -- Data
    print('#interactions in valid: ', len(val_src_ln))

    test_src_ln = sources[valid_test_flag]
    test_dst_ln = destinations[valid_test_flag]
    
    # test_is_old_node_edge = np.array([(a in valid_train_userset and b in valid_train_itemset) for a, b in zip(test_src_l, test_dst_l)])
    # test_src_ln = test_src_l[test_is_old_node_edge]
    # test_dst_ln = test_dst_l[test_is_old_node_edge]

    # test data -- Data
    print('#interaction in test: ', len(test_src_ln))

    train_items = defaultdict(list) # [[] for _ in range(user_item_src.max() + 1)]
    for src, dst in zip(user_item_src, user_item_dst):
        train_items[src].append(dst)
    
    test_set = defaultdict(list) # [[] for _ in range(test_src_ln.max() + 1)]
    for src, dst in zip(test_src_ln, test_dst_ln):
        test_set[src].append(dst)

    f = open(train_path, 'w')
    for k, v in train_items.items():
        f.write(str(u_map[k])+' ')
        for idx, item in enumerate(v):
            if idx != len(v) - 1:
                f.write(str(i_map[item])+' ')
            else:
                f.write(str(i_map[item])+'\n')
    f.close()
    
    f = open(test_path, 'w')
    for k, v in test_set.items():
        f.write(str(u_map[k])+' ')
        for idx, item in enumerate(v):
            if idx != len(v) - 1:
                f.write(str(i_map[item])+' ')
            else:
                f.write(str(i_map[item])+'\n')
    f.close()
    


# run('ml100k')
parser = argparse.ArgumentParser('Interface for TGN data preprocessing')
parser.add_argument('--data', type=str, help='Dataset name (eg. wikipedia or reddit)',
                    default='ml100k')
parser.add_argument('--bipartite', action='store_true', help='Whether the graph is bipartite')

args = parser.parse_args()

run(args.data, bipartite=args.bipartite)

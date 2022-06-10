import json
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from collections import defaultdict


def preprocess(data_name):
    u_list, i_list, ts_list, label_list = [], [], [], []
    feat_l = []
    idx_list = []

    with open(data_name) as f:
        s = next(f)
        for idx, line in enumerate(f):
            e = line.strip().split(',')
            u = int(e[0])
            i = int(e[1])

            ts = float(e[2])
            label = float(e[3])  # int(e[3])

            feat = np.array([float(x) for x in e[4:]])

            u_list.append(u)
            i_list.append(i)
            ts_list.append(ts)
            label_list.append(label)
            idx_list.append(idx)

            feat_l.append(feat)
    return pd.DataFrame({'u': u_list,
                         'i': i_list,
                         'ts': ts_list,
                         'label': label_list,
                         'idx': idx_list}), np.array(feat_l)


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
    else:
        new_df.u += 1
        new_df.i += 1
        new_df.idx += 1

    return new_df


def run(data_name, bipartite=True):
    Path(f"../data/{data_name}").mkdir(parents=True, exist_ok=True)
    PATH = '../data/{}/{}.csv'.format(data_name, data_name)
    OUT_DF = '../data/{}/ml_{}.csv'.format(data_name, data_name)
    OUT_FEAT = '../data/{}/ml_{}.npy'.format(data_name, data_name)
    train_path = '../data/{}/train.txt'.format(data_name)
    test_path = '../data/{}/test.txt'.format(data_name)
    # OUT_NODE_FEAT = '../../Data/{}/ml_{}_node.npy'.format(data_name, data_name)

    df, feat = preprocess(PATH)
    # new_df = reindex(df, bipartite)
    new_df = df.copy()

    # empty = np.zeros(feat.shape[1])[np.newaxis, :]
    # feat = np.vstack([empty, feat])

    # max_idx = max(new_df.u.max(), new_df.i.max())
    # rand_feat = np.zeros((max_idx + 1, 172))

    new_df.to_csv(OUT_DF)
    np.save(OUT_FEAT, feat)
    # np.save(OUT_NODE_FEAT, rand_feat)

    df_path = OUT_DF
    ### Load data and train val test split
    graph_df = pd.read_csv(df_path)

    val_time, test_time = list(np.quantile(graph_df.ts, [0.80, 0.90]))

    sources = graph_df.u.values
    destinations = graph_df.i.values
    edge_idxs = graph_df.idx.values
    labels = graph_df.label.values
    timestamps = graph_df.ts.values

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
    val_src_l = sources[valid_val_flag]
    val_dst_l = destinations[valid_val_flag]
    
    valid_is_old_node_edge = np.array([(a in valid_train_userset and b in valid_train_itemset) for a, b in zip(val_src_l, val_dst_l)])
    val_src_ln = val_src_l[valid_is_old_node_edge]
    val_dst_ln = val_dst_l[valid_is_old_node_edge]

    # valid data -- Data
    print('#interactions in valid: ', len(val_src_ln))

    test_src_l = sources[valid_test_flag]
    test_dst_l = destinations[valid_test_flag]
    
    test_is_old_node_edge = np.array([(a in valid_train_userset and b in valid_train_itemset) for a, b in zip(test_src_l, test_dst_l)])
    test_src_ln = test_src_l[test_is_old_node_edge]
    test_dst_ln = test_dst_l[test_is_old_node_edge]

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
        f.write(str(k)+' ')
        for idx, item in enumerate(v):
            if idx != len(v) - 1:
                f.write(str(item)+' ')
            else:
                f.write(str(item)+'\n')
    f.close()
    
    f = open(test_path, 'w')
    for k, v in test_set.items():
        f.write(str(k)+' ')
        for idx, item in enumerate(v):
            if idx != len(v) - 1:
                f.write(str(item)+' ')
            else:
                f.write(str(item)+'\n')
    f.close()


parser = argparse.ArgumentParser('Interface for TGN data preprocessing')
parser.add_argument('--data', type=str, help='Dataset name (eg. wikipedia or reddit)',
                    default='mooc')
parser.add_argument('--bipartite', action='store_true', help='Whether the graph is bipartite')

args = parser.parse_args()

run(args.data, bipartite=args.bipartite)

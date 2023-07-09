import random
import networkx as nx
import time
import math
import utils
import numpy as np
import warnings

warnings.filterwarnings('ignore')


def create_matrix(num_nodes, inner_dim):
    pro = 1 / (2 * math.sqrt(num_nodes))
    rev_pro = 1 - pro
    matrix = np.zeros((num_nodes, inner_dim))
    for m in range(num_nodes):
        for n in range(inner_dim):
            random_num = random.random()
            if random_num <= pro:
                matrix[m][n] = 1
            elif random_num >= rev_pro:
                matrix[m][n] = -1
    return matrix


def node2vec(walks_list, s_alpha=0.025):
    num_walks = len(walks_list)
    print("length of walks_list: ", num_walks)

    node_num = nx.number_of_nodes(nx_g)
    nodes = list(nx_g.nodes)
    # nodes = []
    # for node in nx_g.nodes:
    #     nodes.append(node)
    A = np.zeros((node_num, node_num))
    for i in range(node_num):
        for j in range(i + 1, node_num):
            if nx_g.has_edge(nodes[i], nodes[j]):
                A[i][j] = 1
                A[j][i] = 1
    R = create_matrix(num_nodes=node_num, inner_dim=dim)
    embeddings = A
    tmp = A
    for k in range(1, order):
        tmp = (A @ tmp) * eps
        embeddings += tmp
    # mean_vec = np.mean(embeddings, axis=0)
    # std_vec = np.std(embeddings, axis=0)
    # embeddings = (embeddings - mean_vec) / std_vec
    # D = np.zeros((node_num, node_num))
    # for i in range(node_num):
    #     D[i][i] = 1 / nx_g.degree(nodes[i])
    # embeddings = D @ embeddings
    D = np.diag(1/embeddings.sum(axis=0))
    embeddings = (D @ (embeddings @ R)) * math.pow(node_num, 0.25)
    center_dict = {}
    context_dict = {}
    for i in range(node_num):
        center_dict[nodes[i]] = embeddings[i]
        context_dict[nodes[i]] = np.zeros(shape=dim)
    output = r'C:\Users\123456\Desktop\data_res\res\fhne_aminer_orig.txt'
    utils.save(output, center_dict)

    neg_tables = utils.Table(nx_g)

    pairs = []
    for i in range(num_walks):
        for pos, token in enumerate(walks_list[i]):
            cur_win = np.random.randint(low=1, high=win + 1)
            start = max(pos - cur_win, 0)
            end = min(pos + cur_win + 1, len(walks_list[i]))
            pre = walks_list[i][start:pos]
            for k in range(len(pre)):
                pair = [token, pre[k]]
                pairs.append(pair)
            post = walks_list[i][pos + 1:end]
            for k in range(len(post)):
                pair = [token, post[k]]
                pairs.append(pair)
    random.shuffle(pairs)
    sum_node = len(pairs)
    print(sum_node)
    count_node = 0
    alpha = s_alpha
    start_time = time.time()
    for item in pairs:
        if count_node % 5000 == 0:
            progress = count_node / sum_node
            alpha = (1 - progress) * s_alpha
            dur = time.time() - start_time
            print("\rprogress: {:.2f}%    elapsed time: {:.2f}s".format(progress * 100, dur), end="")
        # r_num = random.randint(0, 2)
                 # * random_list[r_num]
        gradient = np.zeros(dim)
        z = np.dot(center_dict[item[1]], context_dict[item[0]])
        p = utils.sigmoid(z)
        pre_gradient = alpha * (p - 1) * 2

        context_dict[item[0]] -= pre_gradient * center_dict[item[1]]
        center_dict[item[1]] -= pre_gradient * context_dict[item[0]]
        for target in neg_tables.sample(neg):
            z = np.dot(center_dict[item[1]], context_dict[target])
            p = utils.sigmoid(z)
            pre_gradient = alpha * p

            context_dict[target] -= pre_gradient * center_dict[item[1]]
            center_dict[item[1]] -= pre_gradient * context_dict[target]
        # center_dict[item[1]] += gradient
        count_node += 1

    return center_dict
    # sum_node = num_walks * (walk_len + 1)
    # print(sum_node)
    # count_node = 0
    # alpha = s_alpha
    # start_time = time.time()
    # for i in range(num_walks):
    #     for pos, token in enumerate(walks_list[i]):
    #         if count_node % 3000 == 0:
    #             progress = count_node / sum_node
    #             alpha = (1 - progress) * s_alpha
    #             dur = time.time() - start_time
    #             print("\rprogress: {:.2f}%    elapsed time: {:.2f}s".format(progress * 100, dur), end="")
    #         cur_win = np.random.randint(low=1, high=win + 1)
    #         start = max(pos - cur_win, 0)
    #         end = min(pos + cur_win + 1, len(walks_list[i]))
    #         context = walks_list[i][start:pos] + walks_list[i][pos + 1:end]
    #         for context_node in context:
    #             gradient = np.zeros(dim)
    #             z = np.dot(center_dict[context_node], context_dict[token])
    #             p = utils.sigmoid(z)
    #             pre_gradient = alpha * beta * (p - 1)
    #             context_dict[token] -= pre_gradient * center_dict[context_node]
    #             gradient -= pre_gradient * context_dict[token]
    #             for target in neg_tables.sample(neg):
    #                 z = np.dot(center_dict[context_node], context_dict[target])
    #                 p = utils.sigmoid(z)
    #                 pre_gradient = alpha * p
    #                 gradient -= pre_gradient * context_dict[target]
    #                 context_dict[target] -= pre_gradient * center_dict[context_node]
    #             center_dict[context_node] += gradient
    #         count_node += 1
    # return center_dict


if __name__ == '__main__':
    s_t = time.time()
    print("v_fhne_aminer15:")
    input_path = r'C:\Users\123456\Desktop\data_res\aminer_data\aminer15_edgelist.txt'
    output_path = r'C:\Users\123456\Desktop\data_res\res\fhne_aminer.txt'
    nx_g = nx.read_edgelist(input_path)
    # c_list = ['a', 'p', 'c']
    # class_table = utils.ClassTable(nx_g, c_list)

    walk_nums = 1
    walk_len = 12
    dim = 128
    win = 5
    neg = 3

    order = 3
    eps = 0.8
    beta = 2
    s_a = 0.02

    print("Creating walks...")
    Walk = utils.MetaWalk(nx_g)
    walks = Walk.generate_walks(walk_nums, walk_len)
    random.shuffle(walks)
    print("Generating representation...")
    utils.save(output_path, node2vec(walks, s_a))
    print("\nRepresentation File saved")
    e_t = time.time()
    print("Total time: {:.2f}s".format(e_t - s_t))
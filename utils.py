import random
import math
import time
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def assign_weight(g):
    for (u, v, w) in g.edges(data=True):
        w['weight'] = 1
    return g


def sigmoid(x):
    if x > 6:
        return 1.0
    elif x < -6:
        return 0.0
    else:
        return 1 / (1 + math.exp(-x))


def save(out_file, out_dict):
    fo = open(out_file, mode='w', encoding='utf-8')
    fo.write('%d \n' % len(out_dict))
    for key in out_dict:
        embedding_str = ' '.join([str(s) for s in out_dict[key]])
        fo.write('%s %s\n' % (str(key), embedding_str))
    fo.close()


class MetaWalk:
    def __init__(self, g):
        self.G = g
        self.paper_author = dict()
        self.author_paper = dict()
        self.conf_paper = dict()
        self.paper_conf = dict()
        self.read_data()

    def read_data(self):
        for node in self.G.nodes:
            if node[0] == 'a':
                if node not in self.author_paper:
                    self.author_paper[node] = []
                neighs = self.G[node]
                for neg in neighs:
                    self.author_paper[node].append(neg)
                    if neg not in self.paper_author:
                        self.paper_author[neg] = []
                    self.paper_author[neg].append(node)
            elif node[0] == 'c':
                if node not in self.conf_paper:
                    self.conf_paper[node] = []
                neighs = self.G[node]
                for neg in neighs:
                    self.conf_paper[node].append(neg)
                    if neg not in self.paper_conf:
                        self.paper_conf[neg] = []
                    self.paper_conf[neg].append(node)

    def generate_walks(self, num_walks, walk_length):
        walks = []
        for j in range(num_walks):
            for au in self.author_paper:
                walk = [au]
                author = au
                for i in range(walk_length // 4):
                    paper1 = random.choice(self.author_paper[author])
                    walk.append(paper1)
                    conf = self.paper_conf[paper1][0]
                    walk.append(conf)
                    paper2 = random.choice(self.conf_paper[conf])
                    walk.append(paper2)
                    author = random.choice(self.paper_author[paper2])
                    walk.append(author)
                walks.append(walk)
                walk = [au]
                author = au
                for i in range(walk_length//2):
                    paper1 = random.choice(self.author_paper[author])
                    walk.append(paper1)
                    author = random.choice(self.paper_author[paper1])
                    walk.append(author)
                walks.append(walk)
        return walks


class Walk:
    def __init__(self, g):
        self.G = g

    def random_walk(self, walk_length, start_node):
        walk = [start_node]
        while len(walk) < walk_length:
            cur = walk[-1]
            neighs = list(self.G.neighbors(cur))
            nxt = random.choice(neighs)
            walk.append(nxt)
        return walk

    def batch_walk(self, num_walks, walk_length):
        walks = []
        nodes = list(self.G.nodes)
        for i in range(num_walks):
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self.random_walk(walk_length=walk_length, start_node=node))
        return walks


class Table:
    def __init__(self, g):
        self.graph = g
        self.table = []
        self.create_table()

    def create_table(self):
        g = self.graph
        power = 0.75
        table_size = 1e7
        sum_sample = sum(math.pow(g.degree(node, weight='weight'), power) for node in g.nodes)
        table = []
        p = 0
        i = 0
        for node in g.nodes:
            p += (float(math.pow(g.degree(node, weight='weight'), power)) / sum_sample)
            while i < p * table_size:
                table.append(node)
                i += 1
        self.table = table

    def sample(self, nums):
        random_nums = np.random.randint(low=0, high=len(self.table), size=nums)
        return [self.table[i] for i in random_nums]


def node2vec(walks_list, g, dim=128, win=5, neg=3, walk_len=30, s_alpha=0.025):
    num_walks = len(walks_list)
    print("length of walks_list: ", num_walks)
    context_dict = {}
    center_dict = {}
    for node in g.nodes:
        center_dict[node] = np.random.uniform(low=-0.5 / dim, high=0.5 / dim, size=dim)
        context_dict[node] = np.zeros(shape=dim)
    neg_table = Table(g)
    sum_node = num_walks * walk_len
    print(sum_node)
    count_node = 0
    alpha = 0
    start_time = time.time()
    for i in range(num_walks):
        for pos, token in enumerate(walks_list[i]):
            if count_node % 10000 == 0:
                progress = count_node / sum_node
                alpha = (1 - progress) * s_alpha
                dur = time.time() - start_time
                print("\rprogress: {:.2f}%    elapsed time: {:.2f}s".format(progress * 100, dur), end="")
            cur_win = np.random.randint(low=1, high=win + 1)
            start = max(pos - cur_win, 0)
            end = min(pos + cur_win + 1, len(walks_list[i]))
            context = walks_list[i][start:pos] + walks_list[i][pos + 1:end]
            for context_node in context:
                gradient = np.zeros(dim)
                pairs = [(token, 1)] + [(neg_token, 0) for neg_token in neg_table.sample(neg)]
                for target, label in pairs:
                    z = np.dot(center_dict[context_node], context_dict[target])
                    p = sigmoid(z)
                    pre_gradient = alpha * (label - p)
                    gradient += pre_gradient * context_dict[target]
                    context_dict[target] += pre_gradient * center_dict[context_node]
                center_dict[context_node] += gradient
            count_node += 1
    # output = r'C:\Users\123456\Desktop\data_res\res\aminer_context.txt'
    # save(output, context_dict)
    print("\n")
    return center_dict




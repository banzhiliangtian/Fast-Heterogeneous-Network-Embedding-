import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import normalized_mutual_info_score, f1_score, roc_auc_score, accuracy_score
from sklearn.cluster import KMeans
import warnings
import numpy as np

warnings.filterwarnings('ignore')
random.seed(1)


def multi_classification(x_train, y_train, x_test, y_test):
    classifier = LogisticRegression()
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    micro_f1 = f1_score(y_test, y_pred, average='micro')
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    # print('Macro_F1_score:  {:.4f}'.format(macro_f1))
    # print('Micro_F1_score:  {:.4f}'.format(micro_f1))
    return micro_f1, macro_f1


def binary_classification(x_train, y_train, x_test, y_test):
    classifier = LogisticRegression()
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    # y_pred_proba = classifier.predict_proba(x_test)[:, 1]
    f1 = f1_score(y_test, y_pred)
    # auc_score = roc_auc_score(y_test, y_pred_proba)
    acc = accuracy_score(y_test, y_pred)
    # print('f1:      {:.4f}'.format(f1))
    # print('auc:     {:.4f}'.format(auc_score))
    # print('acc:     {:.4f}'.format(acc))
    return f1, acc


class Evaluation:
    def __init__(self, embeddings_path, flag=True):
        self.embeddings_path = embeddings_path
        self.path_1 = r'C:\Users\123456\Desktop\data_res'
        self.name_emb_dict = {}
        self.flag = flag

    def load_embeddings(self, skip_head=True, factor=1.0, d=128):
        embeddings_file = open(self.path_1 + self.embeddings_path, 'r', encoding='utf-8')
        if skip_head:
            embeddings_file.readline()
        while 1:
            line = embeddings_file.readline()
            if line == '' or line is None:
                break
            embedding = []
            vec = line.strip().split(' ')
            for i in range(1, d+1):
                embedding.append(float(vec[i]) * factor)
            self.name_emb_dict[vec[0]] = embedding
        embeddings_file.close()
        print("embeddings num:", len(self.name_emb_dict))

    def classify(self, label_file, train_size, is_binary):
        x = []
        y = []
        lines = open(self.path_1 + label_file, 'r', encoding='utf-8')
        for line in lines:
            tokens = line.strip().split(' ')
            if self.name_emb_dict.__contains__(tokens[0]) and tokens[0][0] == 'a':
                x.append(self.name_emb_dict[tokens[0]])
                y.append(int(tokens[1]))
        lines.close()
        print("embeddings num with label:", len(x))
        sum_m1 = 0.0
        sum_m2 = 0.0
        for i in range(10):
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1 - train_size, random_state=None)
            if is_binary:
                f1, acc = binary_classification(x_train, y_train, x_test, y_test)
                sum_m1 += f1
                sum_m2 += acc
            else:
                micro, macro = multi_classification(x_train, y_train, x_test, y_test)
                sum_m1 += micro
                sum_m2 += macro
        if is_binary:
            print('f1:      {:.4f}'.format(sum_m1/10))
            print('acc:     {:.4f}'.format(sum_m2/10))
        else:
            print('Macro_F1_score:  {:.4f}'.format(sum_m1/10))
            print('Micro_F1_score:  {:.4f}'.format(sum_m2/10))

    def cluster(self, label_file, cluster_k):
        x = []
        y = []
        lines = open(self.path_1 + label_file, 'r', encoding='utf-8')
        for line in lines:
            tokens = line.strip().split(' ')
            if self.name_emb_dict.__contains__(tokens[0]) and tokens[0][0] == 'a':
                x.append(self.name_emb_dict[tokens[0]])
                y.append(int(tokens[1]))
        lines.close()
        sum_nmi = 0
        for i in range(10):
            km = KMeans(n_clusters=cluster_k)
            km.fit(x, y)
            y_pre = km.predict(x)
            nmi = normalized_mutual_info_score(y, y_pre)
            sum_nmi += nmi
        print('Kmean, k={}, nmi={:.4f}'.format(cluster_k, sum_nmi/10))

    def calculate_sim(self, u, v):
        if self.flag:
            return np.array(u)*np.array(v)
        else:
            return np.abs(np.array(u)*np.array(v))

    def link_prediction(self, edge_file, train_size):
        x = []
        y = []
        with open(self.path_1 + edge_file + '\\pos_edges.txt', 'r') as aa_pos_f:
            for line in aa_pos_f:
                tokens = line.strip().split('\t')
                if self.name_emb_dict.__contains__(tokens[0]) and self.name_emb_dict.__contains__(tokens[1]):
                    pos_1_emb = self.name_emb_dict[tokens[0]]
                    pos_2_emb = self.name_emb_dict[tokens[1]]
                    sim_pos = self.calculate_sim(pos_1_emb, pos_2_emb)
                    x.append(sim_pos)
                    y.append(1)
        aa_pos_f.close()
        with open(self.path_1 + edge_file + '\\neg_edges.txt', 'r') as aa_neg_f:
            for line in aa_neg_f:
                tokens = line.strip().split('\t')
                if self.name_emb_dict.__contains__(tokens[0]) and self.name_emb_dict.__contains__(tokens[1]):
                    neg_1_emb = self.name_emb_dict[tokens[0]]
                    neg_2_emb = self.name_emb_dict[tokens[1]]
                    sim_neg = self.calculate_sim(neg_1_emb, neg_2_emb)
                    x.append(sim_neg)
                    y.append(0)
        aa_neg_f.close()

        sum_m1 = 0
        sum_m2 = 0
        for i in range(10):
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1 - train_size, random_state=None)
            f1, acc = binary_classification(x_train, y_train, x_test, y_test)
            sum_m1 += f1
            sum_m2 += acc
        print('f1:      {:.4f}'.format(sum_m1 / 10))
        print('acc:     {:.4f}'.format(sum_m2 / 10))
        # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1 - train_size, random_state=None)
        # binary_classification(x_train, y_train, x_test, y_test)


if __name__ == '__main__':
    # train_ratio = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    train_ratio = [0.2, 0.5, 0.8]
    # input_file = '\\res\\sahe_aminer.txt'
    input_file = '\\res\\v_idea_aminer.txt'
    # input_file = '\\res\\fhne_aminer.txt'
    print(input_file)
    eva_model = Evaluation(input_file)
    eva_model.load_embeddings(factor=1, d=128)

    print('\n===== classification =====')
    label = '\\aminer_data\\label_15.txt'
    for t_r in train_ratio:
        print("train_ration: {:.0f}%".format(t_r * 100))
        eva_model.classify(label, t_r, False)

    print('\n===== cluster =====')
    label = '\\aminer_data\\label_15.txt'
    eva_model.cluster(label, cluster_k=5)
    #
    print('\n===== link prediction =====')
    egdes_path = '\\aminer_data'
    for t_r in train_ratio:
        print("train_ration: {:.0f}%".format(t_r * 100))
        eva_model.link_prediction(egdes_path, t_r)

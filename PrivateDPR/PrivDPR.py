import random
import math
import tensorflow as tf
import numpy as np
import argparse
import networkx as nx
import scipy.sparse as sp
from utils import graph_util

parser = argparse.ArgumentParser()
parser.add_argument('--embedding_dim', default=128)
parser.add_argument('--batch_size', default=16)
parser.add_argument('--walk_num', default=2)
parser.add_argument('--walk_len', default=16)
parser.add_argument('--learning_rate', default=0.001)
parser.add_argument('--n_epochs', default=1)
parser.add_argument('--n_earlystop', default=2)
parser.add_argument('--tau', default=0.5)
parser.add_argument('--delay_factor', default=0.85)
parser.add_argument('--num_of_gra', default=None)
parser.add_argument('--delta', default=0.00001)
parser.add_argument('--is_GradientClip', default=True)
parser.add_argument('--layer_num', default=1)
parser.add_argument('--epsilon', default=0.1)
parser.add_argument('--hidden_layer_dim', default=64)
parser.add_argument('--g_clip', default=5)
parser.add_argument('--w_clip', default=1/8)

args = parser.parse_args()  # parameters

class DiGraSynModel:
    def __init__(self, graph, Layer_num, node_embed_init=None):
        self.n_node = graph.number_of_nodes()
        self.n_edge = graph.number_of_edges()
        args.num_of_gra = self.n_node
        self.node_emd_init = node_embed_init

        with tf.variable_scope('graph_forward_pass'):
            if self.node_emd_init:
                self.node_embedding_matrix = tf.get_variable(name='node_embedding_mat',
                                                             shape=self.node_emd_init.shape,
                                                             initializer=tf.constant_initializer(self.node_emd_init),
                                                             trainable=True)
            else:
                self.node_embedding_matrix = tf.get_variable(name='node_embedding_mat',
                                                             shape=[self.n_node, args.embedding_dim],
                                                             initializer=tf.contrib.layers.xavier_initializer(
                                                                 uniform=False),
                                                             trainable=True)
            # multi-layer MLP
            self.weights = []
            self.biases = []
            for l in range(Layer_num):
                weight_name = 'weight_' + str(l)
                bias_name = 'bias_' + str(l)
                if l == 0:
                    self.weights.append(tf.get_variable(name=weight_name, shape=[args.embedding_dim, args.hidden_layer_dim],
                                initializer=tf.contrib.layers.xavier_initializer(uniform=False), trainable=True))
                    self.biases.append(tf.get_variable(name=bias_name, shape=[args.hidden_layer_dim],
                                    initializer=tf.contrib.layers.xavier_initializer(uniform=False), trainable=True))
                elif l == Layer_num-1:
                    self.weights.append(tf.get_variable(name=weight_name, shape=[args.hidden_layer_dim, 1],
                                initializer=tf.contrib.layers.xavier_initializer(uniform=False), trainable=True))
                    self.biases.append(tf.get_variable(name=bias_name, shape=[1],
                                    initializer=tf.contrib.layers.xavier_initializer(uniform=False), trainable=True))
                else:
                    self.weights.append(tf.get_variable(name=weight_name, shape=[args.hidden_layer_dim, args.hidden_layer_dim],
                                initializer=tf.contrib.layers.xavier_initializer(uniform=False), trainable=True))
                    self.biases.append(tf.get_variable(name=bias_name, shape=[args.hidden_layer_dim],
                                    initializer=tf.contrib.layers.xavier_initializer(uniform=False), trainable=True))

            self.node_head_ids = tf.placeholder(tf.int64, shape=[None])

            self.node_tail_ids = tf.placeholder(tf.int64, shape=[None])

            self.sample_ids_set = tf.placeholder(tf.int64, shape=[None])

            self.node_head_outDeg = tf.placeholder(tf.float32, shape=[None])

            self.node_tail_inDeg = tf.placeholder(tf.float32, shape=[None])

            self.node_head_embedding = tf.matmul(tf.one_hot(self.node_head_ids, depth=args.num_of_gra),
                                                 self.node_embedding_matrix)

            self.node_tail_embedding = tf.matmul(tf.one_hot(self.node_tail_ids, depth=args.num_of_gra),
                                                 self.node_embedding_matrix)

            self.head_node_score = self.generate_node(self.node_head_embedding, Layer_num)
            self.head_node_score = tf.transpose(self.head_node_score)

            self.tail_node_score = self.generate_node(self.node_tail_embedding, Layer_num)
            self.tail_node_score = tf.transpose(self.tail_node_score)

            # loss function
            self.loss_fir_term = self.node_tail_inDeg * tf.square(args.delay_factor) * \
                                   tf.square(self.head_node_score / self.node_head_outDeg
                                   -self.tail_node_score / (self.node_tail_inDeg * args.delay_factor))
            self.loss_sec_term = (self.head_node_score / self.node_head_outDeg
                                   -self.tail_node_score / (self.node_tail_inDeg * args.delay_factor)) \
                                 * 2 * args.delay_factor * (1-args.delay_factor) / self.n_node
            self.loss_third_term = tf.square(1-args.delay_factor) / (self.node_tail_inDeg *
                                   tf.square(tf.cast(self.n_node, dtype=tf.float32)))
            self.AsyPreser_loss = self.loss_fir_term + self.loss_sec_term + self.loss_third_term

            self.output_upped_w = tf.matmul(self.node_embedding_matrix, tf.transpose(self.node_embedding_matrix))
            # self.output_upped_batch_w = tf.matmul(tf.one_hot(self.sample_ids_set, depth=args.num_of_gra),
            #                                      self.output_upped_w)
            # self.output_discrete_batch_w = gumbel_softmax(self.output_upped_batch_w, temperature=args.tau, hard=True)
            self.output_discrete_batch_w = gumbel_softmax(self.output_upped_w, temperature=args.tau, hard=True)
            self.discrete_batch_w_index = tf.argmax(self.output_discrete_batch_w, axis=1)

            self.loss = tf.reduce_mean(self.AsyPreser_loss)
            self.optimizer = tf.train.AdamOptimizer(args.learning_rate)

            self.params = [v for v in tf.trainable_variables() if 'graph_forward_pass' in v.name]
            if args.is_GradientClip:
                self.var_list = self.params
                self.grads_and_vars = self.optimizer.compute_gradients(self.loss, self.var_list)
                for i, (g, v) in enumerate(self.grads_and_vars):  # for each pair
                    if g is not None and v is not None:
                        if "node_embedding_mat" in v.name:
                            '''
                            Note that when printing the L2 norm of the noisy gradient and 
                            the L2 norm of the true gradient, the noise added to the noise gradient 
                            is many orders of magnitude larger than the true gradient. This leads 
                            to difficulty in maintaining the number of triangles in the generated synthetic 
                            graph. Indeed, the results of TC in Tables 2 and 3 formally confirm this point in the paper, 
                            while the distribution information is relatively well maintained. One important reason 
                            for this situation is that a one-hot encoding was applied before the input embedding matrix, 
                            which may result in many zero gradients in the gradient with respect to the input matrix. Since 
                            gradient information is not released and only the synthetic graph is published, noise can be added 
                            only to the non-zero gradients in the future to help improve the situation.
                            '''
                            noise_g = g + self.Gau_Noise(g, args.g_clip)
                            self.grads_and_vars[i] = (noise_g, v)
                        else:
                            self.grads_and_vars[i] = (g, v)

                self.train_op = self.optimizer.apply_gradients(self.grads_and_vars)

    def generate_node(self, node_embedding, Layer_num):
        global layer_result
        input = tf.reshape(node_embedding, [-1, args.embedding_dim])
        for l in range(Layer_num):
            self.biase = self.biases[l]
            # self.weight = tf.clip_by_norm(self.weights[l], args.w_clip)  # weight clipping
            self.weight = spectral_norm(self.weights[l]) * args.w_clip
            # self.weight = l2_norm(self.weights[l])
            layer_result = tf.nn.sigmoid(tf.add(tf.matmul(input, self.weight), self.biase))
            input = layer_result
        output = layer_result
        return output

    def Gau_Noise(self, tensor, sens_value):
        total_iters = args.n_epochs * math.floor(self.n_node / args.batch_size)
        '''
        Split privacy parameters and epsilon, where epsilon is not divided by T 
        because the sens_value is divided by T according to Eq (13), and T is canceled out
        '''
        sigma = tf.sqrt(2 * tf.log(1.25 / (args.delta / total_iters))) / args.epsilon
        s = tensor.get_shape().as_list()  # get shape of the tensor
        Gau_noise = tf.random_normal(s, mean=0.0, stddev=sigma * sens_value)
        # t = tf.add(tensor, rt)
        return Gau_noise

    def random_walk(self, s, graph):
        walk = []
        p = s
        while len(walk) < args.walk_len:
            # Returns an iterator over successor nodes of n
            # A successor of n is a node m such that there exists a directed edge from n to m.
            if p not in graph.nodes() or len(graph.neighbors(p)) == 0:
                break
            p = random.choice(graph.neighbors(p))
            walk.append(p)
        return walk

    def random_walk_sampling(self, index, node_list, graph):
        node_head_ids = []
        node_tail_ids = []
        node_head_outDeg = []
        node_tail_inDeg = []

        for node_id in node_list[index * args.batch_size: (index + 1) * args.batch_size]:
            for k in range(args.walk_num):
                walk = self.random_walk(node_id, graph)
                for t in walk:
                    node_head_ids.append(node_id)
                    node_tail_ids.append(t)
        # print(graph.out_degree(node_head_ids))

        for head_id in node_head_ids:
            node_head_outDeg.append(graph.out_degree(head_id))
        for tail_id in node_tail_ids:
            node_tail_inDeg.append(graph.in_degree(tail_id))

        return node_head_ids, node_tail_ids, node_head_outDeg, node_tail_inDeg

    def train(self, graph):
        self.node_list = graph.nodes()
        node_count_mat = np.zeros((graph.number_of_nodes(), graph.number_of_nodes()))
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for each_epoch in range(args.n_epochs):
                for index in range(math.floor(len(self.node_list) / args.batch_size)):
                    print('epoch: %d, index: %d' % (each_epoch, index))
                    head_ids, tail_ids, head_outDeg, tail_inDeg = self.random_walk_sampling(index, self.node_list, graph)

                    if len(head_ids) is not 0 or len(tail_ids) is not 0:
                        sample_ids_set = list(set(head_ids) | set(tail_ids))
                        feed_dict = {self.node_head_ids: head_ids, self.node_tail_ids: tail_ids,
                                     self.sample_ids_set: sample_ids_set,
                                     self.node_head_outDeg: head_outDeg, self.node_tail_inDeg: tail_inDeg}
                        # node_embeddings, _, _ = sess.run([self.output_upped_w, self.train_op, self.loss], feed_dict=feed_dict)

                        discrete_batch_w_index = sess.run(self.discrete_batch_w_index, feed_dict=feed_dict)

                    for (i, j) in zip(self.node_list, discrete_batch_w_index):
                        node_count_mat[i, j] += 1

        return node_count_mat

def generate_SynGraphs(SynDigraName, num_of_edge, node_count_mat, node_embeddings):
    sparse_node_count_mat = sp.csr_matrix(node_count_mat)
    syn_graph_adj = graph_from_scores(sparse_node_count_mat, num_of_edge, node_embeddings)
    syn_graph = transform_adj_to_Graph(syn_graph_adj)
    saveGraphToEdgeListTxtn2v(syn_graph, SynDigraName)

def l2_norm(v):
    return v / (tf.reduce_sum(v ** 2) ** 0.5)

def spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.Variable(initial_value=tf.truncated_normal(shape=[1, w_shape[-1]]), trainable=False)
    # u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        """
        power iteration
        Usually iteration = 1 will be enough
        """
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = l2_norm(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = l2_norm(u_)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
    w_norm = w / sigma

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm

def transform_adj_to_Graph(adj):
    n = adj.shape[0]
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    for i in range(n):
        for j in range(n):
            if(i != j):
                if(adj[i, j] > 0):
                    graph.add_edge(i, j, weight=adj[i, j])
    return graph

def saveGraphToEdgeListTxt(graph, file_name):
    with open(file_name, 'w') as f:
        f.write('%d\n' % graph.number_of_nodes())
        f.write('%d\n' % graph.number_of_edges())
        for i, j, w in graph.edges(data='weight', default=1):
            f.write('%d %d\n' % (i, j))

def saveGraphToEdgeListTxtn2v(graph, file_name):
    with open(file_name, 'w') as f:
        for i, j, w in graph.edges(data='weight', default=1):
            f.write('%d %d\n' % (i, j))

def graph_from_scores(scores, n_edges=None):
    target_g = np.zeros(scores.shape)  # initialize target graph
    scores = scores + scores.T
    scores_int = scores.toarray().copy()  # internal copy of the scores matrix
    scores_int[np.diag_indices_from(scores_int)] = 0  # set diagonal to zero
    degrees_int = scores_int.sum(1)   # The row sum over the scores.

    N = target_g.shape[0]
    for n in range(N):  # Iterate over the nodes
        target = np.random.choice(N, p=scores_int[n] / degrees_int[n])
        target_g[n, target] = 1
        target_g[target, n] = 1

    sigmoid_embeddings = 1 / (1 + np.exp(-scores_int))
    upper_triangle_indices = np.triu_indices_from(sigmoid_embeddings, k=1)
    upper_triangle_values = sigmoid_embeddings[upper_triangle_indices]
    estimated_edge_num = np.sum(upper_triangle_values > 0.5) / 250

    diff = np.round((2 * estimated_edge_num - target_g.sum()) / 2)
    if diff > 0:
        triu = np.triu(scores_int)
        triu[target_g.nonzero()] = 0
        triu = triu / triu.sum()

        n_possible = np.count_nonzero(triu)

        triu_ixs = np.triu_indices_from(scores_int)
        extra_edges = np.random.choice(
            triu_ixs[0].shape[0], replace=False, p=triu[triu_ixs], size=min(int(diff), int(n_possible))
        )

        target_g[(triu_ixs[0][extra_edges], triu_ixs[1][extra_edges])] = 1
        target_g[(triu_ixs[1][extra_edges], triu_ixs[0][extra_edges])] = 1

    return target_g

def symmetric(directed_adjacency, clip_to_one=True):
    A_symmetric = directed_adjacency + directed_adjacency.T
    if clip_to_one:
        A_symmetric[A_symmetric > 1] = 1
    return A_symmetric

def gumbel_softmax(logits, temperature, hard=False):
    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        y_hard = tf.cast(tf.equal(y, tf.reduce_max(y, 1, keep_dims=True)), y.dtype)
        y = tf.stop_gradient(y_hard - y) + y
    return y

def sample_gumbel(shape, eps=1e-20):
    U = tf.random_uniform(shape, minval=0, maxval=1)
    return -tf.log(-tf.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(tf.shape(logits))
    return tf.nn.softmax(y / temperature)

if __name__ == '__main__':
    dataset_name = 'chicago'
    Alg_name = 'PrivDPR'
    w_clip_values = [1/8]
    epsilon_values = [0.1]

    Pre_name = 'Processed_'
    train_filename = '../ProcessedData/' + Pre_name + dataset_name + '.txt'
    # Load graph
    OriGraph = graph_util.loadGraphFromEdgeListTxt(train_filename, directed=True)
    # OriGraph = OriGraph.to_directed()
    num_of_edge = OriGraph.number_of_edges()
    num_of_node = OriGraph.number_of_nodes()
    M = (2 * (num_of_node - 1) * args.delay_factor ** 2 + 2 * args.delay_factor \
         + 2 * args.delay_factor * (1 - args.delay_factor) / num_of_node) * (1 + 1 / args.delay_factor)
    num_of_sampledNodePairs = args.batch_size * args.walk_num * args.walk_len
    iterNum_in_each_epoch = math.floor(num_of_node / args.batch_size)
    x = args.g_clip / (num_of_sampledNodePairs * M * args.n_epochs * iterNum_in_each_epoch)
    base = args.w_clip
    Layer_num = math.ceil(math.log(x, base) - 1)
    # print(Layer_num)

    model = DiGraSynModel(OriGraph, Layer_num)
    node_count_mat, node_embeddings = model.train(OriGraph)
    generate_SynGraphs(Alg_name, num_of_edge, node_count_mat, node_embeddings)

    print('performing is end')

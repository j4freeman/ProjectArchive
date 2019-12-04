import networkx as nx
from networkx.algorithms import community, bridges, smallworld
import numpy as np
import matplotlib.pyplot as plt

import collections

from scipy import stats

import imageio

import time

import copy

from tqdm import trange, tqdm

import os

np.random.seed(0)

class ScalarSocialModel:
    def __init__(self, upper=1, mu=1/2, sigma=np.sqrt(1/2) / 2, thresh=0.95, prob_adopted=0.5, neighbor_threshold=5/100, sim_thresh=0.15, product_adoption_jitter=np.sqrt(1/2) / 12, n_init=20, path='runs/'):
        self.lower = 0
        self.upper = upper
        self.mu = mu
        self.sigma = sigma
        self.thresh = thresh
        self.prob_adopted = prob_adopted
        self.neighbor_threshold = neighbor_threshold
        self.sim_thresh = sim_thresh
        self.product_adoption_jitter = product_adoption_jitter
        self.product_dist = stats.truncnorm((self.lower - self.mu)/self.sigma, (self.upper - self.mu)/self.sigma, loc=self.mu, scale=self.sigma)
        self.n_init = n_init
        self.path = path
        self.ims = []

        if not os.path.exists(self.path + 'graphs/'):
            os.makedirs(self.path + 'graphs/')
        if not os.path.exists(self.path + 'frames/'):
            os.makedirs(self.path + 'frames/')
        if not os.path.exists(self.path + 'gifs/'):
            os.makedirs(self.path + 'gifs/')

        self.G = nx.connected_watts_strogatz_graph(self.n_init, 3, np.random.rand())
        products = dict([(i, self.random_product()) for i in range(self.n_init)])
        thresholds = dict([(i, np.random.rand()) for i in range(self.n_init)])
        schellings = dict([(i, self.random_schelling()) for i in range(self.n_init)])

        nx.set_node_attributes(self.G, products, 'apt_markakis')
        nx.set_node_attributes(self.G, thresholds, 'threshold')
        nx.set_node_attributes(self.G, schellings, 'schelling')

        nx.write_graphml(self.G, self.path + 'graphs/m0.graphml')

    def random_product(self):
        return np.random.choice([np.random.rand()*self.upper, -1], p=[self.prob_adopted, 1 - self.prob_adopted])

    def gauss_random_product(self):
        return np.random.choice([self.product_dist.rvs(), -1], p=[self.prob_adopted, 1 - self.prob_adopted])

    def random_schelling(self):
        return np.random.rand() * (self.upper / 8)

    def barabasi_step(self):
        for _ in range(np.random.choice(2, p=[0.3, 0.7])):
            tot = self.G.number_of_edges() * 2
            probs = [val / tot for (_, val) in self.G.degree()]
            rands = np.random.uniform(size=len(probs))
            check = [rands[i] <= probs[i] for i in range(len(probs))]
            while not True in check:
                rands = np.random.uniform(size=len(probs))
                check = [rands[i] <= probs[i] for i in range(len(probs))]
            new_prod = self.random_product()
            self.G.add_node(len(rands), apt_markakis = new_prod, schelling = self.random_schelling(), threshold=np.random.rand())
            self.G.add_edges_from([(len(rands), i) for i in range(len(rands)) if check[i]])

    def threshold_met(self, node):
        neighbors = [self.G.node[x]['apt_markakis'] for x in self.G[node]]
        if neighbors:
            return (sum([1 for x in neighbors if x > -1]) / len(neighbors)) > self.G.node[node]['threshold']
        return False

    def apt_markakis_step_med(self):
        unset = [x for x in self.G.nodes if self.G.node[x]['apt_markakis'] < 0]
        candidates = [x for x in unset if self.threshold_met(x)]
        if candidates:
            tar = np.random.choice(candidates)
            neighbor_med = np.median([self.G.node[x]['apt_markakis'] for x in self.G.nodes if self.G.node[x]['apt_markakis'] > -1])
            adopted = np.random.normal() * self.product_adoption_jitter + neighbor_med
            while adopted < 0:
                adopted = np.random.normal() * self.product_adoption_jitter + neighbor_med
            self.G.node[tar]['apt_markakis'] = adopted

    def apt_markakis_step_avg(self):
        unset = [x for x in self.G.nodes if self.G.node[x]['apt_markakis'] < 0]
        candidates = [x for x in unset if self.threshold_met(x)]
        if candidates:
            tar = np.random.choice(candidates)
            neighbor_avg = np.average([self.G.node[x]['apt_markakis'] for x in self.G.nodes if self.G.node[x]['apt_markakis'] > -1])
            adopted = np.random.normal() * self.product_adoption_jitter + neighbor_avg
            while adopted < 0:
                adopted = np.random.normal() * self.product_adoption_jitter + neighbor_avg
            self.G.node[tar]['apt_markakis'] = adopted

    def bucket_markakis(self, n_buckets):
        unset = [x for x in self.G.nodes if self.G.node[x]['apt_markakis'] < 0]
        candidates = [x for x in unset if self.threshold_met(x)]
        if not candidates:
            return
        tar = np.random.choice(candidates)
        neighbor_list = [self.G.node[x]['apt_markakis'] for x in self.G.nodes if self.G.node[x]['apt_markakis'] > -1]

        buckets = [0] * n_buckets
        for n in neighbor_list:
            idx = int(n * n_buckets)
            buckets[idx] += 1

        buckets = [b / sum(buckets) for b in buckets]

        neighbor_med = np.random.choice(n_buckets, p=buckets) / n_buckets


        adopted = np.random.normal() * self.product_adoption_jitter + neighbor_med
        while adopted < 0 or adopted > self.upper:
            adopted = np.random.normal() * self.product_adoption_jitter + neighbor_med

        self.G.node[tar]['apt_markakis'] = adopted

    def unhappy(self, node):
        neighbors = [self.G.node[x]['apt_markakis'] for x in self.G[node] if self.G.node[x]['apt_markakis'] > -1]
        if not neighbors:
            return False
        dif_neighbors = [x for x in neighbors if np.abs(self.G.node[node]['apt_markakis'] - x) / self.G.node[node]['apt_markakis'] > self.neighbor_threshold]
        return len(dif_neighbors) / len(neighbors) > self.sim_thresh

    def neighbor_dist_avg(self, node):
        neighbors = [self.G.node[x]['apt_markakis'] for x in self.G[node] if self.G.node[x]['apt_markakis'] > -1]
        if neighbors:
            return np.sqrt(np.sum([(neighbor - self.G.node[node]['apt_markakis'])**2 for neighbor in neighbors])) / len(neighbors)
        return np.inf

    def euclidian(self, node, vec):
        return np.sum([np.abs(self.G.node[node]['apt_markakis'] - self.G.node[n]['apt_markakis']) for n in vec]) / len(vec)
    #     return np.sqrt(np.sum([(self.G.node[neighbor]['apt_markakis'] - self.G.node[node]['apt_markakis'])**2 for neighbor in vec])) / len(vec)

    def schelling_step_avg(self):
        adopted = [x for x in self.G.nodes if self.G.node[x]['apt_markakis'] > -1]
        candidates = [x for x in adopted if self.unhappy(x)]
        add_ctr = 0
        rem_ctr = 0
        if candidates:
            tar = np.random.choice(candidates)
            probs = [val for (_, val) in self.G.degree()]
            tot = sum([np.e**x for x in probs])
            probs = [np.e**x / tot for x in probs]
            friends = list(set([n for i in self.G[tar] for n in self.G[i]]))
            friend_probs = [1 / (len(friends)) if x in friends else 0 for x in range(len(probs))]
            probs = [0.5 * probs[i] + 0.5 * friend_probs[i] for i in range(len(probs))]
            test_pts = np.random.choice(len(probs), p=probs, replace=False, size=len(probs))
            test_pts = [test for test in test_pts if self.G.node[test]['apt_markakis'] > -1]
            bs = list(bridges(self.G))
            for test in test_pts:

                neighbors = [x for x in self.G[tar] if self.G.node[x]['apt_markakis'] > -1]
                neighbor_dist = self.euclidian(tar, neighbors)
                if self.G.has_edge(test, tar) and (test, tar) not in bs and (tar, test) not in bs:
                    neighbor_test = copy.deepcopy(neighbors)
                    neighbor_test.remove(test)
                    if self.euclidian(tar, neighbor_test) / neighbor_dist < self.thresh:
                        self.G.remove_edge(test, tar)
                        bs = list(bridges(self.G))
                        rem_ctr += 1
                elif self.euclidian(tar, neighbors + [test]) / neighbor_dist < self.thresh:
                    self.G.add_edge(test, tar)
                    add_ctr += 1
                if not self.unhappy(tar):
                    break

    def similar(self, node):
        set_nodes = [x for x in self.G.nodes() if self.G.node[x]['apt_markakis'] > -1]
        similars =  [x for x in set_nodes if np.abs(self.G.node[node]['apt_markakis'] - self.G.node[x]['apt_markakis']) / self.G.node[node]['apt_markakis'] <= self.neighbor_threshold]
        return [x for x in similars if not (self.G.has_edge(x, node) or self.G.has_edge(node, x)) ]

    def dif_nodes(self, node):
        neighbors = [x for x in self.G[node] if self.G.node[x]['apt_markakis'] > -1]
        difs =  [x for x in neighbors if np.abs(self.G.node[node]['apt_markakis'] - self.G.node[x]['apt_markakis']) / self.G.node[node]['apt_markakis'] > self.neighbor_threshold]
        return difs

    def improved_schelling(self):
        adopted = [x for x in self.G.nodes if self.G.node[x]['apt_markakis'] > -1]
        candidates = [x for x in adopted if self.unhappy(x)]
        add_ctr = 0
        rem_ctr = 0
        if not candidates:
            return

        tar = np.random.choice(candidates)
        similars = self.similar(tar)
        sim_vals = [True for _ in similars]
        difs = self.dif_nodes(tar)
        dif_vals = [False for _ in difs]

        test_pts = similars + difs
        test_vals = sim_vals + dif_vals

        degrees = [self.G.degree[x] for x in test_pts]
        degrees = [degrees[i] / 4 if test_vals[i] else degrees[i] for i in range(len(degrees))]
        degrees = [x/sum(degrees) for x in degrees]

        indices = np.random.choice(len(degrees), p=degrees, replace=False, size=len(degrees))

        bs = list(bridges(self.G))
        for idx in indices:
            if test_vals[idx]:
                self.G.add_edge(tar, test_pts[idx])
            else:
                if not (test_pts[idx], tar) in bs and not (tar, test_pts[idx]) in bs:
                    self.G.remove_edge(test_pts[idx], tar)
                    bs = list(bridges(self.G))
            if not self.unhappy(tar):
                break


    def get_products(self):
        attrs = [nx.get_node_attributes(self.G, 'apt_markakis')[i] for i in self.G.nodes]
        return attrs

    def gif(self):
        files = []
        for im in self.ims:
            files.append(imageio.imread(im))

        gif_name = self.path + 'gifs/' + str(int(time.time())) + '.gif'
        imageio.mimsave(gif_name, files, duration=0.3)


    def run(self, iters=250, make_gif=False):
        if make_gif:
            prods = self.get_products()
            nx.draw_kamada_kawai(self.G, node_color=prods, node_size=20)
            fname = self.path + 'frames/0.png'
            plt.savefig(fname)
            self.ims.append(fname)
            plt.cla()
            plt.clf()
        for i in trange(iters, desc='executing model'):
            if make_gif:
                prods = self.get_products()
                nx.draw_kamada_kawai(self.G, node_color=prods, node_size=20)
                fname = self.path + 'frames/' + str(i + 1) + ".png"
                plt.savefig(fname)
                self.ims.append(fname)
                plt.cla()
                plt.clf()
            self.barabasi_step()
            self.bucket_markakis(100)
            self.improved_schelling()
            nx.write_graphml(self.G, self.path + 'graphs/m' + str(i + 1) + '.graphml')
        if make_gif:
            self.gif()

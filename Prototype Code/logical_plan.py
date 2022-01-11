import torch
import pickle
import torch.utils.data
import time
import dgl
import os
import numpy as np
import csv
from scipy import sparse as sp

class LogicalPlanDGL(torch.utils.data.Dataset):
    def __init__(self, data_dir, split, normalized, all_graphs, undirected):
        self.data_dir = data_dir
        self.split = split
        self.undirected = undirected
        self.graph_labels = []
        self.graph_lists = []
        self.all_graphs = all_graphs
        if normalized:
            self.index = 2
        else:
            self.index = 1
        self._prepare()


    def _prepare(self):
        print("preparing graphs for the %s set..." % ( self.split.upper()))
        with open(self.data_dir + 'holistic_cost_dgl_pickles_job_light/team10_job_light_cardinalities.csv') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            not_first_element = False
            cardinalities_job_light = []
            for row in reader:
                if not_first_element:
                    cardinalities_job_light.append(float(row[self.index]))
                else:
                    not_first_element = True
        with open(self.data_dir + 'holistic_cost_dgl_pickles_synthetic/team10_synthetic_cardinalities.csv') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            not_first_element = False
            cardinalities_synthetic = []
            for row in reader:
                if not_first_element:
                    cardinalities_synthetic.append(float(row[self.index]))
                else:
                    not_first_element = True
        ind = 0
        if self.split == "train":
            name = "holistic_cost_dgl_pickles_synthetic/Synthetic_"
            self.n_samples = 4250
            r = range(1, 4251)
            cardinalities = cardinalities_synthetic
        elif self.split == "test":
            name = "holistic_cost_dgl_pickles_synthetic/Synthetic_"
            self.n_samples = 750
            r = range(4251, 5001)
            cardinalities = cardinalities_synthetic
        else:
            name = "holistic_cost_dgl_pickles_job_light/job_light_"
            self.n_samples = 70
            r = range(1, 71)
            cardinalities = cardinalities_job_light
        for i in r:
            s = self.data_dir + name + str(i) + '.pkl'
            with open(s, "rb") as f:
                g = pickle.load(f)
            g2 = dgl.graph(g.edges())
            g2.ndata["feat"] = g.ndata["feat"].float()
            g2.edata["feat"] = g.edata["feat"].float()
            if self.undirected:
                srcs = g2.edges()[0]
                dsts = g2.edges()[1]
                new_srcs = torch.cat((srcs, dsts), 0)
                new_dsts = torch.cat((dsts, srcs), 0)
                g3 = dgl.graph((new_srcs, new_dsts))
                g3.ndata["feat"] = g2.ndata["feat"].float()
                g3.edata["feat"] = torch.cat((g2.edata["feat"].float(), g2.edata["feat"].float()), 0)
                g2 = g3
            if g2.num_nodes() >= 9 or self.all_graphs:
                self.graph_lists.append(g2)
                self.graph_labels.append(torch.tensor(cardinalities[i-1]))
                ind = ind + 1

        self.n_samples = ind

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.graph_lists[idx], self.graph_labels[idx]


class LogicalPlanDatasetDGL(torch.utils.data.Dataset):
    def __init__(self, name):
        t0 = time.time()
        data_dir = 'data/'
        normalized = True
        all_graphs = True
        undirected = False
        if name == "LogicalPlanUnnormalized":
            normalized = True
        if name == "LogicalPlanBigGraphs":
            all_graphs = False
        if name == "LogicalPlanUndirected":
            undirected = True

        self.train = LogicalPlanDGL(data_dir, 'train', normalized, all_graphs, undirected)
        self.val = LogicalPlanDGL(data_dir, 'val', normalized, all_graphs, undirected)
        self.test = LogicalPlanDGL(data_dir, 'test', normalized, all_graphs, undirected)
        print("Time taken: {:.4f}s".format(time.time() - t0))


class LogicalPlanDataset(torch.utils.data.Dataset):
    def __init__(self, name):
        start = time.time()
        print("[I] Loading dataset %s..." % (name))
        self.name = name
        data_dir = 'data/LogicalPlan/'
        with open(data_dir+name+'.pkl',"rb") as f:
            f = pickle.load(f)
            self.train = f[0]
            self.val = f[1]
            self.test = f[2]
            self.num_atom_type = f[3]
            self.num_bond_type = f[4]
        print('train, test, val sizes :',len(self.train),len(self.test),len(self.val))
        print("[I] Finished loading.")
        print("[I] Data load time: {:.4f}s".format(time.time()-start))

    def collate(self, samples):
        graphs, labels = map(list, zip(*samples))
        labels = torch.tensor(np.array(labels)).unsqueeze(1)
        batched_graph = dgl.batch(graphs)
        return batched_graph, labels

    def _add_self_loops(self):
        # function for adding self loops
        # this function will be called only if self_loop flag is True

        self.train.graph_lists = [self_loop(g) for g in self.train.graph_lists]
        self.val.graph_lists = [self_loop(g) for g in self.val.graph_lists]
        self.test.graph_lists = [self_loop(g) for g in self.test.graph_lists]

    def _add_positional_encodings(self, pos_enc_dim):
        # Graph positional encoding v/ Laplacian eigenvectors
        self.train.graph_lists = [positional_encoding(g, pos_enc_dim) for g in self.train.graph_lists]
        self.val.graph_lists = [positional_encoding(g, pos_enc_dim) for g in self.val.graph_lists]
        self.test.graph_lists = [positional_encoding(g, pos_enc_dim) for g in self.test.graph_lists]


def self_loop(g):
    """
        Utility function only, to be used only when necessary as per user self_loop flag
        : Overwriting the function dgl.transform.add_self_loop() to not miss ndata['feat'] and edata['feat']


        This function is called inside a function in MoleculeDataset class.
    """
    new_g = dgl.DGLGraph()
    new_g.add_nodes(g.number_of_nodes())
    new_g.ndata['feat'] = g.ndata['feat']

    src, dst = g.all_edges(order="eid")
    src = dgl.backend.zerocopy_to_numpy(src)
    dst = dgl.backend.zerocopy_to_numpy(dst)
    non_self_edges_idx = src != dst
    nodes = np.arange(g.number_of_nodes())
    new_g.add_edges(src[non_self_edges_idx], dst[non_self_edges_idx])
    new_g.add_edges(nodes, nodes)

    # This new edata is not used since this function gets called only for GCN, GAT
    # However, we need this for the generic requirement of ndata and edata
    new_g.edata['feat'] = torch.zeros(new_g.number_of_edges())
    return new_g


def positional_encoding(g, pos_enc_dim):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """

    # Laplacian
    A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # Eigenvectors with numpy
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort()  # increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])
    g.ndata['pos_enc'] = torch.from_numpy(EigVec[:, 1:pos_enc_dim + 1]).float()

    # # Eigenvectors with scipy
    # EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR')
    # EigVec = EigVec[:, EigVal.argsort()] # increasing order
    # g.ndata['pos_enc'] = torch.from_numpy(np.abs(EigVec[:,1:pos_enc_dim+1])).float()

    return g

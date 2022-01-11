import torch
import pickle
import torch.utils.data
import time
import dgl
import os
import numpy as np
import csv
from scipy import sparse as sp

class PhysicalPlanDGL(torch.utils.data.Dataset):
    def __init__(self, data_dir, split, name):
        self.data_dir = data_dir
        if name == "PhysicalPlanTime":
            self.dataset = "time"
        elif name == "PhysicalPlanCost":
            self.dataset = "cost"
        elif name == "PhysicalPlanCardinality":
            self.dataset = "cardinalities"
        self.name = name
        self.split = split
        self.graph_labels = []
        self.graph_lists = []
        self._prepare()


    def _prepare(self):
        print("preparing graphs for the %s set..." % (self.split.upper()))
        with open(self.data_dir + 'holistic_cost_dgl_pickles_job_light_physical/team10_job_light_' + str(self.dataset) + '.csv') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            not_first_element = False
            cardinalities_job_light = []
            for row in reader:
                if not_first_element:
                    cardinalities_job_light.append(float(row[2]))
                else:
                    not_first_element = True
        with open(self.data_dir + 'holistic_cost_dgl_pickles_synthetic_physical/team10_synthetic_' + str(self.dataset) + '.csv') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            not_first_element = False
            cardinalities_synthetic = []
            for row in reader:
                if not_first_element:
                    cardinalities_synthetic.append(float(row[2]))
                else:
                    not_first_element = True

        if self.split == "train":
            name = "holistic_cost_dgl_pickles_synthetic_physical/"
            self.n_samples = 4250
            r = range(1, 4251)
            cardinalities = cardinalities_synthetic
        elif self.split == "test":
            name = "holistic_cost_dgl_pickles_synthetic_physical/"
            self.n_samples = 750
            r = range(4251, 5001)
            cardinalities = cardinalities_synthetic
        else:
            name = "holistic_cost_dgl_pickles_job_light_physical/"
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
            self.graph_lists.append(g2)
            self.graph_labels.append(torch.tensor(cardinalities[i-1]))

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.graph_lists[idx], self.graph_labels[idx]


class PhysicalPlanDatasetDGL(torch.utils.data.Dataset):
    def __init__(self, name):
        t0 = time.time()
        data_dir = 'data/'
        print("Loading Dataset " + name)
        self.train = PhysicalPlanDGL(data_dir, 'train', name)
        self.val = PhysicalPlanDGL(data_dir, 'val', name)
        self.test = PhysicalPlanDGL(data_dir, 'test', name)
        print("Time taken: {:.4f}s".format(time.time() - t0))


class PhysicalPlanDataset(torch.utils.data.Dataset):
    #PhysicalPlanTime or PhysicalPlanCost
    def __init__(self, name):
        start = time.time()
        print("[I] Loading dataset %s..." % (name))
        self.name = name
        data_dir = 'data/PhysicalPlan/'
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

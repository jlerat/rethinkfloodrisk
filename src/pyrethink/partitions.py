from itertools import combinations

import numpy as np


class Partitions():
    def __init__(self, nelements):
        if nelements > 10:
            errmsg = "Expected nelements <= 10."
            raise ValueError(errmsg)

        # Initialise
        self._nelements = nelements
        self._data = list(range(nelements))
        self._subsets = []
        self._pair_in_same_cluster = []
        self._subsets_counts = []
        self._nsubsets = 0

        self._probabilities = None

        # Populate
        self.add_subsets(0, [])

        # Convert to arrays
        self._subsets = np.array(self.subsets)
        self._subsets_counts = np.array(self.subsets_counts)
        self._pair_in_same_cluster = np.array(self.pair_in_same_cluster)

    @property
    def nelements(self):
        return self._nelements

    @property
    def data(self):
        return self._data

    @property
    def subsets(self):
        return self._subsets

    @property
    def subsets_counts(self):
        return self._subsets_counts

    @property
    def pair_in_same_cluster(self):
        return self._pair_in_same_cluster

    @property
    def nsubsets(self):
        return len(self._subsets)

    @property
    def npairs(self):
        nel = self.nelements
        return nel * (nel - 1) // 2

    def add_subsets(self, index, ans):
        data = self.data
        nel = self.nelements
        npairs = self.npairs

        if index == len(data):
            combs = np.zeros((nel, nel))
            insame = [0] * npairs
            for ipart, parts in enumerate(ans):
                for d in parts:
                    combs[ipart, d] = 1

                for ic, (a, b) in enumerate(combinations(range(nel), 2)):
                    v = combs[ipart, a] * combs[ipart, b] or insame[ic]
                    insame[ic] = int(v)

            self.pair_in_same_cluster.append(insame)
            self.subsets.append(combs)
            self.subsets_counts.append(len(ans))

            return

        elem = data[index]

        for i in range(len(ans)):
            ans[i].append(elem)
            self.add_subsets(index + 1, ans)
            ans[i].pop()

        ans.append([elem])
        self.add_subsets(index + 1, ans)
        ans.pop()

    def find_subset_index(self, pair_in_same):
        found = []
        for i in range(self.nsubsets):
            same = self.pair_in_same_cluster[i]
            if all(s == p for s, p in zip(same, pair_in_same)):
                found.append(i)

        return found

    def extract_index(self, index):
        return [self.subsets[i] for i in index]

    def find_subset(self, pair_in_same):
        idx = self.find_subset_index(pair_in_same)
        return self.extract_index(idx)

    def ipart2sets(self, ipart):
        part = self.subsets[ipart]
        cnt = self.subsets_counts[ipart]
        part = part[:cnt]
        i1, i2 = np.where(part == 1)
        return i1[np.argsort(i2)]

    def compute_probabilities(self, partitions_id, dirichlet_alpha):
        if dirichlet_alpha < 1:
            errmsg = "Expected dirichlet_alpha >= 1."
            raise ValueError(errmsg)

        ns = self.nsubsets
        probs = np.zeros(ns)
        for pid in np.unique(partitions_id):
            if pid >= ns:
                errmsg = f"Expected partition IDs in [0, {ns}[."
                raise ValueError(errmsg)
            elif pid == -1:
                # Missing data
                continue

            n = (pid == partitions_id).sum()
            probs[pid] = float(n) + dirichlet_alpha - 1.

        return probs / probs.sum()

    def sample(self, probs, nsamples):
        k = np.arange(self.nsubsets)
        return np.random.choice(k, p=probs,
                                size=nsamples)

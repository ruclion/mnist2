from __future__ import print_function

import numpy as np
import heapq
import copy

class node(object):
    def __init__(self, val = None, lrt = None, rrt = None, fa = None, axis = None):
        self.val = val
        self.lrt = lrt
        self.rrt = rrt
        self.fa = fa
        self.axis = axis

class KDTree(object):
    def __init__(self, str = 'simple', data = None):
        self.dist_kind = str
        self.data = data
        # print(data[0])
        if len(data) != 0:
            self.all_axis = len(data[0]) - 1
        self.num = len(data)
        print(self.num)
        self.rt = self.create(0, 0, self.num - 1, -1)
        print('rt', self.rt)
        self.find_times = 0


    #data is list, and data data is also list
    def create(self, axis = None, l = None, r = None, fa = None):
        if l > r:
            return None
        rt = node()
        key = (r - l + 1) // 2
        # print(self.data[0 : 2])
        def tmp_key(x):
            return x[axis]
        self.data[l : r + 1].sort(key=tmp_key)
        # self.data[l : r + 1] = np.sort(self.data[l : r + 1], )
        n_axis = (axis + 1) % self.all_axis
        lrt = self.create(n_axis, l, l + key - 1, rt)
        rrt = self.create(n_axis, l + key + 1, r, rt)
        rt.lrt = lrt
        rt.rrt = rrt
        rt.fa = fa
        rt.axis = axis
        rt.val = l + key
        return rt
    def dist(self, x, y):
        if self.dist_kind == 'simple':
            xx = np.array(x)
            yy = np.array(y)
            return sum((xx - yy) ** 2)
            # len_x = len(x)
            # # print(len_x, len(y))
            # sum = 0
            # for i in range(len_x):
            #     sum += (x[i] - y[i]) ** 2
            # return sum
    def find_dfs(self, x, h, rt, k):
        self.find_times += 1
        dist_val = self.dist(x, self.data[rt.val][0:-1])
        if len(h) < k or dist_val < (-h[0][0]):
            if len(h) == k:
                heapq.heappop(h)
            heapq.heappush(h, (-dist_val, rt.val))

        axis = rt.axis
        lrt = rt.lrt
        rrt = rt.rrt
        # print(len(x), len(self.data[rt.val]), axis)
        if x[axis] > self.data[rt.val][axis]:
            lrt = rt.rrt
            rrt = rt.lrt
        if lrt is not None:
            self.find_dfs(x, h, lrt, k)
        # print('origial', x[axis], self.data[rt.val][axis])
        vir_data = copy.copy(x[:])
        vir_data[axis] = self.data[rt.val][axis]
        # print(x[axis], vir_data[axis])
        # print(self.dist(x, vir_data))
        if rrt is not None and (len(h) < k or self.dist(x, vir_data)  < (-h[0][0])):
            self.find_dfs(x, h, rrt, k)


    def find_k_near(self, x = None, k = 5):
        h = []
        self.find_times = 0
        # print(self.rt)
        self.find_dfs(x, h, self.rt, k)
        print('find_time', self.find_times)
        h_len = len(h)
        cnt = np.arange(0, 10)
        for i in range(h_len):
            t = heapq.heappop(h)
            cnt[self.data[t[1]][-1]] += 1
        return cnt.argmax()

class KDTree_like_sklearn(object):
    def __init__(self, dist_kind = 'simple', k = 5):
        self.dist_kind = dist_kind
        self.k = k
    def fit(self, X, Y):
        x_list = []
        # print(type(x_list), type(X))
        len_x = len(X)
        for i in range(len_x):
            if type(X[i]) is not list:
                t = X[i].tolist()
            else:
                t = copy.copy(X[i][:])
            # print(type(t))
            t.append(Y[i])
            x_list.append(t)
            # x_list[i] = t
            # print(type(x_list[i]))
            # x_list[i].append(Y[i])
        self.kdtree = KDTree(self.dist_kind, x_list)

    def predict(self, X):
        ans = []
        len_x = len(X)

        for i in range(len_x):
            ans.append(self.kdtree.find_k_near(X[i], self.k))
            print('finish: ', i, len_x)
        return ans








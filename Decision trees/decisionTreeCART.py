import numpy as np
import csv
import math
# import sys
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
import matplotlib.gridspec as gridspec
from scipy.stats import mode

class Node:
    def __init__(self, training_data, depth, attrs, max_depth, msg = ""):
        self.training_data = training_data
        self.depth = depth
        self.max_depth = max_depth
        self.attrs = attrs
# 		self.used_attrs = used_attrs
        self.split_attr = None
        self.split_attr_value = None
        self.sons = []
        self.msg = msg
        self.label_value_cnt = self.calculate_label_value_cnt(training_data)
        self.majority_vote_result = max(self.label_value_cnt.keys(), key = lambda x: self.label_value_cnt[x])

        
    def calculate_label_value_cnt(self, training_data):
        label_value = training_data[1]
        label_value_cnt = {}
        for value in label_value:
            if value not in label_value_cnt:
                label_value_cnt[value] = 1
            else:
                label_value_cnt[value] += 1
        return label_value_cnt

    @staticmethod
    def impurity_cost(y):
        """Only computes entropy impurity for now."""
        n = len(y)
        value_counts = {}
        for val in y:
            if val not in value_counts:
                value_counts[val] = 1/n
            else:
                value_counts[val] += 1/n
        
        entropy = 0
        for k, p in value_counts.items():
            entropy += -p*np.log(p)
        return entropy
        
    def split_cost(self, j, s, X, y):
        
        # perform split
        left_set_X = []
        left_set_y = []
        right_set_X = []
        right_set_y = []
        for i in range(len(y)):
            if X[i, j] <= s:
                left_set_X.append(X[i].tolist())
                left_set_y.append(y[i])
            else:
                right_set_X.append(X[i].tolist())
                right_set_y.append(y[i])
                
        left_cost = self.impurity_cost(left_set_y)
        right_cost = self.impurity_cost(right_set_y)
        
        cost = left_cost + right_cost
        split = (np.array(left_set_X), left_set_y), (np.array(right_set_X), right_set_y)
        return cost, split

    def train(self):
        if self.depth == self.max_depth:
            return
        y = self.training_data[1]
        if len(set(y)) == 1:
            # The node is pure
            return
        
        X, y = self.training_data
        
        j_star, s_star = 0, 0
        cost_star = np.inf
        for j in self.attrs:
            # Compute all possible split points
            split_points = X[0:-1,j]
            # split_points.sort()
            split_points = set(split_points)

            for s in split_points:
                c, split_sets = self.split_cost(j, s, X, y)
                
                if c < cost_star:
                    cost_star = c
                    self.j_star = j
                    self.s_star = s
                    split_set_star = split_sets
        
        # catch edge case. In case we accidentally do a degenerate split, don't add children. 
        for i in range(2):
            if len(split_set_star[i][1]) == 0:
                return
        
        for i in range(2):
            son_training_data = split_set_star[i]
            msg = f"j = {j}, s = {s}"
            son = Node(son_training_data, 
                       self.depth + 1, 
                       self.attrs,
                       self.max_depth, 
                       msg)
            self.sons.append(son)
        
        for son in self.sons:
            son.train()
        return

    def predict(self, x):
        
        # if no children, return majority vote
        if len(self.sons) == 0:
            return self.majority_vote_result
        
        if x[self.j_star] <= self.s_star:
            return self.sons[0].predict(x)
        else:
            return self.sons[1].predict(x)
        

    # def print(self):
    #     print("{}{}".format("| " * self.depth + self.msg, self.label_value_cnt))
    #     for son in self.sons:
    #         son.print()
    #     return
		

class DecisionTree:
    def __init__(self, X_train, y_train, max_depth):
        self.training_data = (X_train, y_train)
        n_attributes = X_train.shape[1]
        self.attributes = np.arange(n_attributes)
        #self.training_data, self.attributes, self.label = self.parse_training_data_file(training_data_file)
        self.root = Node(self.training_data, 0, self.attributes, max_depth, [])

    # def parse_training_data_file(self, training_data_file):
    #     with open(training_data_file, "r") as csv_file:
    #         reader = csv.reader(csv_file)
    #         header = next(reader)
    #         attributes = header[:-1]
    #         label = header[-1]
    #         # print(f'header = {header}, \natt = {attributes}, \nlabel = {label}')
    #         # for row in reader:
    #         #     print(row)
    #         training_data = np.array([row for row in reader])
    #     return training_data, attributes, label

    def train(self):
        self.root.train()

    def predict(self, x):
        n_examples = x.shape[0]
        y_hat = np.zeros(n_examples)
        for i, x_prime in enumerate(x):
            y_hat[i] = self.root.predict(x_prime)
        return y_hat

    # def pretty_print(self):
    #     self.root.print()



class RandomForest():
    def __init__(self, tree_list):
        self.tree_list = tree_list
        self.n_trees = len(tree_list)
        
    def predict(self, x):
        
        n_examples = x.shape[0]
        y_hat = np.zeros((n_examples, self.n_trees))
        for i, tree in enumerate(self.tree_list):
            y_hat[:, i] = tree.predict(x)
        
        print('Tree Prediction Complete')
        
        #TODO: Make this array based, remove for loop
        out = np.zeros(n_examples)
        for i, y in enumerate(y_hat):
            out[i] = mode(y)[0].item()
        
        return np.array(out)
        # return y_hat/self.n_trees
        

def spiral_xy(i, spiral_num):
    """
    Create the data for a spiral.

    Arguments:
        i runs from 0 to 96
        spiral_num is 1 or -1
    """
    φ = i/16 * math.pi
    r = 6.5 * ((104 - i)/104)
    x = (r * math.cos(φ) * spiral_num)/13 + 0.5
    y = (r * math.sin(φ) * spiral_num)/13 + 0.5
    return [x, y]

def spiral(spiral_num):
    return [spiral_xy(i, spiral_num) for i in range(97)]

def main():
    
    
    max_depth = np.inf
    
    # iris = load_iris()
    # X = iris.data[:,[0, 2]]
    # y = iris.target
    
    X1 = spiral(1)
    X2 = spiral(-1)
    X = np.array(X1 + X2)
    
    y1 = np.zeros(len(X1), dtype=int)
    y2 = np.ones(len(X2), dtype=int)
    y = np.concatenate((y1, y2))
    
    tree_list = []
    for i in range(2):
        
        n_X = X.shape[0]
        sample_indices = np.random.randint(low=0, high=n_X, size=n_X).tolist()
        X_bootstrap = X[sample_indices,:]
        y_bootstrap = y[sample_indices]
        tree = DecisionTree(X_train=X_bootstrap, 
                        y_train=y_bootstrap, 
                        max_depth=max_depth)
        tree.train()
        tree_list.append(tree)
    print('Training Complete')
    
    RF = RandomForest(tree_list)
    
    # y_hat = RF.predict(np.array([[8, 8]]))
    gs = gridspec.GridSpec(2, 2)

    fig = plt.figure(figsize=(10,8))
    
    # ax = plt.subplot(gs[0, 0])
    # fig = plot_decision_regions(X=X, y=y, clf=tree_list[0], legend=2)
    # plt.title("decision tree")
    
    # ax = plt.subplot(gs[0, 1])
    # fig = plot_decision_regions(X=X, y=y, clf=tree_list[1], legend=2)
    # plt.title("decision tree")
    
    ax = plt.subplot(gs[1, 1])
    fig = plot_decision_regions(X=X, y=y, clf=RF, legend=2)
    plt.title("decision tree")

if __name__ == "__main__":
    main()


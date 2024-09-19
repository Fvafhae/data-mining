import random

import pandas as pd
import numpy as np

class Node():

    x = None
    y = None
    
    left = None
    right = None
    
    split_feature = None
    split_threshold = None
    
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def is_leaf(self):
        if self.left == None and self.right == None:
            return True
        else:
            return False
    
    def predict_y(self):
        return np.round(np.mean(self.y))


def tree_grow(x, y, nmin, minleaf):
    root = Node(x, y)
    grow(root, nmin, minleaf)
    return root


def grow(node, nmin, minleaf):

    if len(node.y) < nmin:
        return

    if len(np.unique(node.y)) == 1:
        return
    # node.split_feature = random.choice([x for x in range(len(node.x[0]))])
    # node.split_threshold = node.x[:, node.split_feature].mean()

    #TODO
    node.split_feature, node.split_threshold, split = search_best_split(node, minleaf)

    # left_split = {
    #     "x": [],
    #     "y": []
    # }
    # right_split = {
    #     "x": [],
    #     "y": []
    # }

    # for i in range(len(node.y)):
    #     if node.x[i, node.split_feature] <= node.split_threshold:
    #         left_split['x'].append(node.x[i])
    #         left_split['y'].append(node.y[i])
    #     else:
    #         right_split['x'].append(node.x[i])
    #         right_split['y'].append(node.y[i])

    node.left = Node(np.array(split[0]), split[1])
    grow(node.left, nmin, minleaf)
    node.right = Node(np.array(split[2]), split[3])
    grow(node.right, nmin, minleaf)


# def search_best_split(node, minleaf):
#     node.split_feature = random.choice([x for x in range(len(node.x[0]))])
#     node.split_threshold = node.x[:, node.split_feature].mean()   

#determines the best split in a node, where x is a data matrix and y is the vector of class labels
def search_best_split(node, minleaf):

    x = node.x
    y = np.array(node.y)
    
    best_quality = np.inf
    best_split = None
    best_feature = None
    best_threshold = None

    for i in range(len(x[0])):
        sorted = np.sort(np.unique(x[:,i]))
        splitpoints = (sorted[0:len(sorted)-1] + sorted[1:])/2
        for j in splitpoints:
            left = y[x[:,i] <= j]
            if len(left) < minleaf:
                continue
            right = y[x[:,i] > j]
            if len(right) < minleaf:
                continue
            quality = len(left)/len(y)*impurity(left) + len(right)/len(y)*impurity(right)
            if quality < best_quality:
                best_quality, best_split, best_feature, best_threshold = quality, split_data(x, y, i, j), i, j
    
    return best_feature, best_threshold, best_split

#computes impurity of a node/array
def impurity(a):
    p = a.sum() / len(a)
    return p * (1-p)

def split_data(x, y, feati, threshold):
    left_x = []
    right_x = []
    left_y = []
    right_y = []

    for i in range(len(x)):
        if x[i,feati] <= threshold:
            left_x.append(x[i])
            left_y.append(y[i])
        else:
            right_x.append(x[i])
            right_y.append(y[i])

    return (left_x, left_y, right_x, right_y)    

def tree_pred(x, tr):
    current_node = tr
    while not current_node.is_leaf():
        if x[current_node.split_feature] <= current_node.split_threshold:
            current_node = current_node.left
        else:
            current_node = current_node.right
    return current_node.predict_y()

df = pd.read_csv("data.csv")
x = df.drop(columns=['class'])
y = df['class']

tree = tree_grow(x.values, y.values, 2, 1)
print(tree_pred(np.array([24,1,1,84,1,0]), tree))
print(1)
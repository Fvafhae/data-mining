import numpy as np
#credit_data = np.genfromtxt('credit.txt', delimiter=',', skip_header=True)

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
        if x[i][feati] <= threshold:
            left_x.append(x[i])
            left_y.append(y[i])
        else:
            right_x.append(x[i])
            right_y.append(y[i])

    return(left_x, left_y, right_x, right_y)        


#determines the best split in a node, where x is a data matrix and y is the vector of class labels
def bestsplit(x,y):
    best_quality = np.inf
    best_split = None
    best_feature = None
    best_threshold = None

    for i in range(len(x[0])):
        sorted = np.sort(np.unique(x[i]))
        splitpoints = (sorted[0:-2] + sorted[1:])/2
        for j in splitpoints:
            left = y[x[i] <= j]
            right = y[x[i] > j]
            quality = len(left)/len(y)*impurity(left) + len(right)/len(y)*impurity(right)
            if quality < best_quality:
                best_quality, best_split, best_feature, best_threshold = quality, split_data(x, y, i, j), i, j
    
    return best_feature, best_threshold, best_split

# x is data matrix, y is vector of class labels(binary)(assumming no missing values)
# nmin is minimal number of observations a node must contain to split, minleaf min n of observations a leaf must contain
# nfeat is number of features that should be considered for a split

# To do
# use gini-index for determining quality split
# Should return a tree object
#    
def tree_grow(x, y, nmin, minleaf, nfeat):
    return

# x is data matrix, tr is tree object created with tree_grow

#todo
# should return y (vector)
def tree_pred(x, tr):
    return


# m is number of bootstrap samples to be drawn
# function returns list of m trees
def tree_grow_b(x, y, nmin, minleaf, nfeat, m):
    return

# tr is a list of trees
# final prediction uses majority vote of m predictions
# returns vector y
def tree_pred_b(x, tr):
    return



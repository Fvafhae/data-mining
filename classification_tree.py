import random
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.metrics import confusion_matrix, classification_report

class Node():
    """
    A class representing a single node in a decision tree.
    
    Attributes:
        x: the data matrix at this node
        y: the class label vector at this node
        left, right: pointers to the left and right children
        split_feature: the feature used for splitting at this node
        split_threshold: the threshold used for splitting on the selected feature
    """
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
        """
        Checks if the current node is a leaf or has children nodes.
        Returns True if the node is a leaf, otherwise False.
        """
        return self.left is None and self.right is None
    
    def predict_y(self):
        """
        Predicts the class label for the current node by returning the mean of the target values.
        This is used in leaf nodes to make predictions.
        """
        return Counter(self.y).most_common(1)[0][0]
        # return np.round(np.mean(self.y))

    def __str__(self):
        tree_string = f"x[{self.split_feature}] <= {self.split_threshold}"
        if not self.is_leaf():
            tree_string += f"\nx[{self.left.split_feature}] <= {self.left.split_threshold} | x[{self.right.split_feature}] <= {self.right.split_threshold}"
        return tree_string


def tree_grow(x, y, nmin, minleaf, nfeat):
    """
    Grows a decision tree recursively, starting from the root.

    Arguments:
        x: the data matrix
        y: the class label vector
        nmin: minimum number of samples required to split a node
        minleaf: minimum number of samples required in any leaf node
        nfeat: number of random features to consider at each split (for random forests)
    
    Returns:
        root: the root node of the decision tree
    """
    root = Node(x, y) #create root node
    grow(root, nmin, minleaf, nfeat) #recursively grow the decision tree

    return root


def tree_pred(x, tr):
    """
    Predicts the class label for a single instance using a decision tree.
    
    Arguments:
        x: a single sample
        tr: the decision tree
    
    Returns:
        The predicted class label
    """
    pred = []
    for i in range(len(x)):
        row = x[i]
        
        current_node = tr
        # Go through the tree until reaching a leaf node
        while not current_node.is_leaf():
            if row[current_node.split_feature] <= current_node.split_threshold:
                current_node = current_node.left
            else:
                current_node = current_node.right
                
        # Return the prediction from the leaf node
        pred.append(current_node.predict_y())
        
    return np.array(pred)


def tree_grow_b(x, y, nmin, minleaf, nfeat, m):
    """
    Function to grow m decision trees on m bootstrap samples for bagging and random forests.
    
    Arguments:
        x: the data matrix
        y: vector of class labels
        nmin = minimum number of data points to attempt a split
        minleaf = minimum number of data points in a leaf
        nfeat = number of features to consider for the split
        m = Number of bootstrap samples and trees to grow
    
    Returns:
        trees = A list containing the m grown trees
    """
    trees = []
    n = len(y)  # number of samples
    for i in range(m):
        # Generate a bootstrap sample
        indices = np.random.choice(range(n), size=n, replace=True)
        x_bootstrap = x[indices]
        y_bootstrap = y[indices]
        
        # Grow a single tree using tree_grow
        tree = tree_grow(x_bootstrap, y_bootstrap, nmin, minleaf, nfeat)
        trees.append(tree)
    return trees


def tree_pred_b(x, trees):
    """
    Function to predict class labels for each instance in x using multiple trees.
    The final prediction is determined by majority voting.
    
    Arguments:
    x: the data matrix
    trees: list of grown trees
    
    Returns:
    y_pred: a vector containing the predicted class labels for each row in x
    """
    # Get predictions for each instance from all trees
    predictions = np.array([tree_pred(x, tree) for tree in trees])
    
    # Reshape predictions to have rows correspond to each instance and columns to each tree
    predictions = predictions.reshape(len(trees), len(x)).T
    
    # Perform majority voting for each instance
    y_pred = [Counter(row).most_common(1)[0][0] for row in predictions]
    
    return np.array(y_pred)


def grow(node, nmin, minleaf, nfeat):
    """
    Function that recursively grows the tree by splitting nodes
    """
    # Stop growing if the node contains fewer than nmin samples or if all target values are the same
    if len(node.y) < nmin or len(np.unique(node.y)) == 1:
        return
    
    # Select a random subset of nfeat features (for random forest) to consider for splitting
    p = node.x.shape[1]  # Total number of features
    feature_indices = np.random.choice(range(p), nfeat, replace=False)
    
    # Find the best split using only the selected features
    node.split_feature, node.split_threshold, split = search_best_split(node, minleaf, feature_indices)

    # If no valid split is found, return
    if node.split_feature is None: 
        return

    # Split the data and create left and right child notes
    node.left = Node(np.array(split[0]), split[1])
    node.right = Node(np.array(split[2]), split[3])
    
    # Recursively grow the left and right subtrees
    grow(node.left, nmin, minleaf, nfeat)
    grow(node.right, nmin, minleaf, nfeat)
 

def search_best_split(node, minleaf, feature_indices):
    """
    Searches for the best feature and threshold to split the node.
    
    Arguments:
    node: the current node being split
    minleaf: minimum number of samples required in a leaf
    feature_indices: indices of the random features to consider for splitting
    
    Returns:
    best_feature: the index of the best feature to split on
    best_threshold: the threshold value for the best split
    best_split: a tuple containing the split data for left and right child nodes
    """
    x = node.x
    y = np.array(node.y)
    
    best_quality = np.inf # Initialize the best quality as infinite because we are minimizing impurity
    best_split = None
    best_feature = None
    best_threshold = None
    
    # Iterate through each randomly selected feature
    for i in feature_indices: 
        # Get all unique sorted values of the feature
        sorted_values = np.sort(np.unique(x[:, i]))
        # Create potential split points
        splitpoints = (sorted_values[0:len(sorted_values)-1] + sorted_values[1:]) / 2
        for j in splitpoints:
            # Split the target values based on the current feature and split point
            left = y[x[:, i] <= j]
            right = y[x[:, i] > j]
            # Skip invalid splits (if any side has fewer samples than minleaf)
            if len(left) < minleaf or len(right) < minleaf:
                continue
            # Calculate the quality of the split
            quality = len(left)/len(y)*impurity(left) + len(right)/len(y)*impurity(right)
            # Keep track of the best split
            if quality < best_quality:
                best_quality, best_split, best_feature, best_threshold = quality, split_data(x, y, i, j), i, j
    return best_feature, best_threshold, best_split


def impurity(a):
    """
    Computes the impurity of a node.
    This implementation assumes binary classification.
    
    Arguments:
    a: array of class labels at the node
    
    Returns:
    Impurity score based on the proportion of the majority class
    """
    p = np.mean(a) # Proportion of class 1 for binary classification
    return p * (1 - p) #Gini-index


def split_data(x, y, feati, threshold):
    """
    Splits the data into left and right subsets based on the given feature and threshold.
    
    Arguments:
    x: the data matrix
    y: the class label vector
    feati: the feature index to split on
    threshold: the threshold value for the split
    
    Returns:
    A tuple containing the left and right subsets of both x and y
    """
    left_x, right_x = [], []
    left_y, right_y = [], []

    for i in range(len(x)):
        if x[i, feati] <= threshold:
            left_x.append(x[i])
            left_y.append(y[i])
        else:
            right_x.append(x[i])
            right_y.append(y[i])

    return (left_x, left_y, right_x, right_y)    

if __name__ == "__main__":
    # TESTING
    df = pd.read_csv("./data/cleaned/eclipse-metrics-packages-2.0.csv", header = 0)
    x = df.drop(columns=['post']).values
    y = df['post'].values

    tree = tree_grow(x, y, 15, 5, 41)
    pred_y = tree_pred(x, tree)


    print(classification_report(y, pred_y))

    data = {
        'actual': y,  
        'predicted': pred_y
    }

    df = pd.DataFrame(data)

    conf_matrix = confusion_matrix(df['actual'], df['predicted'])

    conf_matrix_df = pd.DataFrame(conf_matrix, 
                                index=['Actual Negative (0)', 'Actual Positive (1)'],
                                columns=['Predicted Negative (0)', 'Predicted Positive (1)'])
    print(conf_matrix_df)

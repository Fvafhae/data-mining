#Bagging Implementation

#Instructions
#First run the code in main.py to get the tree_grow and tree_pred functions

def tree_grow_b(x, y, nmin, minleaf, m): # m is for number of bootstraap
    trees = [] # List to store the trees
    sampler = x.shape[0] # Number of samples or number of rows in 2-D array x
    for _ in range(m): # Loop through the number of bootstrap samples
        indices = np.random.choice(sampler, sampler, replace=True) # Bootstrap random sampling with replacement
        x_sample = x[indices] 
        y_sample = y[indices] 
        tree = tree_grow(x_sample, y_sample, nmin, minleaf) # Grow a single tree
        trees.append(tree) # Append the tree to the list of trees
    return trees # Return the list of trees

def tree_pred_b(x, trees):  # x is a 2-D array of samples and trees is a list of trees
    array_pred = np.zeros((x.shape[0], len(trees)))  # Create an array to store the predictions
    i = 0 # Index for the trees
    for tree in trees: # Loop through the trees
        for j in range(x.shape[0]): # Take the number of rows in x
            array_pred[j, i] = tree_pred(x[j], tree) # Make a prediction for each sample
        i += 1  # Increment the tree index
    store_pred = [] # Create a list to store the  predictions
    for row in array_pred: # Loop through the predictions
        unique_elements, counts = np.unique(row, return_counts=True) # Get the unique elements and their counts
        most_common = unique_elements[np.argmax(counts)] # Get the majority for classification exercise        
        store_pred.append(most_common) # Append the majority to the predictions list
    return np.array(store_pred) # Return an array

    
# Get Data
df = pd.read_csv("data.csv")
x = df.drop(columns=['class']).values
y = df['class'].values

# Grow 100 trees using bootstrap samples
trees = tree_grow_b(x, y, nmin=2, minleaf=1, m=100)

# Get Predict using Bagging Classifier
sample = np.array([24, 1, 1, 84, 1, 0])
prediction = tree_pred_b(np.array([sample]), trees)
print(f"Bagged Prediction: {prediction}")
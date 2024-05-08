import pandas as pd
import sys

# Capture the original print function for redirection purposes.
original_print = print

def custom_print(*args, **kwargs):
    """Custom print function to redirect prints to a file instead of the console.
    Inputs:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.
    """
    file_path = kwargs.pop('file', './output.txt')
    mode = kwargs.pop('mode', 'a')  # Default mode is append ('a')

    with open(file_path, mode) as f:
        original_print(*args, file=f, **kwargs)

# Reassign the default print function to the custom one.
print = custom_print
IMPURITY_THRESHOLD = 0.2

class Condition:
    """Represents a condition in a decision tree node, used to split the data."""
    def __init__(self, feature, value, is_numeric=True):
        """Initialize the condition with the feature, value, and type of feature.
        Inputs:
            feature: The feature based on which the split is made.
            value: The threshold value for the split.
            is_numeric: Boolean indicating if the feature is numeric or categorical.
        """
        self.feature = feature
        self.value = value
        self.is_numeric = is_numeric

    def match(self, example):
        """Check if the example meets the condition.
        Input:
            example: A single data point.
        Returns:
            Boolean result of the match condition.
        """
        if self.is_numeric:
            return example[self.feature] < self.value
        else:
            return example[self.feature] == self.value

class Node:
    """Node in the decision tree, which could be either a decision node or a leaf node."""
    def __init__(self, condition, left=None, right=None, is_leaf=False, label=None):
        """Initialize a node with given specifications.
        Inputs:
            condition: The condition based on which the node splits the data.
            left: Left child node.
            right: Right child node.
            is_leaf: Boolean indicating if the node is a leaf.
            label: The class label if the node is a leaf.
        """
        self.condition = condition
        self.left = left
        self.right = right
        self.is_leaf = is_leaf
        self.label = label  # Store the class label at the leaf node

    def is_leaf_node(self):
        """Determine if the node is a leaf node.
        Returns:
            Boolean indicating if the node is a leaf.
        """
        return self.is_leaf

def calc_gini_impurity(left, right):
    """Calculate the weighted Gini impurity for a split.
    Inputs:
        left: DataFrame of the left split.
        right: DataFrame of the right split.
    Returns:
        Weighted Gini impurity of the split.
    """
    left_impurity = calculate_impurity(left)
    right_impurity = calculate_impurity(right)
    total_gini = (len(left) / (len(left) + len(right))) * left_impurity + \
                 (len(right) / (len(left) + len(right))) * right_impurity
    return total_gini

def calculate_impurity(data):
    """Calculate Gini impurity of a dataset.
    Input:
        data: DataFrame whose impurity is to be calculated.
    Returns:
        Gini impurity score.
    """
    counts = data.iloc[:, -1].value_counts().to_dict()
    impurity = 1 - sum((count / len(data)) ** 2 for count in counts.values())
    return impurity

def get_best_split(data, num_features=None, decision_forest_features=None):
    """Find the best split for the data considering the specified features.
    Inputs:
        data: DataFrame to be split.
        num_features: Number of features to consider if not using decision_forest_features.
        decision_forest_features: Specific features to consider for splits.
    Returns:
        The best condition that provides the minimum Gini impurity.
    """
    best_gini = float('inf')
    best_condition = None
    features = data.columns[:-1].tolist()
    if num_features is not None:
        features = pd.Series(features).sample(n=num_features).tolist()
        print("Features considered for the split: ", features)
    if decision_forest_features is not None:
        features = decision_forest_features
    print("Features considered for the split: ", features)
    print("Number of features considered: ", num_features) 
    for feature in features:
        values = data[feature].unique()
        if pd.api.types.is_numeric_dtype(data[feature]):
            sorted_data = data.sort_values(by=feature)
            values = (sorted_data[feature].shift(2) + sorted_data[feature].shift(1) + sorted_data[feature]).dropna() / 3
        for value in values:
            conditions = [Condition(feature, value, is_numeric=pd.api.types.is_numeric_dtype(data[feature]))]
            for condition in conditions:
                left, right = data_split(data, condition)
                if not left.empty and not right.empty:
                    gini = calc_gini_impurity(left, right)
                    if gini < best_gini:
                        best_gini = gini
                        best_condition = condition
    return best_condition

def data_split(data, condition):
    """Split data according to the provided condition.
    Input:
        data: DataFrame to be split.
        condition: Condition used to split the data.
    Returns:
        Tuple of (left, right) DataFrames after the split.
    """
    if condition.is_numeric:
        left = data[data[condition.feature] < condition.value]
        right = data[data[condition.feature] >= condition.value]
    else:
        left = data[data[condition.feature] == condition.value]
        right = data[data[condition.feature] != condition.value]
    return left, right

def build_tree(data, num_features=None, impurity_threshold=IMPURITY_THRESHOLD, decision_tree_features=None):
    """Build a decision tree using recursive partitioning.
    Inputs:
        data: DataFrame from which to build the tree.
        num_features: Optional number of features to consider for each split.
        impurity_threshold: Threshold below which a node becomes a leaf.
        decision_tree_features: Specific features to consider, if not using num_features.
    Returns:
        The root node of the decision tree.
    """
    if data.empty or calculate_impurity(data) < impurity_threshold:
        most_common_class = data.iloc[:, -1].mode()[0]
        return Node(condition=None, is_leaf=True, label=most_common_class)
    best_condition = get_best_split(data, num_features, decision_tree_features)
    if best_condition is None:
        most_common_class = data.iloc[:, -1].mode()[0]
        return Node(condition=None, is_leaf=True, label=most_common_class)
    node = Node(best_condition)
    left_data, right_data = data_split(data, best_condition)
    node.left = build_tree(left_data, num_features)
    node.right = build_tree(right_data, num_features)
    return node

def predict(node, instance):
    """Recursively predict the class of the instance using the decision tree.
    Inputs:
        node: The current node of the tree.
        instance: The data point to classify.
    Returns:
        The class label predicted by the tree.
    """
    if node.is_leaf_node():
        return node.label
    if node.condition.is_numeric:
        if instance[node.condition.feature] < node.condition.value:
            return predict(node.left, instance)
        else:
            return predict(node.right, instance)
    else:
        if instance[node.condition.feature] == node.condition.value:
            return predict(node.left, instance)
        else:
            return predict(node.right, instance)

def calculate_accuracy(node, data):
    """Calculate the accuracy of the decision tree on the provided data.
    Inputs:
        node: The root node of the decision tree.
        data: DataFrame containing the data to test the tree.
    Returns:
        The accuracy percentage of the tree on the test data.
    """
    correct_predictions = 0
    for _, row in data.iterrows():
        if predict(node, row) == row.iloc[-1]:
            correct_predictions += 1
    accuracy = correct_predictions / len(data)
    return accuracy

def print_tree(node, depth=0):
    """Recursively print the structure of the decision tree.
    Inputs:
        node: The current node to print.
        depth: The current depth in the tree (used for indentation).
    """
    if node is None:
        return
    indent = "    " * depth
    if node.is_leaf_node():
        print(f"{indent}[Leaf node: Class {node.label}]")
    else:
        if node.condition.is_numeric:
            print(f"{indent}[{node.condition.feature} < {node.condition.value}]")
        else:
            print(f"{indent}[{node.condition.feature} == {node.condition.value}]")
    print_tree(node.left, depth + 1)
    print_tree(node.right, depth + 1)

def count_used_features_in_tree(node):
    """Count how many times each feature is used in the decision tree.
    Input:
        node: The root or any node of the tree from where counting starts.
    Returns:
        A dictionary with feature names as keys and counts as values.
    """
    features_dict = {}
    if node.is_leaf_node():
        return features_dict
    if node.condition.feature in features_dict:
        features_dict[node.condition.feature] += 1
    else:
        features_dict[node.condition.feature] = 1
    left_features = count_used_features_in_tree(node.left)
    right_features = count_used_features_in_tree(node.right)
    for feature, count in left_features.items():
        if feature in features_dict:
            features_dict[feature] += count
        else:
            features_dict[feature] = count
    for feature, count in right_features.items():
        if feature in features_dict:
            features_dict[feature] += count
        else:
            features_dict[feature] = count
    return features_dict

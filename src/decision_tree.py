
import pandas as pd
import sys

# Capture the original print function
original_print = print

def custom_print(*args, **kwargs):
    file_path = kwargs.pop('file', './output.txt')
    mode = kwargs.pop('mode', 'a')  # Default mode is append ('a')

    # Open the file with the provided path and mode
    with open(file_path, mode) as f:
        # Use the captured original print function with file argument
        original_print(*args, file=f, **kwargs)

# Reassign print to custom_print
print = custom_print
IMPURITY_THRESHOLD=0.2

class Condition:
    def __init__(self, feature, value, is_numeric=True):
        self.feature = feature
        self.value = value
        self.is_numeric = is_numeric

    def match(self, example):
        if self.is_numeric:
            return example[self.feature] < self.value
        else:
            return example[self.feature] == self.value

class Node:
    def __init__(self, condition, left=None, right=None, is_leaf=False, label=None):
        self.condition = condition
        self.left = left
        self.right = right
        self.is_leaf = is_leaf
        self.label = label  # Store the class label at the leaf node

    def is_leaf_node(self):
        return self.is_leaf



def calc_gini_impurity(left, right):
    left_impurity = calculate_impurity(left)
    right_impurity = calculate_impurity(right)
    total_gini = (len(left) / (len(left) + len(right))) * left_impurity + (len(right) / (len(left) + len(right))) * right_impurity
    return total_gini

def calculate_impurity(data):
    counts = data.iloc[:, -1].value_counts().to_dict()
    impurity = 1 - sum((count / len(data)) ** 2 for count in counts.values())
    return impurity

def get_best_split(data,num_features=None,decision_forest_features=None):
    best_gini = float('inf')
    best_condition = None
    if num_features is not None:
        features = data.columns[:-1].tolist()
        features = pd.Series(features).sample(n=num_features).tolist()
    if decision_forest_features is not None:
        features=decision_forest_features
    features = data.columns[:-1]  # Assuming the last column is the target
    for feature in features:
        values = data[feature].unique()
        sorted_data = data.sort_values(by=feature)
        if pd.api.types.is_numeric_dtype(data[feature]):
            values = (sorted_data[feature].shift(2)+sorted_data[feature].shift(1) + sorted_data[feature]).dropna() / 3
        for value in values:
            if pd.api.types.is_numeric_dtype(data[feature]):
                conditions = [Condition(feature, value)]
            else:
                conditions = [Condition(feature, value, is_numeric=False)]
            for condition in conditions:
                left, right = data_split(data, condition)
                if not left.empty and not right.empty:
                    gini = calc_gini_impurity(left, right)
                    if gini < best_gini:
                        best_gini = gini
                        best_condition = condition
    return best_condition


def data_split(data, condition):
    if condition.is_numeric:
        left = data[data[condition.feature] < condition.value]
        right = data[data[condition.feature] >= condition.value]
    else:
        left = data[data[condition.feature] == condition.value]
        right = data[data[condition.feature] != condition.value]
    return left, right



def build_tree(data,num_features=None,impurity_threshold=IMPURITY_THRESHOLD,decision_tree_features=None):
    if data.empty or calculate_impurity(data) < impurity_threshold:
        most_common_class = data.iloc[:, -1].mode()[0]  # Get the most common class label
        return Node(condition=None, is_leaf=True, label=most_common_class)  # This node is a leaf with a class label
    best_condition = get_best_split(data,num_features,decision_tree_features)
    if best_condition is None:
        most_common_class = data.iloc[:, -1].mode()[0]
        return Node(condition=None, is_leaf=True, label=most_common_class)  # No split possible, make it a leaf
    node = Node(best_condition)
    left_data, right_data = data_split(data, best_condition)
    #print(f"Splitting on {best_condition.feature} at {best_condition.value}")
    node.left = build_tree(left_data,num_features)
    node.right = build_tree(right_data,num_features)
    return node

def predict(node, instance):
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
    correct_predictions = 0
    for _, row in data.iterrows():
        if predict(node, row) == row.iloc[-1]:  # Assuming the label is the last column
            correct_predictions += 1
    accuracy = correct_predictions / len(data)
    return accuracy


def print_tree(node, depth=0):
    if node is None:
        return
    indent = "    " * depth  # Increase indentation with depth
    if node.is_leaf_node():
        print(f"{indent}[Leaf node: Class {node.label}]")  # Display the class of the leaf node
    else:
        if node.condition.is_numeric:
            print(f"{indent}[{node.condition.feature} < {node.condition.value}]")
        else:
            print(f"{indent}[{node.condition.feature} == {node.condition.value}]")
    # Recursively print the left and right children
    print_tree(node.left, depth + 1)
    print_tree(node.right, depth + 1)


def count_used_features_in_tree(node):
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
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np


IMPURITY_THRESHOLD = 0.05
FRACTION_OF_DATA_TO__CALC_SPLIT=0.1

def preprocess_dataframe(df):
    df.dropna(inplace=True)
    return df

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

"""
    def get_best_split(data):
    best_gini = float('inf')
    best_condition = None
    features = data.columns[:-1]  # Assuming the last column is the target
    for feature in features:
        values = data[feature].unique()
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
"""

def get_best_split(data):
    best_gini = float('inf')
    best_condition = None
    sample_data = data.sample(frac=FRACTION_OF_DATA_TO__CALC_SPLIT)  # Use only 10% of data to find splits
    features = data.columns[:-1]
    
    for feature in features:
        if pd.api.types.is_numeric_dtype(data[feature]):
            thresholds = np.percentile(sample_data[feature].dropna(), [10, 20, 30, 40, 50, 60, 70, 80, 90])
            conditions = [Condition(feature, threshold) for threshold in thresholds]
        else:
            top_categories = sample_data[feature].value_counts().nlargest(10).index
            conditions = [Condition(feature, category, is_numeric=False) for category in top_categories]
        
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


def build_tree(data):
    if data.empty or calculate_impurity(data) == IMPURITY_THRESHOLD:
        most_common_class = data.iloc[:, -1].mode()[0]  # Get the most common class label
        return Node(condition=None, is_leaf=True, label=most_common_class)  # This node is a leaf with a class label
    best_condition = get_best_split(data)
    if best_condition is None:
        most_common_class = data.iloc[:, -1].mode()[0]
        return Node(condition=None, is_leaf=True, label=most_common_class)  # No split possible, make it a leaf
    node = Node(best_condition)
    left_data, right_data = data_split(data, best_condition)
    print(f"Splitting on {best_condition.feature} at {best_condition.value}")
    node.left = build_tree(left_data)
    node.right = build_tree(right_data)
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


if __name__ == "__main__":
    
    print("Decision Tree Classifier")
    path = "C:/Users/User/git/CART-method_with_randomf-forest_and_decision-forest/Data/adult/adult.csv"
    print(f"Reading data from {path}")
    csv = pd.read_csv(path)
    print("Preprocessing data...")
    csv = preprocess_dataframe(csv)
    print("Splitting data into training and testing sets...")
    train_data, test_data = train_test_split(csv, test_size=0.2, random_state=42)
    print("Building tree...")
    root = build_tree(train_data)
    print("Printing tree...")
    print_tree(root)
    accuracy = calculate_accuracy(root, test_data)
    print(f"Accuracy on test data: {accuracy:.2f}")

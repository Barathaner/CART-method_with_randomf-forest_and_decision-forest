import pandas as pd
from sklearn.model_selection import train_test_split
import graphviz
from graphviz import Digraph
import random
IMPURITY_THRESHOLD=0.2

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

def get_best_split(data):
    best_gini = float('inf')
    best_condition = None
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

def visualize_tree(node, dot=None, parent_id=None, edge_label="", node_id=0):
    if dot is None:
        dot = Digraph()
        dot.attr('node', shape='box')  # Set the shape of nodes to boxes
        initial_label = f'"{node_id}: {node.label if node.is_leaf_node() else node.condition}"'
        dot.node(str(node_id), initial_label)
        parent_id = node_id

    if node.is_leaf_node():
        leaf_label = f'"Leaf {node_id}: {node.label}"'
        dot.node(str(node_id), leaf_label)
        if parent_id is not None:
            dot.edge(str(parent_id), str(node_id), label=edge_label)
    else:
        # Create a label for the condition
        if node.condition.is_numeric:
            condition_label = f"{node.condition.feature} < {node.condition.value}"
        else:
            # For categorical features, the split is binary but visualized as one node
            condition_label = f"{node.condition.feature} == {node.condition.value}"

        node_label = f'"{node_id}: {condition_label}"'
        dot.node(str(node_id), node_label)
        if parent_id is not None:
            dot.edge(str(parent_id), str(node_id), label=edge_label)

        # Recursively visualize the subtree
        # We define "Yes" for the condition being true, and "No" for false
        if node.left:
            left_id = random.randint(0, 1000000)
            visualize_tree(node.left, dot, node_id, "Yes", left_id)
        if node.right:
            right_id = random.randint(0, 1000000)
            visualize_tree(node.right, dot, node_id, "No", right_id)

    return dot


def build_tree(data):
    if data.empty or calculate_impurity(data) < IMPURITY_THRESHOLD:
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
    print("Visualizing the tree...")
    # Assuming the tree has been built and stored in 'root'
    dot = visualize_tree(root)
    # Save the dot source in a file and render it as a PNG image
    dot.render('tree_diagram', format='png', cleanup=True)  # 'cleanup=True' will remove the dot source file after rendering

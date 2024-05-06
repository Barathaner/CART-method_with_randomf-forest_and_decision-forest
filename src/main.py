import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess_dataframe(df):
    df.dropna(inplace=True)
    return df

class Condition:
    def __init__(self, feature, threshold):
        self.feature = feature
        self.threshold = threshold

class Node:
    def __init__(self, condition, left=None, right=None, is_leaf=False, label=None):
        self.condition = condition
        self.left = left
        self.right = right
        self.is_leaf = is_leaf
        self.label = label  # Store the class label at the leaf node

    def is_leaf_node(self):
        return self.is_leaf



def calc_gini_impurity(condition, data):
    left = data[data[condition.feature] < condition.threshold]
    right = data[data[condition.feature] >= condition.threshold]
    left_impurity = calculate_impurity(left)
    right_impurity = calculate_impurity(right)
    total_gini = (len(left) / len(data)) * left_impurity + (len(right) / len(data)) * right_impurity
    return total_gini

def calculate_impurity(data):
    counts = data.iloc[:, -1].value_counts().to_dict()
    impurity = 1 - sum((count / len(data)) ** 2 for count in counts.values())
    return impurity

def get_best_split(data):
    best_gini = float('inf')
    best_condition = None
    for feature in data.columns[:-1]:
        sorted_data = data.sort_values(by=feature)
        thresholds = (sorted_data[feature].shift(1) + sorted_data[feature]).dropna() * 0.5
        for threshold in thresholds:
            condition = Condition(feature, threshold)
            gini = calc_gini_impurity(condition, data)
            if gini < best_gini:
                best_gini = gini
                best_condition = condition
    return best_condition


def print_tree(node, depth=0):
    if node is None:
        return
    indent = "    " * depth  # Increase indentation with depth
    if node.is_leaf_node():
        print(f"{indent}[Leaf node: Class {node.label}]")  # Display the class of the leaf node
    else:
        print(f"{indent}[{node.condition.feature} < {node.condition.threshold}]")
    # Recursively print the left and right children
    print_tree(node.left, depth + 1)
    print_tree(node.right, depth + 1)


def build_tree(data):
    if data.empty or calculate_impurity(data) == 0:
        most_common_class = data.iloc[:, -1].mode()[0]  # Get the most common class label
        return Node(condition=None, is_leaf=True, label=most_common_class)  # This node is a leaf with a class label
    best_condition = get_best_split(data)
    if best_condition is None:
        most_common_class = data.iloc[:, -1].mode()[0]
        return Node(condition=None, is_leaf=True, label=most_common_class)  # No split possible, make it a leaf
    node = Node(best_condition)
    left_data = data[data[best_condition.feature] < best_condition.threshold]
    right_data = data[data[best_condition.feature] >= best_condition.threshold]
    node.left = build_tree(left_data)
    node.right = build_tree(right_data)
    return node

def predict(node, instance):
    if node.is_leaf_node():
        return node.label
    if instance[node.condition.feature] < node.condition.threshold:
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
    csv = pd.read_csv("C:/Users/User/git/CART-method_with_randomf-forest_and_decision-forest/Data/iris/iris.csv")
    csv = preprocess_dataframe(csv)
    train_data, test_data = train_test_split(csv, test_size=0.2, random_state=42)
    root = build_tree(train_data)

    print_tree(root)
    
    accuracy = calculate_accuracy(root, test_data)
    print(f"Accuracy on test data: {accuracy:.2f}")


from decision_tree import build_tree,predict
import pandas as pd
from sklearn.model_selection import train_test_split
NUM_OF_TREES=10
NUMBER_OF_FEATURES=5


def bootstrap(data):
    return data.sample(n=len(data), replace=True)

def predict_data_with_RF(data, forest):
    predictions = []
    # Ensure data is treated as a DataFrame, not a row-by-row operation
    for index, instance in data.iterrows():
        votes = [predict(tree, instance[:-1]) for tree in forest]  # Use instance[:-1] to exclude the label
        # Get the most common class label
        prediction = max(set(votes), key=votes.count)
        predictions.append(prediction)
    return predictions


def accuracy_with_RF(data, forest):
    predictions = predict_data_with_RF(data, forest)
    correct_predictions = sum(1 for i, pred in enumerate(predictions) if pred == data.iloc[i, -1])
    accuracy = correct_predictions / len(data)
    return accuracy


def build_random_forest(data, num_features,number_of_trees=NUM_OF_TREES):
    forest = []
    for i in range(number_of_trees):
        bootstrapped_data = bootstrap(data)
        tree = build_tree(bootstrapped_data, num_features)
        forest.append(tree)
    return forest


def cross_val(data, num_folds, num_trees, num_features):
    fold_size = len(data) // num_folds
    accuracies = []
    for i in range(num_folds):
        print(f"Running fold {i + 1}...")
        start = i * fold_size
        end = (i + 1) * fold_size
        fold_data = data.iloc[start:end]
        test_data,train_data = train_test_split(fold_data, test_size=0.2, random_state=42)
        forest = build_random_forest(train_data, num_features,num_trees)
        accuracy = accuracy_with_RF(test_data, forest)
        accuracies.append(accuracy)
    avg_accuracy = sum(accuracies) / len(accuracies)
    return avg_accuracy
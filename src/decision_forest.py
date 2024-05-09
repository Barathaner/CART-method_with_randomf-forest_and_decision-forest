from decision_tree import build_tree, predict, count_used_features_in_tree, print_tree
from sklearn.model_selection import train_test_split
import pandas as pd
import random
import itertools
from concurrent.futures import ProcessPoolExecutor
# Capture the original print function
original_print = print

def custom_print(*args, **kwargs):
    """Redirects print output to a file instead of the console.
    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments. Contains 'file' and 'mode'.
    """
    file_path = kwargs.pop('file', './output.txt')
    mode = kwargs.pop('mode', 'a')  # Default mode is append ('a')
    with open(file_path, mode) as f:
        original_print(*args, file=f, **kwargs)

# Reassign print to custom_print
print = custom_print

def build_decision_forest(data, num_features, number_of_trees):
    """Builds a decision forest using random feature subsets.
    Args:
        data (DataFrame): The dataset used to build the forest.
        num_features (int): The number of features to consider.
        number_of_trees (int): The number of trees to build.
    Returns:
        tuple: A tuple containing the forest (list of trees) and a dictionary of feature counts.
    """
    forest = []
    forest_features_count_dict = {}
    tasks = [(data, None, pd.Series(data.columns[:-1]).sample(n=random.randint(1, num_features)).tolist()) for _ in range(number_of_trees)]
    with ProcessPoolExecutor() as executor:
        results = executor.map(build_tree_wrapper, tasks)
    for tree, feature_count_dict in results:
        forest.append(tree)
        for feature, count in feature_count_dict.items():
            if feature in forest_features_count_dict:
                forest_features_count_dict[feature] += count
            else:
                forest_features_count_dict[feature] = count
    return forest, forest_features_count_dict

def build_tree_wrapper(args):
    """Wrapper function for building a tree in a separate process.
    Args:
        args (tuple): Tuple containing the data, number of features, and the list of features to use.
    Returns:
        tuple: A tuple containing the built tree and its feature count dictionary.
    """
    data, num_features, features = args
    #print(f"Building tree with features: {features}")
    tree = build_tree(data=data, num_features=None, decision_tree_features=features)
    feature_count_dict = count_used_features_in_tree(tree)
    return tree, feature_count_dict

def predict_data_with_DF(data, forest):
    """Predicts the labels for a dataset using a decision forest.
    Args:
        data (DataFrame): The data to predict.
        forest (list): The decision forest.
    Returns:
        list: Predicted labels for the data.
    """
    predictions = []
    for index, instance in data.iterrows():
        votes = [predict(tree, instance[:-1]) for tree in forest]
        prediction = max(set(votes), key=votes.count)
        predictions.append(prediction)
    return predictions

def print_forest(forest):
    """Prints all the trees in the forest.
    Args:
        forest (list): The decision forest.
    """
    for i, tree in enumerate(forest):
        print(f"Tree {i}:")
        print_tree(tree)

def accuracy_with_DF(data, forest):
    """Calculates the accuracy of the decision forest on a given dataset.
    Args:
        data (DataFrame): The dataset to evaluate.
        forest (list): The decision forest.
    Returns:
        float: The accuracy of the forest.
    """
    predictions = predict_data_with_DF(data, forest)
    correct_predictions = sum(1 for i, pred in enumerate(predictions) if pred == data.iloc[i, -1])
    accuracy = correct_predictions / len(data)
    return accuracy

def hyperparameter_tuning_DF(data, num_trees, num_features):
    """
    Performs hyperparameter tuning for a decision forest by testing all combinations of tree counts and feature counts.
    
    Args:
        data (DataFrame): The dataset to use for building the forest.
        num_trees (list of int): List of the number of trees to use in the forest.
        num_features (list of int): List of the number of features to consider when splitting at each node.
    
    Returns:
        tuple: The best performing forest and its feature count dictionary, along with its accuracy.
    """
    accuracies = {}
    forests = {}
    feature_counts = {}
    test_data, train_data = train_test_split(data, test_size=0.2, random_state=42)

    # Generate all combinations of tree counts and feature counts
    for trees, features in itertools.product(num_trees, num_features):
        print(f"Testing configuration with {trees} trees and {features} features.")
        forest, features_count = build_decision_forest(train_data, features, trees)
        accuracy = accuracy_with_DF(test_data, forest)
        key = f"{trees} trees, {features} features"

        accuracies[key] = accuracy
        forests[key] = forest
        feature_counts[key] = features_count

        print(f"Configuration: {key}, Accuracy: {accuracy:.2f}, Feature counts: {features_count}")

    # Determine the best performing configuration
    best_key = max(accuracies, key=accuracies.get)
    print("Best configuration:", best_key)
    print("Highest accuracy:", accuracies[best_key])
    print("Feature counts for best configuration:", feature_counts[best_key])

    return forests[best_key], feature_counts[best_key], accuracies[best_key]

def generalization_error_with_cross_val_DF(data, num_folds, random_forest):
    """Calculates the generalization error of a decision forest using cross-validation.
    Args:
        data (DataFrame): The dataset to use.
        num_folds (int): The number of folds for cross-validation.
        random_forest (list): The decision forest.
    Returns:
        float: The average accuracy across all folds.
    """
    fold_size = len(data) // num_folds
    accuracies = []
    for i in range(num_folds):
        print(f"Running fold {i + 1}...")
        start = i * fold_size
        end = (i + 1) * fold_size
        fold_data = data.iloc[start:end]
        accuracy = accuracy_with_DF(fold_data, random_forest)
        accuracies.append(accuracy)
    average_accuracy = sum(accuracies) / len(accuracies)
    return average_accuracy

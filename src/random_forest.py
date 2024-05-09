from decision_tree import build_tree, predict, count_used_features_in_tree
import pandas as pd
from sklearn.model_selection import train_test_split
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
import itertools
# Capture the original print function
original_print = print

def custom_print(*args, **kwargs):
    """Customizes the print function to redirect outputs to a file.
    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.
    """
    file_path = kwargs.pop('file', './output.txt')
    mode = kwargs.pop('mode', 'a')  # Default mode is append.
    with open(file_path, mode) as f:
        original_print(*args, file=f, **kwargs)

# Reassign print to the customized print function.
print = custom_print

NUM_OF_TREES = 10
NUMBER_OF_FEATURES = 5

def bootstrap(data):
    """Generates a bootstrap sample from the dataset.
    Args:
        data (DataFrame): The dataset from which to sample.
    Returns:
        DataFrame: A bootstrap sample of the dataset.
    """
    return data.sample(n=len(data), replace=True)

def predict_data_with_RF(data, forest):
    """Predicts labels for each instance in the dataset using a random forest.
    Args:
        data (DataFrame): The dataset to predict.
        forest (list): The random forest model.
    Returns:
        list: The predicted labels.
    """
    predictions = []
    for index, instance in data.iterrows():
        votes = [predict(tree, instance[:-1]) for tree in forest]  # Exclude the label
        prediction = max(set(votes), key=votes.count)
        predictions.append(prediction)
    return predictions

def accuracy_with_RF(data, forest):
    """Calculates the accuracy of the random forest on the provided dataset.
    Args:
        data (DataFrame): The dataset to evaluate.
        forest (list): The random forest model.
    Returns:
        float: The accuracy of the forest.
    """
    predictions = predict_data_with_RF(data, forest)
    correct_predictions = sum(1 for i, pred in enumerate(predictions) if pred == data.iloc[i, -1])
    return correct_predictions / len(data)

def build_tree_and_count_features(task):
    """Wrapper function to build a tree and count used features.
    Args:
        task (tuple): Contains the dataset, number of features.
    Returns:
        tuple: A decision tree and a feature count dictionary.
    """
    data, num_features = task
    bootstrapped_data = bootstrap(data)
    tree = build_tree(bootstrapped_data, num_features=num_features)
    feature_count_dict = count_used_features_in_tree(tree)
    return tree, feature_count_dict

def build_random_forest(data, num_features, number_of_trees=10):
    """Builds a random forest using parallel computation.
    Args:
        data (DataFrame): The dataset to build the forest from.
        num_features (int): The number of features to consider for each tree.
        number_of_trees (int): The number of trees in the forest.
    Returns:
        tuple: A list of decision trees and a feature count dictionary.
    """
    forest = []
    forest_features_count_dict = {}
    tasks = [(data, num_features) for _ in range(number_of_trees)]
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(build_tree_and_count_features, tasks))
    for tree, feature_count_dict in results:
        forest.append(tree)
        for feature, count in feature_count_dict.items():
            if feature in forest_features_count_dict:
                forest_features_count_dict[feature] += count
            else:
                forest_features_count_dict[feature] = count
    return forest, forest_features_count_dict

def hyperparameter_tuning_RF(data, num_trees, num_features):
    """
    Performs hyperparameter tuning for a random forest by testing all combinations of tree counts and feature counts.
    
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
        if features == -1:
            print("Using the function int(math.sqrt(len(features)) for each node to have a different number of features.")
        forest, features_count = build_random_forest(train_data, features, trees)
        accuracy = accuracy_with_RF(test_data, forest)
        
        key = f"{trees} trees, {features} features"

        accuracies[key] = accuracy
        forests[key] = forest
        features_count = dict(sorted(features_count.items(), key=lambda item: item[1], reverse=True))
        feature_counts[key] = features_count

        print(f"Configuration: {key}, Accuracy: {accuracy:.2f}, Feature counts: {features_count}")

    # Determine the best performing configuration
    best_key = max(accuracies, key=accuracies.get)
    print("Best configuration:", best_key)
    print("Highest accuracy:", accuracies[best_key])
    print("Feature counts for best configuration:", feature_counts[best_key])

    return forests[best_key], feature_counts[best_key], accuracies[best_key]

def generalization_error_with_cross_val_RF(data, num_folds, random_forest):
    """Calculates the generalization error of the random forest using cross-validation.
    Args:
        data (DataFrame): The dataset to use.
        num_folds (int): The number of folds for cross-validation.
        random_forest (list): The random forest model.
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
        accuracy = accuracy_with_RF(fold_data, random_forest)
        accuracies.append(accuracy)
    return sum(accuracies) / len(accuracies)

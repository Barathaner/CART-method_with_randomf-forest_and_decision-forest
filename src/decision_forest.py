from decision_tree import build_tree,predict,count_used_features_in_tree,print_tree
from sklearn.model_selection import train_test_split
import pandas as pd
import random
from concurrent.futures import ProcessPoolExecutor
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

def build_decision_forest(data, num_features, number_of_trees):
    forest = []
    forest_features_count_dict = {}
    
    # Erzeuge eine Liste von Tupeln für jeden Baum, wobei jedes Tupel die Daten und eine zufällige Feature-Auswahl enthält
    tasks = [(data, num_features, pd.Series(data.columns[:-1]).sample(n=random.randint(1, num_features)).tolist()) for _ in range(number_of_trees)]
    
    # Verwenden eines Prozesspools, um die Bäume parallel zu erstellen
    with ProcessPoolExecutor() as executor:
        results = executor.map(build_tree_wrapper, tasks)

    # Sammle die Ergebnisse und aktualisiere das Forest-Feature-Count-Dictionary
    for tree, feature_count_dict in results:
        forest.append(tree)
        for feature, count in feature_count_dict.items():
            if feature in forest_features_count_dict:
                forest_features_count_dict[feature] += count
            else:
                forest_features_count_dict[feature] = count

    return forest, forest_features_count_dict

def build_tree_wrapper(args):
    # Diese Funktion wird benötigt, um die Daten und Parameter korrekt an build_tree zu übergeben
    data, num_features, features = args
    tree = build_tree(data=data, num_features=num_features, decision_tree_features=features)
    feature_count_dict = count_used_features_in_tree(tree)
    return tree, feature_count_dict

def predict_data_with_DF(data, forest):
    predictions = []
    # Ensure data is treated as a DataFrame, not a row-by-row operation
    for index, instance in data.iterrows():
        votes = [predict(tree, instance[:-1]) for tree in forest]  # Use instance[:-1] to exclude the label
        # Get the most common class label
        prediction = max(set(votes), key=votes.count)
        predictions.append(prediction)
    return predictions


def print_forest(forest):
    for i, tree in enumerate(forest):
        print(f"Tree {i}:")
        print_tree(tree)


def accuracy_with_DF(data, forest):
    predictions = predict_data_with_DF(data, forest)
    correct_predictions = sum(1 for i, pred in enumerate(predictions) if pred == data.iloc[i, -1])
    accuracy = correct_predictions / len(data)
    return accuracy

def hyperparameter_tuning_DF(data, num_folds, num_trees, num_features):
    accuracies = {}
    forests = {}  # Dictionary to store each forest
    forestfeaturecounts={}
    test_data, train_data = train_test_split(data, test_size=0.2, random_state=42)
    for i in range(num_folds):
        print(f"Running test {i + 1}...")
        print("num_features:",num_features[i],"num_trees:",num_trees[i])
        # Split the data into training and testing sets
        # Build the decision forest with the specified number of features and trees
        forest,forestfeatures = build_decision_forest(train_data, num_features[i], num_trees[i])
        # Evaluate the accuracy of the forest on the test data
        accuracy = accuracy_with_DF(test_data, forest)
        # Create a unique key for each configuration
        key = f"test:{i} num_features:{num_features[i]} num_trees:{num_trees[i]}"
        # Store the accuracy and the corresponding forest
        accuracies[key] = accuracy
        forests[key] = forest
        forestfeaturecounts[key]=forestfeatures
    # Find the key with the highest accuracy
    best_key = max(accuracies, key=accuracies.get)
    print("Decision Forest Hyperparameter Tuning accuracies:", accuracies)
    print("Decision Forest Hyperparameter Tuning FeatureCounts:",forestfeaturecounts)
    print("Best configuration:", best_key, "with accuracy:", accuracies[best_key], "and feature count:",forestfeaturecounts[best_key])

    # Return the forest with the highest accuracy
    return forests[best_key],forestfeaturecounts[best_key]



def generalization_error_with_cross_val_DF(data, num_folds, random_forest):
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
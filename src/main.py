from sklearn.model_selection import train_test_split
from preprocessing import preprocess_dataframe
from decision_tree import build_tree, print_tree, calculate_accuracy,count_used_features_in_tree
from random_forest import build_random_forest,accuracy_with_RF,generalization_error_with_cross_val_RF,hyperparameter_tuning_RF
from visualization import visualize_tree
from decision_forest import build_decision_forest,accuracy_with_DF,generalization_error_with_cross_val_DF,hyperparameter_tuning_DF,print_forest
import pandas as pd
import math
import sys
import random
import time

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

def change_delimiter(input_filename, output_filename, old_delimiter=';', new_delimiter=','):
    """
    Changes the delimiter of a given file from an old delimiter to a new delimiter.

    :param input_filename: str, the name of the input file to process.
    :param output_filename: str, the name of the output file to save the processed data.
    :param old_delimiter: str, the delimiter to replace.
    :param new_delimiter: str, the delimiter to use as replacement.
    """
    try:
        with open(input_filename, 'r', encoding='utf-8') as file:
            data = file.read()
        
        # Replace the old delimiter with the new one
        data = data.replace(old_delimiter, new_delimiter)
        
        with open(output_filename, 'w', encoding='utf-8') as file:
            file.write(data)
        
        print(f"File '{input_filename}' has been processed and saved as '{output_filename}' with the new delimiter.")
    except Exception as e:
        print(f"An error occurred: {e}")



csvdictionary={
               "iris":"C:/Users/User/git/CART-method_with_randomf-forest_and_decision-forest/Data/iris/iris.csv",
               "wdbc":"C:/Users/User/git/CART-method_with_randomf-forest_and_decision-forest/Data/wdbc/wdbc.csv",
               "car":"C:/Users/User/git/CART-method_with_randomf-forest_and_decision-forest/Data/car/car.csv",
               #"studentsuccess":"C:/Users/User/git/CART-method_with_randomf-forest_and_decision-forest/Data/studentsuccess/studentsuccess.csv"
               }
    
if __name__ == "__main__":
    starttime = time.time()
    for dataset_name, path in csvdictionary.items():    
        print(f"Reading data from {path}")
        csv = pd.read_csv(path)
        print("Preprocessing data...")
        csv = preprocess_dataframe(csv)
        print("Splitting data into training and testing sets...")
        """print("Building tree...")
        root = build_tree(data=csv, num_features=1)
        print("Printing tree...")
        print_tree(root)
        accuracy = calculate_accuracy(root, csv)
        print(f"Accuracy on test data: {accuracy:.2f}")
        print("Counting features...")
        features_count_dict=count_used_features_in_tree(root)
        print(features_count_dict)
        print("Visualizing the tree...")
        # Assuming the tree has been built and stored in 'root'
        dot = visualize_tree(root)
        # Save the dot source in a file and render it as a PNG image
        dot.render('tree_diagram', format='png', cleanup=True)  # 'cleanup=True' will remove the dot source file after rendering"""
        
        num_trees = [1, 10, 25, 50, 75, 100]
        # Random Forest
        print("Building random forest for ...",dataset_name)
        print("Hyperparameter tuning for Random Forest...")

        #-1 is signal for the function that changes each node to have a different number of features
        num_features = [1, 2, int(math.log2(len(csv.columns)-1)), -1]

        if len(csv.columns)-1 > 8:
            hyperparametertuningrf, features_count_dict_rf,bestacc = hyperparameter_tuning_RF(csv, num_trees, num_features)
        else:
            simple_features = [1, 2, 3, len(csv.columns)-1,-1]  # Simpler scenario with fewer features
            hyperparametertuningrf, features_count_dict_rf,bestacc = hyperparameter_tuning_RF(csv, num_trees, simple_features)
        #print("Best Configuration: ")
        #print_forest(hyperparametertuningrf)
        random_Tree=hyperparametertuningrf[0]
        dot = visualize_tree(random_Tree)
        # Save the dot source in a file and render it as a PNG image
        name = f"tree_diagram_random_forest_{dataset_name}"
        dot.render(name, format='png', cleanup=True)  # 'cleanup=True' will remove the dot source file after rendering

        #Decision Forest
        print("Building decision forest for ...", dataset_name)
        print("Hyperparameter tuning for Decision Forest...")
        
        num_features = [1, 2, int(math.log2(len(csv.columns)-1)), 0]
        #0 is signal for the function that changes each tree to have a different number of features like the function from the assignment
        if len(csv.columns)-1 > 8:
            hyperparametertuningdf, features_count_dict_df,bestacc = hyperparameter_tuning_DF(csv, num_trees, num_features)
        else:
            simple_features = [1, 2, 3, len(csv.columns)-1,0]  # Simpler scenario with fewer features
            hyperparametertuningdf, features_count_dict_df,bestacc = hyperparameter_tuning_DF(csv, num_trees, simple_features)
        random_Tree=hyperparametertuningdf[0]
        dot = visualize_tree(random_Tree)
        # Save the dot source in a file and render it as a PNG image
        name = f"tree_diagram_decision_forest_{dataset_name}"
        dot.render(name, format='png', cleanup=True)  # 'cleanup=True' will remove the dot source file after rendering
        #print("Best configuration:")
        #print_forest(hyperparametertuningdf)

    endtime = time.time()
    print(f"Total time taken: {endtime - starttime:.2f} seconds")
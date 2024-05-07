from sklearn.model_selection import train_test_split
from preprocessing import preprocess_dataframe
from decision_tree import build_tree, print_tree, calculate_accuracy,count_used_features_in_tree
from random_forest import build_random_forest,accuracy_with_RF,generalization_error_with_cross_val_RF,hyperparameter_tuning_RF
from visualization import visualize_tree
from decision_forest import build_decision_forest,accuracy_with_DF,generalization_error_with_cross_val_DF,hyperparameter_tuning_DF,print_forest
import pandas as pd
import math
import sys
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

NUM_OF_TREES=5
NUMBER_OF_FEATURES=2
IMPURITY_THRESHOLD=0.2
    
csvdictionary={
               "iris":"C:/Users/User/git/CART-method_with_randomf-forest_and_decision-forest/Data/iris/iris.csv",
               "wdbc":"C:/Users/User/git/CART-method_with_randomf-forest_and_decision-forest/Data/wdbc/wdbc.csv",
               "car":"C:/Users/User/git/CART-method_with_randomf-forest_and_decision-forest/Data/car/car.csv",
               "studentsuccess":"C:/Users/User/git/CART-method_with_randomf-forest_and_decision-forest/Data/studentsuccess/studentsuccess.csv",}
    
if __name__ == "__main__":
    starttime = time.time()
    for dataset_name, path in csvdictionary.items():    
        print(f"Reading data from {path}")
        csv = pd.read_csv(path)
        print("Preprocessing data...")
        csv = preprocess_dataframe(csv)
        print("Splitting data into training and testing sets...")
        train_data, test_data = train_test_split(csv, test_size=0.2, random_state=42)
        """print("Building tree...")
        root = build_tree(train_data)
        print("Printing tree...")
        print_tree(root)
        accuracy = calculate_accuracy(root, test_data)
        print(f"Accuracy on test data: {accuracy:.2f}")
        print("Counting features...")
        features_count_dict=count_used_features_in_tree(root)
        print(features_count_dict)
        print("Visualizing the tree...")
        # Assuming the tree has been built and stored in 'root'
        dot = visualize_tree(root)
        # Save the dot source in a file and render it as a PNG image
        dot.render('tree_diagram', format='png', cleanup=True)  # 'cleanup=True' will remove the dot source file after rendering
        """

        # Random Forest
        print("Building random forest for ...",dataset_name)
        #forest,features_count_dict_rf = build_random_forest(train_data, NUMBER_OF_FEATURES, NUM_OF_TREES)
        #print("Counting features for Random Forest:",features_count_dict_rf)
        #print("Calculating accuracy for RF...")
        #accuracy = accuracy_with_RF(test_data,forest)
        #print(f"Accuracy on test data with RF: {accuracy:.2f}")
        print("Hyperparameter tuning for Random Forest...")
        hyperparametertuningrf,features_count_dict_rf=hyperparameter_tuning_RF(train_data, 24, [1,10,25,50,75,100,1,10,25,50,75,100,1,10,25,50,75,100,1,10,25,50,75,100], [1,2,int(math.log2(len(train_data.columns))),int(math.sqrt(len(train_data.columns))),1,2,int(math.log2(len(train_data.columns))),int(math.sqrt(len(train_data.columns))),1,2,int(math.log2(len(train_data.columns))),int(math.sqrt(len(train_data.columns))),1,2,int(math.log2(len(train_data.columns))),int(math.sqrt(len(train_data.columns))),1,2,int(math.log2(len(train_data.columns))),int(math.sqrt(len(train_data.columns))),1,2,int(math.log2(len(train_data.columns))),int(math.sqrt(len(train_data.columns)))])
        #print("Best Configuration: ")
        #print_forest(hyperparametertuningrf)
        print("Featurecountdictionary: ",features_count_dict_rf)
        print("Calculating Generalization Error with Cross validation for Random Forest...")
        #generalizationerrorrf=generalization_error_with_cross_val_RF(csv, 10, hyperparametertuningrf)
        accuracyRF=accuracy_with_RF(test_data,hyperparametertuningrf)
        print(f"Accuracy on Test for Random Forest: {accuracyRF:.2f}")


        #Decision Forest
        print("Building decision forest for ...", dataset_name)
        #decision_forest,features_count_dict_df=build_decision_forest(train_data,NUMBER_OF_FEATURES,NUM_OF_TREES)
        #print("Counting features for Decision Forest:",features_count_dict_df)
        #print("Calculating accuracy for DF...")
        #accueracy_DF=accuracy_with_DF(test_data,decision_forest)
        #print(f"Accuracy on test data with Decision Forest: {accueracy_DF:.2f}")
        print("Hyperparameter tuning for Decision Forest...")
        hyperparametertuningdf,featurecountdict=hyperparameter_tuning_DF(train_data, 24, [1,10,25,50,75,100,1,10,25,50,75,100,1,10,25,50,75,100,1,10,25,50,75,100], [1,2,int(math.log2(len(train_data.columns))),int(math.sqrt(len(train_data.columns))),1,2,int(math.log2(len(train_data.columns))),int(math.sqrt(len(train_data.columns))),1,2,int(math.log2(len(train_data.columns))),int(math.sqrt(len(train_data.columns))),1,2,int(math.log2(len(train_data.columns))),int(math.sqrt(len(train_data.columns))),1,2,int(math.log2(len(train_data.columns))),int(math.sqrt(len(train_data.columns))),1,2,int(math.log2(len(train_data.columns))),int(math.sqrt(len(train_data.columns)))])
        #print("Best configuration:")
        #print_forest(hyperparametertuningdf)
        print("featurescountdict: ",featurecountdict)
        print("Calculating Generalization Error with Cross validation for Random Forest...")
        #generalizationerrorrf=generalization_error_with_cross_val_DF(csv, 10, hyperparametertuningdf)
        accuracyDF=accuracy_with_DF(test_data,hyperparametertuningdf)
        print(f"Accuracy on Test for Decision Forest: {accuracyDF:.2f}")

    endtime = time.time()
    print(f"Total time taken: {endtime - starttime:.2f} seconds")
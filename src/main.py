from sklearn.model_selection import train_test_split
from preprocessing import preprocess_dataframe
#from decision_tree import build_tree, print_tree, calculate_accuracy,predict
from random_forest import build_random_forest,accuracy_with_RF
from visualization import visualize_tree
import pandas as pd
NUM_OF_TREES=5
NUMBER_OF_FEATURES=2
IMPURITY_THRESHOLD=0.2
    
    
    
if __name__ == "__main__":
    path = "C:/Users/User/git/CART-method_with_randomf-forest_and_decision-forest/Data/car/car.csv"
    print(f"Reading data from {path}")
    csv = pd.read_csv(path)
    print("Preprocessing data...")
    csv = preprocess_dataframe(csv)
    print("Splitting data into training and testing sets...")
    train_data, test_data = train_test_split(csv, test_size=0.2, random_state=42)
    """    print("Building tree...")
    root = build_tree(train_data)
    print("Printing tree...")
    print_tree(root)
    accuracy = calculate_accuracy(root, test_data)
    print(f"Accuracy on test data: {accuracy:.2f}")
    print("Visualizing the tree...")
    # Assuming the tree has been built and stored in 'root'
    dot = visualize_tree(root)
    # Save the dot source in a file and render it as a PNG image
    dot.render('tree_diagram', format='png', cleanup=True)  # 'cleanup=True' will remove the dot source file after rendering"""
    print("Building random forest...")
    forest = build_random_forest(train_data, NUMBER_OF_FEATURES, NUM_OF_TREES)
    print("Calculating accuracy...")
    accuracy = accuracy_with_RF(test_data,forest)
    print(f"Accuracy on test data: {accuracy:.2f}")
    
    
    

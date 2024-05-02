import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess_dataframe(df):
    df.dropna(inplace=True)
    return df


class Condition:
    def __init__(self, feature, threshold):
        self.feature = feature
        self.threshold = threshold
        

class Leaf:
    def __init__(self, value=None):
        self.value = value

    def is_leaf_node(self):
        return True

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.left is None

def calc_gini_impurity(condition, data):
    left = data[data[condition.feature] < condition.threshold]
    right = data[data[condition.feature] >= condition.threshold]
    left_counts = left.iloc[:,-1].value_counts().to_dict()
    prob_left = 1
    for key in left_counts:
        prob_left -= (left_counts[key]/len(left))**2
    #print("gini_left:",prob_left)
    prob_right = 1
    right_counts = right.iloc[:,-1].value_counts().to_dict()
    for key in right_counts:
        prob_right -= (right_counts[key]/len(right))**2
    #print("gini_right:",prob_right)
    
    total_gini = (len(left)/len(data))*prob_left + (len(right)/len(data))*prob_right
    #print("total_gini:",total_gini)
    
    return total_gini
if __name__ == "__main__":
    csv = pd.read_csv("C:/Users/User/git/CART-method_with_randomf-forest_and_decision-forest/Data/iris/iris.csv")
    csv = preprocess_dataframe(csv)
    
    # Split the data into training and testing sets
    train_data, test_data = train_test_split(csv, test_size=0.2, random_state=42)
    

    best_fit=None
    best_root_gini = 1
    features=train_data.iloc[:, :-1]
    for feature in features.columns:
        print(feature)
        train_data.sort_values(by=feature, inplace=True)
        avg_sepal_length = (features[feature].shift(1) + features[feature].shift(-1)) / 2
        avg_sepal_length = avg_sepal_length.dropna()
        best_gini = 1
        for i in range(len(avg_sepal_length)):
            gini = calc_gini_impurity(Condition(feature,avg_sepal_length.iloc[i]), train_data)
            if gini < best_gini:
                best_gini = gini
                best_condition = Condition(feature,avg_sepal_length.iloc[i])
        print(feature,best_gini)
        print(best_condition.feature)
        if best_gini < best_root_gini:
            best_root_gini = best_gini
            best_fit = best_condition

        print(best_fit.feature)
        
    root = Node(feature=best_fit.feature, threshold=best_fit.threshold)
    left = train_data[train_data[root.feature] < root.threshold]
    left_counts = left.iloc[:,-1].value_counts().to_dict()
    prob_left = 1
    for key in left_counts:
        prob_left -= (left_counts[key]/len(left))**2
    print("gini_left:",left_counts)
    if prob_left !=0:
        print("left!!!!!")
        best_fit=None
        best_root_gini = 1
        features=left.iloc[:, :-1]
        for feature in features.columns:
            print(feature)
            train_data.sort_values(by=feature, inplace=True)
            avg_sepal_length = (features[feature].shift(1) + features[feature].shift(-1)) / 2
            avg_sepal_length = avg_sepal_length.dropna()
            best_gini = 1
            for i in range(len(avg_sepal_length)):
                gini = calc_gini_impurity(Condition(feature,avg_sepal_length.iloc[i]), left)
                if gini < best_gini:
                    best_gini = gini
                    best_condition = Condition(feature,avg_sepal_length.iloc[i])
            print(feature,best_gini)
            print(best_condition.feature)
            if best_gini < best_root_gini:
                best_root_gini = best_gini
                best_fit = best_condition

            print("left",best_fit.feature)
    else:
        print("left leaf")
        root.left = Leaf(list(left_counts.keys())[0])
    right = train_data[train_data[root.feature] >= root.threshold]
    right_counts = right.iloc[:,-1].value_counts().to_dict()
    prob_right=1
    for key in right_counts:
        prob_right -= (right_counts[key]/len(right))**2
    print("gini_right:",prob_right)
    if prob_right !=0:
        best_fit=None
        best_root_gini = 1
        features=right.iloc[:, :-1]
        for feature in features.columns:
            print(feature)
            train_data.sort_values(by=feature, inplace=True)
            avg_sepal_length = (features[feature].shift(1) + features[feature].shift(-1)) / 2
            avg_sepal_length = avg_sepal_length.dropna()
            best_gini = 1
            for i in range(len(avg_sepal_length)):
                gini = calc_gini_impurity(Condition(feature,avg_sepal_length.iloc[i]), right)
                if gini < best_gini:
                    best_gini = gini
                    best_condition = Condition(feature,avg_sepal_length.iloc[i])
            print(feature,best_gini)
            print(best_condition.feature)
            if best_gini < best_root_gini:
                best_root_gini = best_gini
                best_fit = best_condition

            print("right",best_fit.feature)
    
    root.right = Node(feature=best_fit.feature, threshold=best_fit.threshold)
    root.right.left 
    print(root.left.value)
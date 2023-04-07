import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import os


# Load the dataset
mushrooms_data = pd.read_csv('mushrooms.csv')

# Find constant columns
constant_columns = [col for col in mushrooms_data.columns if len(mushrooms_data[col].unique()) == 1]

# Drop constant columns from the dataset
mushrooms_data = mushrooms_data.drop(constant_columns, axis=1)

# Label Encoding for categorical variables
label_encoder = LabelEncoder()
for col in mushrooms_data.columns:
    mushrooms_data[col] = label_encoder.fit_transform(mushrooms_data[col])

# Split data into training and testing sets
train_data, test_data, train_labels, test_labels = train_test_split(mushrooms_data.drop('class', axis=1), mushrooms_data['class'], test_size=0.2, random_state=42)

# Display the number of training and testing examples
print(f"Number of training examples: {len(train_data)}")
print(f"Number of test examples: {len(test_data)}")

# Create a DecisionTreeClassifier object
clf = DecisionTreeClassifier()

# Train the algorithm
clf.fit(train_data, train_labels)

# Check the performance of the algorithm on the testing data
accuracy = clf.score(test_data, test_labels)
print(f"Accuracy: {accuracy}")

from sklearn.tree import export_graphviz
from graphviz import Source

# Export the graph as a dot file
dot_data = export_graphviz(clf, out_file=None, 
                           feature_names=train_data.columns,  
                           class_names=['edible', 'poisonous'],  
                           filled=True, rounded=True,  
                           special_characters=True)

# Plot the tree
graph = Source(dot_data)
graph.view()
# mushroom-classification-decision-tree-visualization
Readme for Mushroom Classification with Decision Trees
This code uses a Decision Tree Classifier to classify mushrooms as edible or poisonous based on their features.

Dataset
The dataset used in this code is called "Mushroom Classification" and can be found in the UCI Machine Learning Repository. It contains information about various attributes of mushrooms, such as cap shape, odor, and habitat, as well as their class (either edible or poisonous).

Preprocessing
The code first reads in the dataset using Pandas and then identifies any constant columns (columns that have the same value for all rows). These columns are then dropped from the dataset using the drop() function.

Next, categorical variables in the dataset are label encoded using Scikit-learn's LabelEncoder function. This assigns a numerical value to each category in the variable, which allows the algorithm to work with categorical data.

The dataset is then split into training and testing sets using Scikit-learn's train_test_split() function. The training set is used to train the Decision Tree Classifier, while the testing set is used to evaluate its performance.

Model Training and Evaluation
A Decision Tree Classifier is created using Scikit-learn's DecisionTreeClassifier() function. The classifier is then trained using the fit() function on the training data.

The accuracy of the classifier is then evaluated using the score() function on the testing data.

Visualization
The Decision Tree Classifier is visualized using Scikit-learn's export_graphviz() function and the Source function from the graphviz library. The resulting graph is saved as a dot file and displayed as an image using the view() function.

Conclusion
This code provides an example of using a Decision Tree Classifier to classify mushrooms based on their features. The resulting model achieves high accuracy on the testing data and can be visualized as a decision tree.

The decision tree graph generated by the Decision Tree Classifier is saved as a PDF file in Source.gv.pdf.

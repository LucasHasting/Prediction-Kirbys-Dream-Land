#include data wrangling library
import pandas as pd

#include plotting library
import matplotlib.pyplot as plt

#include model libraries
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.naive_bayes import GaussianNB

#include test and accuracy libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#gett data from csv
df = pd.read_csv('kdl.csv')

#split data into dependent/independent variables
y = df["move"]
df.drop("move", axis=1, inplace=True)
X = df
columns = df.columns.to_list()
columns.append("move")


#split data into test/traiing (1/3 - test, 2/3 - training)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = DecisionTreeClassifier(random_state=42) # Initialize the classifier
clf.fit(X_train, y_train) # Train the classifier
y_pred = clf.predict(X_test) # Make predictions on the test set
accuracy = accuracy_score(y_test, y_pred) # Calculate accuracy
print(f"DT Accuracy: {accuracy:.2f}")

'''
gnb = GaussianNB() # Initialize the classifier
gnb.fit(X_train, y_train) # Train the classifier
y_pred = gnb.predict(X_test) # Make predictions on the test set
accuracy = accuracy_score(y_test, y_pred) # Calculate accuracy
print(f"NB Accuracy: {accuracy:.2f}")
'''

#Display Decision Tree
plt.figure(figsize=(500, 500)) # Adjust figure size for better visualization
plot_tree(clf, 
          feature_names=columns, 
          class_names=y.unique(),
          filled=True, 
          rounded=True,
          fontsize=10)
plt.title("Decision Tree Visualization")
plt.savefig("Decision_Tree.png")
plt.close()
print("DONE")

#Name:          Lucas Hasting
#Class:         DA 460
#Date:          12/7/2025
#Instructor:    Dr. Imbrogno
#Description:   Parameter Search for models.py
#Sources:       ChatGPT was used for syntax
#               https://scikit-learn.org/stable/api/index.html
#               https://mlbenchmarks.org/04-holdout-method.html

#include data wrangling library
import pandas as pd

#include plotting library
import matplotlib.pyplot as plt

#include model libraries
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

#include test and accuracy libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#min-max norm function
def min_max_normalization(x, old_min, old_max, new_min, new_max):
    return ((x - old_min) / (old_max - old_min) * (new_max - new_min) + new_min)

#constant for generalization gap threshold
GG_THRESHOLD = 0.05

#get data from csv
df = pd.read_csv('kdl.csv')

#split data into dependent/independent variables
y = df["move"]
df.drop("move", axis=1, inplace=True)

#get dummies for game state and normalize
df = pd.get_dummies(df, columns=['game_state'], prefix='state', dtype=int)
for i in df.columns[df.columns.str.startswith('state_')]:
    lst = df[i].to_list()
    lst = [min_max_normalization(x, 0, 1, 0, 255) for x in lst]
    df[i] = pd.DataFrame(lst)

#min-max norm - kirby_x
lst = df["kirby_x"].to_list()
lst = [min_max_normalization(x, 0, 65535, 0, 255) for x in lst]
df["kirby_x"] = pd.DataFrame(lst)

#min-max norm - kirby_y
lst = df["kirby_y"].to_list()
lst = [min_max_normalization(x, 0, 65535, 0, 255) for x in lst]
df["kirby_y"] = pd.DataFrame(lst)

X = df

#split data into test/training (1/3 - test, 2/3 - training)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Find best parameter: max_depth
train_accuracy = 1
test_accuracy = 0
param = 20 #good starting point

while(abs(train_accuracy - test_accuracy) > GG_THRESHOLD or param == 0):
    clf = DecisionTreeClassifier(random_state=42, max_depth=param) # Initialize the classifier
    clf.fit(X_train, y_train) # Train the classifier
    y_pred = clf.predict(X_test) # Make predictions on the test set
    accuracy = accuracy_score(y_test, y_pred) # Calculate accuracy

    #check for overfitting using the generalization gap
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"param: {param}, acc: {abs(train_accuracy - test_accuracy)} = {train_accuracy} - {test_accuracy}")
    param -= 1

print(f"Best max_depth: {param+1}\n")

#Find best parameter: k
train_accuracy = 1
test_accuracy = 0
param = 3 #good starting point

while(abs(train_accuracy - test_accuracy) > GG_THRESHOLD or param == 100):
    knn = KNeighborsClassifier(n_neighbors=param,metric='euclidean') # Initialize the classifier
    knn.fit(X_train, y_train) # Train the classifier
    y_pred = knn.predict(X_test) # Make predictions on the test set
    accuracy = accuracy_score(y_test, y_pred) # Calculate accuracy

    #check for overfitting using the generalization gap
    y_train_pred = knn.predict(X_train)
    y_test_pred = knn.predict(X_test)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"param: {param}, acc: {abs(train_accuracy - test_accuracy)} = {train_accuracy} - {test_accuracy}")
    param += 2

print(f"Best k: {param - 2}")

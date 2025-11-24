#Name:          Lucas Hasting
#Class:         DA 460
#Date:          12/7/2025
#Instructor:    Dr. Imbrogno
#Description:   Course Project - Build/Test models for Kirby's Dream Land
#Sources:       ChatGPT was used for syntax
#               https://scikit-learn.org/stable/api/index.html

#include data wrangling library
import pandas as pd

#include plotting library
import matplotlib.pyplot as plt

#include model libraries
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.neighbors import KNeighborsClassifier

#include test and accuracy libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay

#min-max norm function
def min_max_normalization(x, old_min, old_max, new_min, new_max):
    return ((x - old_min) / (old_max - old_min) * (new_max - new_min) + new_min)

#get data from csv
df = pd.read_csv('kdl.csv')

#split data into dependent/independent variables
y = df["move"]
df.drop("move", axis=1, inplace=True)
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

#Found max_depth in separate program
clf = DecisionTreeClassifier(random_state=42, max_depth=17) # Initialize the classifier
clf.fit(X_train, y_train) # Train the classifier
y_pred = clf.predict(X_test) # Make predictions on the test set
accuracy = accuracy_score(y_test, y_pred) # Calculate accuracy

#display confusion matrix
fig, ax = plt.subplots(figsize=(200, 200))
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred), display_labels=sorted(y.unique()))
disp.plot(cmap=plt.cm.Blues,ax=ax)
disp.ax_.set_xticks([])
plt.title('Decision Tree - Confusion Matrix')
plt.show()

#display model accuracy
print(f"DT Accuracy: {accuracy:.2f}")
print()

#found K in separate program
knn = KNeighborsClassifier(n_neighbors=7,metric='euclidean') # Initialize the classifier
knn.fit(X_train, y_train) # Train the classifier
y_pred = knn.predict(X_test) # Make predictions on the test set
accuracy = accuracy_score(y_test, y_pred) # Calculate accuracy

#display confusion matrix
fig, ax = plt.subplots(figsize=(200, 200))
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred), display_labels=sorted(y.unique()))
disp.plot(cmap=plt.cm.Reds,ax=ax)
disp.ax_.set_xticks([])
plt.title('k-NN - Confusion Matrix')
plt.show()

#display model accuracy
print(f"KNN Accuracy: {accuracy:.2f}")
print()

#display Decision Tree
print("DECISION TREE:")
print(export_text(clf, feature_names = df.columns.to_list()))

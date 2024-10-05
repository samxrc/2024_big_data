...Simple Handwriting Recognition Tool
Here's a 15-minute AI project idea for beginners: Build a simple handwriting recognition tool using
Python and the scikit-learn library.
Here are the steps to create a basic handwriting recognition tool:
1. Install Python and the scikit-learn library.
2. Import the necessary libraries and modules in your Python script.
3. Load the MNIST dataset of handwritten digits.
4. Split the dataset into training and testing sets.
5. Choose a classification algorithm (e.g. logistic regression, decision tree, or random forest) and
train the model on the training set.
6. Test the model on the testing set and evaluate its performance using metrics such as accuracy
or confusion matrix.
7. Use the trained model to recognize new handwritten digits.


In just 15 minutes, you can create a simple handwriting recognition tool that can recognize
handwritten digits with a decent accuracy. As you gain more experience and skills, you can explore
more advanced techniques such as deep learning and convolutional neural networks to improve the
accuracy of your handwriting recognition model.
...

# ----------1  install python libraries 
pip install scikit-learn matplotlib numpy


# ---- 2 import libraries & modules
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# -------3 Load digits dataset (similar to MNIST, but smaller)
digits = datasets.load_digits()
# Features (pixel values) and labels (digit class)
X = digits.images
y = digits.target
# Reshape the image data to fit the model
n_samples = len(X)
X = X.reshape((n_samples, -1))


# -----4   Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# -----5 Choose a classification algorithm and train the model

#Logistic Regression 
logistic_reg = LogisticRegression(max_iter=10000) 
logistic_reg.fit(X_train, y_train)

#Decision Tree 
tree_clf = DecisionTreeClassifier()
tree_clf.fit(X_train, y_train)

#Random Forest
rf_clf = RandomForestClassifier()
rf_clf.fit(X_train, y_train)

# ------6 3 type of test
# Testing Logistic Regression 
y_pred = logistic_reg.predict(X_test)
print("Logistic Regression Accuracy:", metrics.accuracy_score(y_test, y_pred))
print(metrics.classification_report(y_test, y_pred))

# Testing Decision Tree 

y_pred_tree = tree_clf.predict(X_test)
print("Decision Tree Accuracy:", metrics.accuracy_score(y_test, y_pred_tree))
print(metrics.classification_report(y_test, y_pred_tree))

# Testing Random Forest
y_pred_rf = rf_clf.predict(X_test)
print("Random Forest Accuracy:", metrics.accuracy_score(y_test, y_pred_rf))
print(metrics.classification_report(y_test, y_pred_rf))

#Confusion Matrix
#You can also visualize the confusion matrix for deeper insight into the performance of the model.

conf_matrix = metrics.confusion_matrix(y_test, y_pred_rf)
plt.matshow(conf_matrix, cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

#---------------7Use the trained model to recognize new handwritten digits
# Use the Random Forest model for prediction
some_digit = X_test[0]  # Test on the first digit of the test set
predicted_label = rf_clf.predict([some_digit])
print(f"Predicted label: {predicted_label}")

# Show the image of the digit
plt.imshow(X_test[0].reshape(8, 8), cmap='gray')
plt.title(f"Prediction: {predicted_label[0]}")
plt.show()

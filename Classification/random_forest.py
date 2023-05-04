
# Banafsheh Khazali
# Data: March 04, 2023

"""## Libraries"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


import os
from os import listdir
from os.path import join, isfile

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier

# from google.colab import drive
# drive.mount('/content/drive')

"""## Label Assignment"""

# Set the directory containing the CSV files
directory = "Dataset"

# Define a list to store the data and labels
data = []
labels = []

# Loop through the folders
for folder in os.listdir(directory):
    folder_path = os.path.join(directory, folder)
    if os.path.isdir(folder_path):
        # Get the label (folder name)
        label = folder
        
        # Loop through the CSV files in the folder
        for filename in os.listdir(folder_path):
            if filename.endswith(".csv"):
                # Read the CSV file into a list of lists
                with open(os.path.join(folder_path, filename), "r") as f:
                    lines = f.readlines()
                    # Remove the header line if it exists
                    if lines[0].startswith("datetime"):
                        lines.pop(0)
                    # Extract the data values from the remaining lines
                    data_values = [list(map(float, line.strip().split(",")[1:])) for line in lines if line.strip().split(",")[0] != "time"]
                # Add the data and label to the lists
                data.append(data_values)
                          
                labels.append(label)
print(labels)
print(data)
# Pad the sequences to make them the same length
X_padded = pad_sequences(data, padding="post", dtype="float32")
print("x_padded is:", X_padded)
# Convert the data and labels to numpy arrays
X = np.array(X_padded)
y = np.array(labels)


# Flatten the data into a 2D array
n_samples, n_timesteps, n_features = X.shape


X_flat = X.reshape((n_samples, n_timesteps * n_features))

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X_flat, y, test_size=0.2, random_state=42)

# # Create a random forest classifier and fit it to the training data
# clf = RandomForestClassifier(n_estimators=100, random_state=42)
# clf.fit(X_train, y_train)

# # Evaluate the classifier on the testing data
# score = clf.score(X_test, y_test)
# print("Accuracy:", score)

"""## Classififcation"""

# Pad the sequences to make them the same length
X_padded = pad_sequences(data, padding="post", dtype="float32")

# Convert the data and labels to numpy arrays
X = np.array(X_padded)
y = np.array(labels)

# Flatten the data into a 2D array
n_samples, n_timesteps, n_features = X.shape
X_flat = X.reshape((n_samples, n_timesteps * n_features))

# Split the data into training, validation, and testing sets
X_train_val, X_test, y_train_val, y_test = train_test_split(X_flat, labels, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

# Create a random forest classifier and fit it to the training data
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate the classifier on the training data
y_pred_train = clf.predict(X_train)
print("Training data classification report:")
print(classification_report(y_train, y_pred_train))

# Evaluate the classifier on the validation data
y_pred_val = clf.predict(X_val)
print("Validation data classification report:")
print(classification_report(y_val, y_pred_val))

# Evaluate the classifier on the testing data
y_pred_test = clf.predict(X_test)
print("Testing data classification report:")
print(classification_report(y_test, y_pred_test))

# Calculate and plot the confusion matrix for the testing data
cm = confusion_matrix(y_test, y_pred_test)
plt.figure(figsize=(8, 8))
sns.heatmap(cm, annot=True, cmap="Blues", xticklabels=np.unique(y_train_val), yticklabels=np.unique(y_train_val))
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("RF_confmatrix.png")
plt.show()

"""## Feature Importance"""

# Pad the sequences to make them the same length
X_padded = pad_sequences(data, padding="post", dtype="float32")

# Convert the data and labels to numpy arrays
X = np.array(X_padded)
y = np.array(labels)

# Flatten the data into a 2D array
n_samples, n_timesteps, n_features = X.shape
X_flat = X.reshape((n_samples, n_timesteps * n_features))

# Split the data into training, validation, and testing sets
X_train_val, X_test, y_train_val, y_test = train_test_split(X_flat, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

# Create a random forest classifier and fit it to the training data
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate the classifier on the training data
y_pred_train = clf.predict(X_train)
print("Training data classification report:")
print(classification_report(y_train, y_pred_train))

# Evaluate the classifier on the validation data
y_pred_val = clf.predict(X_val)
print("Validation data classification report:")
print(classification_report(y_val, y_pred_val))

# Evaluate the classifier on the testing data
y_pred_test = clf.predict(X_test)
print("Testing data classification report:")
print(classification_report(y_test, y_pred_test))


# Calculate and plot the confusion matrix for the training data
cm = confusion_matrix(y_train, y_pred_train)
plt.figure(figsize=(8, 8))
sns.heatmap(cm, annot=True, cmap="Blues", xticklabels=np.unique(y_train_val), yticklabels=np.unique(y_train_val))
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - Training Data")
plt.savefig("RF_Conf_train.png")

# Calculate and plot the confusion matrix for the validation data
cm = confusion_matrix(y_val, y_pred_val)
plt.figure(figsize=(8, 8))
sns.heatmap(cm, annot=True, cmap="Blues", xticklabels=np.unique(y_train_val), yticklabels=np.unique(y_train_val))
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - Validation Data")
plt.savefig("RF_Conf_val.png")

# Calculate and plot the confusion matrix for the testing data
cm = confusion_matrix(y_test, y_pred_test)
plt.figure(figsize=(8, 8))
sns.heatmap(cm, annot=True, cmap="Blues", xticklabels=np.unique(y_train_val), yticklabels=np.unique(y_train_val))
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - Testing Data")
plt.savefig("RF_Conf_test.png")

# Get the feature importances and sort them in descending order
importances = clf.feature_importances_
sorted_indices = np.argsort(importances)[::-1]

# Print the top 10 most important features and their importances
print("Top 10 most important features:")
with open(os.path.join(directory, "0", "synthetic_data_157.csv"), "r") as f:
    column_names = f.readline().strip().split(",")[1:]


for i in range(10):
  feature_idx = sorted_indices[i] % 279 # get the feature index from the sorted index
  col_name = column_names[feature_idx] # get the column name from the feature index
  print("%d. %s (%f)" % (i+1, col_name, importances[sorted_indices[i]]))

top_cols = []
top_importances = []
for i in range(10):
    col_idx = sorted_indices[i] % 279  # get the column index from the sorted feature index
    col_name = column_names[col_idx]
    importance = importances[sorted_indices[i]]
    top_cols.append(col_name)
    top_importances.append(importance)
    print("%d. %s (%f)" % (i+1, col_name, importance))

# Create a bar plot of the top 10 most important features
plt.figure(figsize=(10,6))
plt.bar(top_cols, top_importances)
plt.xticks(rotation=90)
plt.ylabel("Feature Importance")
plt.title("Top 10 Most Important Features")
plt.savefig("RF_features.png")
plt.show()

from sklearn.metrics import accuracy_score, precision_score, recall_score
import pandas as pd

# Define a list to store the data and labels
data = []
labels = []

# Loop through the folders
for folder in os.listdir(directory):
    folder_path = os.path.join(directory, folder)
    if os.path.isdir(folder_path):
        # Get the label (folder name)
        label = folder
        
        # Loop through the CSV files in the folder
        for filename in os.listdir(folder_path):
            if filename.endswith(".csv"):
                # Read the CSV file into a list of lists
                with open(os.path.join(folder_path, filename), "r") as f:
                    lines = f.readlines()
                    # Remove the header line if it exists
                    if lines[0].startswith("datetime"):
                        lines.pop(0)
                    # Extract the data values from the remaining lines
                    data_values = [list(map(float, line.strip().split(",")[1:])) for line in lines if line.strip().split(",")[0] != "time"]
                # Add the data and label to the lists
                data.append(data_values)
                          
                labels.append(label)

# Pad the sequences to make them the same length
X_padded = pad_sequences(data, padding="post", dtype="float32")

# Convert the data and labels to numpy arrays
X = np.array(X_padded)
y = np.array(labels)

# Flatten the data into a 2D array
n_samples, n_timesteps, n_features = X.shape
X_flat = X.reshape((n_samples, n_timesteps * n_features))

# Split the data into training, validation, and testing sets
X_train_val, X_test, y_train_val, y_test = train_test_split(X_flat, labels, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

# Create a gradient boosting classifier and fit it to the training data
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate the classifier on the training data
y_pred_train = clf.predict(X_train)
acc_train = accuracy_score(y_train, y_pred_train)
prec_train = precision_score(y_train, y_pred_train, average='macro')
rec_train = recall_score(y_train, y_pred_train, average='macro')

# Evaluate the classifier on the validation data
y_pred_val = clf.predict(X_val)
acc_val = accuracy_score(y_val, y_pred_val)
prec_val = precision_score(y_val, y_pred_val, average='macro')
rec_val = recall_score(y_val, y_pred_val, average='macro')

# Evaluate the classifier on the testing data
y_pred_test = clf.predict(X_test)
acc_test = accuracy_score(y_test, y_pred_test)
prec_test = precision_score(y_test, y_pred_test, average='macro')
rec_test = recall_score(y_test, y_pred_test, average='macro')

# Create a table to display the results
data = {'Dataset': ['Training', 'Validation', 'Testing'], 
        'Accuracy': [acc_train, acc_val, acc_test], 
        'Precision': [prec_train, prec_val, prec_test], 
        'Recall': [rec_train, rec_val, rec_test]}
df = pd.DataFrame(data)

# Create a bar plot to display the results

x = np.arange(len(data['Dataset']))
width = 0.25

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, data['Accuracy'], width, label='Accuracy')
rects2 = ax.bar(x + width/2, data['Recall'], width, label='Recall')
rects3 = ax.bar(x + width/2 + width, data['Precision'], width, label='Precision')
# Add labels, title and legend

ax.set_ylabel('Score')
ax.set_title('Model Performance')
ax.set_xticks(x)
ax.set_xticklabels(data['Dataset'])
ax.legend()
# Add values on top of the bars

def autolabel(rects):
  for rect in rects:
    height = rect.get_height()
    ax.annotate('{}'.format(round(height, 2)),
    xy=(rect.get_x() + rect.get_width() / 2, height),
    xytext=(0, 3), # 3 points vertical offset
    textcoords="offset points",
    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

fig.tight_layout()
plt.savefig("RF_performance.png")
plt.show()

Here's the rewritten code to perform 3-class classification using the RandomForestClassifier algorithm on the given dataset.

python
# Banafsheh Khazali
# Data: March 04, 2023

"""## Libraries"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os
from os import listdir
from os.path import join, isfile

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.sequence import pad_sequences


"""## Label Assignment"""

# Set the directory containing the CSV files
directory = "Dataset"

# Define a list to store the data and labels
data = []
labels = []

# Loop through the folders
for folder in os.listdir(directory):
    folder_path = os.path.join(directory, folder)
    if os.path.isdir(folder_path):
        # Get the label (folder name)
        label = folder
        
        # Loop through the CSV files in the folder
        for filename in os.listdir(folder_path):
            if filename.endswith(".csv"):
                # Read the CSV file into a list of lists
                with open(os.path.join(folder_path, filename), "r") as f:
                    lines = f.readlines()
                    # Remove the header line if it exists
                    if lines[0].startswith("datetime"):
                        lines.pop(0)
                    # Extract the data values from the remaining lines
                    data_values = [list(map(float, line.strip().split(",")[1:])) for line in lines if line.strip().split(",")[0] != "time"]
                # Add the data and label to the lists
                if len(data_values) == 6000 and len(data_values[0]) == 279:
                    data.append(data_values)
                    labels.append(label)
print(labels)
print(data)

# Pad the sequences to make them the same length
X_padded = pad_sequences(data, padding="post", dtype="float32")
print("X_padded is:", X_padded)

# Convert the data and labels to numpy arrays
X = np.array(X_padded)
y = np.array(labels)

# Flatten the data into a 2D array
n_samples, n_timesteps, n_features = X.shape
X_flat = X.reshape((n_samples, n_timesteps * n_features))

# Split the data into training, validation, and testing sets
X_train_val, X_test, y_train_val, y_test = train_test_split(X_flat, labels, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

"""## Classification"""

# Create a random forest classifier and fit it to the training data
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate the classifier on the training data
y_pred_train = clf.predict(X_train)
print("Training data classification report:")
print(classification_report(y_train, y_pred_train))

# Evaluate the classifier on the validation data
y_pred_val = clf.predict(X_val)
print("Validation data classification report:")
print(classification_report(y_val, y_pred_val))

# Evaluate the classifier on the testing data
y_pred_test = clf.predict(X_test)
print("Testing data classification report:")
print(classification_report(y_test, y_pred_test))

# Calculate and plot the confusion matrix for the testing data
cm = confusion_matrix(y_test, y_pred_test)
plt.figure(figsize=(8, 

Banafshe khazali
keep writing from here:
# Calculate and plot the confusion matrix for the testing data
cm = confusion_matrix(y_test, y_pred_test)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (Testing Data)')
plt.show()
# Calculate and print the classification report for the testing data

cr = classification_report(y_test, y_pred_test)
print('Classification Report (Testing Data):\n', cr)
# Calculate and print the accuracy score for the testing data

acc = accuracy_score(y_test, y_pred_test)
print('Accuracy Score (Testing Data):', acc)
# Calculate and plot the ROC curve for the testing data

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_test[:, 1])
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='ROC curve')
plt.plot([0, 1], [0, 1], linestyle='--', label='Random guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (Testing Data)')
plt.legend()
plt.show()
# Calculate and print the area under the ROC curve for the testing data

auc = roc_auc_score(y_test, y_pred_proba_test[:, 1])
print('AUC Score (Testing Data):', auc)
# Save the model as a pickle file

# filename = 'credit_model.pkl'
# with open(filename, 'wb') as file:
# pickle.dump(model, file)

# print('Model saved as', filename)

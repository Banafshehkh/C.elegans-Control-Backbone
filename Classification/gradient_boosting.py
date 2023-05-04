# Banafsheh Khazali
# Data: May 02, 2023

"""## Libraries"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from os import listdir
from os.path import join, isfile
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Set the directory containing the CSV files
directory = "Dataset"
# Define a list to store the data and labels
data = []
labels = []

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

                    # Extract the data values from the remaining lines
                    data_values = []
                    for i, line in enumerate(lines):
                        if i == 0:
                            continue  # Skip the first row
                        data_values.append(list(map(float, line.strip().split(",")[1:])))

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

"""## Classification"""

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_flat, y, test_size=0.2, random_state=42)

# Create a logistic regression model and fit it to the training data
clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X_train, y_train)

# Evaluate the classifier on the testing data
y_pred_test = clf.predict(X_test)
print("Testing data classification report:")
print(classification_report(y_test, y_pred_test))

# Calculate and plot the confusion matrix for the testing data
cm = confusion_matrix(y_test, y_pred_test)
plt.figure(figsize=(8, 8))
sns.heatmap(cm, annot=True, cmap="Blues", xticklabels=np.unique(y_train), yticklabels=np.unique(y_train))
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("LR_confmatrix.png")
plt.show()

"""## Feature Importance"""

# Select the most important features
selector = SelectFromModel(clf, threshold="median")
selector.fit(X_train, y_train)

# Get the indices of the selected features
selected_feature_indices = selector.get_support(indices=True)

# Get the names of the selected features
feature_names = []
for i in selected_feature_indices:
    feature_names.append(f"Feature_{i//n_features+1}_Timestep_{i%n_features+1}")

print(f"Selected features: {feature_names}")


# Print the top 10 most important features and their importances
print("Top 10 most important features:")
for i in range(10):
    col_idx = sorted_indices[i] % 279  # get the column index from the sorted feature index
    print("%d. Column %d (%f)" % (i+1, col_idx, importances[sorted_indices[i]]))



# Create a gradient boosting classifier and fit it to the training data
clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Get the feature importances and sort them in descending order
importances = clf.feature_importances_
sorted_indices = np.argsort(importances)[::-1]

# Print the top 10 most important features and their importances
print("Top 10 most important features:")
with open(os.path.join(directory, "0", "/content/drive/MyDrive/Dataset/0/newAWC_6000_0.csv"), "r") as f:
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
plt.savefig("GB_features.png")
plt.show()









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

cm = confusion_matrix(y_train, y_pred_train)
plt.figure(figsize=(8, 8))
sns.heatmap(cm, annot=True, cmap="Blues", xticklabels=np.unique(y_train_val), yticklabels=np.unique(y_train_val))
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - Training Data")
plt.savefig("GB_Conf_train.png")

# Calculate and plot the confusion matrix for the validation data
cm = confusion_matrix(y_val, y_pred_val)
plt.figure(figsize=(8, 8))
sns.heatmap(cm, annot=True, cmap="Blues", xticklabels=np.unique(y_train_val), yticklabels=np.unique(y_train_val))
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - Validation Data")
plt.savefig("GB_Conf_val.png")

# Calculate and plot the confusion matrix for the testing data
cm = confusion_matrix(y_test, y_pred_test)
plt.figure(figsize=(8, 8))
sns.heatmap(cm, annot=True, cmap="Blues", xticklabels=np.unique(y_train_val), yticklabels=np.unique(y_train_val))
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - Testing Data")
plt.savefig("GB_Conf_test.png")

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
plt.savefig("GB_performance.png")
plt.show()


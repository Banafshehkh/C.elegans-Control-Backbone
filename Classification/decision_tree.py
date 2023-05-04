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


clf = DecisionTreeClassifier(max_depth=5, random_state=42)
clf.fit(X_train, y_train)

y_pred_test = clf.predict(X_test)
print("Testing data classification report:")
print(classification_report(y_test, y_pred_test))

cm = confusion_matrix(y_test, y_pred_test)
plt.figure(figsize=(8, 8))
sns.heatmap(cm, annot=True, cmap="Blues", xticklabels=np.unique(y_train), yticklabels=np.unique(y_train))
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("DT_confmatrix.png")
plt.show()

"""## Feature Importance"""

selector = SelectFromModel(clf, threshold="median")
selector.fit(X_train, y_train)

selected_feature_indices = selector.get_support(indices=True)

feature_names = []
for i in selected_feature_indices:
    feature_names.append(f"Feature_{i//n_features+1}Timestep{i%n_features+1}")

print(f"Selected features: {feature_names}")

# Experiment 5 : Multilayer Perceptron (MLP) Neural Network

## Aim

To implement a Multilayer Perceptron (MLP) neural network for
handwritten digit classification using Python and evaluate its
performance on the test dataset.

> **Note:** The source page contains an inconsistent aim statement
> mentioning K-Means clustering. The experiment title and algorithm
> steps describe an MLP neural network, so this README follows the MLP
> experiment.

## Overview

A **Multilayer Perceptron (MLP)** is a feed-forward artificial neural
network consisting of an input layer, one or more hidden layers, and an
output layer. It learns patterns in data by adjusting weights and biases
during training using backpropagation.

In this experiment, the MLP model is trained on the **Digits dataset**
available in scikit-learn. Each sample is an 8 × 8 image of a
handwritten digit (0--9). The pixel values are used as input features,
and the model predicts the corresponding digit.

## Algorithm

1.  **Start.**
2.  Load the Digits dataset and separate the input features and target
    labels.
3.  Split the dataset into training and testing sets.
4.  Standardize the input features.
5.  Create an MLP classifier with the specified hidden-layer structure.
6.  Train the model using the training data.
7.  Predict the labels for the test data.
8.  Evaluate the model using test accuracy and a classification report.
9.  Plot the training loss curve.
10. **Stop.**

> **Note:** The exact accuracy and number of training iterations may vary
> slightly with changes to the train-test split, random seed, or model
> parameters.

## Program

```python
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

# Step 2: Load the Digits dataset
digits = load_digits()
X, y = digits.data, digits.target
print("Dataset shape:", X.shape)

# Step 3: Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print("Training samples:", X_train.shape[0], "| Test samples:", X_test.shape[0])

# Step 4: Standardize features
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# Steps 5-6: Create and train the MLP
mlp = MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu",
                    solver="adam", max_iter=300, random_state=42)
mlp.fit(X_train_s, y_train)
print("Iterations run:", mlp.n_iter_)

# Steps 7-8: Predict and evaluate
y_pred = mlp.predict(X_test_s)
print("\nTest Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 9: Plot training loss curve
plt.figure(figsize=(6, 4))
plt.plot(mlp.loss_curve_)
plt.title("MLP Training Loss Curve")
plt.xlabel("Iteration (epoch)")
plt.ylabel("Loss")
plt.grid(True)
plt.savefig("loss_curve.png", dpi=120, bbox_inches="tight")
plt.show()
```

## Output

```
Dataset shape: (1797, 64)
Training samples: 1437 | Test samples: 360
Iterations run: 116

Test Accuracy: 0.9667

Classification Report:
               precision    recall  f1-score   support

           0       1.00      0.97      0.99        36
           1       0.92      0.92      0.92        36
           2       0.95      1.00      0.97        35
           3       0.97      0.97      0.97        37
           4       0.95      1.00      0.97        36
           5       0.97      1.00      0.99        37
           6       1.00      0.94      0.97        36
           7       0.97      1.00      0.99        36
           8       0.94      0.91      0.93        35
           9       1.00      0.94      0.97        36

    accuracy                           0.97       360
   macro avg       0.97      0.97      0.97       360
weighted avg       0.97      0.97      0.97       360
```

**Training loss curve**

<img width="655" height="481" alt="image" src="https://github.com/user-attachments/assets/0b123482-9848-4b91-b73e-51f3cb92696b" />


## Result

The program was executed successfully. An MLP classifier with two hidden
layers (64 and 32 neurons, ReLU activation, Adam optimizer) was trained
on 1437 handwritten digit images and tested on 360 unseen images.

-   The model achieved a **test accuracy of 96.67%**.
-   Training stopped after 116 iterations, and the loss curve decreases
    steadily, showing that the network learned the patterns in the data.
-   Precision, recall and F1-score are 0.91 or higher for every digit.
    Digits 1 and 8 were the hardest to classify (F1 of 0.92 and 0.93).

Thus, the MLP neural network classified handwritten digits with high
accuracy on the test dataset.

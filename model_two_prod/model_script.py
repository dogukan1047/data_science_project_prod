import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import pickle
import matplotlib.pyplot as plt

# Read the CSV file
data = pd.read_csv("labels1.csv")

# Separate input (X) and target (y) variables
X = data['image_vector'].apply(lambda x: list(map(int, x.split(',')))).tolist()
y = data['label']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define parameter ranges for hyperparameter optimization
parameters = {
    'hidden_layer_sizes': [(50,), (100,), (200,), (300,)],  # Hidden layer sizes
    'activation': ['relu', 'tanh', 'logistic'],  # Activation functions
    'alpha': [0.0001, 0.001, 0.01, 0.5],  # L2 regularization parameter
    'max_iter': [200, 300, 400],  # Maximum number of iterations
}

# Define the MLPClassifier model
mlp = MLPClassifier(random_state=42)

# Perform hyperparameter optimization with GridSearchCV
grid_search = GridSearchCV(mlp, parameters, cv=5)
grid_search.fit(X_train, y_train)

# Print the best parameters and best score
print("Best parameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)

# Select the best model and train
best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)

# Evaluate the model's performance on the test set
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Test set accuracy score:", accuracy)

# Save the best model to a pickle file
with open('best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

# Let's track the loss values and accuracy values during training
history = best_model.fit(X_train, y_train)

# Get the loss values during training
loss_values = history.loss_curve_

# Calculate accuracy for each iteration
train_accuracy_values = [accuracy_score(y_train, best_model.predict(X_train))]
test_accuracy_values = [accuracy_score(y_test, best_model.predict(X_test))]

for i in range(1, len(history.loss_curve_)):
    best_model.partial_fit(X_train, y_train)
    train_accuracy_values.append(accuracy_score(y_train, best_model.predict(X_train)))
    test_accuracy_values.append(accuracy_score(y_test, best_model.predict(X_test)))

# Plot the loss curve during training
plt.plot(loss_values)
plt.title('Loss Curve During Training')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.savefig('loss_curve.png')
plt.show()

# Plot the accuracy curve during training
plt.plot(train_accuracy_values, label='Training Accuracy')
plt.plot(test_accuracy_values, label='Test Accuracy')
plt.title('Accuracy Curve During Training')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('accuracy_curve.png')
plt.show()

print("Training Set Accuracy (Last Iteration):", train_accuracy_values[-1])
print("Test Set Accuracy (Last Iteration):", test_accuracy_values[-1])
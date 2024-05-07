# Importing libraries
from json import encoder

import numpy as np
import pandas as pd
from scipy.stats import mode
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Reading the train.csv by removing the
# last column since it's an empty column
DATA_PATH = "../heart.csv"
data = pd.read_csv(DATA_PATH).dropna(axis=1)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=24)

print(f"Train: {X_train.shape}, {y_train.shape}")
print(f"Test: {X_test.shape}, {y_test.shape}")


# Defining scoring metric for k-fold cross validation
def cv_scoring(estimator, X, y):
    return accuracy_score(y, estimator.predict(X))


from sklearn.model_selection import GridSearchCV

param_grid_svc = {'kernel': ['linear', 'rbf'], 'C': [1, 10]}
param_grid_gnb = {}
param_grid_rf = {'n_estimators': [10, 50, 100], 'max_depth': [None, 5, 10]}

models = {
    "SVC": GridSearchCV(SVC(), param_grid_svc, cv=5, scoring=cv_scoring),
    "Gaussian NB": GridSearchCV(GaussianNB(), param_grid_gnb, cv=5, scoring=cv_scoring),
    "Random Forest": GridSearchCV(RandomForestClassifier(random_state=18), param_grid_rf, cv=5, scoring=cv_scoring)

}

# Producing cross validation score for the models
for model_name in models:
    model = models[model_name]
    scores = cross_val_score(model, X, y, cv=10,
                             n_jobs=-1,
                             scoring=cv_scoring)
    print("==" * 30)
    print(model_name)
    print(f"Scores: {scores}")
    print(f"Mean Score: {np.mean(scores)}")

# Training and testing SVM Classifier
svm_model = SVC()
svm_model.fit(X_train, y_train)
preds = svm_model.predict(X_test)

print(f"Accuracy on train data by SVM Classifier\
: {accuracy_score(y_train, svm_model.predict(X_train)) * 100}")

print(f"Accuracy on test data by SVM Classifier\
: {accuracy_score(y_test, preds) * 100}")
cf_matrix = confusion_matrix(y_test, preds)

# Training and testing Naive Bayes Classifier
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
preds = nb_model.predict(X_test)
print(f"Accuracy on train data by Naive Bayes Classifier\
: {accuracy_score(y_train, nb_model.predict(X_train)) * 100}")

print(f"Accuracy on test data by Naive Bayes Classifier\
: {accuracy_score(y_test, preds) * 100}")
# Training and testing Random Forest Classifier
rf_model = RandomForestClassifier(random_state=18)
rf_model.fit(X_train, y_train)
preds = rf_model.predict(X_test)
print(f"Accuracy on train data by Random Forest Classifier\
: {accuracy_score(y_train, rf_model.predict(X_train)) * 100}")

print(f"Accuracy on test data by Random Forest Classifier\
: {accuracy_score(y_test, preds) * 100}")
# Training the models on whole data
final_svm_model = SVC()
final_nb_model = GaussianNB()
final_rf_model = RandomForestClassifier(random_state=18)
final_svm_model.fit(X, y)
final_nb_model.fit(X, y)
final_rf_model.fit(X, y)

# Reading the test data
test_data = pd.read_csv("../heart.csv").dropna(axis=1)

test_X = test_data.iloc[:, :-1]
test_Y = test_data.iloc[:, -1]

# Making prediction by take mode of predictions
# made by all the classifiers
# Making prediction by take mode of predictions
# made by all the classifiers
svm_preds = final_svm_model.predict(test_X)
nb_preds = final_nb_model.predict(test_X)
rf_preds = final_rf_model.predict(test_X)

final_preds = [mode([i, j, k])[0] for i, j, k in zip(svm_preds, nb_preds, rf_preds)]

print(f"Accuracy on Test dataset by the combined model: {accuracy_score(test_Y, final_preds) * 100}")
from sklearn.ensemble import VotingClassifier

# Define the models
models = [
    ("SVM", final_svm_model),
    ("Naive Bayes", final_nb_model),
    ("Random Forest", final_rf_model)
]

# Create a Voting Classifier
voting_clf = VotingClassifier(models, voting="hard")

# Train the Voting Classifier
voting_clf.fit(X_train, y_train)

# Evaluate on the test set
voting_preds = voting_clf.predict(X_test)
test_accuracy = accuracy_score(y_test, voting_preds)
print(f"Accuracy on test data by Voting Classifier: {test_accuracy * 100}")

# Now, let's train the Voting Classifier on the whole dataset
voting_clf.fit(X, y)

# Make predictions on the test dataset
final_preds = voting_clf.predict(test_X)

# Calculate accuracy on the test dataset
final_test_accuracy = accuracy_score(test_Y, final_preds)
print(f"Accuracy on Test dataset by the combined model: {final_test_accuracy * 100}")
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Define the parameter grid for GridSearchCV
param_grid_rf = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_features': ['sqrt', 'log2', None],
    'max_depth': [5, 10, 15, 20],  # Reducing max_depth to mitigate overfitting
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Create the Random Forest Classifier
rf_model = RandomForestClassifier(random_state=18)

# Perform GridSearchCV with the updated parameter grid
grid_search_rf = GridSearchCV(rf_model, param_grid_rf, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search_rf.fit(X_train, y_train)

# Get the best parameters
best_params_rf = grid_search_rf.best_params_
print("Best Parameters for Random Forest Classifier:", best_params_rf)

# Use the best estimator for predictions
best_rf_model = grid_search_rf.best_estimator_
rf_preds = best_rf_model.predict(X_test)

# Evaluate the model
accuracy_rf = accuracy_score(y_test, rf_preds)
print("Accuracy on test data by Random Forest Classifier after hyperparameter tuning:", accuracy_rf)

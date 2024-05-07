# imported all library
import numpy as np
from django.shortcuts import render

from .forms import HeartDiseaseForm
from .forms import DiabetesForm


def heart(request):
    from scipy.stats import mode
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.svm import SVC
    import pandas as pd
    from sklearn.naive_bayes import GaussianNB
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix
    # Read the heart disease training data from a CSV file
    df = pd.read_csv('../static/heart.csv')
    data = df.dropna(axis=1)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    value = ''

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
        'n_estimators': [50, 100, 200],
        'max_features': ['sqrt', 'log2', None],
        'max_depth': [5, 10, 15],  # Limiting the max depth to reduce overfitting
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    # Create the Random Forest Classifier
    rf_model = RandomForestClassifier(random_state=18)

    # Perform GridSearchCV with the updated parameter grid
    grid_search_rf = GridSearchCV(rf_model, param_grid_rf, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search_rf.fit(X_train, y_train)

    # Get the best parameters
    best_params_rf = grid_search_rf.best_params_
    print("Best Parameters for Random Forest Classifier:", best_params_rf)

    # Use the best estimator for predictions
    best_rf_model = grid_search_rf.best_estimator_
    rf_preds = best_rf_model.predict(X_test)

    # Evaluate the model
    accuracy_rf = accuracy_score(y_test, rf_preds)
    print("Accuracy on test data by Random Forest Classifier after hyper parameter tuning:", accuracy_rf)
    if request.method == 'POST':
        # Retrieve the user input from the form
        age = float(request.POST['age'])
        sex = float(request.POST['sex'])
        cp = float(request.POST['cp'])
        trestbps = float(request.POST['trestbps'])
        chol = float(request.POST['chol'])
        fbs = float(request.POST['fbs'])
        restecg = float(request.POST['restecg'])
        thalach = float(request.POST['thalach'])
        exang = float(request.POST['exang'])
        oldpeak = float(request.POST['oldpeak'])
        slope = float(request.POST['slope'])
        ca = float(request.POST['ca'])
        thal = float(request.POST['thal'])

        # Create a numpy array with the user's data
        user_data = np.array(
            (age,
             sex,
             cp,
             trestbps,
             chol,
             fbs,
             restecg,
             thalach,
             exang,
             oldpeak,
             slope,
             ca,
             thal)
        ).reshape(1, 13)

        predictions = best_rf_model.predict(user_data)  # Make predictions on the user's data
        print('predictions')
        print(predictions)
        if int(predictions[0]) == 1:
            value = 'have'  # User is predicted to have heart disease
        elif int(predictions[0]) == 0:
            value = "don\'t have"  # User is predicted to not have heart disease

    return render(request,
                  'heart.html',
                  {
                      'context': value,
                      'title': 'Heart Disease Prediction',
                      'active': 'btn btn-success peach-gradient text-white',
                      'heart': True,
                      'form': HeartDiseaseForm(),
                  })


def diabetes(request):
    value = ''
    import warnings
    from tkinter import ttk
    import tkinter as tk

    warnings.filterwarnings('ignore')

    # Import Neccessary libraries
    import numpy as np
    import pandas as pd

    # Import Visualization libraries
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Import Model
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from sklearn.pipeline import Pipeline

    # Import Sampler libraries
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.pipeline import Pipeline as imbPipeline

    # Set the decimal format
    pd.options.display.float_format = "{:.2f}".format
    df = pd.read_csv('../static/diabetes.csv')
    df.head()
    # Handle duplicates
    duplicate_rows_data = df[df.duplicated()]
    print("number of duplicate rows: ", duplicate_rows_data.shape)
    df = df.drop_duplicates()

    # Loop through each column and count the number of distinct values
    for column in df.columns:
        num_distinct_values = len(df[column].unique())
        print(f"{column}: {num_distinct_values} distinct values")
    # Checking null values
    print(df.isnull().sum())
    # Remove Unneccessary value [0.00195%]
    df = df[df['gender'] != 'Other']

    # df.describe().style.format("{:.2f}")
    # Define a function to map the existing categories to new ones
    def recategorize_smoking(smoking_status):
        if smoking_status in ['never', 'No Info']:
            return 'non-smoker'
        elif smoking_status == 'current':
            return 'current'
        elif smoking_status in ['ever', 'former', 'not current']:
            return 'past_smoker'

    # Apply the function to the 'smoking_history' column
    df['smoking_history'] = df['smoking_history'].apply(recategorize_smoking)
    data = df.copy()
    # Check the new value counts
    print(df['smoking_history'].value_counts())

    def perform_one_hot_encoding(df, column_name):
        # Perform one-hot encoding on the specified column
        dummies = pd.get_dummies(df[column_name], prefix=column_name)

        # Drop the original column and append the new dummy columns to the dataframe
        df = pd.concat([df.drop(column_name, axis=1), dummies], axis=1)

        return df

    # Perform one-hot encoding on the gender variable
    data = perform_one_hot_encoding(data, 'gender')

    # Perform one-hot encoding on the smoking history variable
    data = perform_one_hot_encoding(data, 'smoking_history')
    # Compute the correlation matrix
    correlation_matrix = data.corr()
    # Graph I.

    # Graph II
    # Create a heatmap of the correlations with the target column
    corr = data.corr()
    target_corr = corr['diabetes'].drop('diabetes')

    # Sort correlation values in descending order
    target_corr_sorted = target_corr.sort_values(ascending=False)

    # Define resampling
    over = SMOTE(sampling_strategy=0.1)
    under = RandomUnderSampler(sampling_strategy=0.5)
    # Define preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            (
                'num', StandardScaler(),
                ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level', 'hypertension', 'heart_disease']),
            ('cat', OneHotEncoder(), ['gender', 'smoking_history'])
        ])

    # Split data into features and target variable
    X = df.drop('diabetes', axis=1)
    y = df['diabetes']
    # Create a pipeline that preprocesses the data, resamples data, and then trains a classifier
    clf = imbPipeline(steps=[('preprocessor', preprocessor),
                             ('over', over),
                             ('under', under),
                             ('classifier', RandomForestClassifier())])
    # Define the hyperparameters and the values we want to test
    param_grid = {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4]
    }
    # Create Grid Search object
    grid_search = GridSearchCV(clf, param_grid, cv=5)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    grid_search.fit(X_train, y_train)
    # Print the best parameters
    print("Best Parameters: ", grid_search.best_params_)

    y_pred = grid_search.predict(X_test)

    # Evaluate the model
    print("Model Accuracy: ", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    # After fitting the model, we input feature names
    onehot_columns = list(
        grid_search.best_estimator_.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(
            ['gender', 'smoking_history']))

    # Then we add the numeric feature names
    feature_names = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level', 'hypertension',
                     'heart_disease'] + onehot_columns

    # And now let's get the feature importances
    importances = grid_search.best_estimator_.named_steps['classifier'].feature_importances_

    # Create a dataframe for feature importance
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

    # Sort the dataframe by importance
    importance_df = importance_df.sort_values('Importance', ascending=False)

    # Print the feature importances
    print(importance_df)
    if request.method == 'POST':
        # Retrieve the user input from the form
        age = float(request.POST['age'])
        gender = request.POST['gender']
        hypertension = int(request.POST['hypertension'])
        heart_disease = int(request.POST['heart_disease'])
        smoking_history = request.POST['smoking_history']
        bmi = float(request.POST['bmi'])
        hba1c_level = float(request.POST['hba1c_level'])
        blood_glucose_level = float(request.POST['blood_glucose_level'])

        # Create a numpy array with the user's data

        print(smoking_history)
        print(gender)
        user_data = np.array(
            (age,
             hypertension,
             heart_disease,
             bmi,
             hba1c_level,
             blood_glucose_level)
        ).reshape(1, 6)

        predictions = grid_search.predict(user_data)
        print('predictions')
        print(predictions)
        if int(predictions[0]) == 1:
            value = 'have'  # User is predicted to have heart disease
        elif int(predictions[0]) == 0:
            value = "don\'t have"  # User is predicted to not have heart disease

    return render(request,
                  'diabetes.html',
                  {
                      'context': value,
                      'title': 'Diabetes Disease Prediction',
                      'active': 'btn btn-success peach-gradient text-white',
                      'diabetes': True,
                      'form': DiabetesForm(),
                  })


def home(request):
    return render(request,
                  'home.html')

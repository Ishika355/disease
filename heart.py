import inline as inline
import matplotlib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# SVM Base Model Definition
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
svm_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(probability=True))
])
# SVM Hyperparameter Tuning
param_grid_svm = {
    'svm__C': [0.0011, 0.005, 0.01, 0.05, 0.1, 1, 10, 20],
    'svm__kernel': ['linear', 'rbf', 'poly'],
    'svm__gamma': ['scale', 'auto', 0.1, 0.5, 1, 5],
    'svm__degree': [2, 3, 4]
}

from sklearn.model_selection import GridSearchCV, StratifiedKFold
def tune_clf_hyperparameters(clf, param_grid, X_train, y_train, scoring='recall', n_splits=3):
    '''
    This function optimizes the hyperparameters for a classifier by searching over a specified hyperparameter grid.
    It uses GridSearchCV and cross-validation (StratifiedKFold) to evaluate different combinations of hyperparameters.
    The combination with the highest recall for class 1 is selected as the default scoring metric.
    The function returns the classifier with the optimal hyperparameters.
    '''

    # Create the cross-validation object using StratifiedKFold to ensure the class distribution is the same across all the folds
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)

    # Create the GridSearchCV object
    clf_grid = GridSearchCV(clf, param_grid, cv=cv, scoring=scoring, n_jobs=-1)

    # Fit the GridSearchCV object to the training data
    clf_grid.fit(X_train, y_train)

    # Get the best hyperparameters
    best_hyperparameters = clf_grid.best_params_

    # Return best_estimator_ attribute which gives us the best model that has been fitted to the training data
    return clf_grid.best_estimator_, best_hyperparameters

df = pd.read_csv('../heart.csv')
df
df.info()
X= df.drop('target', axis=1)
y= df['target']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2,random_state=42)
# Call the function for hyperparameter tuning
best_svm, best_svm_hyperparams = tune_clf_hyperparameters(svm_pipeline, param_grid_svm, X_train, y_train)
print('SVM Optimal Hyperparameters: \n', best_svm_hyperparams)
# SVM Model Evaluation
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from scipy.stats import boxcox
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


plt.rcParams['figure.dpi'] = 200

# Configure Seaborn plot styles: Set background color and use dark grid
sns.set(rc={'axes.facecolor': '#faded9'}, style='darkgrid')

# Define the continuous features
continuous_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

# Identify the features to be converted to object data type
features_to_convert = [feature for feature in df.columns if feature not in continuous_features]

# Convert the identified features to object data type
df[features_to_convert] = df[features_to_convert].astype('object')

df.dtypes
df.describe().T
df.describe(include='object')
# # Filter out continuous features for the univariate analysis
# df_continuous = df[continuous_features]
#
# # Set up the subplot
# fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
#
# # Loop to plot histograms for each continuous feature
# for i, col in enumerate(df_continuous.columns):
#     x = i // 3
#     y = i % 3
#     values, bin_edges = np.histogram(df_continuous[col],
#                                      range=(np.floor(df_continuous[col].min()), np.ceil(df_continuous[col].max())))
#
#     graph = sns.histplot(data=df_continuous, x=col, bins=bin_edges, kde=True, ax=ax[x, y],
#                          edgecolor='none', color='red', alpha=0.6, line_kws={'lw': 3})
#     ax[x, y].set_xlabel(col, fontsize=15)
#     ax[x, y].set_ylabel('Count', fontsize=12)
#     ax[x, y].set_xticks(np.round(bin_edges, 1))
#     ax[x, y].set_xticklabels(ax[x, y].get_xticks(), rotation=45)
#     ax[x, y].grid(color='lightgrey')
#
#     for j, p in enumerate(graph.patches):
#         ax[x, y].annotate('{}'.format(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height() + 1),
#                           ha='center', fontsize=10, fontweight="bold")
#
#     textstr = '\n'.join((
#         r'$\mu=%.2f$' % df_continuous[col].mean(),
#         r'$\sigma=%.2f$' % df_continuous[col].std()
#     ))
#     ax[x, y].text(0.75, 0.9, textstr, transform=ax[x, y].transAxes, fontsize=12, verticalalignment='top',
#                   color='white', bbox=dict(boxstyle='round', facecolor='#ff826e', edgecolor='white', pad=0.5))
#
# ax[1,2].axis('off')
# plt.suptitle('Distribution of Continuous Variables', fontsize=20)
# plt.tight_layout()
# plt.subplots_adjust(top=0.92)
# plt.show()
# Filter out categorical features for the univariate analysis
categorical_features = df.columns.difference(continuous_features)
df_categorical = df[categorical_features]
# # Set up the subplot for a 4x2 layout
# fig, ax = plt.subplots(nrows=5, ncols=2, figsize=(15, 18))
#
# # Loop to plot bar charts for each categorical feature in the 4x2 layout
# for i, col in enumerate(categorical_features):
#     row = i // 2
#     col_idx = i % 2
#
#     # Calculate frequency percentages
#     value_counts = df[col].value_counts(normalize=True).mul(100).sort_values()
#
#     # Plot bar chart
#     value_counts.plot(kind='barh', ax=ax[row, col_idx], width=0.8, color='red')
#
#     # Add frequency percentages to the bars
#     for index, value in enumerate(value_counts):
#         ax[row, col_idx].text(value, index, str(round(value, 1)) + '%', fontsize=15, weight='bold', va='center')
#
#     ax[row, col_idx].set_xlim([0, 95])
#     ax[row, col_idx].set_xlabel('Frequency Percentage', fontsize=12)
#     ax[row, col_idx].set_title(f'{col}', fontsize=20)
#
# ax[4,1].axis('off')
# plt.suptitle('Distribution of Categorical Variables', fontsize=22)
# plt.tight_layout()
# plt.subplots_adjust(top=0.95)
# plt.show()
# # Set color palette
# sns.set_palette(['#ff826e', 'red'])
#
# # Create the subplots
# fig, ax = plt.subplots(len(continuous_features), 2, figsize=(15,15), gridspec_kw={'width_ratios': [1, 2]})
#
# # Loop through each continuous feature to create barplots and kde plots
# for i, col in enumerate(continuous_features):
#     # Barplot showing the mean value of the feature for each target category
#     graph = sns.barplot(data=df, x="target", y=col, ax=ax[i,0])
#
#     # KDE plot showing the distribution of the feature for each target category
#     sns.kdeplot(data=df[df["target"]==0], x=col, fill=True, linewidth=2, ax=ax[i,1], label='0')
#     sns.kdeplot(data=df[df["target"]==1], x=col, fill=True, linewidth=2, ax=ax[i,1], label='1')
#     ax[i,1].set_yticks([])
#     ax[i,1].legend(title='Heart Disease', loc='upper right')
#
#     # Add mean values to the barplot
#     for cont in graph.containers:
#         graph.bar_label(cont, fmt='         %.3g')
#
# # Set the title for the entire figure
# plt.suptitle('Continuous Features vs Target Distribution', fontsize=22)
# plt.tight_layout()
# plt.show()
# Remove 'target' from the categorical_features
# categorical_features = [feature for feature in categorical_features if feature != 'target']
# fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(15,10))
#
# for i,col in enumerate(categorical_features):
#
#     # Create a cross tabulation showing the proportion of purchased and non-purchased loans for each category of the feature
#     cross_tab = pd.crosstab(index=df[col], columns=df['target'])
#
#     # Using the normalize=True argument gives us the index-wise proportion of the data
#     cross_tab_prop = pd.crosstab(index=df[col], columns=df['target'], normalize='index')
#
#     # Define colormap
#     cmp = ListedColormap(['#ff826e', 'red'])
#
#     # Plot stacked bar charts
#     x, y = i//4, i%4
#     cross_tab_prop.plot(kind='bar', ax=ax[x,y], stacked=True, width=0.8, colormap=cmp,
#                         legend=False, ylabel='Proportion', sharey=True)
#
#     # Add the proportions and counts of the individual bars to our plot
#     for idx, val in enumerate([*cross_tab.index.values]):
#         for (proportion, count, y_location) in zip(cross_tab_prop.loc[val],cross_tab.loc[val],cross_tab_prop.loc[val].cumsum()):
#             ax[x,y].text(x=idx-0.3, y=(y_location-proportion)+(proportion/2)-0.03,
#                          s = f'    {count}\n({np.round(proportion * 100, 1)}%)',
#                          color = "black", fontsize=9, fontweight="bold")
#
#     # Add legend
#     ax[x,y].legend(title='target', loc=(0.7,0.9), fontsize=8, ncol=2)
#     # Set y limit
#     ax[x,y].set_ylim([0,1.12])
#     # Rotate xticks
#     ax[x,y].set_xticklabels(ax[x,y].get_xticklabels(), rotation=0)
#
#
# plt.suptitle('Categorical Features vs Target Stacked Barplots', fontsize=22)
# plt.tight_layout()
# plt.show()
# Check for missing values in the dataset
df.isnull().sum().sum()
continuous_features
Q1 = df[continuous_features].quantile(0.25)
Q3 = df[continuous_features].quantile(0.75)
IQR = Q3 - Q1
outliers_count_specified = ((df[continuous_features] < (Q1 - 1.5 * IQR)) | (df[continuous_features] > (Q3 + 1.5 * IQR))).sum()

outliers_count_specified
# Implementing one-hot encoding on the specified categorical features
df_encoded = pd.get_dummies(df, columns=['cp', 'restecg', 'thal'], drop_first=True)

# Convert the rest of the categorical variables that don't need one-hot encoding to integer data type
features_to_convert = ['sex', 'fbs', 'exang', 'slope', 'ca', 'target']
for feature in features_to_convert:
    df_encoded[feature] = df_encoded[feature].astype(int)

df_encoded.dtypes

# Displaying the resulting DataFrame after one-hot encoding
df_encoded.head()
# Define the features (X) and the output labels (y)
X = df_encoded.drop('target', axis=1)
y = df_encoded['target']
# Splitting data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
continuous_features
# # Adding a small constant to 'oldpeak' to make all values positive
# X_train['oldpeak'] = X_train['oldpeak'] + 0.001
# X_test['oldpeak'] = X_test['oldpeak'] + 0.001
# # Checking the distribution of the continuous features
# fig, ax = plt.subplots(2, 5, figsize=(15,10))
#
# # Original Distributions
# for i, col in enumerate(continuous_features):
#     sns.histplot(X_train[col], kde=True, ax=ax[0,i], color='#ff826e').set_title(f'Original {col}')
#
#
# # Applying Box-Cox Transformation
# # Dictionary to store lambda values for each feature
# lambdas = {}
#
# for i, col in enumerate(continuous_features):
#     # Only apply box-cox for positive values
#     if X_train[col].min() > 0:
#         X_train[col], lambdas[col] = boxcox(X_train[col])
#         # Applying the same lambda to test data
#         X_test[col] = boxcox(X_test[col], lmbda=lambdas[col])
#         sns.histplot(X_train[col], kde=True, ax=ax[1,i], color='red').set_title(f'Transformed {col}')
#     else:
#         sns.histplot(X_train[col], kde=True, ax=ax[1,i], color='green').set_title(f'{col} (Not Transformed)')
#
# fig.tight_layout()
# plt.show()
# Define the base DT model
dt_base = DecisionTreeClassifier(random_state=0)
def tune_clf_hyperparameters(clf, param_grid, X_train, y_train, scoring='recall', n_splits=3):
    '''
    This function optimizes the hyperparameters for a classifier by searching over a specified hyperparameter grid.
    It uses GridSearchCV and cross-validation (StratifiedKFold) to evaluate different combinations of hyperparameters.
    The combination with the highest recall for class 1 is selected as the default scoring metric.
    The function returns the classifier with the optimal hyperparameters.
    '''

    # Create the cross-validation object using StratifiedKFold to ensure the class distribution is the same across all the folds
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)

    # Create the GridSearchCV object
    clf_grid = GridSearchCV(clf, param_grid, cv=cv, scoring=scoring, n_jobs=-1)

    # Fit the GridSearchCV object to the training data
    clf_grid.fit(X_train, y_train)

    # Get the best hyperparameters
    best_hyperparameters = clf_grid.best_params_

    # Return best_estimator_ attribute which gives us the best model that has been fitted to the training data
    return clf_grid.best_estimator_, best_hyperparameters
# Hyperparameter grid for DT
param_grid_dt = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [2,3],
    'min_samples_split': [2, 3, 4],
    'min_samples_leaf': [1, 2]
}
# Call the function for hyperparameter tuning
best_dt, best_dt_hyperparams = tune_clf_hyperparameters(dt_base, param_grid_dt, X_train, y_train)
print('DT Optimal Hyperparameters: \n', best_dt_hyperparams)
svm_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(probability=True))
])

# Step 9.2 | SVM Hyperparameter Tuning
# Let's configure the hyperparameters grid and employ the tune_clf_hyperparameters function to determine the best hyperparameters for our SVM pipeline:

param_grid_svm = {
    'svm__C': [0.0011, 0.005, 0.01, 0.05, 0.1, 1, 10, 20],
    'svm__kernel': ['linear', 'rbf', 'poly'],
    'svm__gamma': ['scale', 'auto', 0.1, 0.5, 1, 5],
    'svm__degree': [2, 3, 4]
}
# Call the function for hyperparameter tuning
best_svm, best_svm_hyperparams = tune_clf_hyperparameters(svm_pipeline, param_grid_svm, X_train, y_train)
print('SVM Optimal Hyperparameters: \n', best_svm_hyperparams)
def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluates the performance of a trained model on test data using various metrics.
    """
    # Make predictions
    y_pred = model.predict(X_test)

    # Get classification report
    report = classification_report(y_test, y_pred, output_dict=True)

    # Extracting metrics
    metrics = {
        "precision_0": report["0"]["precision"],
        "precision_1": report["1"]["precision"],
        "recall_0": report["0"]["recall"],
        "recall_1": report["1"]["recall"],
        "f1_0": report["0"]["f1-score"],
        "f1_1": report["1"]["f1-score"],
        "macro_avg_precision": report["macro avg"]["precision"],
        "macro_avg_recall": report["macro avg"]["recall"],
        "macro_avg_f1": report["macro avg"]["f1-score"],
        "accuracy": accuracy_score(y_test, y_pred)
    }

    # Convert dictionary to dataframe
    df = pd.DataFrame(metrics, index=[model_name]).round(2)

    return df
print(classification_report(y_train, best_svm.predict(X_train)))
print(classification_report(y_test, best_svm.predict(X_test)))
svm_evaluation = evaluate_model(best_svm, X_test, y_test, 'SVM')

print(svm_evaluation)

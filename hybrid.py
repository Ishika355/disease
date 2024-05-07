import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Step 1: Load the dataset
from sklearn.tree import DecisionTreeClassifier

heart_data = pd.read_csv("../heart.csv")

# Step 2: Data Preprocessing
# Assuming the target variable is 'target' and other columns are features
X = heart_data.drop(columns=['target'])
y = heart_data['target']

# Step 3: Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Support Vector Machine (SVM) Model Model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)

# Step 5: LogisticRegression Model
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

# Step 6: DecisionTreeClassifier Model
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)

# Step 7: Random Forest Model
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)


# Step 9: GaussianNB Model
gnb_model = GaussianNB()
gnb_model.fit(X_train, y_train)
gnb_pred = gnb_model.predict(X_test)



# Step 12: ExtraTreesClassifier Model
from sklearn.ensemble import ExtraTreesClassifier

et_model = ExtraTreesClassifier()
et_model.fit(X_train, y_train)
et_pred = et_model.predict(X_test)

# Blending Predictions
# blended_pred = et_pred
blended_pred = np.round((svm_pred + lr_pred + gnb_pred + rf_pred + dt_pred + et_pred) / 6)

# np.round((svm_pred + lr_pred) / 2)
# Now, proceed with evaluating the blended predictions
accuracy = accuracy_score(y_test, blended_pred)
print("Blended Model Accuracy:", accuracy)

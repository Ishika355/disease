import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from tkinter import ttk
import tkinter as tk

from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('pima.csv')

print(df)
X = df.drop('Class', axis=1)
y = df[['Class']]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# from sklearn.linear_model import LogisticRegression
# Instantiate and train the K Nearest Neighbors classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

model = knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
print('accuracy_score(y_pred,y_test)')
print(accuracy_score(y_pred,y_test))
import pickle

Model = pickle.dumps(model)

win = tk.Tk()

win.title('Diabetes Predictions')

# Define a custom style for the entry boxes
style = ttk.Style()
style.configure('Custom.TEntry', padding=(10, 5))  # Adjust the padding as needed

# Define GUI components (labels and entry boxes)
#Column 1
Preg=ttk.Label(win,text="Preg")
Preg.grid(row=0,column=0,sticky=tk.W)
Preg_var=tk.StringVar()
Preg_entrybox=ttk.Entry(win, width=16, textvariable=Preg_var, style='Custom.TEntry', background='lightblue')
Preg_entrybox.grid(row=0,column=1)
#Column 2
Plas=ttk.Label(win,text="Plas")
Plas.grid(row=1,column=0,sticky=tk.W)
Plas_var=tk.StringVar()
Plas_entrybox=ttk.Entry(win, width=16, textvariable=Plas_var, style='Custom.TEntry', background='lightgreen')
Plas_entrybox.grid(row=1,column=1)
#Column 3
Pres=ttk.Label(win,text="Pres")
Pres.grid(row=2,column=0,sticky=tk.W)
Pres_var=tk.StringVar()
Pres_entrybox=ttk.Entry(win, width=16, textvariable=Pres_var, style='Custom.TEntry', background='lightyellow')
Pres_entrybox.grid(row=2,column=1)
#Column 4
Skin=ttk.Label(win,text="Skin")
Skin.grid(row=3,column=0,sticky=tk.W)
Skin_var=tk.StringVar()
Skin_entrybox=ttk.Entry(win, width=16, textvariable=Skin_var, style='Custom.TEntry', background='lightcoral')
Skin_entrybox.grid(row=3,column=1)
#Column 5
Insu=ttk.Label(win,text="Insu")
Insu.grid(row=4,column=0,sticky=tk.W)
Insu_var=tk.StringVar()
Insu_entrybox=ttk.Entry(win, width=16, textvariable=Insu_var, style='Custom.TEntry', background='lightpink')
Insu_entrybox.grid(row=4,column=1)
#Column 6
Mass=ttk.Label(win,text="Mass")
Mass.grid(row=5,column=0,sticky=tk.W)
Mass_var=tk.StringVar()
Mass_entrybox=ttk.Entry(win, width=16, textvariable=Mass_var, style='Custom.TEntry', background='lightblue')
Mass_entrybox.grid(row=5,column=1)
#Column 7
Pedi=ttk.Label(win,text="Pedi")
Pedi.grid(row=6,column=0,sticky=tk.W)
Pedi_var=tk.StringVar()
Pedi_entrybox=ttk.Entry(win, width=16, textvariable=Pedi_var, style='Custom.TEntry', background='lightgreen')
Pedi_entrybox.grid(row=6,column=1)
#Column 8
Age=ttk.Label(win,text="Age")
Age.grid(row=7,column=0,sticky=tk.W)
Age_var=tk.StringVar()
Age_entrybox=ttk.Entry(win, width=16, textvariable=Age_var, style='Custom.TEntry', background='lightyellow')
Age_entrybox.grid(row=7,column=1)

import pandas as pd
DF = pd.DataFrame()
def action():
    global DB
    import pandas as pd
    DF = pd.DataFrame(columns=['Preg','Plas','Pres','Skin','Insu','Mass','Pedi','Age'])
    PREG=Preg_var.get()
    print(Preg_var.get())
    DF.loc[0,'Preg']=PREG
    PLAS=Plas_var.get()
    DF.loc[0,'Plas']=PLAS
    PRES=Pres_var.get()
    DF.loc[0,'Pres']=PRES
    SKIN=Skin_var.get()
    DF.loc[0,'Skin']=SKIN
    INSU=Insu_var.get()
    DF.loc[0,'Insu']=INSU
    MASS=Mass_var.get()
    DF.loc[0,'Mass']=MASS
    PEDI=Pedi_var.get()
    DF.loc[0,'Pedi']=PEDI
    AGE=Age_var.get()
    DF.loc[0,'Age']=AGE

    print(DF.shape)
    print(DF)
    DB=DF
def Output():
    action()
    required_columns = ['Preg','Plas','Pres','Skin','Insu','Mass','Pedi','Age']
    print(DB)
    missing_columns = set(required_columns) - set(DB.columns)
    if missing_columns:
        print("Error: Missing columns -", missing_columns)
        return
    # Call action to update DB with entry values
    DB["Preg"] = pd.to_numeric(DB["Preg"])
    DB["Plas"] = pd.to_numeric(DB["Plas"])
    DB["Pres"] = pd.to_numeric(DB["Pres"])
    DB["Skin"] = pd.to_numeric(DB["Skin"])
    DB["Insu"] = pd.to_numeric(DB["Insu"])
    DB["Mass"] = pd.to_numeric(DB["Mass"])
    DB["Pedi"] = pd.to_numeric(DB["Pedi"])
    DB["Age"] = pd.to_numeric(DB["Age"])
    output = model.predict(DB)
    result = 'Diabetic' if output == ['tested_positive'] else 'Non-Diabetic'
    Predict_entrybox = ttk.Entry(win, width=16, style='Custom.TEntry', background='lightgrey')
    Predict_entrybox.grid(row=20, column=1)
    Predict_entrybox.insert(1, str(result))
    Accuracy_entrybox = ttk.Entry(win, width=16, style='Custom.TEntry', background='lightgrey')
    Accuracy_entrybox.grid(row=22, column=1)
    print(accuracy_score)
    print(accuracy_score(y_pred,y_test))
    print(accuracy_score(y_pred,y_test)*100)
    Accuracy_entrybox.insert(1, f'Accuracy: {round(accuracy_score(y_pred, y_test) * 100, 2)}% ')
Predict_button = ttk.Button(win, text="Predict", command=Output)
Predict_button.grid(row=20, column=0)
win.mainloop()

import pickle

Model = pickle.dumps(model)

win = tk.Tk()

win.title('Diabetes Predictions')

# Define a custom style for the entry boxes
style = ttk.Style()
style.configure('Custom.TEntry', padding=(10, 5))  # Adjust the padding as needed

# Define GUI components (labels and entry boxes)
# Column 1
bmi = ttk.Label(win, text="bmi")
bmi.grid(row=0, column=0, sticky=tk.W)
bmi_var = tk.StringVar()
bmi_entrybox = ttk.Entry(win, width=16, textvariable=bmi_var, style='Custom.TEntry', background='lightblue')
bmi_entrybox.grid(row=0, column=1)
# Column 2
HbA1c_level = ttk.Label(win, text="HbA1c_level")
HbA1c_level.grid(row=1, column=0, sticky=tk.W)
HbA1c_level_var = tk.StringVar()
HbA1c_level_entrybox = ttk.Entry(win, width=16, textvariable=HbA1c_level_var, style='Custom.TEntry',
                                 background='lightgreen')
HbA1c_level_entrybox.grid(row=1, column=1)
# Column 3
blood_glucose_level = ttk.Label(win, text="blood_glucose_level")
blood_glucose_level.grid(row=2, column=0, sticky=tk.W)
blood_glucose_level_var = tk.StringVar()
blood_glucose_level_entrybox = ttk.Entry(win, width=16, textvariable=blood_glucose_level_var, style='Custom.TEntry',
                                         background='lightyellow')
blood_glucose_level_entrybox.grid(row=2, column=1)
# Column 4
hypertension = ttk.Label(win, text="hypertension")
hypertension.grid(row=3, column=0, sticky=tk.W)
hypertension_var = tk.StringVar()
hypertension_entrybox = ttk.Entry(win, width=16, textvariable=hypertension_var, style='Custom.TEntry',
                                  background='lightcoral')
hypertension_entrybox.grid(row=3, column=1)
# Column 5
heart_disease = ttk.Label(win, text="heart_disease")
heart_disease.grid(row=4, column=0, sticky=tk.W)
heart_disease_var = tk.StringVar()
heart_disease_entrybox = ttk.Entry(win, width=16, textvariable=heart_disease_var, style='Custom.TEntry',
                                   background='lightpink')
heart_disease_entrybox.grid(row=4, column=1)
# Column 5
gender = ttk.Label(win, text="gender")
gender.grid(row=5, column=0, sticky=tk.W)
gender_var = tk.StringVar()
gender_entrybox = ttk.Entry(win, width=16, textvariable=gender_var, style='Custom.TEntry', background='lightpink')
gender_entrybox.grid(row=5, column=1)
# Column 5
smoking_history = ttk.Label(win, text="smoking_history")
smoking_history.grid(row=6, column=0, sticky=tk.W)
smoking_history_var = tk.StringVar()
smoking_history_entrybox = ttk.Entry(win, width=16, textvariable=smoking_history_var, style='Custom.TEntry',
                                     background='lightpink')
smoking_history_entrybox.grid(row=6, column=1)
# Column 8
age = ttk.Label(win, text="age")
age.grid(row=7, column=0, sticky=tk.W)
age_var = tk.StringVar()
age_entrybox = ttk.Entry(win, width=16, textvariable=age_var, style='Custom.TEntry', background='lightyellow')
age_entrybox.grid(row=7, column=1)

import pandas as pd

DF = pd.DataFrame()


def action():
    global DB
    import pandas as pd
    DF = pd.DataFrame(
        columns=['age', 'bmi', 'HbA1c_level', 'blood_glucose_level', 'hypertension', 'heart_disease', 'smoking_history',
                 'gender'])
    bmi = bmi_var.get()
    DF.loc[0, 'bmi'] = bmi
    HbA1c_level = HbA1c_level_var.get()
    DF.loc[0, 'HbA1c_level'] = HbA1c_level
    blood_glucose_level = blood_glucose_level_var.get()
    DF.loc[0, 'blood_glucose_level'] = blood_glucose_level
    hypertension = hypertension_var.get()
    DF.loc[0, 'hypertension'] = hypertension
    smoking_history = smoking_history_var.get()
    DF.loc[0, 'smoking_history'] = smoking_history
    heart_disease = heart_disease_var.get()
    DF.loc[0, 'heart_disease'] = heart_disease
    gender = gender_var.get()
    DF.loc[0, 'gender'] = gender
    age = age_var.get()
    DF.loc[0, 'age'] = age
    DB = DF


def Output():
    action()
    required_columns = [
        'age', 'bmi', 'HbA1c_level', 'blood_glucose_level', 'hypertension', 'heart_disease', 'smoking_history',
        'gender']
    missing_columns = set(required_columns) - set(DB.columns)
    if missing_columns:
        print("Error: Missing columns -", missing_columns)
        return
    # Call action to update DB with entry values
    DB["bmi"] = pd.to_numeric(DB["bmi"])
    DB["HbA1c_level"] = pd.to_numeric(DB["HbA1c_level"])
    DB["blood_glucose_level"] = pd.to_numeric(DB["blood_glucose_level"])
    DB["hypertension"] = pd.to_numeric(DB["hypertension"])
    DB["heart_disease"] = pd.to_numeric(DB["heart_disease"])
    DB["smoking_history"] = pd.to_numeric(DB["smoking_history"])
    DB["gender"] = pd.to_numeric(DB["gender"])
    DB["age"] = pd.to_numeric(DB["age"])
    output = model.predict(DB)
    result = 'Diabetic' if output == ['tested_positive'] else 'Non-Diabetic'
    Predict_entrybox = ttk.Entry(win, width=16, style='Custom.TEntry', background='lightgrey')
    Predict_entrybox.grid(row=20, column=1)
    Predict_entrybox.insert(1, str(result))
    Accuracy_entrybox = ttk.Entry(win, width=16, style='Custom.TEntry', background='lightgrey')
    Accuracy_entrybox.grid(row=22, column=1)
    print(accuracy_score)
    print(accuracy_score(y_pred, y_test))
    print(accuracy_score(y_pred, y_test) * 100)
    Accuracy_entrybox.insert(1, f'Accuracy: {round(accuracy_score(y_pred, y_test) * 100, 2)}% ')


Predict_button = ttk.Button(win, text="Predict", command=Output)
Predict_button.grid(row=20, column=0)
win.mainloop()

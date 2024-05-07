from django import forms


class HeartDiseaseForm(forms.Form):
    # Define form fields for heart disease prediction

    age = forms.FloatField(label='Age', widget=forms.NumberInput(attrs={'class': 'form-control'}))
    # Field for age, represented as a float input widget
    sex = forms.FloatField(label='Sex', widget=forms.NumberInput(attrs={'class': 'form-control'}))
    # Field for sex, represented as a float input widget

    cp = forms.FloatField(label='Chest Paint Type', widget=forms.NumberInput(attrs={'class': 'form-control'}))
    # Field for chest pain type (CP), represented as a float input widget

    trestbps = forms.FloatField(label='Resting blood pressure(mm Hg)', widget=forms.NumberInput(attrs={'class': 'form-control'}))
    # Field for resting blood pressure (TRESTBPS), represented as a float input widget

    chol = forms.FloatField(label='Serum cholesterol(mg/dl)', widget=forms.NumberInput(attrs={'class': 'form-control'}))
    # Field for serum cholesterol level (CHOL), represented as a float input widget

    fbs = forms.FloatField(label='Fasting blood sugar level', widget=forms.NumberInput(attrs={'class': 'form-control'}))
    # Field for fasting blood sugar (FBS), represented as a float input widget

    restecg = forms.FloatField(label='Resting electrocardiographic results', widget=forms.NumberInput(attrs={'class': 'form-control'}))
    # Field for resting electrocardiographic results (RESTECG), represented as a float input widget

    thalach = forms.FloatField(label='Maximum heart rate achieved during a stress test', widget=forms.NumberInput(attrs={'class': 'form-control'}))
    # Field for maximum heart rate achieved (THALACH), represented as a float input widget

    exang = forms.FloatField(label='Exercise-induced angina', widget=forms.NumberInput(attrs={'class': 'form-control'}))
    # Field for exercise-induced angina (EXANG), represented as a float input widget

    oldpeak = forms.FloatField(label='ST depression induced by exercise relative to rest', widget=forms.NumberInput(attrs={'class': 'form-control'}))
    # Field for ST depression induced by exercise relative to rest (OLDPEAK), represented as a float input widget

    slope = forms.FloatField(label='Slope of the peak exercise ST segment', widget=forms.NumberInput(attrs={'class': 'form-control'}))
    # Field for the slope of the peak exercise ST segment (SLOPE), represented as a float input widget

    ca = forms.FloatField(label='Number of major vessels (0-4) colored by fluoroscopy', widget=forms.NumberInput(attrs={'class': 'form-control'}))
    # Field for the number of major vessels colored by fluoroscopy (CA), represented as a float input widget

    thal = forms.FloatField(label='Thalium stress test result:', widget=forms.NumberInput(attrs={'class': 'form-control'}))
    # Field for thalassemia (THAL), represented as a float input widget


class DiabetesForm(forms.Form):
    age = forms.FloatField(label='Age', widget=forms.NumberInput(attrs={'class': 'form-control'}))
    # Field for age, represented as a float input widget

    gender = forms.ChoiceField(label='Gender', choices=(('male', 'Male'), ('female', 'Female'), ('other', 'Other')),
                               widget=forms.Select(attrs={'class': 'form-control'}))
    # Field for gender, represented as a choice input widget

    hypertension = forms.ChoiceField(label='Hypertension', choices=((0, 'No'), (1, 'Yes')),
                                     widget=forms.Select(attrs={'class': 'form-control'}))
    # Field for hypertension, represented as a choice input widget

    heart_disease = forms.ChoiceField(label='Heart Disease', choices=((0, 'No'), (1, 'Yes')),
                                      widget=forms.Select(attrs={'class': 'form-control'}))
    # Field for heart disease, represented as a choice input widget

    smoking_history = forms.ChoiceField(label='Smoking History', choices=(
    ('not current', 'Not Current'), ('former', 'Former'), ('No Info', 'No Info'), ('current', 'Current'),
    ('never', 'Never'), ('ever', 'Ever')), widget=forms.Select(attrs={'class': 'form-control'}))
    # Field for smoking history, represented as a choice input widget

    bmi = forms.FloatField(label='BMI', widget=forms.NumberInput(attrs={'class': 'form-control'}))
    # Field for BMI, represented as a float input widget

    hba1c_level = forms.FloatField(label='HbA1c Level', widget=forms.NumberInput(attrs={'class': 'form-control'}))
    # Field for HbA1c level, represented as a float input widget

    blood_glucose_level = forms.FloatField(label='Blood Glucose Level',
                                           widget=forms.NumberInput(attrs={'class': 'form-control'}))
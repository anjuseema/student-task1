from django.shortcuts import render
from .forms import StudentForm
import joblib
import pandas as pd
from .models import Student

# Load your trained model
# path = r'C:\Users\anjuv\OneDrive\Desktop\student_django_task\model.pkl'
# model = joblib.load(path)
import pickle
model = pickle.load(open('model1.pkl','rb'))

def predict_grade(request):
    if request.method == 'POST':
        form = StudentForm(request.POST)
        if form.is_valid():
            data = form.cleaned_data
            input_data = {
                'Ethnicity': [data['ethnicity']],
                'ParentalEducation': [data['parental_education']],
                'StudyTimeWeekly': [data['study_time_weekly']],
                'Absences': [data['absences']],
                'Tutoring': [data['tutoring']],
                'ParentalSupport': [data['parental_support']],
                'Extracurricular': [data['extracurricular']],
                'Sports': [data['sports']],
                'Music': [data['music']],
                'Volunteering': [data['volunteering']],
                'GPA': [data['gpa']]
            }
            df = pd.DataFrame(input_data)
            prediction = model.predict(df)
            grade = dict(Student.GRADE_CHOICES).get(prediction[0])
            return render(request, 'predictions/result.html', {'grade': grade})
    else:
        form = StudentForm()
    return render(request, 'predictions/predict.html', {'form': form})


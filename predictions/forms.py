from django import forms
from .models import Student

class StudentForm(forms.ModelForm):
    class Meta:
        model = Student
        fields = [
            'ethnicity', 'parental_education', 'study_time_weekly', 'absences',
            'tutoring', 'parental_support', 'extracurricular', 'sports', 
            'music', 'volunteering', 'gpa'
        ]

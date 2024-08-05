from django.db import models

# Create your models here.
from django.db import models

class Student(models.Model):
    ETHNICITY_CHOICES = [
        (0, 'Caucasian'),
        (1, 'African American'),
        (2, 'Asian'),
        (3, 'Other'),
    ]

    EDUCATION_CHOICES = [
        (0, 'None'),
        (1, 'High School'),
        (2, 'Some College'),
        (3, 'Bachelor\'s'),
        (4, 'Higher'),
    ]

    SUPPORT_CHOICES = [
        (0, 'None'),
        (1, 'Low'),
        (2, 'Moderate'),
        (3, 'High'),
        (4, 'Very High'),
    ]

    BOOLEAN_CHOICES = [
        (0, 'No'),
        (1, 'Yes'),
    ]

    GRADE_CHOICES = [
        (0, 'A'),
        (1, 'B'),
        (2, 'C'),
        (3, 'D'),
        (4, 'F'),
    ]

    ethnicity = models.IntegerField(choices=ETHNICITY_CHOICES)
    parental_education = models.IntegerField(choices=EDUCATION_CHOICES)
    study_time_weekly = models.IntegerField()
    absences = models.IntegerField()
    tutoring = models.IntegerField(choices=BOOLEAN_CHOICES)
    parental_support = models.IntegerField(choices=SUPPORT_CHOICES)
    extracurricular = models.IntegerField(choices=BOOLEAN_CHOICES)
    sports = models.IntegerField(choices=BOOLEAN_CHOICES)
    music = models.IntegerField(choices=BOOLEAN_CHOICES)
    volunteering = models.IntegerField(choices=BOOLEAN_CHOICES)
    gpa = models.FloatField()
    grade_class = models.IntegerField(choices=GRADE_CHOICES)

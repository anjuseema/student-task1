import pandas as pd
# Load Data
df = pd.read_csv(r"C:\Users\anjuv\Downloads\Student_performance_data _.csv")
print(df)
print(df.head(5))
print(df.shape)
print(df.info())
print(df.isnull().sum())
for i in df.columns:
    print(df[i].unique())
corr_matrix = df.corr()
correlation_threshold = 0.023
low_corr_features = [col for col in corr_matrix.columns if abs(corr_matrix["GradeClass"][col]) < correlation_threshold]
low_corr_features

from sklearn.svm import SVC
from sklearn.datasets import make_classification

from yellowbrick.model_selection import RFECV
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier


x = df.drop(columns=["GradeClass"])
y = df['GradeClass']
# Create a dataset with only 3 informative features
x, y = make_classification(
    n_samples=1000, n_features=15, n_informative=12, n_redundant=2,
    n_repeated=0, n_classes=4, n_clusters_per_class=1, random_state=0
)

# Instantiate RFECV visualizer with a linear SVM classifier
visualizer = RFECV(SVC(kernel='linear', C=1))

visualizer.fit(x, y)        # Fit the data to the visualizer
visualizer.show()           # Finalize and render the figure

import matplotlib.pyplot as plt
import seaborn as sns
correlation_matrix = df.corr()
plt.figure(figsize=(10, 10))
sns.heatmap(correlation_matrix,annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

print(df.columns)
columns=[ 'Age', 'Gender', 'Ethnicity', 'ParentalEducation',
        'Absences', 'Tutoring', 'ParentalSupport',
       'Extracurricular', 'Sports', 'Music', 'Volunteering',
       'GradeClass']
for i in columns:
    df[i].value_counts().plot(kind = 'bar')
    plt.show()
    


sns.kdeplot(df['StudentID'])

df['Age'].drop
df['Gender'].drop
df1 = df.drop(['Gender','Age'], axis=1)
print(df1.head())
df2 = df1.drop(['StudentID'], axis=1)
print(df2.isnull().sum())

x = df2.drop(['GradeClass'],axis = 1)
y = df2['GradeClass']
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=20)

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import cross_val_score
models = {
    'Logistic Regression': LogisticRegression(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Support Vector Machine': SVC(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'Neural Network': MLPClassifier()
}
results = {}
for name, model in models.items():
    scores = cross_val_score(model, x_train, y_train, cv=5, scoring='accuracy')
    results[name] = scores
    print(f"{name}: {scores.mean():.4f} ± {scores.std():.4f}")
    
from sklearn.model_selection import GridSearchCV
param_grid = {
    'n_estimators': [1,5,10],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(x_train, y_train)
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score:", grid_search.best_score_)

model=RandomForestClassifier()
model.fit(x_train, y_train)
from sklearn.metrics import classification_report, accuracy_score,confusion_matrix

y_pred = model.predict(x_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

def grade_from_gpa(y_pred):
    if y_pred >= 3.5:
        return 'A'
    elif 3.0 <= y_pred < 3.5:
        return 'B'
    elif 2.5 <= y_pred < 3.0:
        return 'C'
    elif 2.0 <= y_pred < 2.5:
        return 'D'
    else:
        return 'F'
grade = []
for item in y_pred:
    df3 = grade_from_gpa(item)
    grade.append(df3)

x_test['Grade'] = grade



# import joblib
# joblib.dump(model, 'model.pkl')

import pickle
pickle.dump(model, open('model1.pkl','wb'))
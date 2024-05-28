import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import pickle
df = pd.read_csv(r'khktmain-main\data.csv')
X = df.drop('class', axis=1) 
y = df['class'] 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)
X_train = X_train.dropna()
y_train = y_train.loc[X_train.index]
X_test = X_test.dropna()
y_test = y_test.loc[X_test.index]
pipelines = {
    'lr': Pipeline([('s', SimpleImputer()), ('ss', StandardScaler()), ('lr', LogisticRegression())]),
    'rc': Pipeline([('s', SimpleImputer()), ('ss', StandardScaler()), ('rc', RidgeClassifier())]),
    'rf': Pipeline([('s', SimpleImputer()), ('ss', StandardScaler()), ('rf', RandomForestClassifier())]),
    'gb': Pipeline([('s', SimpleImputer()), ('ss', StandardScaler()), ('gb', GradientBoostingClassifier())]),
}

best_accuracy = 0.0
best_classifier = ''
best_model = None
for name, pipeline in pipelines.items():
    model = pipeline.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_classifier = name
        best_model = model

with open('body_language.pkl', 'wb') as f:
    pickle.dump(best_model, f)
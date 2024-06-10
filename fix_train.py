import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import pickle
from tqdm import tqdm

# Load data
df = pd.read_csv(r'data.csv')

# Drop rows with missing values
df.dropna(inplace=True)

# Split data
X = df.drop('class', axis=1)  # features
y = df['class']  # target value

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)

# Define numerical features
numeric_features = X.select_dtypes(include=['float64', 'int64']).columns

# Define preprocessing steps
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features)])

# Define classifiers
classifiers = {
    'lr': LogisticRegression(),
    'rc': RidgeClassifier(),
    'rf': RandomForestClassifier(),
    'gb': GradientBoostingClassifier()
}

best_accuracy = 0
best_classifier = None
best_model = None

for name, classifier in tqdm(classifiers.items(), desc="Training models"):
    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('classifier', classifier)])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_classifier = name
        best_model = model

# Save the best model
with open('body_language.pkl', 'wb') as f:
    pickle.dump(best_model, f)

print(f'The best model is {best_classifier} with accuracy {best_accuracy}.')

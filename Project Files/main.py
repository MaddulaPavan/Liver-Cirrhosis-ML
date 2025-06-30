import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression, RidgeClassifier, LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import joblib
import kagglehub
from flask import Flask, request, render_template_string

path = kagglehub.dataset_download("bhavanipriya222/liver-cirrhosis-prediction")
print("Path to dataset files:", path)
csv_files = glob.glob(os.path.join(path, '*.csv'))
if not csv_files:
    raise FileNotFoundError('No CSV file found in the dataset directory.')
dataset_path = csv_files[0]
print(f"Using dataset: {dataset_path}")
df = pd.read_csv(dataset_path)
print("Columns in dataset:", df.columns.tolist())
df = df.dropna()
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def models_eval_mm(X_train, X_test, y_train, y_test):
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000),
        'LogisticRegressionCV': LogisticRegressionCV(max_iter=1000),
        'XGBClassifier': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        'RidgeClassifier': RidgeClassifier(),
        'KNeighborsClassifier': KNeighborsClassifier(),
        'RandomForestClassifier': RandomForestClassifier(),
        'DecisionTreeClassifier': DecisionTreeClassifier()
    }
    param_grids = {
        'LogisticRegression': {'C': [0.1, 1, 10]},
        'LogisticRegressionCV': {'Cs': [1, 5, 10]},
        'XGBClassifier': {'n_estimators': [50, 100], 'max_depth': [3, 5]},
        'RidgeClassifier': {'alpha': [0.1, 1, 10]},
        'KNeighborsClassifier': {'n_neighbors': [3, 5, 7]},
        'RandomForestClassifier': {'n_estimators': [50, 100], 'max_depth': [3, 5]},
        'DecisionTreeClassifier': {'max_depth': [3, 5, 7]}
    }
    results = {}
    for name, model in models.items():
        print(f"\nOptimizing {name}...")
        grid = GridSearchCV(model, param_grids[name], cv=3, n_jobs=-1, error_score='raise')
        grid.fit(X_train, y_train)
        y_pred_train = grid.predict(X_train)
        y_pred_test = grid.predict(X_test)
        train_score = accuracy_score(y_train, y_pred_train)
        test_score = accuracy_score(y_test, y_pred_test)
        results[name] = {
            'model': grid.best_estimator_,
            'train_score': train_score,
            'test_score': test_score,
            'best_params': grid.best_params_
        }
        print(f"{name}: Train Score = {train_score:.3f}, Test Score = {test_score:.3f}, Best Params = {grid.best_params_}")
    return results

results = models_eval_mm(X_train, X_test, y_train, y_test)
best_model_name = max(results, key=lambda k: results[k]['test_score'])
best_model = results[best_model_name]['model']
print(f"\nBest model: {best_model_name}")
joblib.dump(best_model, 'best_model.pkl')

app = Flask(__name__)

def get_form_html():
    fields = X.columns.tolist()
    form_fields = '\n'.join([
        f'<label>{col}: <input name="{col}" required></label><br>' for col in fields
    ])
    return f'''
    <h2>Liver Cirrhosis Prediction</h2>
    <form method="post">
        {form_fields}
        <input type="submit" value="Predict">
    </form>
    '''

@app.route('/', methods=['GET', 'POST'])
def predict():
    prediction = None
    if request.method == 'POST':
        try:
            input_data = [float(request.form[col]) for col in X.columns]
            model = joblib.load('best_model.pkl')
            pred = model.predict([input_data])[0]
            prediction = f"Prediction: {'Cirrhosis' if pred == 1 else 'No Cirrhosis'}"
        except Exception as e:
            prediction = f"Error: {e}"
    return render_template_string(get_form_html() + (f'<h3>{prediction}</h3>' if prediction else ''))

if __name__ == '__main__':
    app.run(debug=True)

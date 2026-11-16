import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None

FEATURE_COLUMNS = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
    'exang', 'oldpeak', 'slope', 'ca', 'thal'
]
TARGET_COLUMN = 'target'
MODEL_PATH = Path(__file__).resolve().parent / 'heart_prediction_model.pkl'


def load_data():
    csv_path = Path(__file__).resolve().parent / 'heart.csv'
    df = pd.read_csv(csv_path)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    return X, y


def build_models():
    models = [
        ('decision_tree', DecisionTreeClassifier(random_state=42, max_depth=3, min_samples_leaf=5)),
        ('gradient_boosting', GradientBoostingClassifier(random_state=42, n_estimators=200, learning_rate=0.05, max_depth=3)),
    ]

    if XGBClassifier is not None:
        models.insert(0, (
            'xgboost',
            XGBClassifier(
                n_estimators=300,
                max_depth=3,
                learning_rate=0.05,
                objective='binary:logistic',
                eval_metric='logloss',
                random_state=42,
                use_label_encoder=False,
            )
        ))

    return models


def train_and_save_model(output_path=MODEL_PATH):
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    best_name = None
    best_model = None
    best_score = -1.0

    for name, model in build_models():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        score = accuracy_score(y_test, predictions)
        print(f'{name}: {score:.4f}')

        if score > best_score:
            best_name = name
            best_model = model
            best_score = score

    if best_model is None:
        raise ValueError('No model could be trained from the available algorithms.')

    bundle = {
        'model_name': best_name,
        'model': best_model,
        'feature_names': FEATURE_COLUMNS,
        'accuracy': best_score,
    }

    with open(output_path, 'wb') as model_file:
        pickle.dump(bundle, model_file)

    print(f'Best model saved: {best_name} with accuracy {best_score:.4f}')
    return bundle


if __name__ == '__main__':
    train_and_save_model()

import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Charger les données
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Démarrer un run MLflow
#with mlflow.start_run():
#    clf = RandomForestClassifier(n_estimators=10)
#    clf.fit(X_train, y_train)
#
#    preds = clf.predict(X_test)
#    acc = accuracy_score(y_test, preds)
#
#    # Enregistrer une métrique
#    mlflow.log_param("n_estimators", n_estimators)
#    mlflow.log_metric("accuracy", acc)
#
#    # Enregistrer le modèle
#    mlflow.sklearn.log_model(clf, "model")

mlflow.autolog()

with mlflow.start_run():
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
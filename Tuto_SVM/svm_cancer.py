from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report

# 1. Chargement des données (Cancer du sein - binaire)
cancer = datasets.load_breast_cancer()
X = cancer.data
y = cancer.target

# 2. Séparation train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Création du Pipeline (Mise à l'échelle + SVM)
# C'est la bonne pratique : on scale, puis on entraîne.
# On utilise ici un noyau RBF par défaut.
model = make_pipeline(StandardScaler(), svm.SVC(kernel='rbf', C=1.0, gamma='scale'))

# 4. Entraînement
model.fit(X_train, y_train)

# 5. Prédiction et rapport
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=cancer.target_names))

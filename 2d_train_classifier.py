import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

input_csv = "features_labeled.csv"
model_output = "queue_classifier.pkl"

df = pd.read_csv(input_csv)
df = df[df["label"] != "UNLABELED"]

X = df[["density", "occupancy"]].values
y = df["label"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))
print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))

cv_scores = cross_val_score(clf, X_scaled, y, cv=5)
print(f"\n=== Cross-Validation Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

importances = clf.feature_importances_
feature_names = ["density", "occupancy"]
print("\n=== Feature Importance ===")
for name, imp in zip(feature_names, importances):
    print(f"{name}: {imp:.4f}")

joblib.dump({"model": clf, "scaler": scaler}, model_output)
print(f"\n[INFO] Model tersimpan di {model_output}")

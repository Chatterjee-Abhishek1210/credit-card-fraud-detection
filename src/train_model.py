from sklearn.ensemble import RandomForestClassifier
import joblib

def train_rf(X_train, y_train):
    clf = RandomForestClassifier(class_weight='balanced', random_state=42, n_estimators=100)
    clf.fit(X_train, y_train)
    return clf

def save_model(model, filename='model.pkl'):
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")

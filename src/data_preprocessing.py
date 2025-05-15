import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

def load_data(filepath):
    data = pd.read_csv(filepath)
    return data

def preprocess_data(data):
    X = data.drop('Class', axis=1)
    y = data['Class']

    scaler = StandardScaler()
    X[['Time', 'Amount']] = scaler.fit_transform(X[['Time', 'Amount']])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    return X_train_smote, X_test, y_train_smote, y_test

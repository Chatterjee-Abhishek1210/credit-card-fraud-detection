import joblib
import numpy as np

model = joblib.load('model.pkl')

def predict_transaction(features):
    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)
    prob = model.predict_proba(features)[0][1]
    return prediction[0], prob


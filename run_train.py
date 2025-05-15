from src.data_preprocessing import load_data, preprocess_data
from src.train_model import train_rf, save_model
from src.evaluate_model import evaluate_model

def main():
    print("✅ Script started")

    data_path = 'data/creditcard.csv'
    print("📂 Loading data...")
    data = load_data(data_path)

    print("🧹 Preprocessing data...")
    X_train, X_test, y_train, y_test = preprocess_data(data)

    print("🚀 Training model...")
    model = train_rf(X_train, y_train)
    print("✅ Model training completed!")

    print("💾 Saving model...")
    save_model(model)

    print("📊 Evaluating model...")
    evaluate_model(model, X_test, y_test)
    
    print("✅ Executed successfully!")

if __name__ == "__main__":
    main()

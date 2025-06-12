from sklearn.model_selection import train_test_split
from data_loader import load_churn_data
from preprocess import encode_and_scale
from model import train_classifier
from evaluate import evaluate_classifier

if __name__ == "__main__":
    df = load_churn_data("Telco-Customer-Churn.csv")
    X, y, _ = encode_and_scale(df, target_col='Churn')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_classifier(X_train, y_train)
    evaluate_classifier(model, X_test, y_test)
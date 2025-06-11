from sklearn.model_selection import train_test_split
from data_loader import load_emission_data
from preprocess import prepare_features
from model import train_regressor
from evaluate import evaluate_regressor

if __name__ == "__main__":
    df = load_emission_data("CO2 Emissions_Canada.csv")
    features = ['Engine Size(L)', 'Cylinders', 'Fuel Consumption Comb (L/100 km)']
    target = 'CO2 Emissions(g/km)'
    X, y, _ = prepare_features(df, features, target)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_regressor(X_train, y_train)
    evaluate_regressor(model, X_test, y_test)
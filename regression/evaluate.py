from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_regressor(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
    print("R2 Score:", r2_score(y_test, y_pred))

    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.xlabel("Actual CO2 Emissions")
    plt.ylabel("Predicted CO2 Emissions")
    plt.title("Actual vs Predicted")
    plt.show()

    residuals = y_test - y_pred
    sns.histplot(residuals, bins=30, kde=True)
    plt.title("Residual Plot")
    plt.show()
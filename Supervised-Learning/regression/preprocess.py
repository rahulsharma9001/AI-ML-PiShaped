from sklearn.preprocessing import StandardScaler

def prepare_features(df, features, target):
    X = df[features]
    y = df[target]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler
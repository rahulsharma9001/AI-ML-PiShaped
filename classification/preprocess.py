from sklearn.preprocessing import LabelEncoder, StandardScaler

def encode_and_scale(df, target_col):
    df_encoded = df.copy()
    for column in df_encoded.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df_encoded[column] = le.fit_transform(df_encoded[column])

    X = df_encoded.drop(target_col, axis=1)
    y = df_encoded[target_col]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler
from sklearn.ensemble import RandomForestClassifier

def train_classifier(X_train, y_train):
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    return clf
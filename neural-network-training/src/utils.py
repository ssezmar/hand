def load_data(file_path):
    import pandas as pd
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    features = data[['wrist_x', 'wrist_y', 'wrist_z', 'elbow_x', 'elbow_y', 'elbow_z', 'shoulder_x', 'shoulder_y', 'shoulder_z']]
    labels = data['label']
    scaled_features = scaler.fit_transform(features)
    return scaled_features, labels

def split_data(features, labels, test_size=0.2):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test

def save_processed_data(features, labels, file_path):
    import pandas as pd
    processed_data = pd.DataFrame(features, columns=['wrist_x', 'wrist_y', 'wrist_z', 'elbow_x', 'elbow_y', 'elbow_z', 'shoulder_x', 'shoulder_y', 'shoulder_z'])
    processed_data['label'] = labels.values
    processed_data.to_csv(file_path, index=False)
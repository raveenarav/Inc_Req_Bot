import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

DATA_PATH = "../../knowledgebase/data/training_data.csv"
MODEL_PATH = "incident_request_model.pkl"
ENCODERS_PATH = "label_encoders.pkl"

def train():
    df = pd.read_csv(DATA_PATH)
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(df['description'])
    
    classifiers = {}
    label_encoders = {}

    for col in ['category', 'subcategory', 'type']:
        le = LabelEncoder()
        y = le.fit_transform(df[col])
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X, y)
        classifiers[col] = clf
        label_encoders[col] = le
        print(f"Trained: {col}")

    joblib.dump({"vectorizer": vectorizer, "classifiers": classifiers}, MODEL_PATH)
    joblib.dump(label_encoders, ENCODERS_PATH)
    print("Done!")

if __name__ == "__main__":
    train()
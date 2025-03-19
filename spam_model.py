import argparse
import mlflow
import mlflow.pyfunc
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import List

class SpamClassifier(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        with open(context.artifacts["vectorizer"], "rb") as f:
            self.vectorizer = pickle.load(f)
        with open(context.artifacts["model"], "rb") as f:
            self.model = pickle.load(f)

    def predict(self, context, model_input: List[str]) -> List[int]:
        if isinstance(model_input, pd.Series):
            texts = model_input.tolist()
        elif isinstance(model_input, list):
            texts = model_input
        else:
            raise ValueError("Input should be a list or pandas Series of strings.")

        transformed_texts = self.vectorizer.transform(texts)
        return self.model.predict(transformed_texts)

def main(data_path, classifier):
    
    # Load dataset
    df = pd.read_csv(data_path, encoding="latin-1")[["v1", "v2"]]
    df.columns = ["label", "text"]
    df["label"] = df["label"].map({"ham": 0, "spam": 1})
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)
    
    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    match classifier:
        case "SVC":
            model = SVC(probability=True)
        case "RandomForest":
            model = RandomForestClassifier()
        case "NaiveBayes":
            model = MultinomialNB()
        case _:
            raise ValueError("Unsupported classifier. Please choose 'SVC', 'RandomForest', or 'NaiveBayes'.")
    
    model.fit(X_train_tfidf, y_train)

    # Evaluate model
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"Model Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
    
    # Save the vectorizer and model
    vectorizer_path = "vectorizer.pkl"
    model_path = "model.pkl"
    
    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer, f)
    
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    
    # Log the model using MLflow PyFunc
    artifact_path = "spam_classifier"
    
    with mlflow.start_run():
        mlflow.log_param("classifier", classifier)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        
        artifacts = {
            "vectorizer": vectorizer_path,
            "model": model_path
        }
        
        mlflow.pyfunc.log_model(
            artifact_path=artifact_path,
            python_model=SpamClassifier(),
            artifacts=artifacts,
            registered_model_name="SpamDetectionPyFuncModel"
        )
    
    print("Model Logged Successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and log an SVC spam classifier with MLflow.")
    parser.add_argument("data_path", type=str, help="Path to the dataset CSV file.")
    parser.add_argument("--classifier", type=str, default="SVC", help="Classifier to use: SVC, RandomForest, or NaiveBayes.")
    
    args = parser.parse_args()
    main(args.data_path, args.classifier)

import argparse
import pandas as pd
import pickle
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix


def load_data(filepath):
    """Loads data from file into a pandas DataFrame"""
    return pd.read_excel(filepath) if filepath.endswith('.xlsx') else pd.read_csv(filepath)


def train_and_evaluate_models(df, features, target, classifiers, names, scaler_filepath):
    """Trains and evaluates specified models using k-fold cross-validation"""
    X, y = df[features], df[target]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    with open(scaler_filepath, "wb") as file:
        pickle.dump(scaler, file)

    k = 10
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    model_filepaths = []
    
    # Ensure that the models directory exists
    Path("models").mkdir(parents=True, exist_ok=True)
    
    for classifier, name in zip(classifiers, names):
        results = {'accuracy': [], 'recall': [], 'precision': [], 'f1': []}
        for train_index, test_index in kf.split(X_scaled):
            X_train, X_test = X_scaled[train_index], X_scaled[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)
            results['accuracy'].append(accuracy_score(y_test, y_pred))
            results['recall'].append(recall_score(y_test, y_pred))
            results['precision'].append(precision_score(y_test, y_pred))
            results['f1'].append(f1_score(y_test, y_pred))

        for metric, scores in results.items():
            print(f"{name} Average {metric.capitalize()}: {sum(scores) / k:.4f}")

        # Save models in the models directory
        model_filepath = Path(f"models/{name.replace(' ', '_').lower()}_model.pkl")
        model_filepaths.append(model_filepath)
        with open(model_filepath, "wb") as file:
            pickle.dump(classifier, file)

    return model_filepaths


def predict_new_data(df, model_filepaths, scaler_filepath, true_label):
    """Predicts new data with saved models"""
    with open(scaler_filepath, "rb") as file:
        scaler = pickle.load(file)

    X_scaled = scaler.transform(df[['entropy_difference', 'size_difference']])
    for model_filepath in model_filepaths:
        with open(model_filepath, "rb") as file:
            model = pickle.load(file)

        y_pred = model.predict(X_scaled)
        tn, fp, fn, tp = confusion_matrix([true_label]*len(y_pred), y_pred).ravel()
        accuracy = accuracy_score([true_label]*len(y_pred), y_pred)
        recall = recall_score([true_label]*len(y_pred), y_pred)
        precision = precision_score([true_label]*len(y_pred), y_pred)
        f1 = f1_score([true_label]*len(y_pred), y_pred)

        print(f"Model: {model_filepath.stem}")
        print(f"True Positives: {tp}, True Negatives: {tn}, False Positives: {fp}, False Negatives: {fn}")
        print(f"Accuracy: {accuracy:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}, F1 Score: {f1:.4f}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Train and evaluate classifiers for given datasets.")
    parser.add_argument("-d", "--datasets", nargs="+", required=True,
                        help="Filepaths of datasets. Each dataset is labeled with a label in the same order.")
    parser.add_argument("-l", "--labels", nargs="+", type=int, required=True,
                        help="Labels for each dataset. Labels must be in the same order as datasets.")
    parser.add_argument("-s", "--scaler", default="scaler.pkl",
                        help="Filepath for saving the scaler.")
    parser.add_argument("-t", "--test", default=None,
                        help="Filepath of the test dataset.")
    parser.add_argument("-tl", "--true_label", default=None, type=int,
                        help="True label of the test dataset.")
    args = parser.parse_args()

    combined_df = pd.concat([load_data(fp).assign(label=label) for fp, label in zip(args.datasets, args.labels)])
    features = ['entropy_difference', 'size_difference']
    target = 'label'
    classifiers = [
        RandomForestClassifier(),
        DecisionTreeClassifier(),
        KNeighborsClassifier(),
        XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    ]
    names = ['Random Forest', 'Decision Tree', 'K-Nearest Neighbors', 'XGBoost']
    model_filepaths = train_and_evaluate_models(
        combined_df, features, target, classifiers, names, args.scaler)

    if args.test:
        test_df = load_data(args.test)
        predict_new_data(test_df, model_filepaths, args.scaler, args.true_label)


if __name__ == "__main__":
    main()

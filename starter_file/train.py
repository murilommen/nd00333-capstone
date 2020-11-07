import os
import joblib
import argparse
from azureml.core import Run, Dataset

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import ExtraTreesClassifier

def train():
    
    # Get variable hyperparameters as arguments
    parser = argparse.ArgumentParser(description='Training with specified Hyperparameters')
    
    parser.add_argument('--n_estimators',
                        help='the number of trees in the forest',
                        type=int, default=100)
    
    parser.add_argument('--min_weight_fraction_leaf',
                        help='The minimum weighted fraction of the sum total of weights required to be at a leaf node',
                        type=float, default=0.0)
    
    args = parser.parse_args()
    
    n_estimators = args.n_estimators
    min_weight_fraction_leaf = args.min_weight_fraction_leaf
    
    # Get run context from AzureML
    run = Run.get_context()
    ws = run.experiment.workspace
    
    # Read Data
    df = Dataset.get_by_name(ws, 'heart-disease-uci').to_pandas_dataframe()
    
    X, y = df.drop('target', axis=1), df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define Classifier and Train
    clf = ExtraTreesClassifier(n_estimators = n_estimators, min_weight_fraction_leaf=min_weight_fraction_leaf)
    
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    
    # Log metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    run.log('Accuracy', accuracy)
    run.log('n_estimators', n_estimators)
    run.log('leaf_frac', min_weight_fraction_leaf)
    
    # Dump the model
    
    os.makedirs('./outputs', exist_ok=True)
    
    model_path = "outputs/hp-heart-disease_{}.joblib".format(accuracy)
    
    joblib.dump(clf, model_path)
    

if __name__ == "__main__":
    train()
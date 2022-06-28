import os
import joblib
import pandas as pd
from sklearn import preprocessing
from sklearn import metrics
from src import dispatcher
import numpy as np


TEST_DATA = os.environ.get('TEST_DATA')
model_type = os.environ.get('MODEL')

def predict():
    df = pd.read_csv(TEST_DATA)
    test_idx = df.id.values
    
    predictions = None
    

    for fold in range(5):
        df = pd.read_csv(TEST_DATA)
        encoders = joblib.load(os.path.join('models/', 
        f"{model_type}_{fold}_label_encoder.pkl"))
        clf = joblib.load(os.path.join('models/', 
        f"{model_type}_{fold}.pkl"))
        cols = joblib.load(os.path.join('models/', 
        f"{model_type}_columns.pkl"))
        for c in encoders:
            lbl = encoders.get(c)
            df.loc[:, c] = df.loc[:, c].astype(str).fillna("NONE")
            df.loc[:, c] = lbl.transform(df[c].values.tolist())

        df = df[cols]
        preds = clf.predict_proba(df)[:,1]

        if fold ==0:
            predictions = preds
        else:
            predictions += preds
        predictions /=5

        sub = pd.DataFrame(np.column_stack((test_idx, predictions)), 
        columns=["id", "target"])
        sub['id'] = sub['id'].astype(int)
        return sub

if __name__ == "__main__":
    
    submission = predict()
    submission.to_csv('models/submission_random_forest.csv', index=False)
    
import joblib
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.ensemble import RandomForestClassifier


class Stacked:
    def __init__(self, estimators, final_estimator, df):
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.df = df
        self.output_df = pd.DataFrame()

    def stack_trainer(self):
        for fold in range(5):
            empty_df = pd.DataFrame()
            sec_df = self.df.loc[self.df['kfold']==fold]
            targets = sec_df.target.values
            idx = sec_df.id.values
            for estimator in self.estimators:
                data = self.df.copy(deep=True)
                data = data.loc[data['kfold']==fold]
                encoders = joblib.load(os.path.join('models/', 
                 f"{estimator}_{fold}_label_encoder.pkl"))
                clf = joblib.load(os.path.join('models/', 
                 f"{estimator}_{fold}.pkl"))
                cols = joblib.load(os.path.join('models/', 
                 f"{estimator}_columns.pkl"))

                for c in encoders:
                    lbl = encoders.get(c)
                    data.loc[:, c] = data.loc[:, c].astype(str).fillna("NONE")
                    data.loc[:, c] = lbl.transform(data[c].values.tolist())

                data = data[cols]
                preds = clf.predict_proba(data)[:,1]
                empty_df[f'{estimator}'] = preds
                print(empty_df.shape)

            empty_df['target'] = targets
            empty_df['id'] = idx
            if fold ==0:
                self.output_df = empty_df.copy(deep=True)
            else:
                self.output_df = pd.concat([self.output_df, empty_df])
        print(self.output_df.shape)
        # prep the data
        y = self.output_df.target.values
        X = self.output_df.drop(['id', 'target'], axis=1)
        # fit the model
        self.final_estimator.fit(X,y)
        return self
        
        ########### Prediction starts

    def stack_predict(self, X):
        test_idx = X.id.values
        X = X.drop(['id'], axis=1)
        predictions_df = pd.DataFrame()
        for fold in range(5):
            # empty_df = pd.DataFrame()
            for estimator in self.estimators:
                data = X.copy(deep=True)
                encoders = joblib.load(os.path.join('models/', 
                 f"{estimator}_{fold}_label_encoder.pkl"))
                clf = joblib.load(os.path.join('models/', 
                 f"{estimator}_{fold}.pkl"))
                cols = joblib.load(os.path.join('models/', 
                 f"{estimator}_columns.pkl"))

                for c in encoders:
                    lbl = encoders.get(c)
                    data.loc[:, c] = data.loc[:, c].astype(str).fillna("NONE")
                    data.loc[:, c] = lbl.transform(data[c].values.tolist())
                
                data = data[cols]
                preds = clf.predict_proba(data)[:,1]
                if fold ==0:
                    predictions_df[f'{estimator}'] = preds
                else:
                    predictions_df[f'{estimator}'] += preds
        
        for estimator in self.estimators:
            predictions_df[f'{estimator}'] /=5
        
        final_preds = self.final_estimator.predict_proba(predictions_df)[:,1]

        sub = pd.DataFrame(np.column_stack((test_idx, final_preds)), 
         columns=["id", "target"])
        sub['id'] = sub['id'].astype(int)
        return sub

                ######## Upar vale mei estimator and fold dono daalna hoga in the input data for 
                ######## the final estimator


    
        
        
    
if __name__ == '__main__':
    estimators = ['extratrees', 'randomforest','xgbm','gradientboostingclassifier']
    final_estimator = LogisticRegression()
    first_stacker = Stacked(estimators=estimators,final_estimator=final_estimator,
     df=pd.read_csv("input/train_folds.csv"))
    
    first_stacker.stack_trainer()

    submission = first_stacker.stack_predict(X=pd.read_csv("input/test.csv"))
    submission.to_csv("models/stacked_submission.csv", index=False)

    # first_stacker.stack_predict(X=X)

        
    
import pandas as pd
import joblib

class Utils:
    
    def load_from_csv(self, path):
        return pd.read_csv(path)
    
    def load_from_mysql(self,):
        pass
    
    def feature_target(self, dataset, drop_cols, y):
        X = dataset.drop(drop_cols, axis=1)
        y = dataset[y]
        return X, y

    def model_export(self, clf, score):
        print('Score: ', score)
        joblib.dump(clf, './project/models/best_model.pkl')
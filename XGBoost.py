import numpy as np
import pandas as pd
from xgboost import XGBClassifier as xgbc
from sklearn.pipeline import Pipeline
from tqdm import trange


class classifier:

    def __init__(self, history_to_date):
        # history_to_date of form pandas dataframe
        self.htd = history_to_date

    def train_features(self, n, coi_name):

        X = []
        y = []
        for i in trange(len(self.htd)-n):
            
            window = self.htd.copy()[i:i+(n+1)]  # today is [-2] index to see tomorrow 
            coi = window[coi_name][:-1]  # column of interest up till today (-1 index) 
            
            close = window['close'][-2]
            high = window['high'][-2] 
            low = window['low'][-2]
            open_ = window['open'][-2]
            try:
                volume = window['volume'][-2]
            except:
                pass
            
            open_close = (open_ - close) / open_
            high_low = (high - low) / low
            
            coi_percent_change = percentage_change(coi)
            mean = np.mean(coi_percent_change)
            std = np.std(coi_percent_change)
            
            X.append([open_close, high_low, mean, std])
            # buy if tomorrow's close is higher than today's, else sell:
            y.append(1 if window['close'][-1] > window['close'][-2] else -1)

        X = np.array(X)
        y = np.array(y)
        print(f'Disimilarity between +1 and -1 ys: {100*sum(y)/len(y):.2f}%')
        return [X, y]

    def train(self, n=5, coi_name='close'):
        X, y = self.train_features(n=n, coi_name=coi_name)
        print('--- Generated Features ---')
        self.clf = build_pipeline()
        self.clf.fit(X, y)
        print('--- Training Complete ---')
        return

    def test(self, override_features):
        X = override_features
        self.predictions = self.clf.predict(np.array([X]))
        return self.predictions

# Helper functions:
def percentage_change(array):
    op = []
    for i in range(1, len(array)):
        percentage_change = (array[i] - array[i - 1]) / array[i]
        op.append(percentage_change)
    return op

def build_pipeline():
    pipe = Pipeline([('xgb', xgbc())])
    return pipe
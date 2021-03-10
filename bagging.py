import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier

from sklearn.svm import (LinearSVC, SVC)
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings("ignore")

if __name__== "__main__":
    df_heart = pd.read_csv('./datasets/heart.csv')
    print(df_heart['target'].describe())
    
    X = df_heart.drop(['target'], axis=1)
    y = df_heart['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.3)
    
    knn_class = KNeighborsClassifier().fit(X_train, y_train)
    knn_pred = knn_class.predict(X_test)
    print('='*64)
    print('Accuracy with only KNeighborsClassifier:', accuracy_score(knn_pred, y_test))
    
    #bag_class = BaggingClassifier(base_estimator=KNeighborsClassifier(),
    #                              n_estimators=50).fit(X_train, y_train)
    #bag_pred = bag_class.predict(X_test)
    #print(accuracy_score(bag_pred, y_test))
    #print('='*64)
    
    
    classifier = {
        'KNeighbors': KNeighborsClassifier(),
        'LogisticRegression' : LogisticRegression(),
        'LinearSCV': LinearSVC(),
        'SVC': SVC(),
        'SGDC': SGDClassifier(),
        'DecisionTree': DecisionTreeClassifier(),
        'RandomTreeForest' : RandomForestClassifier(random_state=0)
    }

    for name, estimator in classifier.items():
        bag_class = BaggingClassifier(base_estimator=estimator,
                                      n_estimators=30).fit(X_train, y_train)
        bag_pred = bag_class.predict(X_test)

        print(f'Accuracy Bagging with {name}:', accuracy_score(bag_pred, y_test))
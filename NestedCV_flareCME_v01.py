import warnings
warnings.filterwarnings('always')

import numpy as np
import pandas as pd
import gc
import os
from pathlib import Path
from ast import literal_eval
from collections.abc import Mapping
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

datadir = Path("/home/solarml/")
resultDir = Path("/home/solarml/WideSpace")

names = ['fcclean', 'fcfilnan']
 
## settings
seed_no = 42
np.random.seed(seed_no)

# folds = 5
outerFolds = 5
innerFolds = 5

skf = StratifiedKFold(n_splits=innerFolds, shuffle=True, random_state=seed_no)

svm = SVC()
linsvm = LinearSVC()
logr = LogisticRegression()
dtree = DecisionTreeClassifier()
mlp = MLPClassifier()
rfrst = RandomForestClassifier()
xgboost = XGBClassifier()
xtrees = ExtraTreesClassifier()

MLmodels = {
    'svm' : svm,
    'linsvm' : linsvm,
    'logr' : logr,
    'dtree' : dtree,
    'mlp' : mlp,
    'rfrst' : rfrst,
    'xgboost' : xgboost,
    'xtrees' : xtrees
}

MLparams = {
    'linsvm' : {
        'penalty':['l2', 'l1'],
        'C': list(np.logspace(-4, 3, num=8, endpoint=True)),
        'max_iter': [5000, 10000, 20000],
        'random_state' : [seed_no],
        'class_weight':['balanced', {0: 1, 1: 1}, {0: 1, 1: 10}, {0: 1, 1: 20},
                        {0: 1, 1: 50}, {0: 1, 1: 100}, {0: 1, 1: 150},
                        {0: 1, 1: 300}, {0: 1, 1: 400}, {0: 1, 1: 500}]
    },
    'svm' : {
        'kernel':['rbf', 'poly', 'sigmoid'],
        'C': list(np.logspace(-4, 3, num=8, endpoint=True)),
        'gamma':['scale', 'auto', 0.001, 0.01, 0.1, 1, 10],
        'degree':[2, 3, 4],
        'coef0':[-10, -1, -0.1, -0.01, -0.001, 0.0, 0.001, 0.01, 0.1, 1, 10],
        'max_iter': [5000, 10000, 20000],
        'class_weight':['balanced', {0: 1, 1: 1}, {0: 1, 1: 10}, {0: 1, 1: 20},
                        {0: 1, 1: 50}, {0: 1, 1: 100}, {0: 1, 1: 150},
                        {0: 1, 1: 300}, {0: 1, 1: 400}, {0: 1, 1: 500}]
    },
    'logr' : {
        'penalty':['l2', 'l1'],
        'solver':['saga', 'liblinear'],
        'random_state' : [seed_no],
        'C': list(np.logspace(-4, 3, num=8, endpoint=True)),
        'max_iter': [5000, 10000, 20000],
        'class_weight':['balanced', {0: 1, 1: 1}, {0: 1, 1: 10}, {0: 1, 1: 20},
                        {0: 1, 1: 50}, {0: 1, 1: 100}, {0: 1, 1: 150},
                        {0: 1, 1: 300}, {0: 1, 1: 400}, {0: 1, 1: 500}]
    },
    'dtree' : {
        'criterion':['gini', 'entropy'],
        'min_samples_leaf': range(2, 100, 2),
        'min_samples_split': range(5, 130, 5),
        'max_depth' : [2, 4, 6, 8, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
        'class_weight':['balanced', {0: 1, 1: 1}, {0: 1, 1: 10}, {0: 1, 1: 20},
                        {0: 1, 1: 50}, {0: 1, 1: 100}, {0: 1, 1: 150},
                        {0: 1, 1: 300}, {0: 1, 1: 400}, {0: 1, 1: 500}]
    },
    'mlp' : {
        'hidden_layer_sizes': [(50,), (40,), (30,), (20,), (10,), (8,), (6,),
                               (50, 2), (40, 2), (30, 2), (20, 2), (10, 2), (8, 2), (6, 2),
                               (50, 3), (40, 3), (30, 3), (20, 3), (10, 3), (8, 3), (6, 3),
                               (50, 4), (40, 4), (30, 4), (20, 4), (10, 4), (8, 4), (6, 4),
                               (50, 20), (50, 10), (50, 5), (60,), (60, 20), (60, 10), (60, 5), (60, 4), (60, 3), (60, 2),
                               (4,), (3,), (2,), (4, 4), (4, 2), (2, 1), (2, 2), (3, 1), (3, 2),
                               (60, 30), (50, 25), (40, 20), (30, 15), (20, 10), (10, 5)],
        'learning_rate': ['constant', 'invscaling', 'adaptive'],
        'learning_rate_init': [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05],
        'max_iter': [5000, 10000, 20000],
        'solver': ['adam', 'lbfgs']
    },
    'rfrst' : {
        'criterion':['gini', 'entropy'], 
        'n_estimators':[10, 20, 40, 60, 80, 100, 200, 400, 600, 800, 1000],
        'min_samples_leaf': range(2, 100, 2),
        'min_samples_split': range(5, 130, 5),
        'max_depth' : [2, 4, 6, 8, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
        'max_features': ['auto', 'sqrt', 'log2'],
        'class_weight':['balanced', {0: 1, 1: 1}, {0: 1, 1: 10}, {0: 1, 1: 20},
                        {0: 1, 1: 50}, {0: 1, 1: 100}, {0: 1, 1: 150},
                        {0: 1, 1: 300}, {0: 1, 1: 400}, {0: 1, 1: 500}]
    },
    'xgboost' : {
        'max_depth' : range(2, 60, 2), 
        'n_estimators' :[10, 20, 40, 60, 80, 100, 200, 400, 600, 800, 1000],
        'scale_pos_weight': range(1, 400, 50),
        'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1],
        'alpha' : np.logspace(-5, 1,num=7), 
        'gamma' : np.logspace(-5, 1,num=7), 
        'lambda' : range(1, 22, 1)
    },
    'xtrees' : {
        'random_state' : [seed_no],
        'criterion':['gini', 'entropy'],
        'n_estimators':[10, 20, 40, 60, 80, 100, 200, 400, 600, 800, 1000],
        'min_samples_leaf': range(2, 100, 2),
        'min_samples_split': range(5, 130, 5),
        'max_depth' : [2, 4, 6, 8, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
        'max_features': ['auto', 'sqrt', 'log2', None],
        'class_weight':['balanced', {0: 1, 1: 1}, {0: 1, 1: 10}, {0: 1, 1: 20},
                        {0: 1, 1: 50}, {0: 1, 1: 100}, {0: 1, 1: 150},
                        {0: 1, 1: 300}, {0: 1, 1: 400}, {0: 1, 1: 500}]
    }
}


mtrics = ['precision', 'recall', 'f1']
colnames = []
for m in mtrics:
    mntr = '_'.join(['mean_train', m])
    mnte = '_'.join(['mean_test', m])
    rnk = '_'.join(['rank_test', m])
    colnames.extend([mntr, mnte, rnk])
colnames.append('params')

def binaryClf_scorer(ytrue, ypred):
    scores = {}
    tn, fp, fn, tp = confusion_matrix(ytrue, ypred).ravel()
    prf = precision_recall_fscore_support(ytrue, ypred, beta=1, pos_label=1, average='binary')[:3]
    scores['precision'] = prf[0]
    scores['recall'] = prf[1]
    scores['f1'] = prf[2]
    scores['POD'] = float(tp/(tp+fn))
    scores['FAR'] = float(fp/(tp+fp))
    scores['TSS'] = float((tp/(tp+fn)) - (fp/(tn+fp)))
    N = float(tn + fp + fn + tp)
    expRand = float(((tp + fn)*(tp + fp) + (tn + fn)*(tn + fp))/N)
    scores['HSS'] = float((tp + tn - expRand) / (N - expRand))
    for k, v in scores.items():
        scores[k] = round(v, 2)
    return scores

allMetrics = ['precision', 'recall', 'f1', 'POD', 'FAR', 'TSS', 'HSS']
colsOut = ['_'.join(['mean_train', m]) for m in mtrics] + ['_'.join(['mean_val', m]) for m in mtrics] + ['_'.join(['test', m]) for m in allMetrics]
colsOut.append('params')

class NestedTuner():
    """Class instances are classifiers optimized in the given parameter space"""
    tuned_clfs = []
    def __init__(self, name, model, prams, vectors, targets):
        self.name = name
        self.model = model # classifier
        self.prams = prams # hyper-parameter space
        self.vectors = vectors # data matrix
        self.targets = targets # target vector - labels
        self.inCVresults = self.tune()
        self.inCVbest = self.findBest()
        NestedTuner.tuned_clfs.append(self)

    def tune(self):
        grid_df = pd.DataFrame(columns=colnames)
        print('Tuning Classifier {}: \n'.format(self.name))
        clf = RandomizedSearchCV(self.model, self.prams, random_state=seed_no, 
                                 scoring=mtrics, n_iter=1000, cv=skf, verbose=1, 
                                 refit=False, return_train_score=True, n_jobs=-1)
        model = clf.fit(self.vectors, self.targets.ravel())
        for col in colnames:
            grid_df[col] = model.cv_results_[col]
        return grid_df.sort_values(by='rank_test_f1', ascending=True)
    
    def findBest(self):
        best =self.inCVresults.loc[(self.inCVresults['mean_test_precision']>=0.6) & (self.inCVresults['mean_test_recall']>=0.6), :]
        best.reset_index()
        good =self.inCVresults.loc[(self.inCVresults['mean_test_precision']>=0.5) & (self.inCVresults['mean_test_recall']>=0.5), :]
        good.reset_index()
        if len(best)>0 :
            return best.iloc[0].to_dict()
        elif len(good)>0 :
            return good.iloc[0].to_dict()
        else:
            return self.inCVresults.iloc[0].to_dict()
        
        del good, best
        gc.collect()

class ModelNestedCV():
    '''Class instances are estimates of model performance'''
    def __init__(self, name, model, prams, vectors, targets):
        self.name = name
        self.model = model
        self.prams = prams
        self.vectors = vectors
        self.targets = targets
        self.outerResults = self.outerCV()
        self.performance = self.foldAverage()
        
    def outerCV(self):
        allscores = pd.DataFrame(columns=colsOut)
        outer = StratifiedKFold(n_splits=outerFolds, shuffle=True, random_state=seed_no)
        fld = 0
        for train_ix, test_ix in outer.split(self.vectors, self.targets):
            Xtrain, Xtest = self.vectors[train_ix], self.vectors[test_ix]
            Ytrain, Ytest = self.targets[train_ix], self.targets[test_ix]
            tuned = NestedTuner(self.name, self.model, self.prams, Xtrain, Ytrain)
            inresults = resultDir / '_'.join([self.name, str(fld), 'inCV3.csv'])
            tuned.inCVresults.to_csv(inresults, index=True, header=True)
            innerModels = tuned.inCVbest
            cols = allscores.columns.to_list()
            allscores.loc[fld, 'mean_train_precision'] = innerModels['mean_train_precision']
            allscores.loc[fld, 'mean_train_recall'] = innerModels['mean_train_recall']
            allscores.loc[fld, 'mean_train_f1'] = innerModels['mean_train_f1']
            allscores.loc[fld, 'mean_val_precision'] = innerModels['mean_test_precision']
            allscores.loc[fld, 'mean_val_recall'] = innerModels['mean_test_recall']
            allscores.loc[fld, 'mean_val_f1'] = innerModels['mean_test_f1']
            hparams = innerModels['params']
            allscores.loc[fld, 'params'] = str(hparams) 
            if isinstance(hparams, Mapping):
                hpDict = hparams
                clf = self.model
                clf.set_params(**hpDict)
                clf.fit(Xtrain, Ytrain.ravel())
                rslts = binaryClf_scorer(Ytest.ravel(), clf.predict(Xtest))
                for m in allMetrics:
                    tst = '_'.join(['test', m])
                    allscores.loc[fld, tst] = rslts[m]
            elif np.isnan(hparams):
                for m in allMetrics:
                    tst = '_'.join(['test', m])
                    allscores.loc[fld, tst] = np.nan
            else :
                hpDict = literal_eval(hparams)
                clf = self.model
                clf.set_params(**hpDict)
                clf.fit(Xtrain, Ytrain.ravel())
                rslts = binaryClf_scorer(Ytest.ravel(), clf.predict(Xtest))
                for m in allMetrics:
                    tst = '_'.join(['test', m])
                    allscores.loc[fld, tst] = rslts[m]
            
            fld += 1
        return allscores
    
    def foldAverage(self):
        df = self.outerResults
        foldsOut = df.index.values + 1
        keys = ['_'.join(['fold', str(x)]) for x in foldsOut]
        df['outer_fold'] = keys
        df.set_index('outer_fold', drop=True, inplace=True)
        for col in df.columns.values[:-1]:
            df.loc['average', col] = df[col].mean(skipna=True)
            df.loc['std', col] = df[col].std(skipna=True)
        return df

clfs = ['svm', 'linsvm', 'logr', 'dtree', 'mlp', 'rfrst', 'xgboost', 'xtrees']
datacols = ['HELIO_LONGITUDE', 'logFint', 'logfl', 'duration', 'trise', 'cycle', 
              'WIDTH_2', 'LINEAR_SPEED_2']

for fname in names:
    datafile = datadir / '_'.join([fname, 'corr.csv'])
    savedir = resultDir / fname
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    dataset = pd.read_csv(datafile, sep=',', header=0)
    print(dataset.describe())
    X = dataset[datacols].values
    Y = dataset[['SEP']].values
    del dataset
    gc.collect()
    scaler = MinMaxScaler()
    Xsc = scaler.fit_transform(X)
    for c in clfs:
        mdl = ModelNestedCV(c, MLmodels[c], MLparams[c], Xsc, Y)
        df = mdl.performance
        savefile = savedir / '_'.join([c, 'NCVimbl3.csv'])
        df.to_csv(savefile, index=True, header=True)
        del df, mdl
        gc.collect()
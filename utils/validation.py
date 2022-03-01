import itertools
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import re
import xgboost as xgb
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score


def rmse_cv(estimator, X, y, kfold, raw=False):
    score = np.sqrt(-cross_val_score(estimator, X, y,
                    scoring="neg_mean_squared_error", cv=kfold))
    if raw:
        return score
    else:
        return(score.mean(), score.std())


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


class BoostingTunerCV():

    def __init__(self, estimator, cv, param_grid, num_boost_round, early_stopping_rounds, metrics):
        self.estimator = estimator
        self.cv = cv
        self.param_grid = param_grid
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.metrics = metrics

    def tune(self, X, y):
        self.best_params_ = {}
        for grid in range(len(self.param_grid)):
            self._grid_search_cv(X, y, grid)
            self._set_params(grid)

    def _grid_search_cv(self, X, y, grid):
        kfold_list = list(self.cv.split(X, y))
        combinations = list(itertools.product(*self.param_grid[grid].values()))
        if isinstance(self.estimator, xgb.XGBRegressor):
            dtrain = xgb.DMatrix(X, y)
            cv = xgb.cv
        elif isinstance(self.estimator, lgb.LGBMRegressor):
            dtrain = lgb.Dataset(X, y)
            cv = lgb.cv
        for i, values in enumerate(combinations):
            params = self._get_params_to_cv(grid, values)
            cv_result = cv(params, dtrain, num_boost_round=self.num_boost_round,
                           folds=kfold_list, metrics=self.metrics,
                           early_stopping_rounds=self.early_stopping_rounds,
                           verbose_eval=False)
            if isinstance(cv_result, dict):
                cv_result = pd.DataFrame.from_dict(cv_result).rename({self.metrics+'-mean': 'test-' +
                                                                      self.metrics+'-mean'}, axis='columns')
            if (i == 0):
                self._cv_eval(cv_result, values)
            elif ((cv_result['test-'+self.metrics+'-mean'].min() < self._metrics_mean_min) |
                  ((cv_result['test-'+self.metrics+'-mean'].min() == self._metrics_mean_min) &
                   (cv_result.shape[0] < self._n_estimators))):
                self._cv_eval(cv_result, values)
        self.cv_result_.index.name = "n_estimators"
        self.best_params_.update(
            dict(zip(self.param_grid[grid].keys(), self._best_params)))
        # set n_estimators once all the parameters are optimized
        if grid == len(self.param_grid)-1:
            self.estimator.set_params(n_estimators=self._n_estimators)

    def _cv_eval(self, cv_result, values):
        self._best_params = values
        self._metrics_mean_min = cv_result['test-'+self.metrics+'-mean'].min()
        self._n_estimators = cv_result.shape[0]
        # lgb.cv does not yield train-rmse results
        if isinstance(self.estimator, xgb.XGBRegressor):
            self.cv_result_ = cv_result[[
                'test-'+self.metrics+'-mean', 'train-'+self.metrics+'-mean']]
        else:
            self.cv_result_ = cv_result[['test-'+self.metrics+'-mean']]

    def _get_params_to_cv(self, grid, values):
        params = self.estimator.get_params()
        for i in range(len(values)):
            params[list(self.param_grid.get(grid).keys())[i]] = values[i]
        return params

    def _set_params(self, grid):
        for k in self.param_grid[grid].keys():
            setattr(self.estimator.set_params(), k, self.best_params_[k])

    def plot_importance(self, X, y, feature_names, figsize=None, max_num_features=None):
        if (figsize is None) & (max_num_features is None):
            figsize = (10, 0.2*(X.shape[1]))
        elif (figsize is None) & (max_num_features is not None):
            figsize = (10, 0.5*max_num_features)
        if isinstance(self.estimator, xgb.XGBRegressor):
            _, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
            xgb.plot_importance(xgb.train(params=self.estimator.get_params(),
                                          dtrain=xgb.DMatrix(
                                              X, y, feature_names=feature_names),
                                          num_boost_round=self.estimator.get_params()['n_estimators']),
                                max_num_features=max_num_features, title=re.match(r'\w+', str(self.estimator))[0] +
                                ' feature importance', ax=ax)
        elif isinstance(self.estimator, lgb.LGBMRegressor):
            lgb.plot_importance(lgb.train(params=self.estimator.get_params(),
                                          train_set=lgb.Dataset(
                                              X, y, feature_name=feature_names),
                                          num_boost_round=self.estimator.get_params()['n_estimators']),
                                figsize=figsize, max_num_features=max_num_features,
                                title=re.match(r'\w+', str(self.estimator))[0]+' feature importance')
        plt.show()

    def plot_learning_curve(self):
        self.cv_result_.plot(title=re.match(r'\w+', str(self.estimator))[0]+' learning curve',
                             ylim=[0, 2*self._metrics_mean_min], legend=True)


class PipeTunerCV():

    def __init__(self, estimator, kfold, param_grid, metrics=None, n_iter=None, random_state=0):
        self.estimator = estimator
        self.kfold = kfold
        self.metrics = metrics
        self.n_iter = n_iter
        self.random_state = random_state
        _valid_params = ['alpha', 'bootstrap', 'C', 'coef0', 'gamma', 'l1_ratio', 'max_depth', 'max_features',
                         'min_samples_leaf', 'min_samples_split', 'n_estimators']
        for parameter in param_grid.keys():
            if parameter not in _valid_params:
                raise ValueError(
                    "Parameters to cross-validate are: {}".format(_valid_params))
        self.param_grid = param_grid

    def tune(self, X, y):
        self._grid_search_cv(X, y)
        self._set_params(self.best_params_.values())

    def _grid_search_cv(self, X, y):
        combinations = list(itertools.product(*self.param_grid.values()))
        if (self.n_iter is not None):
            if (self.n_iter < len(combinations)):
                random.seed(self.random_state)
                combinations = random.sample(combinations, self.n_iter)
        predictions = np.zeros([len(combinations), self.kfold.n_splits])
        for i, parameters in enumerate(combinations):
            self._set_params(parameters)
            for j, (train_index, holdout_index) in enumerate(self.kfold.split(X)):
                if self.metrics:
                    predictions[i, j] = self.metrics(y.iloc[holdout_index], self.estimator.fit(
                        X.iloc[train_index], y.iloc[train_index]).predict(X.iloc[holdout_index]))
                else:
                    self.estimator.fit(
                        X.iloc[train_index], y.iloc[train_index])
                    predictions[i, j] = self.estimator.score(
                        X.iloc[holdout_index], y.iloc[holdout_index])
        indexer = predictions.mean(axis=1).argsort()
        self.metrics_cv_ = predictions[indexer, :][-int(not(self.metrics))]
        index = np.array([param for param in combinations])
        self.best_params_ = dict(
            zip(self.param_grid.keys(), index[indexer][-int(not(self.metrics))]))

    def _set_params(self, parameters):
        params = dict(zip([list(self.estimator.named_steps.keys())[-1]+'__'+param for
                           param in self.param_grid.keys()], parameters))
        self.estimator.set_params(**params)

    def plot(self, X, y, feature_names, figsize=None, slice_=None):
        if not hasattr(self, 'best_params_'):
            raise ValueError(
                "Tune PipeTunerCV before plotting magnitudes or importances")
        if not isinstance(self.estimator.named_steps[list(self.estimator.named_steps.keys())[-1]], (Ridge, Lasso, ElasticNet, RandomForestRegressor)):
            raise ValueError("Method not available for "+re.match(r'\w+', str(
                self.estimator.named_steps[list(self.estimator.named_steps.keys())[-1]]))[0])
        estimator = clone(self.estimator)
        estimator.fit(X, y)
        if isinstance(estimator.named_steps[list(estimator.named_steps.keys())[-1]], RandomForestRegressor):
            values = pd.Series(estimator.named_steps[list(
                estimator.named_steps.keys())[-1]].feature_importances_, index=feature_names)
            values = self._arrange_values(values, slice_)
            if figsize is None:
                figsize = (10, 0.3*len(values))
            values.plot(kind='barh', figsize=figsize, title=re.match(
                r'\w+', str(estimator.named_steps[list(estimator.named_steps.keys())[-1]]))[0]+" feature importance", color='seagreen')
        else:
            values = pd.Series(estimator.named_steps[list(
                estimator.named_steps.keys())[-1]].coef_, index=feature_names)
            values = self._arrange_values(values, slice_)
            if figsize is None:
                figsize = (10, 0.3*len(values))
            if isinstance(estimator.named_steps[list(estimator.named_steps.keys())[-1]], Ridge):
                values.plot(kind='barh', figsize=figsize, title=re.match(r'\w+', str(estimator.named_steps[list(
                    estimator.named_steps.keys())[-1]]))[0]+" coefficient magnitude", color='seagreen')
            else:
                print("Number of features used: {} ({} %)".format(np.sum(estimator.named_steps[list(estimator.named_steps.keys())[-1]].coef_ != 0),
                                                                  (np.sum(estimator.named_steps[list(estimator.named_steps.keys())[-1]].coef_ != 0)/len(feature_names))*100))
                values.plot(kind='barh', figsize=figsize, title=re.match(r'\w+', str(estimator.named_steps[list(
                    estimator.named_steps.keys())[-1]]))[0]+" coefficient magnitude", color='seagreen')
        plt.show()

    def _arrange_values(self, values, slice_):
        values.sort_values(inplace=True)
        if slice_ is None:
            values = values[values != 0]
        else:
            values = values[values != 0].iloc[slice_]
        return values


class Weighted_Average():

    def __init__(self, estimators, kfold, step, metrics):
        self.estimators = [clone(estimator) for estimator in estimators]
        self.kfold = kfold
        self.step = step
        self.metrics = metrics

    def fit(self, X, y):
        rates = list(itertools.chain.from_iterable(set(itertools.permutations(c)) for c in
                                                   [c for c in itertools.combinations_with_replacement(np.arange(0, 1+(self.step/2), self.step),
                                                                                                       len(self.estimators)) if sum(c) == 1]))
        self._cv_predictions = np.zeros(
            [len(rates), self.kfold.get_n_splits()])
        self._indexer = np.zeros(len(rates))
        self._weights = np.zeros(len(self.estimators))
        for i, (train_index, holdout_index) in enumerate(self.kfold.split(X)):
            predictions = np.zeros([len(holdout_index), len(self.estimators)])
            for j, estimator in enumerate(self.estimators):
                estimator.fit(X.loc[train_index], y[train_index])
                predictions[:, j] = estimator.predict(X.loc[holdout_index])
            self._cv_predictions[:, i] = [self.metrics(y[holdout_index], np.sum(
                pred, axis=1)) for pred in ([weights*predictions for weights in rates])]
        self._indexer = self._cv_predictions.mean(axis=1).argsort()
        self._index = np.array([' + '.join([(str(weights[j]) + ' * ' + re.match(r'\w+', str(self.estimators[j]))[0] + '_' +
                                            re.match(r'\w+', str(self.estimators[j].named_steps[list(self.estimators[j].named_steps.keys())[-1]]))[0])
                                           for j in range(len(self.estimators))]) for i, weights in enumerate(rates)])
        self._weights = re.findall(r'\d\.\d+', self._index[self._indexer][0])
        for estimator in self.estimators:
            estimator.fit(X, y)
        return self

    def get_cvpreds(self):
        if not hasattr(self, '_cv_predictions'):
            raise ValueError(
                "Fit Weighted_Average before plotting coefficients")
        cv_preds = pd.DataFrame(self._cv_predictions[self._indexer], index=self._index[self._indexer],
                                columns=["Fold "+str(fold+1) for fold in range(self.kfold.get_n_splits())])
        cv_preds["RMSE CV"] = cv_preds.mean(axis=1)
        cv_preds["std()"] = cv_preds.std(axis=1)
        return cv_preds

    def predict(self, X):
        if not hasattr(self, '_weights'):
            raise ValueError(
                "Fit Weighted_Average before plotting coefficients")
        predictions = np.zeros([X.shape[0], len(self.estimators)])
        for estimator in range(len(self.estimators)):
            predictions[:, estimator] = float(
                self._weights[estimator]) * self.estimators[estimator].predict(X)
        return np.add.reduce(predictions, axis=1)

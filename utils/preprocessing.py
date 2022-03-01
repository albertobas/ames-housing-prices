import numpy as np
import pandas as pd
from scipy.stats import boxcox
from numbers import Number
from sklearn.base import BaseEstimator, TransformerMixin


class Get_Categorical(BaseEstimator, TransformerMixin):

    def __init__(self, categories, qualitative):
        self.categories = categories
        self.qualitative = qualitative

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_categorical = X.copy()
        for feature in self.qualitative:
            feature_categories = [
                cat for cat in self.categories if (feature + '_') in cat]
            feature_categories = [value.replace(
                feature + '_', '') for value in feature_categories]
            X_categorical[feature] = pd.Categorical(
                X_categorical[feature], categories=feature_categories).codes
        return X_categorical.values


class Get_Dummies(BaseEstimator, TransformerMixin):

    def __init__(self, categories, qualitative, to_string=None):
        self.categories = categories
        self.qualitative = qualitative
        self.to_string = to_string

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_dummies = X.copy()
        if self.to_string is not None:
            for var in self.to_string:
                X_dummies[var] = X_dummies[var].apply(str)
        for feature in self.qualitative:
            if not hasattr(X_dummies[feature], 'cat'):
                feature_categories = [
                    cat for cat in self.categories if (feature + '_') in cat]
                feature_categories = [value.replace(
                    feature + '_', '') for value in feature_categories]
                X_dummies[feature] = X_dummies[feature].astype(
                    pd.api.types.CategoricalDtype(categories=feature_categories))
        X_dummies = pd.get_dummies(X_dummies)
        X_dummies.loc[:, self.categories] = X_dummies[self.categories].replace({
                                                                               0: -1})
        return X_dummies.values


class HousePrices_AvgFeat(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transformed = X.copy()
        X_transformed['_Neighbs_GarageYrBlt_mean'] = X_transformed.groupby(
            ['Neighborhood'], as_index=False)['GarageYrBlt'].transform(lambda g: g.mean())
        X_transformed['_Neighbs_GrLivArea_mean'] = X_transformed.groupby(
            ['Neighborhood'], as_index=False)['GrLivArea'].transform(lambda g: g.mean())
        X_transformed['_Neighbs_LotArea_mean'] = X_transformed.groupby(
            ['Neighborhood'], as_index=False)['LotArea'].transform(lambda g: g.mean())
        X_transformed['_Neighbs_LotFrontage_mean'] = X_transformed.groupby(
            ['Neighborhood'], as_index=False)['LotFrontage'].transform(lambda g: g.mean())
        X_transformed['_Neighbs_OverallCond_mode'] = X_transformed.groupby(
            ['Neighborhood'], as_index=False)['OverallCond'].transform(lambda g: g.value_counts().index[0])
        X_transformed['_Neighbs_OverallQual_mode'] = X_transformed.groupby(
            ['Neighborhood'], as_index=False)['OverallQual'].transform(lambda g: g.value_counts().index[0])
        X_transformed['_Neighbs_GarageYrBlt_offset'] = X_transformed['GarageYrBlt'] - \
            X_transformed['_Neighbs_GarageYrBlt_mean']
        X_transformed['_Neighbs_GrLivArea_offset'] = X_transformed['GrLivArea'] - \
            X_transformed['_Neighbs_GrLivArea_mean']
        X_transformed['_Neighbs_LotArea_offset'] = X_transformed['LotArea'] - \
            X_transformed['_Neighbs_LotArea_mean']
        X_transformed['_Neighbs_LotFrontage_offset'] = X_transformed['LotFrontage'] - \
            X_transformed['_Neighbs_LotFrontage_mean']
        X_transformed['_Neighbs_OverallCond_offset'] = X_transformed['OverallCond'] - \
            X_transformed['_Neighbs_OverallCond_mode']
        X_transformed['_Neighbs_OverallQual_offset'] = X_transformed['OverallQual'] - \
            X_transformed['_Neighbs_OverallQual_mode']
        return X_transformed


class HousePrices_BoxCox(BaseEstimator, TransformerMixin):

    def __init__(self, features, lmbda=None):
        self.features = features
        if isinstance(lmbda, Number):
            self.lmbda = pd.Series(np.repeat(lmbda, len(features)))
        else:
            self.lmbda = lmbda

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transformed = X.copy()
        for i, feature in enumerate(self.features):
            if X_transformed[feature].min() < 0:
                raise ValueError(
                    "Do not transform features with negative values")
            elif X_transformed[feature].min() == 0:
                if self.lmbda is None:
                    X_transformed[feature], _ = boxcox(
                        (X_transformed[feature]+0.1))
                else:
                    X_transformed[feature] = boxcox(
                        (X_transformed[feature]+0.1), self.lmbda.loc[i])
            else:
                if self.lmbda is None:
                    X_transformed[feature], _ = boxcox(X_transformed[feature])
                else:
                    X_transformed[feature] = boxcox(
                        X_transformed[feature], self.lmbda.loc[i])
        return X_transformed


class Imputer(BaseEstimator, TransformerMixin):

    def __init__(self, columns, fillna, groupby=None):
        self.columns = columns
        self.fillna = fillna
        self.groupby = groupby
        if (groupby is not None) & (isinstance(fillna, Number)):
            raise ValueError("Do not group if imputing values directly")
        if (not isinstance(fillna, Number)) & (fillna not in ['median', 'mean']):
            raise ValueError(
                "The fillna methods allowed are 'median' and 'mean'.")

    def fit(self, X, y=None):
        if self.groupby is None:
            self._vals = pd.Series()
        else:
            self._vals = pd.DataFrame()
        for column in self.columns:
            if self.groupby is None:
                if self.fillna == 'median':
                    self._vals.loc[column] = X[column].median()
                elif self.fillna == 'mean':
                    self._vals.loc[column] = X[column].mean()
            else:
                if self.fillna == 'median':
                    self._vals.loc[:, column] = X.groupby(self.groupby)[
                        column].median()
                elif self.fillna == 'mean':
                    self._vals.loc[:, column] = X.groupby(self.groupby)[
                        column].mean()
        return self

    def transform(self, X):
        X_imputed = X.copy()
        for column in X_imputed[self.columns].columns[X_imputed[self.columns].isnull().any()]:
            if self.groupby is None:
                if isinstance(self.fillna, Number):
                    X_imputed[column] = X_imputed[column].fillna(
                        self.fillna).values
                else:
                    X_imputed[column] = X_imputed[column].fillna(
                        self._vals.loc[column]).values
            else:
                X_imputed[column] = X_imputed.set_index(
                    self.groupby)[column].fillna(self._vals.loc[:, column]).values
        return X_imputed

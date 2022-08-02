import logging

import pandas as pd
from pandas import DataFrame

from features.types import R_DATETIME, S_DATETIME_AS_OBJECT
from .abstract import AbstractFeatureGenerator

logger = logging.getLogger(__name__)


class DatetimeFeatureGenerator(AbstractFeatureGenerator):
    """将datetime特征转换为数字特征

    Parameters
    ----------
    features : list, optional
        需要进行转换的datetime单位列表
        有关pandas.Series.dt里的方法的全部时间单位选项列表参阅 https://pandas.pydata.org/docs/reference/api/pandas.Series.html
    """
    def __init__(self, 
            features: list = ['year', 'month', 'day', 'dayofweek'],
            **kwargs
    ):
        super().__init__(**kwargs)
        self.features = features

    def _fit_transform(self, X: DataFrame, **kwargs) -> (DataFrame, dict):
        self._fillna_map = self._compute_fillna_map(X)
        X_out = self._transform(X)
        type_family_groups_special = dict(
            datetime_as_int=list(X_out.columns)
        )
        return X_out, type_family_groups_special

    def _transform(self, X: DataFrame) -> DataFrame:
        return self._generate_features_datetime(X)

    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        return dict(required_raw_special_pairs=[
            (R_DATETIME, None),
            (None, [S_DATETIME_AS_OBJECT])
        ])

    def _compute_fillna_map(self, X: DataFrame):
        fillna_map = dict()
        for datetime_feature in self.features_in:
            datetime_series = pd.to_datetime(X[datetime_feature], errors='coerce')

            # Best guess is currently to fill by the mean.
            fillna_datetime = datetime_series.mean()
            fillna_map[datetime_feature] = fillna_datetime
        return fillna_map


    def _generate_features_datetime(self, X: DataFrame) -> DataFrame:
        X_datetime = DataFrame(index=X.index)
        for datetime_feature in self.features_in:
            X_datetime[datetime_feature] = pd.to_datetime(X[datetime_feature], errors='coerce').fillna(self._fillna_map[datetime_feature])
            # X_datetime[datetime_feature] = pd.to_timedelta(X_datetime[datetime_feature]).dt.total_seconds()
            # 将日期转换成衍生的形式
            for feature in self.features:
                X_datetime[datetime_feature + '.' + feature] = getattr(X_datetime[datetime_feature].dt, feature).astype(int)

            X_datetime[datetime_feature] = pd.to_numeric(X_datetime[datetime_feature])

        return X_datetime

    def _remove_features_in(self, features: list):
        super()._remove_features_in(features)
        if self._fillna_map:
            for feature in features:
                if feature in self._fillna_map:
                    self._fillna_map.pop(feature)

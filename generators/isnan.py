import logging
import numpy as np
import pandas as pd
from pandas import DataFrame
from features.types import R_OBJECT, S_BOOL
from .abstract import AbstractFeatureGenerator

logger = logging.getLogger(__name__)


class IsNanFeatureGenerator(AbstractFeatureGenerator):
    """
    将特征转换为isnull标签

    Parameters
    ----------
    null_map : dict, default {'object': ''}
        指定什么值作为NaN值的的映射字典
        键Key是 self.feature_metadata_in.type_map_raw里的特征的原始类型
        如果特征的原始类型没有在null_map中，np.nan将会被当作NaN
        如果有其他值被定义为NaN值，np.nan将不会被当作NaN
    **kwargs :
        参阅 :class:`AbstractFeatureGenerator`注释文档获取可用的关键值参数
    """
    def __init__(self, null_map=None, **kwargs):
        super().__init__(**kwargs)
        if null_map is None:
            null_map = {R_OBJECT: ''}
        self.null_map = null_map
        self._null_feature_map = None

    def _fit_transform(self, X: DataFrame, **kwargs) -> (DataFrame, dict):
        features = self.feature_metadata_in.get_features()
        self._null_feature_map = dict()
        for feature in features:
            feature_raw_type = self.feature_metadata_in.get_feature_type_raw(feature)
            if feature_raw_type in self.null_map:
                self._null_feature_map[feature] = self.null_map[feature_raw_type]
        X_out = self._transform(X)
        type_family_groups_special = {S_BOOL: list(X_out.columns)}
        return X_out, type_family_groups_special

    def _transform(self, X: DataFrame) -> DataFrame:
        is_nan_features = dict()
        for feature in self.features_in:
            if feature in self._null_feature_map:
                null_val = self._null_feature_map[feature]
                is_nan_features['__nan__.' + feature] = (X[feature] == null_val).astype(np.uint8)
            else:
                is_nan_features['__nan__.' + feature] = X[feature].isnull().astype(np.uint8)
        return pd.DataFrame(is_nan_features, index=X.index)

    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        return dict()

    def _remove_features_in(self, features: list):
        super()._remove_features_in(features)
        if self._null_feature_map:
            for feature in features:
                if feature in self._null_feature_map:
                    self._null_feature_map.pop(feature)

    def _more_tags(self):
        return {'feature_interactions': False}

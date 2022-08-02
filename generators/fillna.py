import logging

import numpy as np
from pandas import DataFrame
from features.types import R_OBJECT
from .abstract import AbstractFeatureGenerator

logger = logging.getLogger(__name__)


class FillNaFeatureGenerator(AbstractFeatureGenerator):
    """
    填补数据中的缺失值

    Parameters
    ----------
    fillna_map : dict, default {'object': ''}
        用来定义用什么填充值去填充NaNs的映射字典
        键Key是self.feature_metadata_in.type_map_raw.里的原始类型特征
        如果特征的原始类型不在fillna-map的字典中，则它所对应的NaN值会被赋值为fillna_default参数成为np.nan
    fillna_default, default np.nan
        默认的空值填充值，如果特征的原始类型不在fillna-map的字典中将会用到该值
        将该值设置除np.nan之外的其他值时候请注意，不是所有的原始类型都能接受int,float或者string
    inplace : bool, default False
        如果为True，则空值会被直接替换进行填充而不会复制输入的数据
        这会导致输入数据里的值直接发生变动
    **kwargs :
        参阅 :class:`AbstractFeatureGenerator` 的注释获得有效的关键参数
    """
    def __init__(self, fillna_map=None, fillna_default=np.nan, inplace=False, **kwargs):
        super().__init__(**kwargs)
        if fillna_map is None:
            fillna_map = {R_OBJECT: ''}
        self.fillna_map = fillna_map
        self.fillna_default = fillna_default
        self._fillna_feature_map = None
        self.inplace = inplace

    def _fit_transform(self, X: DataFrame, **kwargs) -> (DataFrame, dict):
        features = self.feature_metadata_in.get_features()
        self._fillna_feature_map = dict()
        for feature in features:
            feature_raw_type = self.feature_metadata_in.get_feature_type_raw(feature)
            feature_fillna_val = self.fillna_map.get(feature_raw_type, self.fillna_default)
            if feature_fillna_val is not np.nan:
                self._fillna_feature_map[feature] = feature_fillna_val
        return self._transform(X), self.feature_metadata_in.type_group_map_special

    def _transform(self, X: DataFrame) -> DataFrame:
        if self._fillna_feature_map:
            if self.inplace:
                X.fillna(self._fillna_feature_map, inplace=True, downcast=False)
            else:
                X = X.fillna(self._fillna_feature_map, inplace=False, downcast=False)
        return X

    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        return dict()

    def _remove_features_in(self, features):
        super()._remove_features_in(features)
        if features:
            for feature in features:
                self._fillna_feature_map.pop(feature, None)

    def _more_tags(self):
        return {'feature_interactions': False}

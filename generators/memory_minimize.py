import logging

import numpy as np
from pandas import DataFrame, RangeIndex

from features.types import R_CATEGORY, R_INT

from features.generators import AbstractFeatureGenerator
from features.utils import clip_and_astype

logger = logging.getLogger(__name__)


class CategoryMemoryMinimizeFeatureGenerator(AbstractFeatureGenerator):
    """
    通过将多分类特征转换成单调递增的int值来最小化多分类特征的内存使用
    这对于包含字符串特征值的会占用很多内存的但是在下游不需要使用的多分类特征非常重要
    """

    def _fit_transform(self, X: DataFrame, **kwargs) -> (DataFrame, dict):
        self._category_maps = self._get_category_map(X=X)

        X_out = self._transform(X)
        return X_out, self.feature_metadata_in.type_group_map_special

    def _transform(self, X: DataFrame) -> DataFrame:
        return self._minimize_categorical_memory_usage(X)

    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        return dict(valid_raw_types=[R_CATEGORY])

    def _get_category_map(self, X: DataFrame) -> dict:
        category_maps = {}
        for column in X:
            old_categories = list(X[column].cat.categories.values)
            new_categories = RangeIndex(len(old_categories))  # 内存优化categories
            category_maps[column] = new_categories
        return category_maps

    def _minimize_categorical_memory_usage(self, X: DataFrame):
        if self._category_maps:
            X_renamed = dict()
            for column in self._category_maps:
                # rename_categories(inplace=True) 更快但是pandas 1.3.0没有
                X_renamed[column] = X[column].cat.rename_categories(self._category_maps[column])
            X = DataFrame(X_renamed)
        return X

    def _remove_features_in(self, features: list):
        super()._remove_features_in(features)
        if self._category_maps:
            for feature in features:
                if feature in self._category_maps:
                    self._category_maps.pop(feature)

    def _more_tags(self):
        return {'feature_interactions': False}


class NumericMemoryMinimizeFeatureGenerator(AbstractFeatureGenerator):
    """
    对int类型的特征进行裁剪并转换特征类型从而减少内存使用
    dtype_out : np.dtype, default np.uint8
        裁剪完需要转换的输出数据类型,默认为np.uint8
    **kwargs :
        参阅:class:`AbstractFeatureGenerator`注释和文档获取可用的关键参数
    """
    def __init__(self, dtype_out=np.uint8, **kwargs):
        super().__init__(**kwargs)
        self.dtype_out, self._clip_min, self._clip_max = self._get_dtype_clip_args(dtype_out)

    def _fit_transform(self, X: DataFrame, **kwargs) -> (DataFrame, dict):
        X_out = self._transform(X)
        return X_out, self.feature_metadata_in.type_group_map_special

    def _transform(self, X):
        return self._minimize_numeric_memory_usage(X)

    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        return dict(valid_raw_types=[R_INT])

    @staticmethod
    def _get_dtype_clip_args(dtype) -> (np.dtype, int, int):
        try:
            dtype_info = np.iinfo(dtype)
        except ValueError:
            dtype_info = np.finfo(dtype)
        return dtype_info.dtype, dtype_info.min, dtype_info.max

    def _minimize_numeric_memory_usage(self, X: DataFrame):
        return clip_and_astype(df=X, clip_min=self._clip_min, clip_max=self._clip_max, dtype=self.dtype_out)

    def _more_tags(self):
        return {'feature_interactions': False}

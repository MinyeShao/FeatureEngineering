import logging
from typing import Union
from collections import defaultdict
from pandas import DataFrame
from features.types import R_INT, R_FLOAT, R_CATEGORY, R_BOOL
from features.generators.abstract import AbstractFeatureGenerator

logger = logging.getLogger(__name__)


class DropDuplicatesFeatureGenerator(AbstractFeatureGenerator):
    """
    删除与其他特征完全相同的特征，只留下一个数据实例。

    Parameters
    ----------
    sample_size_init : int, default 1000
        对重复特征候选进行初始过滤时要采样的行数。
        通常可以使用较少的行数过滤掉大部分特征，从而大大加快最终检查的计算速度。
        如果为None 或大于行数，则不会开始这个初始过滤。这可能会增加fit（将generator应用在）大型数据集的时间。
    sample_size_final : int, default 20000
        用于最终过滤确定重复特征名的采样行数
        理论上会有极小概率情况下非常接近重复但并不是真的重复的特征被移除，
        如果为None或者超出数据的总行数，将会遍历整个数据集进行重复特征采样（非常消耗资源）
        建议将这个值设定在100000以下以确保合理的fit（应用）时长
    **kwargs :
        参阅:class:`AbstractFeatureGenerator` 注释有关有效关key参数的详细信息。
    """
    def __init__(self, sample_size_init=1000, sample_size_final=20000, **kwargs):
        super().__init__(**kwargs)
        self.sample_size_init = sample_size_init
        self.sample_size_final = sample_size_final

    def _fit_transform(self, X: DataFrame, **kwargs) -> (DataFrame, dict):
        if self.sample_size_init is not None and len(X) > self.sample_size_init:
            features_to_check = self._drop_duplicate_features(X, self.feature_metadata_in, keep=False, sample_size=self.sample_size_init)
            X_candidates = X[features_to_check]
        else:
            X_candidates = X
        features_to_drop = self._drop_duplicate_features(X_candidates, self.feature_metadata_in, sample_size=self.sample_size_final)
        self._remove_features_in(features_to_drop)
        if features_to_drop:
            self._log(15, f'\t{len(features_to_drop)} duplicate columns removed: {features_to_drop}')
        X_out = X[self.features_in]
        return X_out, self.feature_metadata_in.type_group_map_special

    def _transform(self, X: DataFrame) -> DataFrame:
        return X

    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        return dict()

    @classmethod
    def _drop_duplicate_features(cls, X: DataFrame, feature_metadata_in, keep: Union[str, bool] = 'first', sample_size=None) -> list:
        if sample_size is not None and len(X) > sample_size:
            X = X.sample(sample_size, random_state=0)
        features_to_remove = []

        X_columns = set(X.columns)
        features_to_check_numeric = feature_metadata_in.get_features(valid_raw_types=[R_INT, R_FLOAT])
        features_to_check_numeric = [feature for feature in features_to_check_numeric if feature in X_columns]
        if features_to_check_numeric:
            features_to_remove += cls._drop_duplicate_features_numeric(X=X[features_to_check_numeric], keep=keep)
            X = X.drop(columns=features_to_check_numeric)

        X_columns = set(X.columns)
        features_to_check_categorical = feature_metadata_in.get_features(valid_raw_types=[R_CATEGORY, R_BOOL])
        features_to_check_categorical = [feature for feature in features_to_check_categorical if feature in X_columns]
        if features_to_check_categorical:
            features_to_remove += cls._drop_duplicate_features_categorical(X=X[features_to_check_categorical], keep=keep)
            X = X.drop(columns=features_to_check_categorical)

        if len(X.columns) > 0:
            features_to_remove += cls._drop_duplicate_features_generic(X=X, keep=keep)

        return features_to_remove

    @classmethod
    def _drop_duplicate_features_generic(cls, X: DataFrame, keep: Union[str, bool] = 'first'):
        """
        通用的重复删除方法，可以处理所有数据类型
        """
        X_columns = list(X.columns)
        features_to_keep = set(X.T.drop_duplicates(keep=keep).T.columns)
        features_to_remove = [column for column in X_columns if column not in features_to_keep]
        return features_to_remove

    @classmethod
    def _drop_duplicate_features_numeric(cls, X: DataFrame, keep: Union[str, bool] = 'first'):
        X_columns = list(X.columns)
        feature_sum_map = defaultdict(list)
        for feature in X_columns:
            feature_sum_map[round(X[feature].sum(), 2)].append(feature)

        features_to_remove = []
        for key in feature_sum_map:
            if len(feature_sum_map[key]) <= 1:
                continue
            features_to_keep = set(X[feature_sum_map[key]].T.drop_duplicates(keep=keep).T.columns)
            features_to_remove += [feature for feature in feature_sum_map[key] if feature not in features_to_keep]

        return features_to_remove

    @classmethod
    def _drop_duplicate_features_categorical(cls, X: DataFrame, keep: Union[str, bool] = 'first'):
        """
        如果重复特征包含相同的信息，则删除它们，忽略特征中的实际值。
        例如，['a', 'b', 'b'] 被认为是 ['b', 'a', 'a'] 的重复，但不是 ['a', 'b', 'a']的重复 .
        """
        X_columns = list(X.columns)
        mapping_features_val_dict = {}
        features_unique_count_dict = defaultdict(list)
        features_to_remove = []
        for feature in X_columns:
            feature_unique_vals = X[feature].unique()
            mapping_features_val_dict[feature] = dict(zip(feature_unique_vals, range(len(feature_unique_vals))))
            features_unique_count_dict[len(feature_unique_vals)].append(feature)

        for feature_unique_count in features_unique_count_dict:
            # 只需要确认拥有相同数量唯一值的特征
            features_to_check = features_unique_count_dict[feature_unique_count]
            if len(features_to_check) <= 1:
                continue
            mapping_features_val_dict_cur = {feature: mapping_features_val_dict[feature] for feature in features_to_check}
            # 将['a', 'd', 'f', 'a'] 转换为 [0, 1, 2, 0]
            # 将[5, 'a', np.nan, 5] 转换为 [0, 1, 2, 0], 这两个会被认为是重复的因为他们拥有相同的信息
            X_cur = X[features_to_check].replace(mapping_features_val_dict_cur)
            features_to_remove += cls._drop_duplicate_features_numeric(X=X_cur, keep=keep)

        return features_to_remove

    def _more_tags(self):
        return {'feature_interactions': False}

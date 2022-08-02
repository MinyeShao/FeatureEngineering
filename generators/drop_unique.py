import logging
from pandas import DataFrame
from features.types import R_CATEGORY, R_OBJECT, S_TEXT, S_IMAGE_PATH
from features.feature_metadata import FeatureMetadata
from .abstract import AbstractFeatureGenerator

logger = logging.getLogger(__name__)



class DropUniqueFeatureGenerator(AbstractFeatureGenerator):

    """剔除只包含一个独立值的全重复特征或者几乎没有重复值(基于max_unique_ratio进行判断)的category种类或者object类型的特征 """
    def __init__(self, max_unique_ratio=0.99, **kwargs):
        super().__init__(**kwargs)
        self.max_unique_ratio = max_unique_ratio

    def _fit_transform(self, X: DataFrame, **kwargs) -> (DataFrame, dict):
        features_to_drop = self._drop_unique_features(X, self.feature_metadata_in, max_unique_ratio=self.max_unique_ratio)
        self._remove_features_in(features_to_drop)
        X_out = X[self.features_in]
        return X_out, self.feature_metadata_in.type_group_map_special

    def _transform(self, X: DataFrame) -> DataFrame:
        return X

    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        return dict()


    @staticmethod
    def _drop_unique_features(X: DataFrame, feature_metadata: FeatureMetadata, max_unique_ratio) -> list:
        features_to_drop = []
        X_len = len(X)
        max_unique_value_count = X_len * max_unique_ratio
        for column in X:
            unique_value_count = len(X[column].unique())
            # 剔除特征值一致重复的特征
            if unique_value_count == 1:
                features_to_drop.append(column)
            elif feature_metadata.get_feature_type_raw(column) in [R_CATEGORY, R_OBJECT]\
                    and (unique_value_count > max_unique_value_count):
                special_types = feature_metadata.get_feature_types_special(column)
                if S_TEXT in special_types:
                    #不应该剔除文本类型的特征
                    continue
                elif S_IMAGE_PATH in special_types:
                    #不应该剔除一个图像路径的特征
                    continue
                else:
                    features_to_drop.append(column)
        return features_to_drop

    def _more_tags(self):
        return {'feature_interactions': False}
import copy
import logging

import pandas as pd
from pandas import DataFrame
from pandas.api.types import CategoricalDtype
from features.types import R_BOOL, R_CATEGORY, R_OBJECT, S_DATETIME_AS_OBJECT, S_IMAGE_PATH, S_TEXT, S_TEXT_AS_CATEGORY
from .abstract import AbstractFeatureGenerator
from features.generators.memory_minimize import CategoryMemoryMinimizeFeatureGenerator

logger = logging.getLogger(__name__)



class CategoryFeatureGenerator(AbstractFeatureGenerator):
    """
    是用来将对象类型的特征转换为多分类类型的特征，同时移除罕见的分类种类并优化内存的使用
    在应用之后，转换时之前没见过的分类将会被视为缺失值
    Parameters
    ----------
    stateful_categories : bool, default True
        如果为 True，则将训练中的类别应用于转换后的数据，输入数据中的任何未知类别都将被视为缺失值。
        建议将这个值设定为True从而避免下游出现问题
    minimize_memory : bool, default True
        如果为True, 通过将多分类特征转换成单调递增的int值来最小化多分类特征的内存使用
        推荐将该参数设定为True以便动态地减少内存的使用
    cat_order : str, default 'original'
        指定分类存储的顺序
        当minimize_memory为True时这个参数就非常重要，这个顺序会指定不同分类转换为什么整数值
        有效的value:
            'original' :保持原来的顺序。如果该特征最初是一个object，则这相当于“alphanumeric”。
            'alphanumeric' : 按照字母数字顺序升序排列.
            'count' : 按照分类的频次进行排序(从频次最低的用0表示)
    minimum_cat_count : int, default None
        设定一个分类出现频次的判断是否为罕见分类的最小阈值以避免使其被认为是罕见的分类，因为罕见的分类将会被剔除并被视作缺失值
        如果为None，则不需要minimum count。这包括从未出现在数据中但作为可能的类别存在于类别对象中的类别。
    maximum_num_cat : int, default None
        被视为非罕见分类的分类出现频次的最大值
        按出现次数排序，如果maximum_num_cat=N，则最多保留N个最高次数的类别。所有其他将被视为罕见类别。
    fillna : str, default None
        用来处理缺失值的方法，只有当stateful_categories=True时有效
        缺失值包括原始类型就是NaN的特征也包括通过由某些参数导致的转换过后形成的NaN类型的特征，比如maximum_num_cat
        Valid values:
             None : 将缺失值保留，没有任何分类值会将其填充
            'mode' : 缺失值将会被分类中出现频次最高的分类所替代
    **kwargs :
        参阅 :class:`AbstractFeatureGenerator` 获取有效的关键参数信息
    """
    def __init__(self, stateful_categories=True, minimize_memory=True, cat_order='original', minimum_cat_count: int = 2, maximum_num_cat: int = None, fillna: str = None, **kwargs):
        super().__init__(**kwargs)
        self._stateful_categories = stateful_categories
        if minimum_cat_count is not None and minimum_cat_count < 1:
            minimum_cat_count = None
        if cat_order not in ['original', 'alphanumeric', 'count']:
            raise ValueError(f"cat_order must be one of {['original', 'alphanumeric', 'count']}, but was: {cat_order}")
        self.cat_order = cat_order
        self._minimum_cat_count = minimum_cat_count
        self._maximum_num_cat = maximum_num_cat
        self.category_map = None
        if fillna is not None:
            if fillna not in ['mode']:
                raise ValueError(f"fillna={fillna} is not a valid value. Valid values: {[None, 'mode']}")
        self._fillna = fillna
        self._fillna_flag = self._fillna is not None
        self._fillna_map = None

        if minimize_memory:
            self._post_generators = [CategoryMemoryMinimizeFeatureGenerator()] + self._post_generators

    def _fit_transform(self, X: DataFrame, **kwargs) -> (DataFrame, dict):
        if self._stateful_categories:
            X_out, self.category_map, self._fillna_map = self._generate_category_map(X=X)
            if self._fillna_map is not None:
                for column in self._fillna_map:
                    X_out[column] = X_out[column].fillna(self._fillna_map[column])
        else:
            X_out = self._transform(X)
        feature_metadata_out_type_group_map_special = copy.deepcopy(self.feature_metadata_in.type_group_map_special)
        if S_TEXT in feature_metadata_out_type_group_map_special:
            text_features = feature_metadata_out_type_group_map_special.pop(S_TEXT)
            feature_metadata_out_type_group_map_special[S_TEXT_AS_CATEGORY] += [feature for feature in text_features if feature not in feature_metadata_out_type_group_map_special[S_TEXT_AS_CATEGORY]]
        return X_out, feature_metadata_out_type_group_map_special

    def _transform(self, X: DataFrame) -> DataFrame:
        return self._generate_features_category(X)

    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        return dict(
            valid_raw_types=[R_OBJECT, R_CATEGORY, R_BOOL],
            invalid_special_types=[S_DATETIME_AS_OBJECT, S_IMAGE_PATH]
        )

    def _generate_features_category(self, X: DataFrame) -> DataFrame:
        if self.features_in:
            X_category = dict()
            if self.category_map is not None:
                for column, column_map in self.category_map.items():
                    X_category[column] = pd.Categorical(X[column], categories=column_map)
                X_category = DataFrame(X_category, index=X.index)
                if self._fillna_map is not None:
                    for column, column_map in self._fillna_map.items():
                        X_category[column].fillna(column_map, inplace=True)
        else:
            X_category = DataFrame(index=X.index)
        return X_category

    def _generate_category_map(self, X: DataFrame) -> (DataFrame, dict):
        if self.features_in:
            fill_nan_map = dict()
            category_map = dict()
            X_category = X.astype('category')
            for column in X_category:
                rank = X_category[column].value_counts().sort_values(ascending=True)
                if self._minimum_cat_count is not None:
                    rank = rank[rank >= self._minimum_cat_count]
                if self._maximum_num_cat is not None:
                    rank = rank[-self._maximum_num_cat:]
                if self.cat_order == 'count' or self._minimum_cat_count is not None or self._maximum_num_cat is not None:
                    category_list = list(rank.index)  # category_list 通过 'count' 进行排序
                    if len(category_list) > 1:
                        if self.cat_order == 'original':
                            original_cat_order = list(X_category[column].cat.categories)
                            set_category_list = set(category_list)
                            category_list = [cat for cat in original_cat_order if cat in set_category_list]
                        elif self.cat_order == 'alphanumeric':
                            category_list.sort()
                    X_category[column] = X_category[column].astype(CategoricalDtype(categories=category_list))
                    X_category[column] = X_category[column].cat.reorder_categories(category_list)
                elif self.cat_order == 'alphanumeric':
                    category_list = list(X_category[column].cat.categories)
                    category_list.sort()
                    X_category[column] = X_category[column].astype(CategoricalDtype(categories=category_list))
                    X_category[column] = X_category[column].cat.reorder_categories(category_list)
                category_map[column] = copy.deepcopy(X_category[column].cat.categories)
                if self._fillna_flag:
                    if self._fillna == 'mode':
                        if len(rank) > 0:
                            fill_nan_map[column] = list(rank.index)[-1]
            if not self._fillna_flag:
                fill_nan_map = None
            return X_category, category_map, fill_nan_map
        else:
            return DataFrame(index=X.index), None, None

    def _remove_features_in(self, features: list):
        super()._remove_features_in(features)
        if self.category_map:
            for feature in features:
                if feature in self.category_map:
                    self.category_map.pop(feature)
        if self._fillna_map:
            for feature in features:
                if feature in self._fillna_map:
                    self._fillna_map.pop(feature)

    def _more_tags(self):
        return {'feature_interactions': False}

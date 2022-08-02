import copy
import logging
import re
from typing import List

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from features.types import S_IMAGE_PATH, S_TEXT, S_TEXT_SPECIAL
from features.generators.abstract import AbstractFeatureGenerator
from features.generators.binned import BinnedFeatureGenerator

logger = logging.getLogger(__name__)


class TextSpecialFeatureGenerator(AbstractFeatureGenerator):
    """
    TextSpecialFeatureGenerator从原始文本特征生成文本特定特征
    其中包括词计数，字符计数，符号计数，大写字母比例、小写字母比例、数字比例等等。
    通过这个generator产生的特征将会有一个'text_special'的特殊类型

    Parameters
    ----------
    symbols : List[str], optional
        用于计算作为特征的计数和比率的字符串符号列表。
        如果没有指定，默认为['!', '?', '@', '%', '$', '*', '&', '#', '^', '.', ':', ' ', '/', ';', '-', '=']
    min_occur_ratio : float, default 0.01
        被视为特征的出现符号的频率最小比例
        如果一个符号的出现次数小于1/min_occur_ratio*样本总量，那它不会被当作是一个特征
    min_occur_offset : int, default 10
        被视为特征的最少符号出现次数，和 min_occur_ratio类似但不同的是其作为常数加入计算的阈值中。
    bin_features : bool, default True
        如果为True,将BinnedFeatureGenerator加到post_generators的前列，所有经过这个generator产生的特征都将被分箱处理
        这对于‘text_special’特征来说非常重要它将降低模型在数据上的过拟合并且降低内存使用
    post_drop_duplicates : bool, default True
        等同于AbstractFeatureGenerator的post_drop_duplicates方法,除非该方法默认为True而不是False（原本默认是False）
        当符号没有出现在数据中时这将帮助清洗这个generator的输出
    **kwargs :
        参阅AbstractFeatureGenerator 注释获取更多有关关键参数的信息.
    """
    def __init__(self, symbols: List[str] = None, min_occur_ratio=0.01, min_occur_offset=10, bin_features: bool = True, post_drop_duplicates: bool = True, **kwargs):
        super().__init__(post_drop_duplicates=post_drop_duplicates, **kwargs)
        if symbols is None:
            symbols = ['!', '?', '@', '%', '$', '*', '&', '#', '^', '.', ':', ' ', '/', ';', '-', '=']
        self._symbols = symbols  # 生成计数和特征占比的符号
        self._symbols_per_feature = dict()
        self._min_occur_ratio = min_occur_ratio
        self._min_occur_offset = min_occur_offset
        if bin_features:
            self._post_generators = [BinnedFeatureGenerator()] + self._post_generators

    def _fit_transform(self, X: DataFrame, **kwargs) -> (DataFrame, dict):
        self._symbols_per_feature = self._filter_symbols(X, self._symbols)
        self._feature_names_dict = self._compute_feature_names_dict()
        X_out = self._transform(X)
        type_family_groups_special = {
            S_TEXT_SPECIAL: list(X_out.columns)
        }
        return X_out, type_family_groups_special

    def _transform(self, X: DataFrame) -> DataFrame:
        return self._generate_features_text_special(X)

    def _compute_feature_names_dict(self) -> dict:
        feature_names = dict()
        for feature in self.features_in:
            feature_names_cur = dict()
            for feature_name_base in ['char_count', 'word_count', 'capital_ratio', 'lower_ratio', 'digit_ratio', 'special_ratio']:
                feature_names_cur[feature_name_base] = f'{feature}.{feature_name_base}'
            symbols = self._symbols_per_feature[feature]
            for symbol in symbols:
                feature_names_cur[symbol] = {}
                feature_names_cur[symbol]['count'] = f'{feature}.symbol_count.{symbol}'
                feature_names_cur[symbol]['ratio'] = f'{feature}.symbol_ratio.{symbol}'
            feature_names[feature] = feature_names_cur
        return feature_names

    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        return dict(required_special_types=[S_TEXT], invalid_special_types=[S_IMAGE_PATH])

    def _filter_symbols(self, X: DataFrame, symbols: list):
        symbols_per_feature = dict()
        if self.features_in:
            num_samples = len(X)
            occur_threshold = min(np.ceil(self._min_occur_offset + num_samples * self._min_occur_ratio), np.ceil(num_samples / 2))
            for nlp_feature in self.features_in:
                symbols_feature = []
                nlp_feature_str = X[nlp_feature].astype(str)
                for symbol in symbols:
                    occur_symbol = np.sum([value.count(symbol) != 0 for value in nlp_feature_str])
                    if occur_symbol >= occur_threshold:
                        symbols_feature.append(symbol)
                symbols_per_feature[nlp_feature] = np.array(symbols_feature)
        return symbols_per_feature

    def _generate_features_text_special(self, X: DataFrame) -> DataFrame:
        if self.features_in:
            X_text_special_combined = dict()
            for nlp_feature in self.features_in:
                X_text_special_combined = self._generate_text_special(X[nlp_feature], nlp_feature, symbols=self._symbols_per_feature[nlp_feature], X_dict=X_text_special_combined)
            X_text_special_combined = pd.DataFrame(X_text_special_combined, index=X.index)
        else:
            X_text_special_combined = pd.DataFrame(index=X.index)
        return X_text_special_combined

    def _generate_text_special(self, X: Series, feature: str, symbols: list, X_dict: dict) -> dict:
        fn = self._feature_names_dict[feature]
        X_str = X.astype(str)

        X_dict[fn['char_count']] = np.array([self.char_count(value) for value in X_str], dtype=np.uint32)
        X_dict[fn['word_count']] = np.array([self.word_count(value) for value in X_str], dtype=np.uint32)
        X_dict[fn['capital_ratio']] = np.array([self.capital_ratio(value) for value in X_str], dtype=np.float32)
        X_dict[fn['lower_ratio']] = np.array([self.lower_ratio(value) for value in X_str], dtype=np.float32)
        X_dict[fn['digit_ratio']] = np.array([self.digit_ratio(value) for value in X_str], dtype=np.float32)
        X_dict[fn['special_ratio']] = np.array([self.special_ratio(value) for value in X_str], dtype=np.float32)

        char_count = X_dict[fn['char_count']]
        char_count_valid = char_count != 0
        shape = char_count.shape
        for symbol in symbols:
            X_dict[fn[symbol]['count']] = np.array([value.count(symbol) for value in X_str], dtype=np.uint32)
            X_dict[fn[symbol]['ratio']] = np.divide(X_dict[fn[symbol]['count']], char_count, out=np.zeros(shape, dtype=np.float32), where=char_count_valid)

        return X_dict

    @staticmethod
    def word_count(string: str) -> int:
        return len(string.split())

    @staticmethod
    def char_count(string: str) -> int:
        return len(string)

    @staticmethod
    def special_ratio(string: str) -> float:
        string = string.replace(' ', '')
        if not string:
            return 0
        new_str = re.sub(r'[\w]+', '', string)
        return len(new_str) / len(string)

    @staticmethod
    def digit_ratio(string: str) -> float:
        string = string.replace(' ', '')
        if not string:
            return 0
        return sum(c.isdigit() for c in string) / len(string)

    @staticmethod
    def lower_ratio(string: str) -> float:
        string = string.replace(' ', '')
        if not string:
            return 0
        return sum(c.islower() for c in string) / len(string)

    @staticmethod
    def capital_ratio(string: str) -> float:
        string = string.replace(' ', '')
        if not string:
            return 0
        return sum(1 for c in string if c.isupper()) / len(string)

    def _remove_features_in(self, features: list):
        super()._remove_features_in(features)
        if self._symbols_per_feature:
            for feature in features:
                if feature in self._symbols_per_feature:
                    self._symbols_per_feature.pop(feature)

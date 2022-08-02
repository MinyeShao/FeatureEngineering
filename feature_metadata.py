import copy
import logging
from typing import Dict, List, Tuple
from collections import defaultdict

import pandas as pd

from .infer_types import get_type_map_raw, get_type_group_map_special

logger = logging.getLogger(__name__)


class FeatureMetadata:
    """
    特征元数据包含有关在原始数据本身中不直接显现的特征的信息。
    这使特征生成功能feature generators能够正确处理特征，并允许下游模型在训练和推理期间正确处理特征。
    Parameters
    ----------
    type_map_raw : Dict[str, str]
        特征名称及其原始类型映射关系的字典
        其中value可以是任何形式，通常推荐是 ['int', 'float', 'object', 'category', 'datetime']中的一种
    type_group_map_special : Dict[str, List[str]], optional
        特殊类型及特征名称列表映射关系的字典
        其中key可以是任何形式，通常推荐是 ['binned', 'datetime_as_int', 'datetime_as_object', 'text', 'text_as_category', 'text_special', 'text_ngram', 'image_path', 'stack']中的一种
        查看每种特殊特征类型，移步：‘ModelArt.features.types'
        出现在value列表的特征名称必须同时也存在于type_map_raw的key中
        特征名称不是必须要求有特殊类型
        只能指定 type_group_map_special 和 type_map_special 的其中之一。
    type_map_special : Dict[str, List[str]], optional
        特征名称与特殊类型列表映射关系的字典
        这是一种特殊类的代替展现方式
        只能指定 type_group_map_special 和 type_map_special 的其中之一。
    """
    def __init__(self, type_map_raw: Dict[str, str], type_group_map_special: Dict[str, List[str]] = None, type_map_special: Dict[str, List[str]] = None):
        if type_group_map_special is None:
            if type_map_special is not None:
                type_group_map_special = self.get_type_group_map_special_from_type_map_special(type_map_special)
            else:
                type_group_map_special = defaultdict(list)
        elif type_map_special is not None:
            raise ValueError('Only one of type_group_map_special and type_map_special can be specified in init.')
        if not isinstance(type_group_map_special, defaultdict):
            type_group_map_special = defaultdict(list, type_group_map_special)

        self.type_map_raw = type_map_raw
        self.type_group_map_special = type_group_map_special

        self._validate()

    # 确定输入是否合理
    def _validate(self):
        type_group_map_special_expanded = []
        for key in self.type_group_map_special:
            type_group_map_special_expanded += self.type_group_map_special[key]

        features_invalid = []
        type_map_raw_keys = self.type_map_raw.keys()
        for feature in type_group_map_special_expanded:
            if feature not in type_map_raw_keys:
                features_invalid.append(feature)
        if features_invalid:
            raise AssertionError(f"{len(features_invalid)} features are present in type_group_map_special but not in type_map_raw. Invalid features: {features_invalid}")

    # 注意：推理过程中不要依赖这个函数。
    def get_features(self, valid_raw_types: list = None, valid_special_types: list = None, invalid_raw_types: list = None, invalid_special_types: list = None,
                     required_special_types: list = None, required_raw_special_pairs: List[Tuple[str, List[str]]] = None, required_exact=False, required_at_least_one_special=False) -> List[str]:
        """
            通过可用参数筛选之后，返回特征元数据对象中保存的特征列表。

        Parameters
        ----------
        valid_raw_types : list, default None
            如果某个特征的原始类型不在此列表中，则会对其进行去除。
            如果为None，则不会通过此逻辑修剪任何特征。
        valid_special_types : list, default None
            如果某个特征具有不在此列表中的特殊类型，则会对其进行去除。
            没有特殊类型的特征永远不会通过这种逻辑进行去除。
            如果为None，则不会通过此逻辑去除任何特征。
        invalid_raw_types : list, default None
            如果特征的原始类型在这个列表中，将会被去除
            如果为None，则不会通过该逻辑去除任何特征
        invalid_special_types : list, default None
            如果特征有的特殊类型存在于这个列表中，将会被去除
            没有特殊类型的特征将永远不会通过这个逻辑被剔除
            如果为None，则所有特征都不会通过这个逻辑被剔除
        required_special_types : list, default None
            如果特征没有包含列表中的所有特殊类型，将会被剔除
            没有特殊类型的特征通过这个逻辑将会被直接剔除
            如果为None，则不会通过该逻辑去除任何特征
        required_raw_special_pairs : List[Tuple[str, List[str]]], default None
            如果特征不满足至少一个列表里元素的（原始类型raw_type，特殊类型special_types）要求，将会被剔除
            与为 required_raw_special_pairs 中 (raw_type, special_types) 的每个元素调用 get_features(valid_raw_types=[raw_type], required_special_types=special_types) 的集合相同
            如果raw_type为None，那么任何特征都将满足原始类型raw type的要求
            如果special_type为None，那么任何特征都将满足特殊类型 special type的要求（包括那些没有special types的特征）
        required_exact : bool, default False
            如果为True则如果一个特征不包含与required_special_types完全相同的所有特殊类型（同时没有超出的额外的特殊类型），将会被剔除
            如果指定，同时会被应用到required_raw_special_pairs
            如果required_special_types和required_raw_special_pairs为None将不会产生任何作用
        required_at_least_one_special : bool, default False
            如果为True，则如果特征没有特殊类型，将会被剔除
        Returns
        -------
        features : 满足参数规定的所有检查的特征元数据中的特征名称列表.

        """
        features = list(self.type_map_raw.keys())

        if valid_raw_types is not None:
            features = [feature for feature in features if self.get_feature_type_raw(feature) in valid_raw_types]
        if valid_special_types is not None:
            valid_special_types_set = set(valid_special_types)
            features = [feature for feature in features if not valid_special_types_set.isdisjoint(self.get_feature_types_special(feature)) or not self.get_feature_types_special(feature)]
        if invalid_raw_types is not None:
            features = [feature for feature in features if self.get_feature_type_raw(feature) not in invalid_raw_types]
        if invalid_special_types is not None:
            invalid_special_types_set = set(invalid_special_types)
            features = [feature for feature in features if invalid_special_types_set.isdisjoint(self.get_feature_types_special(feature))]
        if required_special_types is not None:
            required_special_types_set = set(required_special_types)
            if required_exact:
                features = [feature for feature in features if required_special_types_set == set(self.get_feature_types_special(feature))]
            else:
                features = [feature for feature in features if required_special_types_set.issubset(self.get_feature_types_special(feature))]
        if required_at_least_one_special:
            features = [feature for feature in features if self.get_feature_types_special(feature)]
        if required_raw_special_pairs is not None:
            features_og = copy.deepcopy(features)
            features_to_keep = []
            for valid_raw, valid_special in required_raw_special_pairs:
                if valid_special is not None:
                    valid_special = set(valid_special)
                features_to_keep_inner = []
                for feature in features:
                    feature_type_raw = self.get_feature_type_raw(feature)
                    feature_types_special = set(self.get_feature_types_special(feature))
                    if valid_raw is None or feature_type_raw == valid_raw:
                        if valid_special is None:
                            features_to_keep_inner.append(feature)
                        elif required_exact:
                            if valid_special == feature_types_special:
                                features_to_keep_inner.append(feature)
                        elif valid_special.issubset(feature_types_special):
                            features_to_keep_inner.append(feature)
                features = [feature for feature in features if feature not in features_to_keep_inner]
                features_to_keep += features_to_keep_inner
            features = [feature for feature in features_og if feature in features_to_keep]

        return features

    def get_feature_type_raw(self, feature: str) -> str:
        return self.type_map_raw[feature]

    def get_feature_types_special(self, feature: str) -> list:
        if feature not in self.type_map_raw:
            raise KeyError(f'{feature} does not exist in {self.__class__.__name__}.')
        return self._get_feature_types(feature=feature, feature_types_dict=self.type_group_map_special)

    def get_type_map_special(self) -> dict:
        return {feature: self.get_feature_types_special(feature) for feature in self.get_features()}

    @staticmethod
    def get_type_group_map_special_from_type_map_special(type_map_special: Dict[str, List[str]]):
        type_group_map_special = defaultdict(list)
        for feature in type_map_special:
            for type_special in type_map_special[feature]:
                type_group_map_special[type_special].append(feature)
        return type_group_map_special

    def get_type_group_map_raw(self):
        type_group_map_raw = defaultdict(list)
        for feature, dtype in self.type_map_raw.items():
            type_group_map_raw[dtype].append(feature)
        return type_group_map_raw

    def remove_features(self, features: list, inplace=False):
        """从features中的元数据中删除所有特征"""
        if inplace:
            metadata = self
        else:
            metadata = copy.deepcopy(self)
        features_invalid = [feature for feature in features if feature not in self.get_features()]
        if features_invalid:
            raise KeyError(f'remove_features was called with a feature that does not exist in feature metadata. Invalid Features: {features_invalid}')
        metadata._remove_features_from_type_map(d=metadata.type_map_raw, features=features)
        metadata._remove_features_from_type_group_map(d=metadata.type_group_map_special, features=features)
        return metadata

    def keep_features(self, features: list, inplace=False):
        """除了features中的特征之外移除metadata中的所有特征"""
        features_invalid = [feature for feature in features if feature not in self.get_features()]
        if features_invalid:
            raise KeyError(f'keep_features was called with a feature that does not exist in feature metadata. Invalid Features: {features_invalid}')
        features_to_remove = [feature for feature in self.get_features() if feature not in features]
        return self.remove_features(features=features_to_remove, inplace=inplace)

    def add_special_types(self, type_map_special: Dict[str, List[str]], inplace=False):
        """
        向特征添加特殊类型

        Parameters
        ----------
        type_map_special : Dict[str, List[str]]
            特征字典 -> 需要增添的特殊类型的列表
            字典里的特征必须已经存在于元数据（FeatureMetadata）对象中
        inplace : bool, default False
            如果为True，更新并返回self
            如果为False，更新self的复制并返回复制
        Returns
        -------
        :class:`FeatureMetadata` 对象.

        Examples
        --------
        >>> from ModelArt.features.feature_metadata import FeatureMetadata
        >>> feature_metadata = FeatureMetadata({'FeatureA': 'int', 'FeatureB': 'object'})
        >>> feature_metadata = feature_metadata.add_special_types({'FeatureA': ['MySpecialType'], 'FeatureB': ['MySpecialType', 'text']})
        """
        if inplace:
            metadata = self
        else:
            metadata = copy.deepcopy(self)
        valid_features = set(self.get_features())

        for feature, special_types in type_map_special.items():
            if feature not in valid_features:
                raise ValueError(f'"{feature}" does not exist in this FeatureMetadata object. Only existing features can be assigned special types.')
            for special_type in special_types:
                metadata.type_group_map_special[special_type].append(feature)
        return metadata

    @staticmethod
    def _remove_features_from_type_group_map(d, features):
        for key, features_orig in d.items():
            d[key] = [feature for feature in features_orig if feature not in features]

    @staticmethod
    def _remove_features_from_type_map(d, features):
        for feature in features:
            if feature in d:
                d.pop(feature)

    def rename_features(self, rename_map: dict, inplace=False):
        """将作为 rename_map 中键的元数据中的所有特征重命名为其值。"""
        if inplace:
            metadata = self
        else:
            metadata = copy.deepcopy(self)
        before_len = len(metadata.type_map_raw.keys())
        metadata.type_map_raw = {rename_map.get(key, key): val for key, val in metadata.type_map_raw.items()}
        after_len = len(metadata.type_map_raw.keys())
        if before_len != after_len:
            raise AssertionError(f'key names conflicted during renaming. Do not rename features to exist feature names.')
        for dtype in metadata.type_group_map_special:
            metadata.type_group_map_special[dtype] = [rename_map.get(feature, feature) for feature in metadata.type_group_map_special[dtype]]
        return metadata


    def join_metadata(self, metadata, shared_raw_features='error'):

        """将两个FeatureMetadata对象拼接在一起, 返回一个新的FeatureMetadata对象"""
        if shared_raw_features not in ['error', 'error_if_diff', 'overwrite']:
            raise ValueError(f"shared_raw_features must be one of {['error', 'error_if_diff', 'overwrite']}, but was: '{shared_raw_features}'")
        type_map_raw = copy.deepcopy(self.type_map_raw)
        shared_features = []
        shared_features_diff_types = []
        for key, features in metadata.type_map_raw.items():
            if key in type_map_raw:
                shared_features.append(key)
                if type_map_raw[key] != metadata.type_map_raw[key]:
                    shared_features_diff_types.append(key)
        if shared_features:
            if shared_raw_features == 'error':
                logger.error('ERROR: Conflicting metadata:')
                logger.error('Metadata 1:')
                self.print_feature_metadata_full(log_prefix='\t', log_level=40)
                logger.error('Metadata 2:')
                metadata.print_feature_metadata_full(log_prefix='\t', log_level=40)
                raise AssertionError(f"Metadata objects to join share raw features, but `shared_raw_features='error'`. Shared features: {shared_features}")
            if shared_features_diff_types:
                if shared_raw_features == 'overwrite':
                    logger.log(20, f'Overwriting type_map_raw during FeatureMetadata join. Shared features with conflicting types: {shared_features_diff_types}')
                    shared_features = []
                elif shared_raw_features == 'error_if_diff':
                    logger.error('ERROR: Conflicting metadata:')
                    logger.error('Metadata 1:')
                    self.print_feature_metadata_full(log_prefix='\t', log_level=40)
                    logger.error('Metadata 2:')
                    metadata.print_feature_metadata_full(log_prefix='\t', log_level=40)
                    raise AssertionError(f"Metadata objects to join share raw features but do not agree on raw dtypes, and `shared_raw_features='error_if_diff'`. Shared conflicting features: {shared_features_diff_types}")
        type_map_raw.update({key: val for key, val in metadata.type_map_raw.items() if key not in shared_features})

        type_group_map_special = self._add_type_group_map_special([self.type_group_map_special, metadata.type_group_map_special])

        return FeatureMetadata(type_map_raw=type_map_raw, type_group_map_special=type_group_map_special)

    @staticmethod
    def _add_type_group_map_special(type_group_map_special_lst: List[dict]) -> dict:
        if not type_group_map_special_lst:
            return defaultdict(list)
        type_group_map_special_combined = copy.deepcopy(type_group_map_special_lst[0])
        for type_group_map_special in type_group_map_special_lst[1:]:
            for key, features in type_group_map_special.items():
                if key in type_group_map_special_combined:
                    features_to_add = [feature for feature in features if feature not in type_group_map_special_combined[key]]
                    type_group_map_special_combined[key] += features_to_add
                else:
                    type_group_map_special_combined[key] = features
        return type_group_map_special_combined

    @staticmethod
    def _get_feature_types(feature: str, feature_types_dict: dict) -> list:
        feature_types = []
        for dtype_family in feature_types_dict:
            if feature in feature_types_dict[dtype_family]:
                feature_types.append(dtype_family)
        feature_types = sorted(feature_types)
        return feature_types

    # 将元数据（metadata）对象列表连接在一起，返回一个新的元数据(metada)对象
    @staticmethod
    def join_metadatas(metadata_list, shared_raw_features='error'):
        metadata_new = copy.deepcopy(metadata_list[0])
        for metadata in metadata_list[1:]:
            metadata_new = metadata_new.join_metadata(metadata, shared_raw_features=shared_raw_features)
        return metadata_new

    def to_dict(self, inverse=False) -> dict:
        if not inverse:
            feature_metadata_dict = dict()
        else:
            feature_metadata_dict = defaultdict(list)

        for feature in self.get_features():
            feature_type_raw = self.type_map_raw[feature]
            feature_types_special = tuple(self.get_feature_types_special(feature))
            if not inverse:
                feature_metadata_dict[feature] = (feature_type_raw, feature_types_special)
            else:
                feature_metadata_dict[(feature_type_raw, feature_types_special)].append(feature)

        if inverse:
            feature_metadata_dict = dict(feature_metadata_dict)

        return feature_metadata_dict

    def print_feature_metadata_full(self, log_prefix='', print_only_one_special=False, log_level=20, max_list_len=5, return_str=False):
        feature_metadata_dict = self.to_dict(inverse=True)
        if not feature_metadata_dict:
            if return_str:
                return ''
            else:
                return
        keys = list(feature_metadata_dict.keys())
        keys = sorted(keys)
        output = [((key[0], list(key[1])), feature_metadata_dict[key]) for key in keys]
        output_str = ''
        if print_only_one_special:
            for i, ((raw, special), features) in enumerate(output):
                if len(special) == 1:
                    output[i] = ((raw, special[0]), features)
                elif len(special) > 1:
                    output[i] = ((raw, special[0]), features)
                    logger.warning(f'Warning: print_only_one_special=True was set, but features with {len(special)} special types were found. Invalid Types: {output[i]}')
                else:
                    output[i] = ((raw, None), features)
        max_key_len = max([len(str(key)) for key, _ in output])
        max_val_len = max([len(str(len(val))) for _, val in output])
        for key, val in output:
            key_len = len(str(key))
            val_len = len(str(len(val)))
            max_key_minus_cur = max(max_key_len - key_len, 0)
            max_val_minus_cur = max(max_val_len - val_len, 0)
            if max_list_len is not None:
                features = str(val[:max_list_len])
                if len(val) > max_list_len:
                    features = features[:-1] + ', ...]'
            else:
                features = str(val)
            if val:
                message = f'{log_prefix}{key}{" " * max_key_minus_cur} : {" " * max_val_minus_cur}{len(val)} | {features}'
                if return_str:
                    output_str += message + '\n'
                else:
                    logger.log(log_level, message)
        if return_str:
            if output_str[-1] == '\n':
                output_str = output_str[:-1]
            return output_str

    @classmethod
    def from_df(cls, df: pd.DataFrame):
        """
        根据推断的输入特征类型构造 FeatureMetadata :class:`pd.DataFrame`。

        Parameters
        ----------
        df : :class:`pd.DataFrame`

            推理FeatureMetadata.所用到的数据dataframe

        Returns
        -------
        :class:`FeatureMetadata` 对象.
        """
        type_map_raw = get_type_map_raw(df)
        type_group_map_special = get_type_group_map_special(df)
        return cls(type_map_raw=type_map_raw, type_group_map_special=type_group_map_special)

    def __str__(self):
        return self.print_feature_metadata_full(return_str=True)

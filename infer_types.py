import logging
from collections import defaultdict
from typing import List

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

logger = logging.getLogger(__name__)


def get_type_family_raw(dtype) -> str:
    """通过数据类型找数据类型父类"""
    try:
        if isinstance(dtype, pd.SparseDtype):
            dtype = dtype.subtype
        if dtype.name == 'category':
            return 'category'
        if 'datetime' in dtype.name:
            return 'datetime'
        if 'string' in dtype.name:
            return 'object'
        elif np.issubdtype(dtype, np.integer):
            return 'int'
        elif np.issubdtype(dtype, np.floating):
            return 'float'
    except Exception as err:
        logger.error(f'Warning: dtype {dtype} is not recognized as a valid dtype by numpy! ModelArt may incorrectly handle this feature...')
        logger.error(err)

    if dtype.name in ['bool', 'bool_']:
        return 'bool'
    elif dtype.name in ['str', 'string', 'object']:
        return 'object'
    else:
        return dtype.name


# 真实数据类型
def get_type_map_real(df: DataFrame) -> dict:
    features_types = df.dtypes.to_dict()
    return {k: v.name for k, v in features_types.items()}


# 原始数据类型（真实数据类型父类）
def get_type_map_raw(df: DataFrame) -> dict:
    features_types = df.dtypes.to_dict()
    return {k: get_type_family_raw(v) for k, v in features_types.items()}


def get_type_map_special(X: DataFrame) -> dict:
    type_map_special = {}
    for column in X:
        types_special = get_types_special(X[column])
        if types_special:
            type_map_special[column] = types_special
    return type_map_special


def get_types_special(X: Series) -> List[str]:
    types_special = []
    if isinstance(X.dtype, pd.SparseDtype):
        types_special.append('sparse')
    if check_if_datetime_as_object_feature(X):
        types_special.append('datetime_as_object')
    elif check_if_nlp_feature(X):
        types_special.append('text')
    return types_special


def get_type_group_map(type_map: dict) -> defaultdict:
    type_group_map = defaultdict(list)
    for key, val in type_map.items():
        if isinstance(val, list):
            for feature_type in val:
                type_group_map[feature_type].append(key)
        else:
            type_group_map[val].append(key)
    return type_group_map


def get_type_group_map_real(df: DataFrame) -> defaultdict:
    type_map_real = get_type_map_real(df)
    return get_type_group_map(type_map_real)


def get_type_group_map_raw(df: DataFrame) -> defaultdict:
    type_map_raw = get_type_map_raw(df)
    return get_type_group_map(type_map_raw)


def get_type_group_map_special(df: DataFrame) -> defaultdict:
    type_map_special = get_type_map_special(df)
    return get_type_group_map(type_map_special)


def check_if_datetime_as_object_feature(X: Series) -> bool:
    type_family = get_type_family_raw(X.dtype)
    if X.isnull().all():
        return False
    if type_family != 'object':
        return False
    try:
        pd.to_numeric(X)
    except:
        try:
            if len(X) > 500:
                # 抽样加加速推理
                X = X.sample(n=500, random_state=0)
            result = pd.to_datetime(X, errors='coerce')
            if result.isnull().mean() > 0.8:  # 如果超过80%的行都是空值
                return False
            return True
        except:
            return False
    else:
        return False


def check_if_nlp_feature(X: Series) -> bool:
    type_family = get_type_family_raw(X.dtype)
    if type_family != 'object':
        return False
    if len(X) > 5000:
        # 抽样加速推理
        X = X.sample(n=5000, random_state=0)
    X_unique = X.unique()
    num_unique = len(X_unique)
    num_rows = len(X)
    unique_ratio = num_unique / num_rows
    if unique_ratio <= 0.01:
        return False
    try:
        avg_words = Series(X_unique).str.split().str.len().mean()
    except AttributeError:
        return False
    if avg_words < 3:
        return False

    return True


def get_bool_true_val(series: pd.Series):
    """
    调用时，根据replace_val参数，将pandas series格式输入转换为bool
    series_bool = series == replace_val
    因此转换为bool时，除了 `replace_val` 之外的任何值都将设置为 `False`。
    series数组必须含有至少两个独立值
    基于假设在两个选项之间选择为“True”的值大部分是任意的，除了 np.nan 不会被视为“True”。
    在可能的情况下，尝试对值进行排序，以便 (0, 1) 选择 1 作为 True，但理想情况下，该决定不会对下游模型产生太大影响。
    推理时任何新的看不见的值（包括 nan）都将自动映射到“False”。
    在此代码中，0 和 0.0（int 和 float）被视为相同的值。与任何其他整数和浮点数（例如 1 和 1.0）类似。

    """
    # 这是一个safety_net，以防混合唯一类型（例如str 和 int）。在这种情况下，会引发异常，因此我们使用未排序的值。
    try:
        uniques = series.unique()
        # 对值进行排序以避免在确定哪个值映射到 `True` 时依赖行顺序。
        uniques.sort()
    except:
        uniques = series.unique()
    replace_val = uniques[1]
    try:
            # 这是为了确保不会将 np.nan 映射到布尔值中的 `True`。
        is_nan = np.isnan(replace_val)
    except:
        is_nan = False
    if is_nan:
        replace_val = uniques[0]
    return replace_val

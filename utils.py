import logging

import numpy as np
from pandas import DataFrame, Series

logger = logging.getLogger(__name__)


def clip_and_astype(df: DataFrame, columns: list = None, clip_min=0, clip_max=255, dtype: str = 'uint8') -> DataFrame:
    """
    对里的列进行裁剪使列里面的值落入指定的最大最小值的区间范围中，然后转换数据类型

    Parameters
    ----------
    df : DataFrame
        输入的DataFrame
    columns : list, optional
        要应用 clip_and_astype 逻辑的 df 的列子集。如果未指定，则使用 df 的所有列。
    clip_min : int or float, default 0
        将列值裁剪到的最小值。所有小于此值的值都将设置为 clip_min。
    clip_max : int or float, default 255
        将列值剪切到的最大值。所有大于此的值都将设置为 clip_max。
    dtype : dtype, default 'uint8'
        裁剪完之后强制进行转换的数据类型

    Returns
    -------
    df_clipped : DataFrame
        返回裁剪并且转换数据类型之后的输入的df
    """
    if columns is None:
        df = np.clip(df, clip_min, clip_max).astype(dtype)
    elif columns:
        df[columns] = np.clip(df[columns], clip_min, clip_max).astype(dtype)
    return df


def is_useless_feature(X: Series) -> bool:
    """如果一个特征每一行的值都一样，那么这个特征将被视为不包含有用信息的特征。"""
    return len(X.unique()) <= 1


def get_smallest_valid_dtype_int(min_val: int, max_val: int):
    """基于特征的最小值和最大值，返回最小的有效 dtype 来表示特征。"""
    if min_val < 0:
        dtypes_to_check = [np.int8, np.int16, np.int32, np.int64]
    else:
        dtypes_to_check = [np.uint8, np.uint16, np.uint32, np.uint64]
    for dtype in [np.uint8, np.uint16, np.uint32, np.uint64]:
        if max_val <= np.iinfo(dtype).max and min_val >= np.iinfo(dtype).min:
            return dtype
    raise ValueError(f'Value is not able to be represented by {dtypes_to_check[-1].__name__}. (min_val, max_val): ({min_val}, {max_val})')
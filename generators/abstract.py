import copy
import inspect
import logging
import time
from collections import defaultdict
from typing import Dict, List
from pandas import DataFrame, Series
from features.infer_types import get_type_map_raw, get_type_map_real, get_type_group_map_special
from features.feature_metadata import FeatureMetadata
from utils.savers import save_pkl
from features.utils import is_useless_feature
logger = logging.getLogger(__name__)



class AbstractFeatureGenerator:
    """
    所有ModelArt feature generator里的特征处理方法都继承于这个AbstractFeatureGenerator进行实现
    设计feature generator的初衷是为了将脏数据以格式化的方式从一个未处理的格式转换成训练能用的格式/数据类型
    首先，generator使用各种参数初始化，这些参数决定了生成特征的方式。
    然后，使用通常采用 pandas DataFrame 格式的训练数据，通过 `.fit()` 或 `.fit_transform()` 方法应用 generator。
    最后，generator可以通过 `.transform()` 方法以与训练数据相同的初始格式转换新数据。

    Parameters
    ----------
    features_in : list, default None
        generator将在 fit（应用） 和 transform（转换） 方法中使用的特征名称列表。
        如果传入 DataFrame 中未出现在 features_in 中的任何特征都将被删除，并且不会影响转换逻辑。
        如果为None，则在应用期间从 _infer_features_in 方法进行推断。
        等价于 feature_metadata_in.get_features()方法。
    feature_metadata_in : :class:`ModelArt.features.feature_metadata.FeatureMetadata`, default None
        :class:`FeatureMetadata`对象与训练集数据的输入特征相关联.
        如果为None
        If None, 在应用期间通过调用 _infer_feature_metadata_in 方法进行推断。
        features_in 中不存在的任何feature（如果提供）将从 feature_metadata_in 中删除。
    post_generators : list of FeatureGenerators, default None
        FeatureGenerators 将在此对象的转换逻辑之后按顺序应用和转换，将其输出作为下一个generator的输入进行传递。
        最后一个FeatureGenerator的输出结果将会被用作转换完成的特征输出
    pre_enforce_types : bool, default False
        如果为 True，则训练数据的原始类型（int64、float32 等）将被强行进行转换，转换的过程中要么成功将类型转换为训练所需类型要么在不能处理时抛出异常。
        该参数设置为 True 很重要，可以确保不正确的 dtype 不会被传递到下游，但在pipeline内的内部feature generator上使用时通常是多余的
    pre_drop_useless : bool, default False
        如果为True，在featrue generator应用过程中如果一个特征里的所有值都是重复且使唯一值时，features_in的list里面将会剔除这个特征.
    post_drop_duplicates : bool, default False
        如果为 True，类drop_duplicates.DropDuplicatesFeatureGenerator 将附加到 post_generators。
        此feature generator将删除在数据中找到的任何重复特征，在任何重复特征集中仅保留一个特征。
        Warning: 对于具有许多特征的大型数据集，这可能在计算上消耗非常大，甚至在计算上不可行。
    reset_index : bool, default False
        如果为 True，则在应用feature generator和转换期间，对于 N 行的数据集，输入数据的索引将重置为从 0 到 N-1 单调递增。
        在应用feature generator和转换结束期间，原始的index索引会被重新应用在输出的数据集上
        对于feature generation pipline里的外部feature generator将该参数设置为True是非常重要的，如果任何内部feature generator没有正确处理非默认index索引，则确保非默认index索引不会导致内部feature generator 损坏。
        这个index索引的重置同时也会被应用在y目标label上（如果y提供了的的话）
    column_names_as_str : bool, default True
        如果为 True，则输入数据的列名将转换为字符串（如果它们尚未转换）。
        这解决了与无法处理整数列名的下游 FeatureGenerator 和模型相关的任何问题，并允许列名前缀和后缀操作以避免错误。
        请注意，出于性能目的，如果列名在适合时不是字符串，则仅在转换时转换。确保一致的列名作为输入以避免错误。
    name_prefix : str, default None
        对所有输出的特征名称进行添加的名称前缀
    name_suffix : str, default None
        对所有输出的特征名称进行添加的名称后缀
    infer_features_in_args : dict, default None
        在推断 self.features_in 时，用作 FeatureMetadata.get_features(**kwargs) 的 kwargs 输入。
        根据 infer_features_in_args_strategy 的值,该字典与 self.get_default_infer_features_in_args() 的输出字典合并。
        仅在 features_in 为 None 时使用。
        如果为 None，则直接使用 self.get_default_infer_features_in_args()。
        有关字典中有效的Key的的完整描述和解释，请参阅 FeatureMetadata.get_features 注释。
        请注意：这是大多数情况下不需要的高级功能。
    infer_features_in_args_strategy : str, default 'overwrite'
        确定 infer_features_in_args 和 self.get_default_infer_features_in_args() 如何组合以产生 self._infer_features_in_args,这决定了 features_in 推理逻辑。
        如果为'overwrite': infer_features_in_args 被独占使用并且 self.get_default_infer_features_in_args() 被忽略。
        如果为‘update’：self.get_default_infer_features_in_args() 将是由 infer_features_in_args 更新的字典。
        如果 infer_features_in_args 为 None，则将其忽略。
    banned_feature_special_types : List[str], default None
        要从输入中另外排除的特征特殊类型列表。将更新 self.get_default_infer_features_in_args()。
    log_prefix : str, default ''
        generator 产生的所有logging声明的前缀字符串
    verbosity : int, default 2
        控制logging打印量的参数
        0的时候将不会打印logs，1的时候将只会打印warnings，2的时候将会打印相对较为重要的信息，3的时候将记录信息级别信息并提供详细的特征类型输入和输出信息。
        日志记录仍由global logger配置控制，因此verbosity=3 并不能保证将输出日志。
    Attributes
    ----------
    features_in : list of str
        generator将在 fit 和 transform 方法中使用的特征名称列表。
        等价于feature_metadata_in.get_features()方法应用之后.
    features_out : list of str
        fit_transform 和 transform 方法的输出中存在的特征名称列表。
        等价于feature_metadata.get_features()方法应用之后
    feature_metadata_in : FeatureMetadata
        数据预转换的 FeatureMetadata（用作应用feature generator和转换方法的输入的数据）。
    feature_metadata : FeatureMetadata
        数据转换后的 FeatureMetadata（由 fit_transform 和 transform 方法输出的数据）。
    feature_metadata_real : FeatureMetadata
        数据转换后的 FeatureMetadata 由确切的 dtypes 组成，而不是在 feature_metadata_in 中找到的分组原始 dtypes：用分组原始 dtypes 代替特殊 dtypes。
        这仅用于 print_feature_metadata_info 方法当作自我检查。建议设置为 None 以减少内存和磁盘使用后适配。

    """
    def __init__(
        self,
        features_in: list = None,
        feature_metadata_in: FeatureMetadata = None,
        post_generators: list = None,
        pre_enforce_types=False,
        pre_drop_useless=False,
        post_drop_duplicates=False,
        reset_index=False,
        column_names_as_str=True,
        name_prefix: str = None,
        name_suffix: str = None,
        infer_features_in_args: dict = None,
        infer_features_in_args_strategy='overwrite',
        banned_feature_special_types: List[str] = None,
        log_prefix='',
        verbosity=2
    ):
        self._is_fit = False  # feature generator 是否已经被应用
        self.features_in = features_in  #   用作feature generation 输入的原始特征
        self.features_out = None  # 转换后的最终特征列表
        self.feature_metadata_in: FeatureMetadata = feature_metadata_in  # 基于原始输入特征的 FeatureMetadata 对象。
        self.feature_metadata: FeatureMetadata = None  # 基于处理后的特征的FeatureMetadata 对象。传递给模型以启用高级功能。
        self.feature_metadata_real: FeatureMetadata = None  # 基于处理后的特征的FeatureMetadata 对象，包含真实的原始dtype信息（如int32、float64等）。传递给模型以启用高级功能。
        self._feature_metadata_before_post = None  #在应用 self._post_generators 之前的FeatureMetadata。
        self._infer_features_in_args = self.get_default_infer_features_in_args()
        if infer_features_in_args is not None:
            if infer_features_in_args_strategy == 'overwrite':
                self._infer_features_in_args = copy.deepcopy(infer_features_in_args)
            elif infer_features_in_args_strategy == 'update':
                self._infer_features_in_args.update(infer_features_in_args)
            else:
                raise ValueError(f"infer_features_in_args_strategy must be one of: {['overwrite', 'update']}, but was: '{infer_features_in_args_strategy}'")
        if banned_feature_special_types:
            if 'invalid_special_types' not in self._infer_features_in_args:
                self._infer_features_in_args['invalid_special_types'] = banned_feature_special_types
            else:
                for f in banned_feature_special_types:
                    if f not in self._infer_features_in_args['invalid_special_types']:
                        self._infer_features_in_args['invalid_special_types'].append(f)

        if post_generators is None:
            post_generators = []
        elif not isinstance(post_generators, list):
            post_generators = [post_generators]
        self._post_generators: list = post_generators
        if post_drop_duplicates:
            from .drop_duplicates import DropDuplicatesFeatureGenerator
            self._post_generators.append(DropDuplicatesFeatureGenerator(post_drop_duplicates=False))
        if name_prefix or name_suffix:
            from .rename import RenameFeatureGenerator
            self._post_generators.append(RenameFeatureGenerator(name_prefix=name_prefix, name_suffix=name_suffix, inplace=True))

        if self._post_generators:
            if not self.get_tags().get('allow_post_generators', True):
                raise AssertionError(f'{self.__class__.__name__} is not allowed to have post_generators, but found: {[generator.__class__.__name__ for generator in self._post_generators]}')

        self.pre_enforce_types = pre_enforce_types
        self._pre_astype_generator = None
        self.pre_drop_useless = pre_drop_useless
        self.reset_index = reset_index
        self.column_names_as_str = column_names_as_str
        self._useless_features_in: list = None
        self._is_updated_name = False  # 特征名是否被name_prefix或者name_suffix修改过
        self.log_prefix = log_prefix
        self.verbosity = verbosity

        self.fit_time = None

    def fit(self, X: DataFrame, **kwargs):
        """
        将generator应用到提供的数据上.
        由于feature generator跟踪输出特征和类型的方法，要求数据在应用feature generator过程中（之前）已经做好转换，因此除了对 fit_transform 的经常调用之外，fit函数很少有直接用。
        Parameters
        ----------
        X : DataFrame
            应用在generator上输入的数据
        **kwargs
            应用generator的时候任何可用的额外的参数
            参阅 fit_transform方法获得常用的kwargs值

        """
        self.fit_transform(X, **kwargs)

    def fit_transform(self, X: DataFrame, y: Series = None, feature_metadata_in: FeatureMetadata = None, **kwargs) -> DataFrame:
        """
        将generator应用到提供的数据并返回数据的转换版本，就像使用相同的数据先调用应用再进行转换一样。（相当于2步骤合并）
        通常比单独调用 fit 和 transform 更有效，如果 fit 过程需要转换数据，则速度可以提高两倍。
        这不能在generator fit（应用） 后调用，并且会导致 AssertionError。
        Parameters

        ----------
        X : DataFrame
            应用在generator上输入的数据
        y : Series, optional
            用于fit generator的输入特征数据X的目标值label y。大多数generator不使用label。
            y.index 必须等于 X.index 以避免错位。
        feature_metadata_in : FeatureMetadata, optional
            与在generator初始化期间提供 feature_metadata_in 相同。如果 self.feature_metadata_in 已经指定，则忽略。
            如果两者都没有设置，则 feature_metadata_in 将从 _infer_feature_metadata_in 方法中推断出来。
        **kwargs
            特定generator应用期间可以使用的任何附加参数。传递给 _fit_transform 和 _fit_generators 方法。

        Returns
        -------
        X_out : DataFrame
        从输入数转换过后的DataFrame对象

        """
        start_time = time.time()
        self._log(20, f'Fitting {self.__class__.__name__}...')
        if self._is_fit:
            raise AssertionError(f'{self.__class__.__name__} is already fit.')
        self._pre_fit_validate(X=X, y=y, feature_metadata_in=feature_metadata_in, **kwargs)

        if self.reset_index:
            X_index = copy.deepcopy(X.index)
            X = X.reset_index(drop=True)
            if y is not None and isinstance(y, Series):
                y = y.reset_index(drop=True)
        else:
            X_index = None
        if self.column_names_as_str:
            columns_orig = list(X.columns)
            X.columns = X.columns.astype(str)  # 确保所有列名为字符串
            columns_new = list(X.columns)
            if columns_orig != columns_new:
                rename_map = {orig: new for orig, new in zip(columns_orig, columns_new)}
                if feature_metadata_in is not None:
                    feature_metadata_in.rename_features(rename_map=rename_map)
                self._rename_features_in(rename_map)
            else:
                self.column_names_as_str = False  # 列名已经使字符串，因此不需要进行转换
        self._ensure_no_duplicate_column_names(X=X)
        self._infer_features_in_full(X=X, feature_metadata_in=feature_metadata_in)
        if self.pre_drop_useless:
            self._useless_features_in = self._get_useless_features(X, columns_to_check=self.features_in)
            if self._useless_features_in:
                self._remove_features_in(self._useless_features_in)
        if self.pre_enforce_types:
            from .astype import AsTypeFeatureGenerator
            self._pre_astype_generator = AsTypeFeatureGenerator(features_in=self.features_in, feature_metadata_in=self.feature_metadata_in, log_prefix=self.log_prefix + '\t')
            self._pre_astype_generator.fit(X)

        X_out, type_family_groups_special = self._fit_transform(X[self.features_in], y=y, **kwargs)

        type_map_raw = get_type_map_raw(X_out)
        self._feature_metadata_before_post = FeatureMetadata(type_map_raw=type_map_raw, type_group_map_special=type_family_groups_special)
        if self._post_generators:
            X_out, self.feature_metadata, self._post_generators = self._fit_generators(X=X_out, y=y, feature_metadata=self._feature_metadata_before_post, generators=self._post_generators, **kwargs)
        else:
            self.feature_metadata = self._feature_metadata_before_post
        type_map_real = get_type_map_real(X_out)
        self.features_out = list(X_out.columns)
        self.feature_metadata_real = FeatureMetadata(type_map_raw=type_map_real, type_group_map_special=self.feature_metadata.get_type_group_map_raw())

        self._post_fit_cleanup()
        if self.reset_index:
            X_out.index = X_index
        self._is_fit = True
        end_time = time.time()
        self.fit_time = end_time - start_time
        if self.verbosity >= 3:
            self.print_feature_metadata_info(log_level=20)
            self.print_generator_info(log_level=20)
        elif self.verbosity == 2:
            self.print_feature_metadata_info(log_level=15)
            self.print_generator_info(log_level=15)
        return X_out

    def transform(self, X: DataFrame) -> DataFrame:
        """
        将输入的数据转换为输出数据的格式
        如果generator已经通过调用fit或者fit_transform方法进行应用（fit）则将会抛出AssertionError异常

        Parameters
        ----------
        X : DataFrame
            generator 输入的将要被转换的数据
            输入数据必须包含 features_in 中的所有feature，并且应该与提供的数据具有相同的 dtypes 以进行fit。
            X 中存在的列但不存在于 features_in 中的额外的特征列将被忽略并且不会影响输出。

        Returns
        -------
        X_out : DataFrame对象类型的输入数据X的转换后的版本数据
        """
        if not self._is_fit:
            raise AssertionError(f'{self.__class__.__name__} is not fit.')
        if self.reset_index:
            X_index = copy.deepcopy(X.index)
            X = X.reset_index(drop=True)
        else:
            X_index = None
        if self.column_names_as_str:
            X.columns = X.columns.astype(str)  # 确保所有列都是字符串
        try:
            X = X[self.features_in]
        except KeyError:
            missing_cols = []
            for col in self.features_in:
                if col not in X.columns:
                    missing_cols.append(col)
            raise KeyError(f'{len(missing_cols)} required columns are missing from the provided dataset to transform using {self.__class__.__name__}. Missing columns: {missing_cols}')
        if self._pre_astype_generator:
            X = self._pre_astype_generator.transform(X)
        X_out = self._transform(X)
        if self._post_generators:
            X_out = self._transform_generators(X=X_out, generators=self._post_generators)
        if self.reset_index:
            X_out.index = X_index
        return X_out

    def _fit_transform(self, X: DataFrame, y: Series, **kwargs) -> (DataFrame, dict):

        raise NotImplementedError

    def _transform(self, X: DataFrame) -> DataFrame:

        raise NotImplementedError

    def _infer_features_in_full(self, X: DataFrame, feature_metadata_in: FeatureMetadata = None):
        """
        推理所有X输入相关的特征信息
        当需要超出 feature_metadata_in 和 features_in 的其他输入信息时，可以扩展此功能。
        例如，AsTypeFeatureGenerator 扩展了这个方法来计算输入的精确原始特征类型以供以后使用。
        此方法返回结果后，self.features_in 和 self.feature_metadata_in 将被设置为适当的值。
        该方法在调用 _fit_transform 之前由 fit_transform 调用。

        Parameters
        ----------
        X : DataFrame
            用来应用在generator上的输入的数据
        feature_metadata_in : FeatureMetadata, optional
            如果传递了这个参数，则self.feature_metadata_in 将会被设置成feature_metadata_in,假设 self.feature_metadata_in 之前是 None 。
            如果self.feature_metadata_in和feature_metadata_in都为None，那么self.feature_metadata_in会由_infer_feature_metadata_in(X)进行推断
        """
        if self.feature_metadata_in is None:
            self.feature_metadata_in = feature_metadata_in
        elif feature_metadata_in is not None:
            self._log(30, '\tWarning: feature_metadata_in passed as input to fit_transform, but self.feature_metadata_in was already set. Ignoring feature_metadata_in.')
        if self.feature_metadata_in is None:
            self._log(20, f'\tInferring data type of each feature based on column values. Set feature_metadata_in to manually specify special dtypes of the features.')
            self.feature_metadata_in = self._infer_feature_metadata_in(X=X)
        if self.features_in is None:
            self.features_in = self._infer_features_in(X=X)
            self.features_in = [feature for feature in self.features_in if feature in X.columns]
        self.feature_metadata_in = self.feature_metadata_in.keep_features(features=self.features_in)


    def _infer_features_in(self, X: DataFrame) -> list:
        """
        通过训练特征数据X推断features_in
        如果在fit之前没有提供features_in的话就会调用这个方法
        这可以在新的generator中被覆盖以使用新的推断逻辑。
        self.feature_metadata_in 在调用此方法时可用。


        Parameters
        ----------
        X : DataFrame
            用来应用在generator上的输入的特征数据

        Returns
        -------
        feature_in : 由输入的训练特征数据X推断出的由特征名称字符串组成的列表 list of str
        """
        return self.feature_metadata_in.get_features(**self._infer_features_in_args)


    @staticmethod
    def _infer_feature_metadata_in(X: DataFrame) -> FeatureMetadata:
        """
        通过特征训练集X推断feature_metadata_in
        如果在fit之前没有提供feature_metadata_in的话就会调用这个方法
        这可以在新generator 中被覆盖以使用新的推断逻辑，但最好保留默认逻辑以与其他generator保持一致。

        Parameters
        ----------
        X : DataFrame
             用来应用在generator上的输入的数据

        Returns
        -------
        feature_metadata_in : 从输入的数据X推断而来的FeatureMetadata对象
        """
        type_map_raw = get_type_map_raw(X)
        type_group_map_special = get_type_group_map_special(X)
        return FeatureMetadata(type_map_raw=type_map_raw, type_group_map_special=type_group_map_special)

    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        raise NotImplementedError

    def _fit_generators(self, X, y, feature_metadata, generators: list, **kwargs) -> (DataFrame, FeatureMetadata, list):
        """
        按顺序拟合 AbstractFeatureGenerator 对象列表，将 generators[i] 的输出作为 generators[i+1] 的输入
        这个方法被调用的时候通过_fit_transform的输出按顺序转换self._post_generators 的 generators 从而获得generator的最终输出结果
        这个方法不应该被 AbstractFeatureGenerator 重写。
        """
        for generator in generators:
            generator.verbosity = min(self.verbosity, generator.verbosity)
            generator.set_log_prefix(log_prefix=self.log_prefix + '\t', prepend=True)
            X = generator.fit_transform(X=X, y=y, feature_metadata_in=feature_metadata, **kwargs)
            feature_metadata = generator.feature_metadata
        return X, feature_metadata, generators

    @staticmethod
    def _transform_generators(X, generators: list) -> DataFrame:
        """
        按顺序通过 AbstractFeatureGenerator 对象列表转换数据X，将 generators[i] 的输出作为 generators[i+1] 的输入
        这个方法被调用的时候通过_transform的输出按顺序转换self._post_generators 的 generators 从而获得generator的最终输出结果
        这个方法不应该被 AbstractFeatureGenerator 重写。
        """
        for generator in generators:
            X = generator.transform(X=X)
        return X

    def _remove_features_in(self, features: list):
        """
        从代表输入数据内容或输入特征如何使用的所有相关对象中删除特征。
        例如，DropDuplicatesFeatureGenerator 在 _fit_transform 期间使用重复的特征列表调用此方法。
        这将允许DropDuplicatesFeatureGenerator的_transform方法简单的直接返回特征数据X，因为重复的特征不在self.features_in中所以已经在transform方法中被剔除了

        Parameters
        ----------
        features : list of str
            要从预期想要的输入中删除的特征名称列表。
        """
        if features:
            if self._feature_metadata_before_post:
                feature_links_chain = self.get_feature_links_chain()
                for feature in features:
                    feature_links_chain[0].pop(feature)
                features_to_keep = set()
                for features_out in feature_links_chain[0].values():
                    features_to_keep = features_to_keep.union(features_out)
                self._feature_metadata_before_post = self._feature_metadata_before_post.keep_features(features_to_keep)

            self.feature_metadata_in = self.feature_metadata_in.remove_features(features=features)
            self.features_in = self.feature_metadata_in.get_features()
            if self._pre_astype_generator:
                self._pre_astype_generator._remove_features_out(features)


    def _remove_features_out(self, features: list):
        """
        从输出的数据中移除特征
        这用于在应用（fit）一系列generator后清理不必要操作的复杂pipeline。
        AbstractFeatureGenerator 的应用不需要更改此方法。

        Parameters
        ----------
        features : list of str
            从self.transform()的输出中移除的特征名称的列表

        """
        feature_links_chain = self.get_feature_links_chain()
        if features:
            self.feature_metadata = self.feature_metadata.remove_features(features=features)
            self.feature_metadata_real = self.feature_metadata_real.remove_features(features=features)
            self.features_out = self.feature_metadata.get_features()
            feature_links_chain[-1] = {feature_in: [feature_out for feature_out in features_out if feature_out not in features] for feature_in, features_out in feature_links_chain[-1].items()}
        self._remove_unused_features(feature_links_chain=feature_links_chain)

    def _remove_unused_features(self, feature_links_chain):
        unused_features = self._get_unused_features(feature_links_chain=feature_links_chain)
        self._remove_features_in(features=unused_features[0])
        for i, generator in enumerate(self._post_generators):
            for feature in unused_features[i + 1]:
                if feature in feature_links_chain[i + 1]:
                    feature_links_chain[i + 1].pop(feature)
            generated_features = set()
            for feature_in in feature_links_chain[i + 1]:
                generated_features = generated_features.union(feature_links_chain[i + 1][feature_in])
            features_out_to_remove = [feature for feature in generator.features_out if feature not in generated_features]
            generator._remove_features_out(features_out_to_remove)

    def _rename_features_in(self, column_rename_map: dict):
        if self.feature_metadata_in is not None:
            self.feature_metadata_in = self.feature_metadata_in.rename_features(column_rename_map)
        if self.features_in is not None:
            self.features_in = [column_rename_map.get(col, col) for col in self.features_in]

    def _pre_fit_validate(self, X: DataFrame, y: Series, **kwargs):
        """
        fit之前的数据可用性检查
        """
        if y is not None and isinstance(y, Series):
            if list(y.index) != list(X.index):
                raise AssertionError(f'y.index and X.index must be equal when fitting {self.__class__.__name__}, but they differ.')

    def _post_fit_cleanup(self):

        pass

    def _ensure_no_duplicate_column_names(self, X: DataFrame):
        if len(X.columns) != len(set(X.columns)):
            count_dict = defaultdict(int)
            invalid_columns = []
            for column in list(X.columns):
                count_dict[column] += 1
            for column in count_dict:
                if count_dict[column] > 1:
                    invalid_columns.append(column)
            raise AssertionError(f'Columns appear multiple times in X. Columns must be unique. Invalid columns: {invalid_columns}')

    @staticmethod
    def _get_useless_features(X: DataFrame, columns_to_check: List[str] = None) -> list:
        useless_features = []
        if columns_to_check is None:
            columns_to_check = list(X.columns)
        for column in columns_to_check:
            if is_useless_feature(X[column]):
                useless_features.append(column)
        return useless_features


    def set_log_prefix(self, log_prefix, prepend=False):
        if prepend:
            self.log_prefix = log_prefix + self.log_prefix
        else:
            self.log_prefix = log_prefix

    def set_verbosity(self, verbosity: int):
        self.verbosity = verbosity

    def _log(self, level, msg, log_prefix=None, verb_min=None):
        if self.verbosity == 0:
            return
        if verb_min is None or self.verbosity >= verb_min:
            if log_prefix is None:
                log_prefix = self.log_prefix
            logger.log(level, f'{log_prefix}{msg}')

    def is_fit(self):
        return self._is_fit


    def is_valid_metadata_in(self, feature_metadata_in: FeatureMetadata):
        """
        如果具有 feature_metadata_in 特征元数据的输入数据为非空，则为True。
        具体就是调用feature_metadata_in.get_features(**self._infer_features_in_args)方法的时候返回不为空
        如果 feature_metadata_in 中的特征不包含generator中的任何可用类型，则为 False。
        例如，如果只有numeric特征作为输入传递给需要文本输入特征的 TextSpecialFeatureGenerator，这将返回 False。
        但是，如果同时传递了numeric和文本特征，这将返回 True，因为其中包含的文本特征将是有效输入（numeric特征将被直接删除）。
        """

        features_in = feature_metadata_in.get_features(**self._infer_features_in_args)
        if features_in:
            return True
        else:
            return False

    def get_feature_links(self) -> Dict[str, List[str]]:
        """返回包括之前和之后的所有generator的特征关系."""
        return self._get_feature_links_from_chain(self.get_feature_links_chain())

    def _get_feature_links(self, features_in: List[str], features_out: List[str]) -> Dict[str, List[str]]:
        """返回忽略之前和之后的所有generator的特征关系"""
        feature_links = {}
        if self.get_tags().get('feature_interactions', True):
            for feature_in in features_in:
                feature_links[feature_in] = features_out
        else:
            for feat_old, feat_new in zip(features_in, features_out):
                feature_links[feat_old] = feature_links.get(feat_old, []) + [feat_new]
        return feature_links

    def get_feature_links_chain(self) -> List[Dict[str, List[str]]]:
        """获取此generator和其之后所有generator的特征依赖关系链"""
        features_out_internal = self._feature_metadata_before_post.get_features()
        generators = [self] + self._post_generators
        features_in_list = [self.features_in] + [generator.features_in for generator in self._post_generators]
        features_out_list = [features_out_internal] + [generator.features_out for generator in self._post_generators]

        feature_links_chain = []
        for i in range(len(features_in_list)):
            generator = generators[i]
            features_in = features_in_list[i]
            features_out = features_out_list[i]
            feature_chain = generator._get_feature_links(features_in=features_in, features_out=features_out)
            feature_links_chain.append(feature_chain)
        return feature_links_chain

    @staticmethod
    def _get_feature_links_from_chain(feature_links_chain: List[Dict[str, List[str]]]) -> Dict[str, List[str]]:
        """
        通过遍历特征关系链获取最终的输入和输出特征关系
        """
        features_out = []
        for val in feature_links_chain[-1].values():
            if val not in features_out:
                features_out.append(val)
        features_in = list(feature_links_chain[0].keys())
        feature_links = feature_links_chain[0]
        for i in range(1, len(feature_links_chain)):
            feature_links_new = {}
            for feature in features_in:
                feature_links_new[feature] = set()
                for feature_out in feature_links[feature]:
                    feature_links_new[feature] = feature_links_new[feature].union(feature_links_chain[i].get(feature_out, []))
                feature_links_new[feature] = list(feature_links_new[feature])
            feature_links = feature_links_new
        return feature_links

    def _get_unused_features(self, feature_links_chain: List[Dict[str, List[str]]]):
        features_in_list = [self.features_in]
        if self._post_generators:
            for i in range(len(self._post_generators)):
                if i == 0:
                    features_in = self._feature_metadata_before_post.get_features()
                else:
                    features_in = self._post_generators[i-1].features_out
                features_in_list.append(features_in)
        return self._get_unused_features_generic(feature_links_chain=feature_links_chain, features_in_list=features_in_list)


    @staticmethod
    def _get_unused_features_generic(feature_links_chain: List[Dict[str, List[str]]], features_in_list: List[List[str]]) -> List[List[str]]:
        unused_features = []
        unused_features_by_stage = []
        for i, chain in enumerate(reversed(feature_links_chain)):
            stage = len(feature_links_chain) - i
            used_features = set()
            for key in chain.keys():
                new_val = [val for val in chain[key] if val not in unused_features]
                if new_val:
                    used_features.add(key)
            features_in = features_in_list[stage - 1]
            unused_features = []
            for feature in features_in:
                if feature not in used_features:
                    unused_features.append(feature)
            unused_features_by_stage.append(unused_features)
        unused_features_by_stage = list(reversed(unused_features_by_stage))
        return unused_features_by_stage

    def print_generator_info(self, log_level: int = 20):
        """
        打印输出generator的详细log信息，例如fit运行时间

        Parameters
        ----------
        log_level : int, default 20
            logging声明的log等级
        """
        if self.fit_time:
            self._log(log_level, f'\t{round(self.fit_time, 1)}s = Fit runtime')
            self._log(log_level, f'\t{len(self.features_in)} features in original data used to generate {len(self.features_out)} features in processed data.')

    def print_feature_metadata_info(self, log_level: int = 20):
        """
        输出应用(fit)特征 generator的详细日志，包括输入和输出 FeatureMetadata 对象的特征类型。

        Parameters
        ----------
        log_level : int, default 20
              logging声明的log等级
        """
        self._log(log_level, '\tTypes of features in original data (raw dtype, special dtypes):')
        self.feature_metadata_in.print_feature_metadata_full(self.log_prefix + '\t\t', log_level=log_level)
        if self.feature_metadata_real:
            self._log(log_level-5, '\tTypes of features in processed data (exact raw dtype, raw dtype):')
            self.feature_metadata_real.print_feature_metadata_full(self.log_prefix + '\t\t', print_only_one_special=True, log_level=log_level-5)
        self._log(log_level, '\tTypes of features in processed data (raw dtype, special dtypes):')
        self.feature_metadata.print_feature_metadata_full(self.log_prefix + '\t\t', log_level=log_level)

    def save(self, path: str):
        save_pkl.save(path=path, object=self)

    def _more_tags(self) -> dict:
        return {}

    def get_tags(self) -> dict:
        """获取此generator的标签。"""
        collected_tags = {}
        for base_class in reversed(inspect.getmro(self.__class__)):
            if hasattr(base_class, '_more_tags'):
                # 需要if进行判断因为mixins可能没有_more_tags
                more_tags = base_class._more_tags(self)
                collected_tags.update(more_tags)
        return collected_tags

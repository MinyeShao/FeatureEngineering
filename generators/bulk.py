import logging
from typing import List

import pandas as pd
from pandas import DataFrame

from features.feature_metadata import FeatureMetadata

from .abstract import AbstractFeatureGenerator

logger = logging.getLogger(__name__)


class BulkFeatureGenerator(AbstractFeatureGenerator):
    """
    BulkFeatureGenerator 是用来构建由多个特征generators组成的复杂特征生成piplines，其中一些generators需要将另一些generators的输出作为输入进行数据处理（多步骤generation）
    对于机器学习问题，是期望使用 BulkFeatureGenerator 实例或继承自 BulkFeatureGenerator 的特征generator进行数据处理的，因为单个特征generator通常不能满足所有输入数据类型的特征处理需求。

    Parameters
    ----------
    generators : List[List[:class:`AbstractFeatureGenerator`]]
        generators 是一个generator groups列表，一个generator group是一个generators列表（简单说就是两层list包含关系）
        所有在generator组里的generators[i]特征generator 都是在相同的数据上进行应用（fit)的，他们的输出合并连接在一起形成generators[i] 的输出。
        然后generator[i+1]基于generators[i]的输出进行应用（fit）。最后一个generator组的输出来自于_fit_transform 和_transform方法的输出
        由于generators的灵活性，在初始化时，generators将先前置pre_generators，如果post_generators不为None则添加post_generators。
        如果pre/post generators已经指定，那么它们提供generators的逻辑如下：
                pre_generators = [[pre_generator] for pre_generator in pre_generators]
                post_generators = [[post_generator] for post_generator in self._post_generators]
                self.generators: List[List[AbstractFeatureGenerator]] = pre_generators + generators + post_generators
                self._post_generators = []
        这意味着 self._post_generators 将是空的，因为 post_generators 将被合并到 self.generators 中。
        请注意，如果generator group 中的generators生成具有相同名称的特征，则会抛出 AssertionError，因为具有相同名称的特征不能出出现在同一个的DataFrame输出中。
            如果两个特征都是需要被保留下来的，在一个generators种指定name_prefix参数来防止命名冲突
            如果在不同的generator groups 进行实验，如果尝试使用不同的generator group，建议尝试在没有任何 ML 模型训练的情况下将实验性特征generator拟合到数据，以确保有效性并避免名称冲突。
    pre_generators: List[AbstractFeatureGenerator], optional
        pre_generators是按照顺序优先进行应用（fit）的generators
        pre_generators是在generators中按照顺序进行优先fit的generator
        函数与 post_generators 参数相同，但 pre_generators 在generators里顺序前置调用，而 post_generators 在generators顺序后置调用。
        为了更方便从BulkFeatureGenerator进行继承而创建的参数
        通常pre_generator中包括:class:`AsTypeFeatureGenerator`和:class:`FillNaFeatureGenerator`去清洗特征数据而不是生成新的特征
    **kwargs :
       参阅:class:`AbstractFeatureGenerator` 注释获取可用的关键参数的细节

    Examples
    --------
    >>> from ModelArt.dataset import ModelArtDataset
    >>> from ModelArt.features.generators import AsTypeFeatureGenerator, BulkFeatureGenerator, CategoryFeatureGenerator, DropDuplicatesFeatureGenerator, FillNaFeatureGenerator, IdentityFeatureGenerator
    >>> from ModelArt.common.features.types import R_INT, R_FLOAT
    >>>
    >>> generators = [
    >>>     [AsTypeFeatureGenerator()],
    >>>     [FillNaFeatureGenerator()],
    >>>     [
    >>>         CategoryFeatureGenerator(),
    >>>         IdentityFeatureGenerator(infer_features_in_args=dict(valid_raw_types=[R_INT, R_FLOAT])),
    >>>     ],
    >>>     [DropDuplicatesFeatureGenerator()]
    >>> ]
    >>> feature_generator = BulkFeatureGenerator(generators=generators, verbosity=3)
    >>>
    >>> label = 'class'
    >>> train_data = ModelArtDataset('your_path_to/train.csv')
    >>> X_train = train_data.drop(labels=[label], axis=1)
    >>> y_train = train_data[label]
    >>>
    >>> X_train_transformed = feature_generator.fit_transform(X=X_train, y=y_train)
    >>>
    >>> test_data = ModelArtDataset('your_path_to/test.csv')
    >>>
    >>> X_test_transformed = feature_generator.transform(test_data)
    """
    def __init__(self, generators: List[List[AbstractFeatureGenerator]], pre_generators: List[AbstractFeatureGenerator] = None, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(generators, list):
            generators = [[generators]]
        elif len(generators) == 0:
            raise AssertionError('generators must contain at least one AbstractFeatureGenerator.')
        generators = [generator_group if isinstance(generator_group, list) else [generator_group] for generator_group in generators]
        if pre_generators is None:
            pre_generators = []
        elif not isinstance(pre_generators, list):
            pre_generators = [pre_generators]
        if self.pre_enforce_types:
            from .astype import AsTypeFeatureGenerator
            pre_generators = [AsTypeFeatureGenerator()] + pre_generators
            self.pre_enforce_types = False
        pre_generators = [[pre_generator] for pre_generator in pre_generators]

        if self._post_generators is not None:
            post_generators = [[post_generator] for post_generator in self._post_generators]
            self._post_generators = []
        else:
            post_generators = []
        self.generators: List[List[AbstractFeatureGenerator]] = pre_generators + generators + post_generators

        for generator_group in self.generators:
            for generator in generator_group:
                if not isinstance(generator, AbstractFeatureGenerator):
                    raise AssertionError(f'generators contains an object which is not an instance of AbstractFeatureGenerator. Invalid generator: {generator}')

        self._feature_metadata_in_unused: FeatureMetadata = None  # FeatureMetadata 对象基于任何特征generator未使用的原始输入特征。

    def _fit_transform(self, X: DataFrame, **kwargs) -> (DataFrame, dict):
        feature_metadata = self.feature_metadata_in
        for i in range(len(self.generators)):
            self._log(20, f'\tStage {i + 1} Generators:')
            feature_df_list = []
            generator_group_valid = []
            for generator in self.generators[i]:
                if generator.is_valid_metadata_in(feature_metadata):
                    if generator.verbosity > self.verbosity:
                        generator.verbosity = self.verbosity
                    generator.set_log_prefix(log_prefix=self.log_prefix + '\t\t', prepend=True)
                    feature_df_list.append(generator.fit_transform(X, feature_metadata_in=feature_metadata, **kwargs))
                    generator_group_valid.append(generator)
                else:
                    self._log(15, f'\t\tSkipping {generator.__class__.__name__}: No input feature with required dtypes.')

            self.generators[i] = generator_group_valid

            self.generators[i] = [generator for j, generator in enumerate(self.generators[i]) if feature_df_list[j] is not None and len(feature_df_list[j].columns) > 0]
            feature_df_list = [feature_df for feature_df in feature_df_list if feature_df is not None and len(feature_df.columns) > 0]

            if self.generators[i]:
                # 如果generators对同一个特征期望不同的原始输入类型则抛出异常
                FeatureMetadata.join_metadatas([generator.feature_metadata_in for generator in self.generators[i]], shared_raw_features='error_if_diff')

            if self.generators[i]:
                feature_metadata = FeatureMetadata.join_metadatas([generator.feature_metadata for generator in self.generators[i]], shared_raw_features='error')
            else:
                feature_metadata = FeatureMetadata(type_map_raw=dict())

            if not feature_df_list:
                X = DataFrame(index=X.index)
            elif len(feature_df_list) == 1:
                X = feature_df_list[0]
            else:
                X = pd.concat(feature_df_list, axis=1, ignore_index=False, copy=False)

        self._remove_features_out(features=[])
        # 移除没用的generators
        for i in range(len(self.generators)):
            generator_group_valid = []
            for j in range(len(self.generators[i])):
                if self.generators[i][j].features_out:
                    generator_group_valid.append(self.generators[i][j])
            self.generators[i] = generator_group_valid

        return X, feature_metadata.type_group_map_special

    def _transform(self, X: DataFrame) -> DataFrame:
        for generator_group in self.generators:
            feature_df_list = []
            for generator in generator_group:
                feature_df_list.append(generator.transform(X))

            if not feature_df_list:
                X = DataFrame(index=X.index)
            elif len(feature_df_list) == 1:
                X = feature_df_list[0]
            else:
                X = pd.concat(feature_df_list, axis=1, ignore_index=False, copy=False)
        X_out = X
        return X_out

    def get_feature_links_chain(self):
        feature_links_chain = []
        for i in range(len(self.generators)):
            feature_links_group = {}
            for generator in self.generators[i]:
                feature_links = generator.get_feature_links()
                for feature_in, features_out in feature_links.items():
                    if feature_in in feature_links_group:
                        feature_links_group[feature_in] += features_out
                    else:
                        feature_links_group[feature_in] = features_out
            feature_links_chain.append(feature_links_group)
        return feature_links_chain

    def _remove_unused_features(self, feature_links_chain):
        unused_features_by_stage = self._get_unused_features(feature_links_chain)
        if unused_features_by_stage:
            unused_features_in = [feature for feature in self.feature_metadata_in.get_features() if feature in unused_features_by_stage[0]]
            feature_metadata_in_unused = self.feature_metadata_in.keep_features(features=unused_features_in)
            if self._feature_metadata_in_unused:
                self._feature_metadata_in_unused = self._feature_metadata_in_unused.join_metadata(feature_metadata_in_unused)
            else:
                self._feature_metadata_in_unused = feature_metadata_in_unused
            self._remove_features_in(features=unused_features_in)

        for i, generator_group in enumerate(self.generators):
            unused_features_in_stage = unused_features_by_stage[i]
            unused_features_out_stage = [feature_links_chain[i][feature_in] for feature_in in unused_features_in_stage if feature_in in feature_links_chain[i]]
            unused_features_out_stage = list(set([feature for sublist in unused_features_out_stage for feature in sublist]))
            for generator in generator_group:
                unused_features_out_generator = [feature for feature in generator.features_out if feature in unused_features_out_stage]
                generator._remove_features_out(features=unused_features_out_generator)

    def _get_unused_features(self, feature_links_chain):
        features_in_list = []
        for i in range(len(self.generators)):
            stage = i + 1
            if stage > 1:
                if self.generators[stage - 2]:
                    features_in = FeatureMetadata.join_metadatas([generator.feature_metadata for generator in self.generators[stage - 2]], shared_raw_features='error').get_features()
                else:
                    features_in = []
            else:
                features_in = self.features_in
            features_in_list.append(features_in)
        return self._get_unused_features_generic(feature_links_chain=feature_links_chain, features_in_list=features_in_list)

    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        return dict()

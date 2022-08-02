import logging
from features.types import R_INT, R_FLOAT, S_TEXT, R_OBJECT, S_IMAGE_PATH
from features.generators import PipelineFeatureGenerator,CategoryFeatureGenerator,DatetimeFeatureGenerator, IdentityFeatureGenerator,TextNgramFeatureGenerator\
,IsNanFeatureGenerator,TextSpecialFeatureGenerator
logger = logging.getLogger(__name__)


class ModelArtFeatureEngineering(PipelineFeatureGenerator):
    """
    具有简化参数的pipeline特征generator可充分处理大多数表格数据，包括文本和日期。
    如需自定义选项，参阅:class:`PipelineFeatureGenerator` and :class:`BulkFeatureGenerator`.的相关注释
    更多自定义选项参数如下：

    Parameters
    ----------
    enable_numeric_features : bool, default True
       是否保留’int‘和’float’类型的特征的原始特征类型
        这些特征将不做改变直接传递给模型
        将IdentityFeatureGenerator(infer_features_in_args=dict(valid_raw_types=['int', 'float']))) 添加到generator group.
    enable_categorical_features : bool, default True
        是否保留‘object‘和’category’类型的特征的原始特征类型
        这些特征会被处理成内存优化类型的'category'特征
        将CategoryFeatureGenerator()添加到generator group
    enable_datetime_features : bool, default True
        是否保留 'datetime' 原始类型的特征和被标记为 'datetime_as_object'类型的 'object' 特征。
        这些特征将被转换为“int”特征，表示毫秒数。
        将DatetimeFeatureGenerator()添加到generator group
    enable_text_special_features : bool, default True
        是否使用被认定为“text”特征的“object”特征来生成“text_special”特征，例如字数、大写字母比例和符号数。
        将TextSpecialFeatureGenerator()添加到generator group
    enable_text_ngram_features : bool, default True
        是否使用被认定为'text'的’object'特征去生成‘text_ngram'特征
        将TextNgramFeatureGenerator(vectorizer=vectorizer, text_ngram_params)添加到generator group，参阅text_ngram.py注释来确定可用的参数
    enable_raw_text_features : bool, default False
        是否使用原始text特征，生成的原始text特征将会以’_raw_text‘作为后缀结束
        例如, 'sentence' --> 'sentence_raw_text'
    enable_vision_features : bool, default True
        是否保留被定义为'image_path'特殊类型的’object‘特征.这种格式的特征的值应该包含一个图像文件的路径。
        是否保留标识为“image_path”特殊类型的“object”特征。这种形式的特征应该有一个图像文件的字符串路径作为它们的值。
        只有视觉模型可以利用这些特征，并且这些特征不会被视为分类特征。
        注意：不会自动推断“image_path”特征。这些功能必须在自定义 FeatureMetadata 对象中明确指定。
        注意：建议字符串路径使用绝对路径而不是相对路径，因为它可能更稳定。
    vectorizer : :class:`sklearn.feature_extraction.text.CountVectorizer`, default CountVectorizer(min_df=30, ngram_range=(1, 3), max_features=10000, dtype=np.uint8)
        sklearn CountVectorizer 对象在:class:`TextNgramFeatureGenerator`中使用.
        只有当`enable_text_ngram_features=True`时启用生效.
    **kwargs ：
        参阅:class:`AbstractFeatureGenerator` 注释获取可用关键值参数

    Examples
    --------
    >>> from ModelArt.dataset import ModelArtDataset
    >>> from ModelArt.features.generators.feature_engineering import  ModelArtFeatureEngineering
    >>>
    >>> feature_engineering = ModelArtFeatureEngineering()
    >>>
    >>> label = 'class'
    >>> train_data = ModelArtDataset('your_path_to/train.csv')
    >>> X_train = train_data.drop(labels=[label], axis=1)
    >>> y_train = train_data[label]
    >>>
    >>> X_train_transformed = feature_engineering.fit_transform(X=X_train, y=y_train)
    >>>
    >>> test_data = ModelArtDataset('your_path_to/test.csv')
    >>>
    >>> X_test_transformed = feature_engineering.transform(test_data)
    """
    def __init__(self,
                 enable_numeric_features=True,
                 enable_categorical_features=True,
                 enable_datetime_features=True,
                 enable_text_special_features=True,
                 enable_text_ngram_features=True,
                 enable_raw_text_features=False,
                 enable_vision_features=True,
                 vectorizer=None,
                 text_ngram_params=None,
                 **kwargs):
        if 'generators' in kwargs:
            raise KeyError(f'generators is not a valid parameter to {self.__class__.__name__}. Use {PipelineFeatureGenerator.__name__} to specify custom generators.')
        if 'enable_raw_features' in kwargs:
            enable_numeric_features = kwargs.pop('enable_raw_features')
            logger.warning(f"'enable_raw_features is a deprecated parameter, use 'enable_numeric_features' instead. Specifying 'enable_raw_features' will raise an exception starting in 0.1.0")

        self.enable_numeric_features = enable_numeric_features
        self.enable_categorical_features = enable_categorical_features
        self.enable_datetime_features = enable_datetime_features
        self.enable_text_special_features = enable_text_special_features
        self.enable_text_ngram_features = enable_text_ngram_features
        self.enable_raw_text_features = enable_raw_text_features
        self.enable_vision_features = enable_vision_features
        self.text_ngram_params = text_ngram_params if text_ngram_params else {}
        generators = self._get_default_generators(vectorizer=vectorizer)
        super().__init__(generators=generators, **kwargs)

    def _get_default_generators(self, vectorizer=None):
        generator_group = []
        if self.enable_numeric_features:
            generator_group.append(IdentityFeatureGenerator(infer_features_in_args=dict(
                valid_raw_types=[R_INT, R_FLOAT])))
        if self.enable_raw_text_features:
            generator_group.append(IdentityFeatureGenerator(infer_features_in_args=dict(
                required_special_types=[S_TEXT], invalid_special_types=[S_IMAGE_PATH]), name_suffix='_raw_text'))
        if self.enable_categorical_features:
            generator_group.append(CategoryFeatureGenerator())
        if self.enable_datetime_features:
            generator_group.append(DatetimeFeatureGenerator())
        if self.enable_text_special_features:
            generator_group.append(TextSpecialFeatureGenerator())
        if self.enable_text_ngram_features:
            generator_group.append(TextNgramFeatureGenerator(vectorizer=vectorizer, **self.text_ngram_params))
        if self.enable_vision_features:
            generator_group.append(IdentityFeatureGenerator(infer_features_in_args=dict(
                valid_raw_types=[R_OBJECT], required_special_types=[S_IMAGE_PATH],
            )))
            generator_group.append(IsNanFeatureGenerator(infer_features_in_args=dict(
                valid_raw_types=[R_OBJECT], required_special_types=[S_IMAGE_PATH],
            )))
        generators = [generator_group]
        return generators

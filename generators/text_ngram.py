import copy
import logging
import traceback

import numpy as np
import pandas as pd
import psutil
from pandas import DataFrame, Series
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from features.types import S_IMAGE_PATH, S_TEXT, S_TEXT_NGRAM

from features.generators.abstract import AbstractFeatureGenerator
from features.vectorizers import get_ngram_freq, downscale_vectorizer, vectorizer_auto_ml_default

logger = logging.getLogger(__name__)


class TextNgramFeatureGenerator(AbstractFeatureGenerator):
    """
    从文本特征中生成ngram特征

    Parameters
    ----------
    vectorizer : :class:`sklearn.feature_extraction.text.CountVectorizer`, default CountVectorizer(min_df=30, ngram_range=(1, 3), max_features=10000, dtype=np.uint8)
        sklearn CountVectorizer通过文本数据生成Ngram词频统计向量
    vectorizer_strategy : str, default 'combined'
        如果为‘combined’则将所有文本特征拼接在一起去fit vectorizer。通过这种方法生成的特征会在名称上加上'__nlp__.'的名称前缀
        如果为’separate‘则所有文本特征会分别去fit各种的vectorizer的复制，最后每个文本特征得出的ngram会拼接在一起形成输出
        如果为’both‘则将‘combined’ 和’separate‘的输出结果拼在一起形成最终输出
        通常建议将 vectorizer_strategy 保持‘conbined’，除非文本特征彼此不相关，因为fit单独的vectorizer可能会增加内存使用和fit时间。
        有效的values: ['combined', 'separate', 'both']
    max_memory_ratio : float, default 0.15
        避免模型训练时内存溢出的计量比例阈值
        生成的 ngram 数量将被限制为最多占总可用内存的 max_memory_ratio 比例，将 ngram 视为 float32 值。
        ngram特征将会按照最低频次到最高频次的规则进行移除
        提示：如果vectorizer_strategy设定的值不是’combined‘，返回的结果ngram大小可能多于这个比例
        如果确定更高的值不会导致内存不足错误，则建议仅将此值增加到 0.15 左右。
    **kwargs :
        参阅 :class:`AbstractFeatureGenerator` 注释获得更多有关有效参数的细节
    """
    def __init__(self, vectorizer=None, vectorizer_strategy='combined', max_memory_ratio=0.15, prefilter_tokens=False, prefilter_token_count=100, **kwargs):
        super().__init__(**kwargs)
        self.vectorizers = []
        self.max_memory_ratio = max_memory_ratio  # 输出 ngram 特征允许以dense int32 形式使用的最大内存比率。
        if vectorizer is None:
            self.vectorizer_default_raw = vectorizer_auto_ml_default()
        else:
            self.vectorizer_default_raw = vectorizer

        if vectorizer_strategy not in ['combined', 'separate', 'both']:
            raise ValueError(f"vectorizer_strategy must be one of {['combined', 'separate', 'both']}, but value is: {vectorizer_strategy}")
        self.vectorizer_strategy = vectorizer_strategy
        self.vectorizer_features = None
        self.prefilter_tokens = prefilter_tokens 
        self.prefilter_token_count = prefilter_token_count 
        self.token_mask = None
        self._feature_names_dict = dict()

    def _fit_transform(self, X: DataFrame, y: Series = None, problem_type: str = None, **kwargs) -> (DataFrame, dict):
        X_out = self._fit_transform_ngrams(X)
        
        if self.prefilter_tokens and self.prefilter_token_count >= X_out.shape[1]:
            logger.warning('`prefilter_tokens` was enabled but `prefilter_token_count` larger than the vocabulary. Disabling `prefilter_tokens`.')
            self.prefilter_tokens = False

        if self.prefilter_tokens and problem_type not in ['binary', 'regression']:
            logger.warning('`prefilter_tokens` was enabled but invalid `problem_type`. Disabling `prefilter_tokens`.')
            self.prefilter_tokens = False

        if self.prefilter_tokens and y is None:
            logger.warning('`prefilter_tokens` was enabled but `y` values were not provided to fit_transform. Disabling `prefilter_tokens`.')
            self.prefilter_tokens = False

        if self.prefilter_tokens:
            scoring_function = f_classif if problem_type == 'binary' else f_regression
            selector = SelectKBest(scoring_function, k=self.prefilter_token_count)
            selector.fit(X_out, y)
            self.token_mask = selector.get_support()
            X_out = X_out[X_out.columns[self.token_mask]]  # 选择与y相关性最高的列

        type_family_groups_special = {
            S_TEXT_NGRAM: list(X_out.columns)
        }
        return X_out, type_family_groups_special

    def _transform(self, X: DataFrame) -> DataFrame:
        if not self.features_in:
            return DataFrame(index=X.index)
        try:
            X_out = self._generate_ngrams(X=X)
            if self.prefilter_tokens:
                X_out = X_out[X_out.columns[self.token_mask]]  # 选择训练期间指定的列
        except Exception:
            self._log(40, '\tError: OOM error during NLP feature transform, unrecoverable. Increase memory allocation or reduce data size to avoid this error.')
            raise
        return X_out

    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        return dict(required_special_types=[S_TEXT], invalid_special_types=[S_IMAGE_PATH])

    def _fit_transform_ngrams(self, X):
        if not self.features_in:
            return DataFrame(index=X.index)
        features_nlp_to_remove = []
        if self.vectorizer_strategy == 'combined':
            self.vectorizer_features = ['__nlp__']
        elif self.vectorizer_strategy == 'separate':
            self.vectorizer_features = copy.deepcopy(self.features_in)
        elif self.vectorizer_strategy == 'both':
            self.vectorizer_features = ['__nlp__'] + copy.deepcopy(self.features_in)
        else:
            raise ValueError(f"vectorizer_strategy must be one of {['combined', 'separate', 'both']}, but value is: {self.vectorizer_features}")
        self._log(20, f'Fitting {self.vectorizer_default_raw.__class__.__name__} for text features: ' + str(self.features_in), self.log_prefix + '\t')
        self._log(15, f'{self.vectorizer_default_raw}', self.log_prefix + '\t\t')
        for nlp_feature in self.vectorizer_features:

            if nlp_feature == '__nlp__':  # 将文本信息拼接起来
                features_in_str = X[self.features_in].astype(str)
                text_list = list(set(['. '.join(row) for row in features_in_str.values]))
            else:
                text_list = list(X[nlp_feature].astype(str).drop_duplicates().values)
            vectorizer_raw = copy.deepcopy(self.vectorizer_default_raw)
            try:
                vectorizer_fit, _ = self._train_vectorizer(text_list, vectorizer_raw)  # Don't use transform_matrix output because it may contain fewer rows due to drop_duplicates call.
                self._log(20, f'{vectorizer_fit.__class__.__name__} fit with vocabulary size = {len(vectorizer_fit.vocabulary_)}', self.log_prefix + '\t')
            except ValueError:
                self._log(30, f"Removing text_ngram feature due to error: '{nlp_feature}'", self.log_prefix + '\t')
                if nlp_feature == '__nlp__':
                    self.vectorizer_features = []
                    features_nlp_to_remove = self.features_in
                    break
                else:
                    features_nlp_to_remove.append(nlp_feature)
            else:
                self.vectorizers.append(vectorizer_fit)
        self._remove_features_in(features_nlp_to_remove)

        downsample_ratio = None
        nlp_failure_count = 0
        X_text_ngram = None
        keep_trying_nlp = True
        while keep_trying_nlp:
            try:
                X_text_ngram = self._generate_ngrams(X=X, downsample_ratio=downsample_ratio)
                keep_trying_nlp = False
            except Exception as err:
                nlp_failure_count += 1
                traceback.print_tb(err.__traceback__)

                X_text_ngram = None
                skip_nlp = False
                for vectorizer in self.vectorizers:
                    vocab_size = len(vectorizer.vocabulary_)
                    if vocab_size <= 50:
                        skip_nlp = True
                        break
                else:
                    if nlp_failure_count >= 3:
                        skip_nlp = True

                if skip_nlp:
                    self._log(30, 'Warning: ngrams generation resulted in OOM error, removing ngrams features. If you want to use ngrams for this problem, increase memory allocation for AutoGluon.', self.log_prefix + '\t')
                    self._log(10, str(err))
                    self.vectorizers = []
                    self.features_in = []
                    keep_trying_nlp = False
                else:
                    self._log(20, 'Warning: ngrams generation resulted in OOM error, attempting to reduce ngram feature count. If you want to optimally use ngrams for this problem, increase memory allocation for AutoGluon.', self.log_prefix + '\t')
                    self._log(10, str(err))
                    downsample_ratio = 0.25
        if X_text_ngram is None:
            X_text_ngram = DataFrame(index=X.index)
        return X_text_ngram

    def _generate_ngrams(self, X, downsample_ratio: int = None):
        X_nlp_features_combined = []
        for nlp_feature, vectorizer_fit in zip(self.vectorizer_features, self.vectorizers):
            if nlp_feature == '__nlp__':
                X_str = X.astype(str)
                text_data = ['. '.join(row) for row in X_str.values]
            else:
                nlp_feature_str = X[nlp_feature].astype(str)
                text_data = nlp_feature_str.values
            transform_matrix = vectorizer_fit.transform(text_data)

            if not self._is_fit:
                transform_matrix = self._adjust_vectorizer_memory_usage(transform_matrix=transform_matrix, text_data=text_data, vectorizer_fit=vectorizer_fit, downsample_ratio=downsample_ratio)
                nlp_features_names = vectorizer_fit.get_feature_names_out()
                nlp_features_names_final = np.array([f'{nlp_feature}.{x}' for x in nlp_features_names]
                                                    + [f'{nlp_feature}._total_']
                                                    )
                self._feature_names_dict[nlp_feature] = nlp_features_names_final

            transform_array = transform_matrix.toarray()
            # This count could technically overflow in absurd situations. Consider making dtype a variable that is computed.
            nonzero_count = np.count_nonzero(transform_array, axis=1).astype(np.uint16)
            transform_array = np.append(transform_array, np.expand_dims(nonzero_count, axis=1), axis=1)
            X_nlp_features = pd.DataFrame(transform_array, columns=self._feature_names_dict[nlp_feature], index=X.index)
            X_nlp_features_combined.append(X_nlp_features)

        if X_nlp_features_combined:
            if len(X_nlp_features_combined) == 1:
                X_nlp_features_combined = X_nlp_features_combined[0]
            else:
                X_nlp_features_combined = pd.concat(X_nlp_features_combined, axis=1)
        else:
            X_nlp_features_combined = DataFrame(index=X.index)

        return X_nlp_features_combined


    def _adjust_vectorizer_memory_usage(self, transform_matrix, text_data, vectorizer_fit, downsample_ratio: int = None):
        # This assumes that the ngrams eventually turn into int32/float32 downstream
        predicted_ngrams_memory_usage_bytes = len(text_data) * 4 * (transform_matrix.shape[1] + 1) + 80
        mem_avail = psutil.virtual_memory().available
        mem_rss = psutil.Process().memory_info().rss
        predicted_rss = mem_rss + predicted_ngrams_memory_usage_bytes
        predicted_percentage = predicted_rss / mem_avail
        if downsample_ratio is None:
            if self.max_memory_ratio is not None and predicted_percentage > self.max_memory_ratio:
                downsample_ratio = self.max_memory_ratio / predicted_percentage
                self._log(30, 'Warning: Due to memory constraints, ngram feature count is being reduced. Allocate more memory to maximize model quality.')

        if downsample_ratio is not None:
            if (downsample_ratio >= 1) or (downsample_ratio <= 0):
                raise ValueError(f'downsample_ratio must be >0 and <1, but downsample_ratio is {downsample_ratio}')
            vocab_size = len(vectorizer_fit.vocabulary_)
            downsampled_vocab_size = int(np.floor(vocab_size * downsample_ratio))
            self._log(20, f'Reducing Vectorizer vocab size from {vocab_size} to {downsampled_vocab_size} to avoid OOM error')
            ngram_freq = get_ngram_freq(vectorizer=vectorizer_fit, transform_matrix=transform_matrix)
            downscale_vectorizer(vectorizer=vectorizer_fit, ngram_freq=ngram_freq, vocab_size=downsampled_vocab_size)
            transform_matrix = vectorizer_fit.transform(text_data)

        return transform_matrix

    @staticmethod
    def _train_vectorizer(text_data: list, vectorizer):
        transform_matrix = vectorizer.fit_transform(text_data)
        vectorizer.stop_words_ = None  # Reduces object size by 100x+ on large datasets, no effect on usability
        return vectorizer, transform_matrix

    def _remove_features_in(self, features):
        super()._remove_features_in(features)
        if features:
            self.vectorizer_features = [feature for feature in self.vectorizer_features if feature not in features]

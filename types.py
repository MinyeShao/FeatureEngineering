
# 原始类型: 分好父类族的原始数据类型信息.
# 比如: uint8, int8, int16, int32, int64 数据类型的特征 映射为'int'类型
R_INT = 'int'
R_FLOAT = 'float'
R_OBJECT = 'object'
R_CATEGORY = 'category'
R_DATETIME = 'datetime'
R_BOOL = 'bool'

# 特殊类型：有关原始数据中不存在的特征的特殊含义的元信息。
# 功能已转换为只有 0 和 1 个值的 int8 原始数据类型（2 个唯一值，缺少转换为 0）
# 这与 R_BOOL 不同，因为 R_BOOL 指的是具有 bool 原始数据类型 的功能（值为 False 和 True 而不是 0 和 1）
S_BOOL = 'bool'

# 特征已从其原始表示中分箱成离散整数值
S_BINNED = 'binned'

# 特征初始为datetime类型可以转换为numeric类型
S_DATETIME_AS_INT = 'datetime_as_int'

# 特征是对象形式的日期时间（字符串日期），可以通过 pd.to_datetime 转换为日期时间
S_DATETIME_AS_OBJECT = 'datetime_as_object'

# 包含文本信息的可用于nlp的特征类型
S_TEXT = 'text'

# 分类型特征，初始值是文本信息。数据中可能包含也可能不包含原始文本。
S_TEXT_AS_CATEGORY = 'text_as_category'

# 特征是基于文本特征的生成特征，但不是 ngram。示例包括字符数、字数、符号数等。
S_TEXT_SPECIAL = 'text_special'

# 特征是基于文本特征的生成特征，该文本特征是一个 ngram。
S_TEXT_NGRAM = 'text_ngram'

# 特征是一种对象类型，其中包含可以在计算机视觉中使用的图像的字符串路径
S_IMAGE_PATH = 'image_path'

# 特征是基于 ML 模型对行的标签列的预测概率生成的特征。
# 任何以堆栈特征作为输入的模型都是堆栈集成。
S_STACK = 'stack'

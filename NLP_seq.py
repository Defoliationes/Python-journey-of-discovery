class Word2Sequence:
    UNK_TAG = "UNK"
    PAD_TAG = "PAD"

    UNK = 0
    PAD = 1

    def __init__(self):
        self.dict = {
            self.UNK_TAG: self.UNK,
            self.PAD_TAG: self.PAD
        }

        self.count = {}

    def fit(self, sentence):
        """
        把单个句子保存到dict中
        :param sentence: [word1, word2, word3,.....]
        :return:
        """
        for word in sentence:
            self.count[word] = self.count.get(word, 0) + 1


    def build_vocab(self, min=5, max=None, max_features=None):
        """
        生成词典
        :param min:最小出现的次数
        :param max: 最大出现的次数
        :param max_features: 一共保留多少个词语
        :return:
        """

        #删除count中词频小于min的word
        if min is not None:
            self.count = {word:value for word, value in self.count.items() if value>min}

        if max is not None:
            self.count = {word:value for word, value in self.count.items() if value<max}

        if max_features is not None:
            temp = sorted(self.count.items(), key=lambda x:x[-1], reverse=True)[:max_features]
            self.count = dict(temp)

        for word in self.count:
            self.dict[word] = len(self.dict)

        # 获取一个反转的字典
        self.inverse_dict = dict(zip(self.dict.values(), self.dict.keys()))

    def transform(self, sentence, max_len=None):
        """
        把句子转化为序列
        :param sentence:
        :param max_len: int,对句子进行填充或者裁剪
        :return:
        """
        # for word in sentence:
        #     self.dict.get(word, self.UNK)
        if max_len is not None:
            if max_len > len(sentence):
                sentence = sentence + [self.PAD_TAG] * (max_len - len(sentence))
            if max_len < len(sentence):
                sentence = sentence[:max_len]

        return [self.dict.get(word, self.UNK) for word in sentence]

    def inverse_transform(self, indices):
        """
        把序列转化为句子
        :param indices:【1，2，3，。。。。。】
        :return:
        """
        return [self.inverse_dict.get(idx) for idx in indices]

    def __len__(self):
        return len(self.dict)

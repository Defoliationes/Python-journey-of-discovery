import os
import json

class HMM:
    def __init__(self, wordtxt, algorithm):
        # 初始化参数
        self.wordtxt = wordtxt
        self.algorithm = algorithm

        self.trans_prob = {}  #转移概率
        self.emit_prob = {}   #发射概率
        self.init_prob = {}   #初始概率
        self.Count_dict = {}  #状态出现次数
        self.state_list = ['B', 'M', 'E', 'S']

        for state in self.state_list:
            trans = {}
            for s in self.state_list:
                trans[s] = 0
            self.trans_prob[state] = trans
            self.emit_prob[state] = {}
            self.init_prob[state] = 0
            self.Count_dict[state] = 0

    def train(self):
        count = -1
        # 读取并处理单词、计算概率矩阵
        path = self.wordtxt
        for line in open(path, 'r'):
            count += 1
            line = line.strip()
            if not line:
                continue

            word_list = [] # 读取每一行的单词
            for i in line:
                if i != ' ':
                    word_list.append(i)

            word_label = [] # 标注每个单词的位置标签
            for word in line.split():
                label = []
                if len(word) == 1:
                    label.append('S')
                else:
                    label += ['B'] + ['M'] * (len(word) - 2) + ['E']
                word_label.extend(label)

            # 统计各个位置状态下的出现次数，用于计算概率
            for index, value in enumerate(word_label):
                self.Count_dict[value] += 1
                if index == 0:
                    self.init_prob[value] += 1
                else:
                    self.trans_prob[word_label[index - 1]][value] += 1
                    self.emit_prob[word_label[index]][word_list[index]] = self.emit_prob[word_label[index]].get(word_list[index], 0) + 1.0

        for key, value in self.init_prob.items(): # 初始概率
            self.init_prob[key] = value * 1 / count

        for key, value in self.trans_prob.items(): # 转移概率
            for k, v in value.items():
                value[k] = v / self.Count_dict[key]
            self.trans_prob[key] = value

        for key, value in self.emit_prob.items(): # 发射概率，采用加1平滑
            for k, v in value.items():
                value[k] = (v + 1) / self.Count_dict[key]
            self.emit_prob[key] = value

            # 将三个概率矩阵保存至json文件
            model = 'hmm_model.json'
            f = open(model, 'a+')
            f.write(json.dumps(self.trans_prob) + '\n' + json.dumps(self.emit_prob) + '\n' + json.dumps(self.init_prob))
            f.close()

    def viterbi(self, text):
        V = [{}]
        path = {}
        # 初始概率
        for state in self.state_list:
            V[0][state] = self.init_prob[state] * self.emit_prob[state].get(text[0], 0)
            path[state] = [state]

        # 当前语料中所有的字
        key_list = []
        for key, value in self.emit_prob.items():
            for k, v in value.items():
                key_list.append(k)

        # 计算待分词文本的状态概率值，得到最大概率序列
        for t in range(1, len(text)):
            V.append({})
            newpath = {}
            for state in self.state_list:
                if text[t] in key_list:
                    emit_count = self.emit_prob[state].get(text[t], 0)
                else:
                    emit_count = 1
                (prob, a) = max([(V[t - 1][s] * self.trans_prob[s].get(state, 0) * emit_count, s) for s in self.state_list if
                                 V[t - 1][s] > 0])
                V[t][state] = prob
                newpath[state] = path[a] + [state]
            path = newpath

        # 根据末尾字的状态，判断最大概率状态序列
        if self.emit_prob['M'].get(text[-1], 0) > self.emit_prob['S'].get(text[-1], 0):
            (prob, a) = max([(V[len(text) - 1][s], s) for s in ('E', 'M')])
        else:
            (prob, a) = max([(V[len(text) - 1][s], s) for s in self.state_list])

        return prob, path[a]


    def cut(self, text):
        self.train()
        # 利用维特比算法，求解最大概率状态序列
        if self.algorithm == 'viterbi':
            prob, pos_list = self.viterbi(text)
        else:
            pass
        # 判断待分词文本每个字的状态，输出结果
        begin, follow = 0, 0
        for index, char in enumerate(text):
            state = pos_list[index]
            if state == 'B':
                begin = index
            elif state == 'E':
                yield text[begin: index + 1]
                follow = index + 1
            elif state == 'S':
                yield char
                follow = index + 1
        if follow < len(text):
            yield text[follow:]

        return text

if __name__ == '__main__':
    wordtxt = 'trainCorpus.txt'
    algorithm = 'viterbi'
    model = HMM(wordtxt, algorithm)
    text = '深航客机攀枝花机场遇险：机腹轮胎均疑受损，跑道灯部分损坏！'
    model.cut(text)
    print(text)
    print(str(list(model.cut(text))))
    pass
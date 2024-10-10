from NLP_seq import Word2Sequence
import pickle
import os
from NLP_data import tokenlize
from tqdm import tqdm

if __name__ == '__main__':
    # ws = Word2Sequence()
    # ws.fit(["我", "是", "谁"])
    # ws.fit(["我", "是", "我"])
    # ws.build_vocab(min=0)
    # print(ws.dict)

    # ret = ws.transform(["我", "爱", "中国"], max_len=10)
    # print(ret)
    # ret = ws.inverse_transform(ret)
    # print(ret)

    ws = Word2Sequence()
    path = r"C:\Users\cai03\PycharmProjects\pythonProject\NLP\movie_data\aclImdb\train"
    temp_data_path = [os.path.join(path, 'pos'), os.path.join(path, "neg")]

    for data_path in temp_data_path:

        file_paths = [os.path.join(data_path, file_name) for file_name in os.listdir(data_path)]

        for file_path in tqdm(file_paths):
            sentence = tokenlize(open(file_path, errors='ignore').read())
            ws.fit(sentence)

    ws.build_vocab(min=10, max_features=10000)
    pickle.dump(ws, open("C:/Users/cai03/PycharmProjects/pythonProject/NLP/ws.pkl", "wb"))
    print(len(ws))
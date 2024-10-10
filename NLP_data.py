from torch.utils.data import DataLoader, Dataset
import NLP_lib as lib
import torch
import os
import re
#数据获取
def tokenlize(content):
    content = re.sub("<.*?>", " ",content)
    fileters = ['\.', ':', '\t', '\n', '\x97', '\x96', '\)', '\(', '#', '$', '%', '&']
    content = re.sub("|".join(fileters), " ", content)
    tokens = [i.strip().lower() for i in content.split()]
    return tokens

class ImdbDataset(Dataset):
    def __init__(self, train=True):
        self.train_data_path = r"C:\Users\cai03\PycharmProjects\pythonProject\NLP\movie_data\aclImdb\train"
        self.test_data_path = r"C:\Users\cai03\PycharmProjects\pythonProject\NLP\movie_data\aclImdb\test"
        data_path = self.train_data_path if train else self.test_data_path

        # 把所有的文件都放入列表中
        temp_data_path = [os.path.join(data_path, 'pos'), os.path.join(data_path, "neg")]
        self.total_file_path = [] # 所有的评论的文件的path
        for path in temp_data_path:
            file_name_list = os.listdir(path)
            file_path_list = [os.path.join(path, i) for i in file_name_list if i.endswith(".txt")]
            self.total_file_path.extend(file_path_list)

    def __getitem__(self, index):
        file_path = self.total_file_path[index]
        # return file_path
        # 获取label
        label_str = file_path.split("\\")[-2]
        # print(label_str)
        label = 0 if label_str =="neg" else 1
        # 获取内容
        with open(file_path, 'rb') as file:
            cont = file.read().decode('utf-8', errors='ignore')  # 使用 UTF-8 编码解码文件内容
        tokens = tokenlize(cont)
        # tokens = tokenlize(open(file_path).read())
        return tokens, label

    def __len__(self):
        return len(self.total_file_path)


def collate_fn(batch):
    """

    :param batch: (一个getitem的结果，一个getitem的结果。。。。）([tokens, label], [tokens, label]..........)
    :return:
    """
    content, label = list(zip(*batch))
    content = [lib.ws.transform(i, max_len=lib.max_len) for i in content]
    content = torch.LongTensor(content)
    label = torch.LongTensor(label)
    return content, label

def get_dataloader(batch_size = lib.batch_size, train=True):
    imdb_dataset = ImdbDataset(train)
    data_loader = DataLoader(imdb_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    return data_loader

if __name__ =='__main__':
   # imdb_dataset = ImdbDataset()
   # print(imdb_dataset[0])
   # my_str = "Once again Mr. Costner has dragged out a movie for far longer than necessary. Aside from the terrific sea rescue sequences, of which there are very few I just did not care about any of the characters. Most of us have ghosts in the closet, and Costner's character are realized early on, and then forgotten until much later"
   # print(tokenlize(my_str))
   for idx, (input, target) in enumerate(get_dataloader()):
       print(idx)
       print(input)
       print(target)
       break
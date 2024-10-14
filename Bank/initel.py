import pickle

# 创建一个空的pkl文件
empty_file = 'Bank_users.pkl'
empty_dict = {}
with open(empty_file, 'wb') as file:
    pickle.dump(empty_dict, file)
# 创建一个空的txt文件

empty_txt_file = 'ID.txt'
with open(empty_txt_file, 'w') as file:
    file.write("12340000")
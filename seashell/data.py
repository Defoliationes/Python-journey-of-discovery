import re
import pandas as pd
import requests
from bs4 import BeautifulSoup
data = []
for i in range(100):
    url = f"https://bj.zu.ke.com/zufang/pg{i}/#contentList"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    for text in soup.find_all("div", class_ = "content__list--item"):
        text_1 = text.find("img").get("alt")
        leixing = text_1.replace("·", " ").split(' ')
        text_2 = text.find("p", class_="content__list--item--des").text
        shuxing = text_2.replace("/", "").replace(" ", "").replace("\n\n", "\n").split('\n')
        # 分解后，列表开头和最后，会有一个空字符元素
        shuxing.pop(0)
        shuxing.pop(-1)
        text_3 = text.find("p", class_="content__list--item--bottom oneline").text
        tedian = re.sub(r'[^\w\s]', '', text_3).replace(" ", "").replace("\n", ",")
        text_4 = text.find("p", class_="content__list--item--brand oneline").text
        laiyuan = re.sub(r'[^\w\s]', '', text_4).replace(" ", "").split('\n')
        text_5 = text.find("span", class_="content__list--item-price").text
        jiage = text_5.split(' ')[0]
        data.append([shuxing, leixing[0], jiage, laiyuan[2], tedian[1:-1]])
df = pd.DataFrame(data, columns=['属性', '租贷类型', '价格（元/月）', '来源', '特点'])
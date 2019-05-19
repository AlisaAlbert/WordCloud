#!/usr/bin/env python
# encoding: utf-8
# author: AlisaAlbert
# 2019/4/24 21:34

import pandas as pd
import numpy as np
import wordcloud
import jieba
import jieba.posseg as pseg
import sqlite3
import matplotlib.pyplot as plt
from collections import Counter as cr
import re
from PIL import Image
from wordcloud import WordCloud, ImageColorGenerator, STOPWORDS
import codecs
from scipy.misc import imread

#读入记录并过滤字符
def read_chat_history(word):
    conn = sqlite3.connect(word)
    wordtmp = pd.read_sql('select * from table;', conn)
    wordtmp2 = wordtmp[['Message']].copy()
    # 过滤英文和字符
    wordtmp2['message'] = wordtmp2.Message.apply(
        lambda x: re.sub('[A-Za-z0-9\[\`\~\!\@\#\$\^\&\*\(\)\=\|\{\}\'\:\：\;\；\、\'\,\[\]\.\<\>\/\?\~\。\@\#\\\&\*\%]'
                         , '',str(x)))
    del wordtmp2['Message']
    return wordtmp2

#切词过滤停用词
def deal_with_word(word,swpath):
    # 切词
    word['message_tmp'] = word.message.apply(lambda x: ' '.join(jieba.cut(str(x))))
    stopwords = pd.read_csv(swpath, index_col=False, quoting=3, sep='\t', names=['stopword'], encoding='utf-8')
    words_df = word[~word['message_tmp'].isin(stopwords.stopword)]
    words_df2 = words_df.join(
        words_df['message_tmp'].str.split(' ', expand=True).stack().reset_index(level=1, drop=True).rename('message_tmp2'))
    words_df2.reset_index(drop=True, inplace=True)
    words_df2 = words_df2[~words_df2['message_tmp'].isin(stopwords.stopword)]
    return words_df2

# 统计词频
def word_count(word):
    words_count = word.groupby(by=['message_tmp2'])['message_tmp2'].agg({'count': np.size})
    words_count = words_count.reset_index().sort_values(["count"], ascending=False)
    #words_dict = words_count.set_index("message_tmp2").to_dict()
    words_dict = ' '.join(words_count['message_tmp2'].tolist())
    return words_dict

#制作词云
def get_wordcloud(image,font,sw,word,result):
    wordcloud = WordCloud(scale=15, font_path=font, mask=image, stopwords=sw, background_color='white',
                          max_words=80000,max_font_size=10, random_state=42)
    wordcloud.generate(word)
    img_colors = ImageColorGenerator(image)
    plt.imshow(wordcloud.recolor(color_func=img_colors))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()
    wordcloud.to_file(result)
    print('Task Done!')

if __name__ == '__main__':
    # 读入
    wordpath = r'path\MM.sqlite'
    chat = read_chat_history(wordpath)
    # 切词
    swpath = r'path\stopword.txt'
    word_split = deal_with_word(chat, swpath)
    # 词频
    words_dict = word_count(word_split)
    # 加载图片和字体
    imagepath = r'path\xsp.jpg'
    bg_image = plt.imread(imagepath)  # 背景图片
    font = r'C:\Windows\Fonts\simhei.ttf'  # 字体
    sw = set(STOPWORDS)  # 停用词
    resultpath = r'path\xspcute.jpg'
    get_wordcloud(bg_image, font, sw, words_dict, resultpath)
import jieba
import os  # 用于处理文件路径
import numpy as np
from gensim.models import Word2Vec
import re
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib.font_manager import FontProperties

def content_deal(content):
    ad = '本书来自www.cr173.com免费txt小说下载站\n更多更新免费电子书请关注www.cr173.com'  #去除无意义的广告词
    content = content.replace(ad, '')
    content = content.replace("\u3000", '')
    return content

def read_novel(path_in, path_out):  # 读取语料内容
    content = []
    print(os.getcwd())
    # names = os.listdir(path_in)
    names = ['白马啸西风', '碧血剑', '飞狐外传', '连城诀', '鹿鼎记', '三十三剑客图', '射雕英雄传', '神雕侠侣',
                  '书剑恩仇录','天龙八部', '侠客行', '笑傲江湖', '雪山飞狐', '倚天屠龙记', '鸳鸯刀', '越女剑']
    for name in names:
        tmp = []
        novel_name = path_in + '\\' + name + '.txt'
        # fenci_name = path_out + '\\' + name
        for line in open(novel_name, 'r', encoding='ANSI'):
            line.strip('\n')
            line = re.sub("[A-Za-z0-9\：\·\—\，\。\“\”\\n \《\》\！\？\、\...]", "", line)
            line = content_deal(line)
            con = list(jieba.cut(line, cut_all=False))  # 结巴分词
            tmp.append(con)
        content.append(tmp)
    return content, names


if __name__ == '__main__':

    book_names = ['白马啸西风', '碧血剑', '飞狐外传', '连城诀', '鹿鼎记', '三十三剑客图', '射雕英雄传', '神雕侠侣',
                  '书剑恩仇录',
                  '天龙八部', '侠客行', '笑傲江湖', '雪山飞狐', '倚天屠龙记', '鸳鸯刀', '越女剑']
    [data_txt, files] = read_novel("./data", "./output")

    for i in range(len(book_names)):
        name = "output/" + files[i]
        print(name)
        word2vec_model = Word2Vec(data_txt[i], hs=1, min_count=5, window=5, vector_size=200, sg=0, epochs=100)

        # 提取所有词的词向量
        # if name == 'output/天龙八部.txt':
        #     words = list(word2vec_model.wv.index_to_key)
        #     word_vectors = word2vec_model.wv[words]
        #     similarity_1 = word2vec_model.wv.similarity('萧峰', '乔峰')
        #     print("Similarity between '萧峰' and '乔峰':", similarity_1)
        #
        #     similarity_1 = word2vec_model.wv.similarity('萧峰', '萧远山')
        #     print("Similarity between '萧峰' and '萧远山':", similarity_1)
        #
        #     similarity_1 = word2vec_model.wv.similarity('萧峰', '虚竹')
        #     print("Similarity between '萧峰' and '虚竹':", similarity_1)
        #
        #     similarity_1 = word2vec_model.wv.similarity('萧峰', '段誉')
        #     print("Similarity between '萧峰' and '段誉':", similarity_1)
        #
        #     similarity_1 = word2vec_model.wv.similarity('萧峰', '慕容复')
        #     print("Similarity between '萧峰' and '慕容复':", similarity_1)
        #
        #     similarity_1 = word2vec_model.wv.similarity('萧峰', '逍遥子')
        #     print("Similarity between '萧峰' and '逍遥子':", similarity_1)
        #
        #
        #     similarity_1 = word2vec_model.wv.similarity('师兄', '师弟')
        #     print("Similarity between '师兄 and '师弟':", similarity_1)

        words = list(word2vec_model.wv.index_to_key)[:1000]
        word_vectors = word2vec_model.wv[words][:1000]

        # 使用 t-SNE 将词向量降到2D
        tsne = TSNE(n_components=2, random_state=0)
        reduced_vectors = tsne.fit_transform(word_vectors)

        font_path = 'C:/Windows/Fonts/simhei.ttf'  # 你可以根据实际路径选择适合的字体
        font_prop = FontProperties(fname=font_path)

        # 绘制散点图
        plt.figure(figsize=(24, 20))
        for j, word in enumerate(words):
            plt.scatter(reduced_vectors[j, 0], reduced_vectors[j, 1])
            plt.annotate(word, xy=(reduced_vectors[j, 0], reduced_vectors[j, 1]), fontproperties=font_prop)

        plt.savefig(f"picture/{files[i]}.png")
        plt.show()
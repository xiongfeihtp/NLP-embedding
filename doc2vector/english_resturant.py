import xml.etree.ElementTree as ET
import sys
import random
import sys
import gensim
import numpy as np

from gensim.models.doc2vec import Doc2Vec,LabeledSentence
from sklearn.model_selection import train_test_split
LabeledSentence = gensim.models.doc2vec.LabeledSentence
try:
    tree = ET.parse('Restaurants_Train.xml')  # 打开xml文档
    # root = ET.fromstring(country_string) #从字符串传递xml
    root = tree.getroot()  # 获得root节点
except Exception as e:
    print("Error:cannot parse file:country.xml.")
    sys.exit(1)
train_dict = {}
#如何在xml文本格式下解析样本和标签
for i, country in enumerate(root.findall('sentence')):  # 找到root节点下的所有country节点
    if not i % 100:
        print(i)
    text = country.find('text').text  # 子节点下节点rank的值
    label= [(x.get('category'),x.get('polarity')) for x in country.find('aspectCategories').findall('aspectCategory')]
    train_dict[i] = [text,label]

def cleanText(corpus):
    punctuation = """.,?!:;(){}[]"""
    # treat punctuation as individual words
    for c in punctuation:
        corpus = [z.replace(c, ' %s ' % c) for z in corpus]
    corpus = [z.split() for z in corpus]
    return corpus
input_list=[]
label_list=[]
for item in train_dict.items():
    input_list.append(item[1][0])
cleaned_input=cleanText(input_list)
def labelizeReviews(reviews, label_type):
    labelized = []
    for i, v in enumerate(reviews):
        label = '%s_%s' % (label_type, i)
        labelized.append(LabeledSentence(v, [label]))
    return labelized
x_train=labelizeReviews(cleaned_input,'train')

size,epoch_num = 400,10
##对数据进行训练
#实例DM和DBOW模型
model_dm = gensim.models.Doc2Vec(min_count=1, window=10, size=size, sample=1e-3, negative=5, workers=3)
model_dbow = gensim.models.Doc2Vec(min_count=1, window=10, size=size, sample=1e-3, negative=5, dm=0, workers=3)
#使用所有的数据建立词典
model_dm.build_vocab(x_train)
model_dbow.build_vocab(x_train)
#进行多次重复训练，每一次都需要对训练数据重新打乱，以提高精度
for epoch in range(epoch_num):
    random.shuffle(x_train)
    model_dm.train(x_train,epochs=1,total_examples=model_dm.corpus_count)
    model_dbow.train(x_train,epochs=1,total_examples=model_dm.corpus_count)

#提取信息
"""
np.random.permutation(length)的随机方法针对np的数组
model_dm.docvecs[]->docvecs
model_dm[]->word
"""
##读取向量
def getVecs(model, corpus, size):
    vecs = [np.array(model.docvecs[z.tags[0]]).reshape((1, size)) for z in corpus]
    return np.concatenate(vecs)
train_vecs_dm=getVecs(model_dm,x_train,size)
train_vecs_dbow =getVecs(model_dbow, x_train, size)
train_vecs = np.hstack((train_vecs_dm, train_vecs_dbow))
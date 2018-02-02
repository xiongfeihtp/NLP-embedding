
# coding: utf-8

# In[1]:


import jieba
from gensim.models.doc2vec import Doc2Vec,TaggedDocument
import glob
import re
import time
from tqdm import tqdm
import os.path
import json
# In[3]:


class Config():
    def __init__(self,configPath):
        f = open(configPath,encoding = 'utf-8-sig').read()
        self.conf = json.loads(f) 
    def get(self,name):
        return self.conf[name]

# In[4]:

def get_stopwords():
    stopwords_file = open(r'C:\Users\Administrator\Py\AlphaInsight\News_Cluster/CN_stopwords.txt',
                          'r',encoding='utf-8')
    stopwords = []  
    for word in stopwords_file.read().split('\n'):  
        if word not in stopwords:  
            stopwords.append(word)
    stopwords_file.close()
    return stopwords
# In[9]:
def get_corpus(topics, stopwords):
    for topic in tqdm(topics):
        file_paths = glob.glob(r'D:\DATA\AlphaInsight_data\News_Cluster\labeled_news\%s/*.txt'%topic)
        for file_path in file_paths:
            words = []
            news_file = open(file_path,'r',encoding='utf-8')
            content = news_file.read()
            content = re.sub(u'[\n]','',content)
            content = re.sub(u' ','',content)
            content = re.sub(u'<.+?>','',content)
            content = re.sub(u'[0-9]+','0',content)
            temp = jieba.cut(content)
            for w in temp:
                if not w in stopwords:
                    words.append(w)
            if not len(words) == 0:
                yield TaggedDocument(words=words, tags=[os.path.basename(file_path)])
# In[8]:
config=Config(r'C:\Users\Administrator\Py\AlphaInsight\News_Cluster/config.json')
topics=config.get('topics')
stopwords=get_stopwords()
# In[11]:
start=time.clock()
corpus=list(get_corpus(topics = topics, stopwords = stopwords))
end=time.clock()
print ('News process cost : %f s'%(end - start))

start=time.clock()
model = Doc2Vec(size = 300, min_count = 5, workers = 2)
model.build_vocab(corpus)
model.train(corpus, total_examples=model.corpus_count, epochs=model.iter)
model.save(r'D:\DATA\AlphaInsight_data\News_Cluster\model/doc2vec_0904.d2v')
end = time.clock()
print ('Model train cost : %f s'%(end - start))
# In[2]:
#显示进度条
# from tqdm import tqdm
# for i in tqdm(range(100)):
#     i


import gensim

LabeledSentence=gensim.models.doc2vec.LabeledSentence

from os import listdir
from os.path import isfile,join


#load txt
docLabels=[f for f in listdir("myPath") if f.endswith('.txt')]
#merge the txt
data=[]
for doc in docLabels:
    data.append(open("myDirPath/"+doc,'r'))

#句子集合
class LabeledLineSentence(object):
    def __init__(self,filename):
        self.filename=filename
    def __iter__(self):
        for uid,line in enumerate(open(self.filename)):
            yield LabeledSentence(words=line.split(),labels=['SENT_%s'%uid])

class LabeledLineDoc(object):
    def __init__(self,doc_list,labels_list):
        self.labels_list=labels_list
        self.doc_list=doc_list
    def __iter__(self):
        for idx,doc in enumerate(self.doc_list):
            yield  LabeledSentence(words=doc.split(),labels=[self.labels_list[idx]])






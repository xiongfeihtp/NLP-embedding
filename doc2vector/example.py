# import locale
# import glob
# import os.path
# import requests
# import tarfile
# import sys
# import codecs
# import smart_open
#
# dirname = 'aclImdb'
# filename = 'aclImdb_v1.tar.gz'
# locale.setlocale(locale.LC_ALL, 'C')
#
#
# control_chars = [chr(0x85)]
#
# # Convert text to lower-case and strip punctuation/symbols from words
# def normalize_text(text):
#     norm_text = text.lower()
#     # Replace breaks with spaces
#     norm_text = norm_text.replace('<br />', ' ')
#     # Pad punctuation with spaces on both sides
#     for char in ['.', '"', ',', '(', ')', '!', '?', ';', ':']:
#         norm_text = norm_text.replace(char, ' ' + char + ' ')
#     return norm_text
#
# import time
# start = time.clock()
#
# if not os.path.isfile('aclImdb/alldata-id.txt'):
#     if not os.path.isdir(dirname):
#         if not os.path.isfile(filename):
#             # Download IMDB archive
#             print("Downloading IMDB archive...")
#             url = u'http://ai.stanford.edu/~amaas/data/sentiment/' + filename
#             r = requests.get(url)
#             with open(filename, 'wb') as f:
#                 f.write(r.content)
#         tar = tarfile.open(filename, mode='r')
#         tar.extractall()
#         tar.close()
#
#     # Concatenate and normalize test/train data
#     print("Cleaning up dataset...")
#     folders = ['train/pos', 'train/neg', 'test/pos', 'test/neg', 'train/unsup']
#     alldata = u''
#     for fol in folders:
#         temp = u''
#         output = fol.replace('/', '-') + '.txt'
#         # Is there a better pattern to use?
#         txt_files = glob.glob(os.path.join(dirname, fol, '*.txt'))
#         for txt in txt_files:
#             with smart_open.smart_open(txt, "rb") as t:
#                 t_clean = t.read().decode("utf-8")
#                 for c in control_chars:
#                     t_clean = t_clean.replace(c, ' ')
#                 temp += t_clean
#             temp += "\n"
#         temp_norm = normalize_text(temp)
#         with smart_open.smart_open(os.path.join(dirname, output), "wb") as n:
#             n.write(temp_norm.encode("utf-8"))
#         alldata += temp_norm
#
#     with smart_open.smart_open(os.path.join(dirname, 'alldata-id.txt'), 'wb') as f:
#         for idx, line in enumerate(alldata.splitlines()):
#             num_line = u"_*{0} {1}\n".format(idx, line)
#             f.write(num_line.encode("utf-8"))
#
# end = time.clock()
# print ("Total running time: ", end-start)



#preprocessing the data
import gensim
from gensim.models.doc2vec import TaggedDocument
from collections import namedtuple

SentimentDocument = namedtuple('SentimentDocument', 'words tags split sentiment')

alldocs = []  # Will hold all docs in original order
with open('aclImdb/alldata-id.txt', encoding='utf-8') as alldata:
    for line_no, line in enumerate(alldata):
        tokens = gensim.utils.to_unicode(line).split()
        words = tokens[1:]
        tags = [line_no] # 'tags = [tokens[0]]' would also work at extra memory cost
        #split的一种技巧a
        split = ['train', 'test', 'extra', 'extra'][line_no//25000]  # 25k train, 25k test, 25k extra
        sentiment = [1.0, 0.0, 1.0, 0.0, None, None, None, None][line_no//12500] # [12.5K pos, 12.5K neg]*2 then unknown
        alldocs.append(SentimentDocument(words, tags, split, sentiment))

train_docs = [doc for doc in alldocs if doc.split == 'train']
test_docs = [doc for doc in alldocs if doc.split == 'test']
doc_list = alldocs[:]  # For reshuffling per pass

print('%d docs: %d train-sentiment, %d test-sentiment' % (len(doc_list), len(train_docs), len(test_docs)))




from collections import defaultdict
from pip._vendor.distlib.compat import raw_input
import xml.etree.ElementTree as ET
import math
import csv
from nltk.tokenize import word_tokenize
import numpy as np

# Dictionary class
class my_dictionary(dict):
    def __init__(self):
        self = dict()

    def add(self, key, value):
        self[key] = value

# Get question number from user
question_number = raw_input("Enter question number : ")
tree = ET.parse('questions.xml')
root = tree.getroot()
# Read csv file and take all abstracts as corpus
filename = "/home/oguzhan/Desktop/2020-04-10/04-10-mag-mapping.csv"
all_documents = []
rows = []
with open(filename, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        rows.append(row)
        all_documents.append(row[8])
# Read relationdoc and find full relevant id's
with open("relationdoc.txt") as f:
    content = f.readlines()

id_list = []
for lines in content:
    lines = ' '.join(lines.split())
    line = lines.split(" ")
    if line[0] == question_number and line[3] == "2":
        id_list.append(line[2])
# Create doc_list according to id_list
doc_list = []
for id in id_list:
    counter = 0
    for row in rows:
        if rows[counter][0] == id:
            doc_list.append(rows[counter][8])
        counter += 1
# Create idf dictionary according to the corpus
idf_dict = defaultdict(int)
for document in all_documents:
    dataset = word_tokenize(document)
    for word in set(dataset):
        idf_dict[word] += 1

N = len(all_documents)
for word in idf_dict:
    idf_dict[word] = math.log10(N / float(idf_dict[word]))

# Find tf scores in the documents
def tf(word, document):
    if len(document.split(None)) == 0:
        return 0
    return document.split().count(word) / float(len(document.split(None)))

# Compute cosine similarity between 2 vector
def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)

# Compute cosine similarity between 2 documents
def documents_similarity(document1, document2):
    bow1 = document1.split(" ")
    bow2 = document2.split(" ")
    wordSet = set(bow1).union(set(bow2))
    wordDict1 = dict.fromkeys(wordSet, 0)
    wordDict2 = dict.fromkeys(wordSet, 0)
    for word in bow1:
        wordDict1[word] = tf(word, document1) * idf_dict[word]
    for word in bow2:
        wordDict2[word] = tf(word, document2) * idf_dict[word]
    array1 = np.array(list(wordDict1.values()))
    array2 = np.array(list(wordDict2.values()))
    return cosine_similarity(array1, array2)
# Create similarity_array for determined documents
similarity_array = np.zeros((len(id_list), len(id_list)))
for i in range(len(doc_list)):
    for j in range(i+1):
        similarity_array[i][j] = documents_similarity(doc_list[i], doc_list[j])

for i in range(len(doc_list)):
    for j in range(i+1):
        similarity_array[j][i] = similarity_array[i][j]



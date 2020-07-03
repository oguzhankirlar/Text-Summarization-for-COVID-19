from idf import id_list, similarity_array, doc_list, documents_similarity
import networkx as nx
from nltk.tokenize import sent_tokenize
import numpy as np
from collections import Counter

G = nx.Graph()  # Create document graph and add nodes set id as name
for id in id_list:
    G.add_node(id)

for i in range(len(id_list)):  # add edge between nodes according to similarity array
    for j in range(len(id_list)):
        if 0.1 < similarity_array[i][j] < 0.9:
            G.add_edge(id_list[i], id_list[j])


# Page Rank Algorith Implementation
def pagerank(G, alpha=0.85, personalization=None,
             max_iter=100, tol=1.0e-5, nstart=None, weight='weight',
             dangling=None):
    if len(G) == 0:
        return {}
    if not G.is_directed():
        D = G.to_directed()
    else:
        D = G
    W = nx.stochastic_graph(D, weight=weight)
    N = W.number_of_nodes()
    if nstart is None:
        x = dict.fromkeys(W, 1.0 / N)
    else:
        s = float(sum(nstart.values()))
        x = dict((k, v / s) for k, v in nstart.items())
    if personalization is None:
        p = dict.fromkeys(W, 1.0 / N)
    else:
        missing = set(G) - set(personalization)
        if missing:
            raise nx.NetworkXError('Personalization dictionary '
                                   'must have a value for every node. '
                                   'Missing nodes %s' % missing)
        s = float(sum(personalization.values()))
        p = dict((k, v / s) for k, v in personalization.items())
    if dangling is None:
        dangling_weights = p
    else:
        missing = set(G) - set(dangling)
        if missing:
            raise nx.NetworkXError('Dangling node dictionary '
                                   'must have a value for every node. '
                                   'Missing nodes %s' % missing)
        s = float(sum(dangling.values()))
        dangling_weights = dict((k, v / s) for k, v in dangling.items())
    dangling_nodes = [n for n in W if W.out_degree(n, weight=weight) == 0.0]
    for _ in range(max_iter):
        xlast = x
        x = dict.fromkeys(xlast.keys(), 0)
        danglesum = alpha * sum(xlast[n] for n in dangling_nodes)
        for n in x:
            for nbr in W[n]:
                x[nbr] += alpha * xlast[n] * W[n][nbr][weight]
            x[n] += danglesum * dangling_weights[n] + (1.0 - alpha) * p[n]
        err = sum([abs(x[n] - xlast[n]) for n in x])
        if err < N * tol:
            return x
    raise nx.NetworkXError('pagerank: power iteration failed to converge '
                           'in %d iterations.' % max_iter)

# Determine most 10 important documents according to PageRank Scores
most_important_docs = []
pr = nx.pagerank(G, 0.15)
sorted_pr = {k: v for k, v in sorted(pr.items(), key=lambda item: item[1])}
for x in list(reversed(list(sorted_pr)))[0:10]:
    most_important_docs.append(x)
# print most important documents and pagerank scores
pr_scores = Counter(pr)
high = pr_scores.most_common(10)
for i in high:
    print(i[0], " :", i[1], " ")

# Find content of the documents
def find_document_from_id(id):
    index = id_list.index(id)
    return doc_list[index]
# Create sentence str from the 10 documents
important_docs_content = ""
for doc in most_important_docs:
    important_docs_content += find_document_from_id(doc) + " "

sentences = sent_tokenize(important_docs_content)
for sen in sentences:
    if sen == "All rights reserved.":
        sentences.remove(sen)
for sen in sentences:
    if sen == "This article is protected by copyright.":
        sentences.remove(sen)
# Create sentence similarity array
sentence_array = np.zeros((len(sentences), len(sentences)))
for i in range(len(sentences)):
    for j in range(i+1):
        sentence_array[i][j] = documents_similarity(sentences[i], sentences[j])

for i in range(len(sentences)):
    for j in range(i+1):
        sentence_array[j][i] = sentence_array[i][j]
# Create graph for sentences
G1 = nx.Graph()
for x in range(len(sentences)):
    G1.add_node(x)
# Add edge between nodes according to sentence similarity array
for i in range(len(sentences)):
    for j in range(len(sentences)):
        if 0.1 < sentence_array[i][j] < 0.9:
            G1.add_edge(i, j)
# Determine most 20 sentences
most_important_sent_id = []
pr1 = nx.pagerank(G1, 0.15)
sorted_pr1 = {k: v for k, v in sorted(pr1.items(), key=lambda item: item[1])}
for x in list(reversed(list(sorted_pr1)))[0:20]:
    most_important_sent_id.append(x)

# Determine most important documents and pagerank scores
pr_scores1 = Counter(pr1)
high = pr_scores1.most_common(20)
for i in high:
    print(i[0], " :", i[1], " ")
# Print 20 sentences according to PageRank scores
sent20 = []
summary = ""
for s in most_important_sent_id:
    sent20.append(sentences[s])
    summary = summary + " " + sentences[s]
print(sent20)

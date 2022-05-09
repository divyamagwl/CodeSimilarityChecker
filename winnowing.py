import nltk
import sys
import math
import pickle
from nltk import word_tokenize
from nltk.util import ngrams
from nltk import cluster
from collections import Counter
from statistics import mean
import networkx as nx
import sys
import csv


def cosine_similarity(l1, l2):

    vec1 = Counter(l1)
    vec2 = Counter(l2)
    
    intersection = set(vec1.keys()) & set(vec2.keys())
    
    # print(intersection)
    
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x]**2 for x in vec1.keys()])
    sum2 = sum([vec2[x]**2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0

    return float(numerator) / denominator


# Could be changed to include rightmost minimum too
def get_min(get_key = lambda x: x):
    def rightmost_minimum(l):
        minimum = float('inf')
        minimum_index = -1
        pos = 0
        
        while(pos < len(l)):
            if (get_key(l[pos]) < minimum):
                minimum = get_key(l[pos])
                minimum_index = pos
            pos += 1
        
        return l[minimum_index]
    return rightmost_minimum

# Have used the inbuilt hash function (Should try a self defined rolling hash function)
def winnowing(kgrams, k, t):
    modified_min_func = get_min(lambda key_value: key_value[0])
    
    document_fingerprints = []
    
    # print(kgrams)
    hash_table = [ (hash(kgrams[i]) , i)  for i in range(len(kgrams)) ]
    # print(len(hash_table))
    
    window_length = t - k + 1
    window_begin = 0
    window_end = window_length
    
    minimum_hash = None

    while (window_end < len(hash_table)):
        window = hash_table[window_begin:window_end]
        window_minimum = modified_min_func(window)
        
        if(minimum_hash != window_minimum):
            # print(window_minimum)
            document_fingerprints.append(window_minimum[0]) #not taking positions into consideration
            minimum_hash = window_minimum

        window_begin = window_begin + 1
        window_end = window_end + 1

    return document_fingerprints

def generate_kgrams(data, k):
    for text in data :
        token = nltk.word_tokenize(text)
        kgrams = ngrams(token, k)
        lst_kgrams = list(kgrams)
        # print("Kgrams : ", lst_kgrams)
        return lst_kgrams


# only conversion to lowercase for now
def preprocess(document):
    preprocessed_document = []
    for item in document :
        item = item.lower()
        preprocessed_document.append(item) 
    return preprocessed_document

def generate_fingerprints(file_name, k, t) :
    data = []
    f = open(file_name)
    data.append(f.read())
    
    preprocessed_data = preprocess(data)
    kgrams = generate_kgrams(preprocessed_data, k)
    # print(len(kgrams))
    document_fingerprints = winnowing(kgrams, k, t)
    return document_fingerprints

program1 = sys.argv[1]
program2 = sys.argv[2]

lev0s = []
lev1s = []
lev2s = []

for i in range(10):
    fingerprints1_0 = generate_fingerprints((program1+"_lev0.txt"), 13, 17)
    fingerprints2_0 = generate_fingerprints((program2+"_lev0.txt"), 13, 17)
    cosine_similarity_lev0 = cosine_similarity(fingerprints1_0, fingerprints2_0)
    lev0s.append(cosine_similarity_lev0)

    fingerprints1_1 = generate_fingerprints((program1+"_lev1.txt"), 13, 17)
    fingerprints2_1 = generate_fingerprints((program2+"_lev1.txt"), 13, 17)
    cosine_similarity_lev1 = cosine_similarity(fingerprints1_1, fingerprints2_1)
    lev1s.append(cosine_similarity_lev1)

    fingerprints1_2 = generate_fingerprints((program1+"_lev2.txt"), 13, 17)
    fingerprints2_2 = generate_fingerprints((program2+"_lev2.txt"), 13, 17)
    cosine_similarity_lev2 = cosine_similarity(fingerprints1_2, fingerprints2_2)
    lev2s.append(cosine_similarity_lev2)
# print(len(fingerprints1_0))
# print(len(fingerprints2_0))

# print(len(fingerprints1_1))
# print(len(fingerprints2_1))

# print(len(fingerprints1_2))
# print(len(fingerprints2_2))

final_cosine_similarity_lev0 = round(mean(lev0s), 2)
final_cosine_similarity_lev1 = round(mean(lev1s), 2)
final_cosine_similarity_lev2 = round(mean(lev2s), 2)

# print("Cosine similarity Level 0 : \n", final_cosine_similarity_lev0)
# print("Cosine similarity Level 1 : \n", final_cosine_similarity_lev1)
# print("Cosine similarity Level 2 : \n", final_cosine_similarity_lev2)

a_file = open(program1+"_count.txt")
b_file = open(program2+"_count.txt")
count_values=[]
for c in range(3):
    a = int(a_file.readline())
    b = int(b_file.readline())
    count_values.append([a,b])

normalization_score = 0
t=0
for c in range(3):
    x=count_values[c][0]
    y=count_values[c][1]
    if(x + y)!=0:
        t=t+1
        if x>y:
            s = 1-((x-y)/(x+y))
        else:
            s = 1-((y-x)/(x+y))    
        normalization_score+=(10*s)

if t!=0:
    normalization_score=normalization_score/(t*10)
    total_similarity_score_win = ((0.5*final_cosine_similarity_lev0) + (0.3*final_cosine_similarity_lev1) + (0.2*final_cosine_similarity_lev2))
    normalization_score = normalization_score
    # print("Winnowing similarity score : \n", total_similarity_score_win)
    # print("Normalization score : \n", normalization_score)
    final_score = (total_similarity_score_win*60)+(normalization_score*40)
    print("Similarity score = : \n", final_score)
else:
    total_similarity_score_win = ((0.5*final_cosine_similarity_lev0) + (0.3*final_cosine_similarity_lev1) + (0.2*final_cosine_similarity_lev2))
    # print("Winnowing similarity score : \n", total_similarity_score_win)
    # print("Invalid norm: \n", )
    final_score = (total_similarity_score_win*100)
    print("Similarity score = : \n", final_score)


program1_name = open("program1.txt", "r").readline()
program2_name = open("program2.txt", "r").readline()

row = [program1_name, program2_name, total_similarity_score_win, final_score]
with open("experiments_python-loops.csv", "a") as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(row)
# Cosine similarity seems to be highest for k = 11 and t = 15, should try others.

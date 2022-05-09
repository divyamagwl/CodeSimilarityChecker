import nltk
from nltk.util import ngrams
import math
from collections import Counter

# FGPT: Fignerprint
class Winnowing:
    def __init__(self, ast1, ast2):
        self.program1 = ast1
        self.program2 = ast2

    def cosineSimilarity(self, l1, l2):
        vec1, vec2 = Counter(l1), Counter(l2)
        intersection = set(vec1.keys()) & set(vec2.keys())

        numerator = sum([vec1[count] * vec2[count] for count in intersection])

        sum1 = sum([vec1[count] ** 2 for count in vec1.keys()])
        sum2 = sum([vec2[count] ** 2 for count in vec2.keys()])
        denominator = math.sqrt(sum1) * math.sqrt(sum2)

        try: 
            result = float(numerator) / denominator
        except:
            result = 0.0

        return result

    def right_min(self, l):
        index = len(l) - l[::-1].index(min(l)) - 1 
        return l[index]
    
    def generateKgrams(self, text, k):
        token = nltk.word_tokenize(text)
        kgrams = ngrams(token, k)
        return list(kgrams)
    
    # TODO: Add more preprocessing steps
    def preprocess(self, text):
        text = text.lower()
        return text

    # Rolling window method with hashing to generate representative fingerprints
    def winnowing(self, kgrams, k, t):
        fingerprints = [] #fingerprints array
        
        hashes = [(hash(kgrams[i]), i)  for i in range(len(kgrams))] #hashing each k-gram
        num_hashes = len(hashes)
 
        window_size = t - k + 1 #size of window = threshold - kgram_size + 1

        minimum_hash = None #to prevent duplicate hash addition from 2 adjacent windows
        
        for begin,end in zip(range(0, num_hashes), range(window_size, num_hashes)): #window loop
            
            window_min = self.right_min(hashes[begin:end]) #getting rightmost minimum of the hash window
            
            if(minimum_hash != window_min): #checking for duplicate
                fingerprints.append(window_min[0]) #(hash(kgrams[i]),i) we are not using position
                minimum_hash = window_min #storing minimum hash for next window check

        return fingerprints #returning the final fingerprints representating our text

    def generateFGPT(self, data, k, t):
        cleaned_data = self.preprocess(data)
        kgrams = self.generateKgrams(cleaned_data, k)
        dataFGPT = self.winnowing(kgrams, k, t)
        return dataFGPT

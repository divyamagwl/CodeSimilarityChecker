import nltk
from nltk.util import ngrams

# FGPT: Fignerprint
class Winnowing:
    
    def generateKgrams(self, text, k):
        token = nltk.word_tokenize(text)
        kgrams = ngrams(token, k)
        return list(kgrams)
    
    def preprocess(self, text):
        text = text.lower()
        return text

    def right_min(self, l):
        index = len(l) - l[::-1].index(min(l)) - 1 
        return l[index]

    # Rolling window method with hashing to generate representative fingerprints
    def winnowing(self, kgrams, k, threshold):
        fingerprints = [] #fingerprints array
        
        hashes = [(hash(kgrams[i]), i)  for i in range(len(kgrams))] #hashing each k-gram
        num_hashes = len(hashes)
 
        window_size = threshold - k + 1 #size of window = threshold - kgram_size + 1

        minimum_hash = None #to prevent duplicate hash addition from 2 adjacent windows
        
        for begin,end in zip(range(0, num_hashes), range(window_size, num_hashes)): #window loop
            
            window_min = self.right_min(hashes[begin:end]) #getting rightmost minimum of the hash window
            
            if(minimum_hash != window_min): #checking for duplicate
                fingerprints.append(window_min[0]) #(hash(kgrams[i]),i) we are not using position
                minimum_hash = window_min #storing minimum hash for next window check

        return fingerprints #returning the final fingerprints representating our text

    def generateFGPT(self, data, k, threshold):
        cleaned_data = self.preprocess(data)
        kgrams = self.generateKgrams(cleaned_data, k)
        dataFGPT = self.winnowing(kgrams, k, threshold)
        return dataFGPT
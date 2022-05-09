import math
from collections import Counter

def readFile(filename):
    with open(filename) as f:
        contents = f.read()
        return contents

def calculateNormScores(ast1_constructs, ast2_constructs, constructFlag):
    norm_values = []

    for i in range(len(ast1_constructs)):
        p1_count = ast1_constructs[i]
        p2_count = ast2_constructs[i]

        if p1_count != 0 and p2_count != 0 and constructFlag[i] != '0':
            N = 1 - (abs(p1_count - p2_count) / (p1_count + p2_count))
            norm_values.append(N)

    return norm_values

def cosineSimilarity(l1, l2):
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

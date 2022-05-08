import ast
import sys
from astor import to_source
import nltk
from nltk.util import ngrams
import sys
import math
from collections import Counter
from statistics import mean
import sys
import csv

class RewriteVariableName(ast.NodeTransformer):
    def visit_Name(self, node):
        if(isinstance(node, ast.Name)):
            value = ast.Name(id='x')
            return ast.copy_location(value, node)
        return node

class AST:
    def __init__(self, maxLevel=3):
        self.maxLevel = maxLevel
        self.levels = [[] for _ in range(maxLevel)]
        self.levelParentChild = [[] for _ in range(maxLevel - 1)]

    def createAst(self, sourceCode):
        inputASTtree = ast.parse(sourceCode)
        newASTTree = RewriteVariableName().visit(inputASTtree)
        newSourceCode = to_source(newASTTree)
        # print(newSourceCode)
        newASTTreeWithCtx = ast.parse(newSourceCode)
        # print(ast.dump(newASTTree))
        # print(ast.dump(newASTTreeWithCtx))
        return newASTTreeWithCtx

    def countExprType(self, tree, exprType):
        count = 0
        for node in ast.walk(tree):
            if(isinstance(node, exprType)):
                count += 1
        return count

    def getLevels(self, node, level=0):
        if(level < self.maxLevel):
            self.levels[level].append(ast.dump(node))

        for _, value in ast.iter_fields(node):
            if(isinstance(value, ast.AST)):
                value = [value]

            if(isinstance(value, list)):
                for item in value:
                    if(isinstance(item, ast.AST)):
                        self.getLevels(item, level=level+1)
        
    def getChildren(self, node):
        parent = ast.dump(node)
        children = []
        for child_node in ast.iter_child_nodes(node):
            children.append(ast.dump(child_node))
        return parent, children

    def getParentChildRelations(self, root, level=0):
        for _, value in ast.iter_fields(root):
            if(isinstance(value, ast.AST)):
                value = [value]

            if(isinstance(value, list)):
                for item in value:
                    if(isinstance(item, ast.AST)):
                        p, c = self.getChildren(item)

                        if(level < self.maxLevel - 1):
                            self.levelParentChild[level].append([p, c])

                        self.getParentChildRelations(item, level=level+1)

# FGPT: Fignerprint
class Winnowing:
    def __init__(self, ast1, ast2):
        self.program1 = ast1
        self.program2 = ast2

    def cosine_similarity(self, l1, l2):

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

    def get_min(self, get_key = lambda x: x):
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

    def generateKgrams(self, text, k):
        token = nltk.word_tokenize(text)
        kgrams = ngrams(token, k)
        return list(kgrams)
    
    # TODO: Add more preprocessing steps
    def preprocess(self, text):
        text = text.lower()
        return text

    # Have used the inbuilt hash function (Should try a self defined rolling hash function)
    def winnowing(self, kgrams, k, t):
        modified_min_func = self.get_min(lambda key_value: key_value[0])
        
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

    def generateFGPT(self, data, k, t):
        cleaned_data = self.preprocess(data)
        kgrams = self.generateKgrams(cleaned_data, k)
        # print(len(kgrams))
        document_fingerprints = self.winnowing(kgrams, k, t)
        return document_fingerprints


def readFile(filename):
    with open(filename) as f:
        contents = f.read()
        return contents


if __name__ == '__main__':
    file1 = sys.argv[1]
    file2 = sys.argv[2]

    ast1 = AST()
    ast2 = AST()

    sourceCode1 = readFile(file1)
    sourceCode2 = readFile(file2)

    generatedAST1 = ast1.createAst(sourceCode1)
    generatedAST2 = ast2.createAst(sourceCode2)

    ast1_counts = {
        "loopsCount" : ast1.countExprType(generatedAST1, (ast.For, ast.While)),
        "ifCount" : ast1.countExprType(generatedAST1, ast.If),
        "funcCount" : ast1.countExprType(generatedAST1, ast.FunctionDef)
    }
   
    ast2_counts = {
        "loopsCount" : ast2.countExprType(generatedAST2, (ast.For, ast.While)),
        "ifCount" : ast2.countExprType(generatedAST2, ast.If),
        "funcCount" : ast2.countExprType(generatedAST2, ast.FunctionDef)
    }

    ast1.getLevels(generatedAST1)
    ast2.getLevels(generatedAST2)

    ast1.getParentChildRelations(generatedAST1)
    ast2.getParentChildRelations(generatedAST2)
    
    for i in range(ast1.maxLevel - 1):
        ast1.levelParentChild[i].sort
    
    for i in range(ast2.maxLevel - 1):
        ast2.levelParentChild[i].sort

    print(ast1_counts["loopsCount"], ast1_counts["ifCount"], ast1_counts["funcCount"])
    print(ast2_counts["loopsCount"], ast2_counts["ifCount"], ast2_counts["funcCount"])

    input_fp1_list1 = []
    input_fp1_list2 = []
    for element in ast1.levelParentChild[0]:
        input_fp1_list1.append(element[0])
        for item in element[1]:
            input_fp1_list2.append(item)

    input_fp2_list1 = []
    input_fp2_list2 = []
    for element in ast2.levelParentChild[0]:
        input_fp2_list1.append(element[0])
        for item in element[1]:
            input_fp2_list2.append(item)


    winnow = Winnowing(ast1, ast2)

    lev0s = []
    lev1s = []
    lev2s = []

    for i in range(1):
        fingerprints1_0 = winnow.generateFGPT('\n'.join(ast1.levels[0]), 13, 17)
        fingerprints2_0 = winnow.generateFGPT('\n'.join(ast2.levels[0]), 13, 17)
        cosine_similarity_lev0 = winnow.cosine_similarity(fingerprints1_0, fingerprints2_0)
        lev0s.append(cosine_similarity_lev0)

        fingerprints1_1 = winnow.generateFGPT('\n'.join(input_fp1_list1), 13, 17)
        fingerprints2_1 = winnow.generateFGPT('\n'.join(input_fp2_list1), 13, 17)
        cosine_similarity_lev1 = winnow.cosine_similarity(fingerprints1_1, fingerprints2_1)
        lev1s.append(cosine_similarity_lev1)

        fingerprints1_2 = winnow.generateFGPT('\n'.join(input_fp1_list2), 13, 17)
        fingerprints2_2 = winnow.generateFGPT('\n'.join(input_fp2_list2), 13, 17)
        cosine_similarity_lev2 = winnow.cosine_similarity(fingerprints1_2, fingerprints2_2)
        lev2s.append(cosine_similarity_lev2)

    final_cosine_similarity_lev0 = round(mean(lev0s), 2)
    final_cosine_similarity_lev1 = round(mean(lev1s), 2)
    final_cosine_similarity_lev2 = round(mean(lev2s), 2)


    count_values = []
    for i in range(3):
        a = list(ast1_counts.values())[i]
        b = list(ast2_counts.values())[i]
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
        final_score = (total_similarity_score_win*60)+(normalization_score*40)
        print("Similarity score = : \n", final_score)
    else:
        total_similarity_score_win = ((0.5*final_cosine_similarity_lev0) + (0.3*final_cosine_similarity_lev1) + (0.2*final_cosine_similarity_lev2))
        final_score = (total_similarity_score_win*100)
        print("Similarity score = : \n", final_score)

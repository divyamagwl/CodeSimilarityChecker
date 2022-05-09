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

# Change all variable names to x
class RewriteVariableName(ast.NodeTransformer):
    def visit_Name(self, node):
        if(isinstance(node, ast.Name)):
            value = ast.Name(id='x')
            return ast.copy_location(value, node)
        return node

class AST:
    def __init__(self, maxLevel=3):
        self.maxLevel = maxLevel
        self.level0 = []
        self.levelParentChild = [[] for _ in range(maxLevel)]
        self.parents = [[] for _ in range(maxLevel)]
        self.children = [[] for _ in range(maxLevel)]

    # create an AST given a source code
    def createAst(self, sourceCode):
        inputASTtree = ast.parse(sourceCode)
        newASTTree = RewriteVariableName().visit(inputASTtree)
        
         # rewriting variable names lead to loss of context
        # So, we againg change it to source code, parse it with context
        # (Load, Store, Del)
        # and dump it in form of string in self.level0
        newSourceCode = to_source(newASTTree)
        newASTTreeWithCtx = ast.parse(newSourceCode)
        self.level0 = ast.dump(newASTTreeWithCtx)
        return newASTTreeWithCtx

    
    # counts the number of expressions of the given type in the 
    # given AST
    def countExprType(self, tree, exprType):
        count = 0
        for node in ast.walk(tree):
            if(isinstance(node, exprType)):
                count += 1
        return count
        
    def getParentChildRelations(self, root, level=0):
        for _, value in ast.iter_fields(root):
            if(isinstance(value, ast.AST)):
                value = [value]

            if(isinstance(value, list)):
                for item in value:
                    if(isinstance(item, ast.AST)):
                        parent = ast.dump(item)
                        children = []
                        for child_node in ast.iter_child_nodes(item):
                            children.append(ast.dump(child_node))

                        if(level < self.maxLevel):
                            self.parents[level].append(parent)
                            for child in children:
                                self.children[level].append(child)
                            self.levelParentChild[level].append([parent, children])

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

    def right_min(self,l):
        index = len(l)- l[::-1].index(min(l)) -1 
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
                fingerprints.append(window_min[0]) #(hash(kgrams[i]),i) we are not usin position
                minimum_hash = window_min #storing minimum hash for next window check

        return fingerprints #returning the final fingerprints representating our text

    def generateFGPT(self, data, k, t):
        cleaned_data = self.preprocess(data)
        kgrams = self.generateKgrams(cleaned_data, k)
        # print(len(kgrams))
        dataFGPT = self.winnowing(kgrams, k, t)
        return dataFGPT


def readFile(filename):
    with open(filename) as f:
        contents = f.read()
        return contents

def calculateNormScores(ast1_constructs, ast2_constructs):
    norm_values = []

    for i in range(len(ast1_constructs)):
        p1_count = ast1_constructs[i]
        p2_count = ast2_constructs[i]

        if p1_count != 0 and p2_count != 0:
            N = 1 - (abs(p1_count - p2_count) / (p1_count + p2_count))
            norm_values.append(N)

    return norm_values


if __name__ == '__main__':
    file1 = sys.argv[1]
    file2 = sys.argv[2]
    file3 = "011"# sys.argv[3]

    ast1 = AST()
    ast2 = AST()

    sourceCode1 = readFile(file1)
    sourceCode2 = readFile(file2)

    generatedAST1 = ast1.createAst(sourceCode1)
    generatedAST2 = ast2.createAst(sourceCode2)

    ast1_counts = {
        "loopsCount" : ast1.countExprType(generatedAST1, (ast.For, ast.While)),
        "ifCount" : ast1.countExprType(generatedAST1, ast.If),
        "controlFlow" : ast1.countExprType(generatedAST1, (ast.Break,ast.Continue)),
        "funcCount" : ast1.countExprType(generatedAST1, ast.FunctionDef),
        "excepCount":ast1.countExprType(generatedAST1, ast.ExceptHandler)    
    }
   
    ast2_counts = {
        "loopsCount" : ast2.countExprType(generatedAST2, (ast.For, ast.While)),
        "ifCount" : ast2.countExprType(generatedAST2, ast.If),
        "controlFlow" : ast2.countExprType(generatedAST2, (ast.Break,ast.Continue)),
        "funcCount" : ast2.countExprType(generatedAST2, ast.FunctionDef),
        "excepCount":ast2.countExprType(generatedAST2, ast.ExceptHandler)
    }

    ast1.getParentChildRelations(generatedAST1)
    ast2.getParentChildRelations(generatedAST2)
    

    print(ast1_counts["loopsCount"], ast1_counts["ifCount"],ast1_counts["controlFlow"], ast1_counts["funcCount"],ast1_counts["excepCount"])
    print(ast2_counts["loopsCount"], ast2_counts["ifCount"],ast2_counts["controlFlow"], ast2_counts["funcCount"],ast2_counts["excepCount"])


    for level in range(ast1.maxLevel):
        print(f"-----------------------------LEVEL{level+1} -> LEVEL{level+2}-----------------------------------------------------")
        for pc_i in range(len(ast1.levelParentChild[level])):
            print("Parent = ", ast1.levelParentChild[level][pc_i][0], "\n\nChildren = ", ast1.levelParentChild[level][pc_i][1])
            print("\n")
        print("--------------------------------------------------------------------------------------------------\n")

    winnow = Winnowing(ast1, ast2)
    k, t = 13, 17

    fingerprints1_0 = winnow.generateFGPT(''.join(ast1.level0), k, t)
    fingerprints2_0 = winnow.generateFGPT(''.join(ast2.level0), k, t)
    final_cosine_similarity_lev0 = round(winnow.cosine_similarity(fingerprints1_0, fingerprints2_0), 2)

    fingerprints1_1 = winnow.generateFGPT(''.join(ast1.parents[0]), k, t)
    fingerprints2_1 = winnow.generateFGPT(''.join(ast2.parents[0]), k, t)
    final_cosine_similarity_lev1 = round(winnow.cosine_similarity(fingerprints1_1, fingerprints2_1), 2)

    fingerprints1_2 = winnow.generateFGPT(''.join(ast1.children[0]), k, t)
    fingerprints2_2 = winnow.generateFGPT(''.join(ast2.children[0]), k, t)
    final_cosine_similarity_lev2 = round(winnow.cosine_similarity(fingerprints1_2, fingerprints2_2), 2)


    ast1_constructs = list(ast1_counts.values())
    ast2_constructs = list(ast2_counts.values())

    norm_values = calculateNormScores(ast1_constructs, ast2_constructs)

    total_similarity_score_win = ((0.5 * final_cosine_similarity_lev0) + (0.3 * final_cosine_similarity_lev1) + (0.2 * final_cosine_similarity_lev2))

    alpha = 60
    if(len(norm_values) != 0):
        final_norm_score = (sum(norm_values) / len(norm_values))
        final_score = (total_similarity_score_win * alpha) + (final_norm_score * (100 - alpha))
    else:
        final_score = (total_similarity_score_win * 100)
        
    print("Similarity score = ", final_score)
    
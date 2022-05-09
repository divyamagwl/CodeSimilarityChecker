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

class RewriteVariableName(ast.NodeTransformer):
    def visit_Name(self, node):
        if(isinstance(node, ast.Name)):
            value = ast.Name(id='x')
            return ast.copy_location(value, node)
        return node

class AST:
    def __init__(self, maxLevel=6):
        self.maxLevel = maxLevel
        self.level0 = []
        self.levelParentChild = [[] for _ in range(maxLevel)]
        self.parents = [[] for _ in range(maxLevel)]
        self.children = [[] for _ in range(maxLevel)]

    def createAst(self, sourceCode):
        inputASTtree = ast.parse(sourceCode)
        newASTTree = RewriteVariableName().visit(inputASTtree)
        newSourceCode = to_source(newASTTree)
        newASTTreeWithCtx = ast.parse(newSourceCode)
        self.level0 = ast.dump(newASTTreeWithCtx)
        return newASTTreeWithCtx

    def countExprType(self, tree, exprType):
        count = 0
        for node in ast.walk(tree):
            if(isinstance(node, exprType)):
                count += 1
        return count
        
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

                        if(level < self.maxLevel):
                            self.levelParentChild[level].append([p, c])

                        self.getParentChildRelations(item, level=level+1)

    def separateParentChild(self, level):
        for parent, children in self.levelParentChild[level]:
            self.parents[level].append(parent)
            for child in children:
                self.children[level].append(child)


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
        
        docFGPT = []
        
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
                docFGPT.append(window_minimum[0]) #not taking positions into consideration
                minimum_hash = window_minimum

            window_begin = window_begin + 1
            window_end = window_end + 1

        return docFGPT

    def generateFGPT(self, data, k, t):
        cleaned_data = self.preprocess(data)
        kgrams = self.generateKgrams(cleaned_data, k)
        # print(len(kgrams))
        docFGPT = self.winnowing(kgrams, k, t)
        return docFGPT


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
    }
   
    ast2_counts = {
        "loopsCount" : ast2.countExprType(generatedAST2, (ast.For, ast.While)),
        "ifCount" : ast2.countExprType(generatedAST2, ast.If),
        "controlFlow" : ast2.countExprType(generatedAST2, (ast.Break,ast.Continue)),
        "funcCount" : ast2.countExprType(generatedAST2, ast.FunctionDef),
    }

    ast1.getParentChildRelations(generatedAST1)
    ast2.getParentChildRelations(generatedAST2)
    
    for i in range(ast1.maxLevel):
        ast1.levelParentChild[i].sort
    
    for i in range(ast2.maxLevel):
        ast2.levelParentChild[i].sort

    ast1.separateParentChild(0)
    ast2.separateParentChild(0)

    print(ast1_counts["loopsCount"], ast1_counts["ifCount"],ast1_counts["controlFlow"], ast1_counts["funcCount"])
    print(ast2_counts["loopsCount"], ast2_counts["ifCount"],ast2_counts["controlFlow"], ast2_counts["funcCount"])


    for level in range(ast1.maxLevel):
        print(f"-----------------------------LEVEL{level+1} -> LEVEL{level+2}-----------------------------------------------------")
        for pc_i in range(len(ast1.levelParentChild[level])):
            print("Parent = ", ast1.levelParentChild[level][pc_i][0], "\n\nChildren = ", ast1.levelParentChild[level][pc_i][1])
            print("\n")
        print("--------------------------------------------------------------------------------------------------\n")

    winnow = Winnowing(ast1, ast2)
    k, t = 13, 17

    fingerprints1_0 = winnow.generateFGPT('\n'.join(ast1.level0), k, t)
    fingerprints2_0 = winnow.generateFGPT('\n'.join(ast2.level0), k, t)
    final_cosine_similarity_lev0 = round(winnow.cosine_similarity(fingerprints1_0, fingerprints2_0), 2)

    fingerprints1_1 = winnow.generateFGPT('\n'.join(ast1.parents[0]), k, t)
    fingerprints2_1 = winnow.generateFGPT('\n'.join(ast2.parents[0]), k, t)
    final_cosine_similarity_lev1 = round(winnow.cosine_similarity(fingerprints1_1, fingerprints2_1), 2)

    fingerprints1_2 = winnow.generateFGPT('\n'.join(ast1.children[0]), k, t)
    fingerprints2_2 = winnow.generateFGPT('\n'.join(ast2.children[0]), k, t)
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
    
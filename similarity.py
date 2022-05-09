import sys
from generateAST import *
from winnowing import *


def readFile(filename):
    with open(filename) as f:
        contents = f.read()
        return contents

def calculateNormScores(ast1_constructs, ast2_constructs, code):
    norm_values = []

    for i in range(len(ast1_constructs)):
        p1_count = ast1_constructs[i]
        p2_count = ast2_constructs[i]

        if p1_count != 0 and p2_count != 0 and code[i] != '0':
            N = 1 - (abs(p1_count - p2_count) / (p1_count + p2_count))
            norm_values.append(N)

    return norm_values


if __name__ == '__main__':
    file1 = sys.argv[1]
    file2 = sys.argv[2]
    code = sys.argv[3]
    maxLevel = sys.argv[4]

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
        "arith" : ast1.countExprType(generatedAST1, (ast.UnaryOp,ast.BinOp)),
        "excepCount":ast1.countExprType(generatedAST1, ast.ExceptHandler)    
    }
   
    ast2_counts = {
        "loopsCount" : ast2.countExprType(generatedAST2, (ast.For, ast.While)),
        "ifCount" : ast2.countExprType(generatedAST2, ast.If),
        "controlFlow" : ast2.countExprType(generatedAST2, (ast.Break,ast.Continue)),
        "funcCount" : ast2.countExprType(generatedAST2, ast.FunctionDef),
        "arith" : ast2.countExprType(generatedAST2, (ast.UnaryOp,ast.BinOp)),
        "excepCount":ast2.countExprType(generatedAST2, ast.ExceptHandler)
    }

    ast1.getParentChildRelations(generatedAST1)
    ast2.getParentChildRelations(generatedAST2)
    

    print(ast1_counts["loopsCount"], ast1_counts["ifCount"],ast1_counts["controlFlow"], ast1_counts["funcCount"],ast1_counts["excepCount"])
    print(ast2_counts["loopsCount"], ast2_counts["ifCount"],ast2_counts["controlFlow"], ast2_counts["funcCount"],ast2_counts["excepCount"])


    # for level in range(ast1.maxLevel):
    #     print(f"-----------------------------LEVEL{level+1} -> LEVEL{level+2}-----------------------------------------------------")
    #     for pc_i in range(len(ast1.levelParentChild[level])):
    #         print("Parent = ", ast1.levelParentChild[level][pc_i][0], "\n\nChildren = ", ast1.levelParentChild[level][pc_i][1])
    #         print("\n")
    #     print("--------------------------------------------------------------------------------------------------\n")

    winnow = Winnowing(ast1, ast2)
    k, t = 13, 17

    min_level = min(ast1.maxLevel, ast2.maxLevel) - 1

    fingerprints1 = [] #level0, level 1,...,min_level parents, level min_level children
    fingerprints2 = []
    
    fingerprints1.append(winnow.generateFGPT(''.join(ast1.level0), k, t))
    fingerprints2.append(winnow.generateFGPT(''.join(ast2.level0), k, t))

    for i in range(min_level):
         fingerprints1.append(winnow.generateFGPT(''.join(ast1.parents[i]), k, t))
         fingerprints2.append(winnow.generateFGPT(''.join(ast2.parents[i]), k, t))
    
    fingerprints1.append(winnow.generateFGPT(''.join(ast1.children[min_level-1]), k, t))
    fingerprints2.append(winnow.generateFGPT(''.join(ast2.children[min_level-1]), k, t))
    

    final_cosine_similarities = []

    for i in range(min_level+2):
        final_cosine_similarities.append(round(winnow.cosineSimilarity(fingerprints1[i], fingerprints2[i]), 2))

    ast1_constructs = list(ast1_counts.values())
    ast2_constructs = list(ast2_counts.values())

    norm_values = calculateNormScores(ast1_constructs, ast2_constructs, code)

    
    weight = 1/(min_level+2)
    total_similarity_score_win = sum([i*weight for i in final_cosine_similarities])

    alpha = 60
    if(len(norm_values) != 0):
        final_norm_score = (sum(norm_values) / len(norm_values))
        final_score = (total_similarity_score_win * alpha) + (final_norm_score * (100 - alpha))
    else:
        final_score = (total_similarity_score_win * 100)

    print("Similarity score = ", final_score)
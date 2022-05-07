import ast
import sys
from astor import to_source
import nltk
import sys
import math
from collections import Counter
from statistics import mean
import sys
import csv
class RewriteVariableName(ast.NodeTransformer):
    def visit_Name(self, node):
        if(isinstance(node, ast.Name)):
            value = ast.Name(id='var')
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
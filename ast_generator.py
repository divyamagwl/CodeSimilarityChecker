import sys
import ast
from astor import to_source

class RewriteVariableName(ast.NodeTransformer):
    def visit_Name(self, node):
        if(isinstance(node, ast.Name)):
            result = ast.Name(id='var')
            return ast.copy_location(result, node)
        return node

def readFile(filename):
    with open(filename) as f:
        contents = f.read()
        return contents

def createAst(sourceCode):
    inputASTtree = ast.parse(sourceCode)
    newASTTree = RewriteVariableName().visit(inputASTtree)
    newSourceCode = to_source(newASTTree)
    # print(newSourceCode)
    newASTTreeWithCtx = ast.parse(newSourceCode)
    # print(ast.dump(newASTTree))
    # print(ast.dump(newASTTreeWithCtx))
    return newASTTreeWithCtx

def countExprType(tree, exprType):
    count = 0
    for node in ast.walk(tree):
        if(isinstance(node, exprType)):
            count += 1
    return count


if __name__ == '__main__':
    filename = sys.argv[1]
    sourceCode = readFile(filename)
    generatedASTtree = createAst(sourceCode)
    
    loopsCount = countExprType(generatedASTtree, (ast.For, ast.While))
    ifCount = countExprType(generatedASTtree, ast.If)
    funcCount = countExprType(generatedASTtree, ast.FunctionDef)

    print(loopsCount, ifCount, funcCount)
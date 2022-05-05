import ast
import sys
from astor import to_source

class RewriteVariableName(ast.NodeTransformer):
    def visit_Name(self, node):
        if(isinstance(node, ast.Name)):
            value = ast.Name(id='var')
            return ast.copy_location(value, node)
        return node

class GenerateAST:
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

def readFile(self, filename):
    with open(filename) as f:
        contents = f.read()
        return contents


if __name__ == '__main__':
    filename = sys.argv[1]

    generator = GenerateAST()
    
    sourceCode = readFile(filename)
    generatedASTtree = generator.createAst(sourceCode)
    
    loopsCount = generator.countExprType(generatedASTtree, (ast.For, ast.While))
    ifCount = generator.countExprType(generatedASTtree, ast.If)
    funcCount = generator.countExprType(generatedASTtree, ast.FunctionDef)

    generator.getLevels(generatedASTtree)

    generator.getParentChildRelations(generatedASTtree)
    for i in range(generator.maxLevel - 1):
        generator.levelParentChild[i].sort
        
    print(loopsCount, ifCount, funcCount)
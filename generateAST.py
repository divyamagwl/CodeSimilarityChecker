import ast
import sys
from astor import to_source


class RewriteVariableName(ast.NodeTransformer):
    def visit_Name(self, node):
        if(isinstance(node, ast.Name)):
            result = ast.Name(id='var')
            return ast.copy_location(result, node)
        return node
class generateAST:
    def __init__(self):
        self.level_0 = []
        self.level_1 = []
        self.level_2 = []
        self.level_0_parents = []
        self.level_0_children = []
        self.level_1_parents = []
        self.level_1_children = []
    
    def readFile(self, filename):
        with open(filename) as f:
            contents = f.read()
            return contents

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
        if level==0:
            self.level_0.append(ast.dump(node))
        elif level==1:
            self.level_1.append(ast.dump(node))
        elif level==2:
            self.level_2.append(ast.dump(node))        

        for _, value in ast.iter_fields(node):
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.AST):
                        self.getLevels(item, level=level+1)
            elif isinstance(value, ast.AST):
                self.getLevels(value, level=level+1)
        
    def getChildren(self, node):
        parent = ast.dump(node)
        children = []
        for child_node in ast.iter_child_nodes(node):
            children.append(ast.dump(child_node))
        return parent, children

    def getParentChildRelations(self, root, level=0):
        for _, value in ast.iter_fields(root):
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.AST):
                        p, c = self.getChildren(item)
                        if level == 0:
                            self.level_0_parents.append(p)
                            self.level_0_children.append(c)
                        elif level == 1:
                            self.level_1_parents.append(p)
                            self.level_1_children.append(c)       
                        self.getParentChildRelations(item, level=level+1)
            elif isinstance(value, ast.AST):
                p, c = self.getChildren(value)
                if level == 0:
                    self.level_0_parents.append(p)
                    self.level_0_children.append(c)
                elif level == 1:
                    self.level_1_parents.append(p)
                    self.level_1_children.append(c)
                self.getParentChildRelations(value, level=level+1)

if __name__ == '__main__':
    filename = sys.argv[1]

    generator = generateAST()
    
    sourceCode = generator.readFile(filename)
    generatedASTtree = generator.createAst(sourceCode)
    
    loopsCount = generator.countExprType(generatedASTtree, (ast.For, ast.While))
    ifCount = generator.countExprType(generatedASTtree, ast.If)
    funcCount = generator.countExprType(generatedASTtree, ast.FunctionDef)

    print(loopsCount, ifCount, funcCount)
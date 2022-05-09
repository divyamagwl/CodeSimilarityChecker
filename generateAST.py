import ast
from astor import to_source

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

    # counts the number of expressions of the given type in the given AST
    def countExprType(self, tree, exprType):
        count = 0
        for node in ast.walk(tree):
            if(isinstance(node, exprType)):
                count += 1
        return count
        
    def generateParentChild(self, root, level=0):
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

                            self.generateParentChild(item, level=level+1)
# CodeSimilarityChecker
A tool which can tell how similar two programs are and assigns a score to their similarity.

# Installation guide
    python3 -m venv venv
    . venv/bin/activate
    pip3 install -r requirement.txt

# Run program on API
    python3 API.py

Set max_level, loop_construct, if_construct, control_construct, function_construct, arithOP_construct and excep_construct.

# Example:

max_level = 3

loop_construct = 1

if_construct = 1 

control_construct = 0 

function_construct = 1

arithOP_construct = 0

excep_construct = 0

file1 = test/1.py

file2 = test/2.py

# Run program on CLI
    python3 generateAST.py <file-path> <file-path> <construct flags> <max levels>

# Example:
    python3 generateAST.py test/1.py test/2.py 110100 3

# Construct Flag

It is a 6 character long binary string code.

Index | Explaination                                         | Denoted By
---   | ---                                                  | --- 
0     |   Do you want to include loops count?                | For + While
1     |   Do you want to include if count?                   | If
2     |   Do you want to include control count?              | Break + Continue
3     |   Do you want to include function count?             | Function Def
4     |   Do you want to include Arithmetic Operation count? | UnaryOp+BinOp
5     |   Do you want to include Exception Handler count?    | ExceptHandler

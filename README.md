# CodeSimilarityChecker
A tool which can tell how similar two programs are and assigns a score to their similarity.

# Installation guide
    python3 -m venv venv
    . venv/bin/activate
    pip3 install -r requirement.txt

# Run program
    python3 generateAST.py <file-path> <file-path>

# Example:
    python3 generateAST.py test/1.py test/2.py code

# Code

It is a 6 character long binary string code.

Index | Explaination
---   | --- 
0     |   Do you want to include loops count?                : For + While
1     |   Do you want to include if count?                   : If
2     |   Do you want to include control count?              : Break + Continue
3     |   Do you want to include function count?             : Function Def
4     |   Do you want to include Arithmetic Operation count? : UnaryOp+BinOp
5     |   Do you want to include Exception Handler count?    : ExceptHandler
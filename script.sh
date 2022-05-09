rm test/generateAST.py
cat generateAST.py >> test/generateAST.py
echo >> test/generateAST.py
cat winnowing.py >> test/generateAST.py
echo >> test/generateAST.py
cat similarity.py >> test/generateAST.py

python3 similarity.py test/generateAST.py test/mix.py 111111 3


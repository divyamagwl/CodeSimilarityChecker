all : prog1 prog2 winnowing

prog1 : 
	python3 generate_ast.py 1 Test_Codes/test1a.py

prog2 : prog1
	python3 generate_ast.py 2 Test_Codes/test1b.py

winnowing : prog1 prog2
	python3 winnowing.py program1 program2

clean:
	rm *.txt *.csv

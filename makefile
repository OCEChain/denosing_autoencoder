# target: dependencies
#  [tab] system command

all:
	g++ -O3 utils.cpp DAE.cpp Softmax.cpp StackedAE.cpp main.cpp -I /Users/ruizhang/Downloads/eigen-eigen-10219c95fe65/ 

clean:
	rm -f *.o a.out


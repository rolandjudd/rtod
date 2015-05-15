all: main.cpp
	g++ -g -o rtod main.cpp `pkg-config --libs opencv`

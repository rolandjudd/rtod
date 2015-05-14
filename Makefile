all: main.cpp
	g++ -g main.cpp `pkg-config --libs opencv`

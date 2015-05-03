all: main.cpp
	g++ main.cpp `pkg-config --libs opencv`

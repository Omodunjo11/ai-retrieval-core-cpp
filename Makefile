CXX = clang++
CXXFLAGS = -O3 -march=native -ffast-math -std=c++17

SRC = src/main.cpp
OUT = build/engine

all:
	mkdir -p build
	$(CXX) $(CXXFLAGS) $(SRC) -o $(OUT)

run: all
	./$(OUT)

clean:
	rm -rf build
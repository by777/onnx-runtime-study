#!/bin/bash
g++ -std=c++11 04_trace.cpp -I../mnn-src/include -L../mnn-src/build -lMNN -o 04_trace
LD_LIBRARY_PATH=../mnn-src/build ./04_trace

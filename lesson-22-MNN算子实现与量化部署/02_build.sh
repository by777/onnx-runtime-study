#!/bin/bash
g++ -std=c++11 03_run.cpp -I../mnn-src/include -L../mnn-src/build -lMNN -o 03_run
LD_LIBRARY_PATH=../mnn-src/build ./03_run
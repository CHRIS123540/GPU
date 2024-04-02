#!/bin/bash

cd build
rm *
cmake ..
make
./basicfwd -a 81:00.0 -l 16-19

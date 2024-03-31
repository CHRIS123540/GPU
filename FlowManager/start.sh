#!/bin/bash

echo  "ethtool -K 网卡 large-receive-offload on"
make clean
make
./build/FlowManager -l 0-10 -a 03:00.0,representor=[65535] -a 03:00.1,representor=[65535]  10

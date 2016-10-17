#!/bin/bash

#export CXX=icpc
#export CC=icc

{
mkdir -p release
pushd release
cmake .. -DCMAKE_BUILD_TYPE=Release
popd
}


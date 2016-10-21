#!/bin/bash

git clone https://github.com/gflags/gflags.git
cd gflags
git checkout d701ceac73be2c43b6e7b97474184e626fded88b
cd ..


mkdir -p hlda_test/lib
cd hlda_test/lib/
wget https://github.com/google/googletest/archive/release-1.8.0.zip
unzip release-1.8.0.zip
rm release-1.8.0.zip

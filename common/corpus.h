#ifndef __CORPUS_H
#define __CORPUS_H

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include "types.h"

class Corpus {
public:
    Corpus(const char *vocabPath, const char *dataPath);

    Corpus(const Corpus &from, int start, int end);

    std::vector<std::vector<TWord>> w;
    std::vector<std::string> vocab;
    std::unordered_map<std::string, TWord> word2id;

    TDoc D;
    TWord V;
    TSize T;
};

#endif

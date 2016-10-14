//
// Created by jianfei on 10/14/16.
//

#ifndef HLDA_EXTERNALHLDA_H
#define HLDA_EXTERNALHLDA_H

#include <map>
#include <string>
#include "CollapsedSampling.h"

class ExternalHLDA : public CollapsedSampling {
public:
    ExternalHLDA(Corpus &corpus, int L,
                 TProb alpha, TProb beta, std::vector<TProb> gamma,
                 std::string prefix);

    void Initialize();

    void Estimate();

private:
    void ReadTree();

    void ReadPath();

    void ReadLevel();

    std::string prefix;
    std::map<int, int> node_id_map;
    std::vector<int> doc_id_map;
    std::vector<Tree::Node *> nodes;
};


#endif //HLDA_EXTERNALHLDA_H

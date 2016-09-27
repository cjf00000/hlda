//
// Created by jianfei on 9/2/16.
//

#ifndef HLDA_DIRECTSAMPLINGHDP_H
#define HLDA_DIRECTSAMPLINGHDP_H

#include <vector>
#include "types.h"
#include "Matrix.h"
#include "IDPool.h"
#include "xorshift.h"

class Corpus;

class DirectSamplingHDP {
    struct Document {
        std::vector<TTopic> z;
        std::vector<TWord> w;
    };

    DirectSamplingHDP(Corpus &corpus, int n_iter, double gamma, double alpha, double eta);

    void Initialize();

    void Estimate();

private:
    void SampleBeta(int n_end);

    void SampleDoc(Document &doc);

    double Perplexity();

    Corpus &corpus;
    int n_iter;
    double gamma;
    double alpha;
    double eta;

    std::vector<Document> docs;

    Matrix<TCount> cwk;
    Matrix<double> log_stirling;
    std::vector<TCount> ck;
    std::vector<TProb> beta;
    TProb beta_u;

    IDPool pool;
    xorshift generator;
};


#endif //HLDA_DIRECTSAMPLINGHDP_H

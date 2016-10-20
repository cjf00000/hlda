//
// Created by jianfei on 16-10-20.
//

#include "corpus.h"
#include "InstantiatedWeightSampling.h"
#include "mkl_vml.h"
#include <iostream>

using namespace std;

InstantiatedWeightSampling::InstantiatedWeightSampling(Corpus &corpus, int L,
                                                       std::vector<TProb> alpha, std::vector<TProb> beta,
                                                       std::vector<TProb> gamma,
                                                       int num_iters, int mc_samples, int mc_iters,
                                                       size_t minibatch_size, int topic_limit,
                                                       int threshold, int branching_factor) :
        PartiallyCollapsedSampling(corpus, L, alpha, beta, gamma, num_iters, mc_samples, mc_iters,
                                   minibatch_size, topic_limit, threshold),
        branching_factor(branching_factor) {
    new_topic = false;
    tree.default_is_collapsed = false;
    tree.GetRoot()->is_collapsed = false;
}

void InstantiatedWeightSampling::SamplePhi() {
    // Add instantiated nodes
    tree.Instantiate(tree.GetRoot(), branching_factor);

    auto nodes = tree.GetAllNodes();
    for (auto *node: nodes)
        cout << node->is_collapsed;
    cout << endl;

    for (TLen l = 0; l < L; l++) {
        TTopic K = tree.NumNodes(l);

        phi[l].SetC(K);
        log_phi[l].SetC(K);
        count[l].SetC(K);
        while (ck[l].size() < (size_t) K) ck[l].push_back(0);

        for (TTopic k = 0; k < K; k++) {
            TProb inv_sum = 1. / (beta[l] * corpus.V + ck[l][k]);
            for (TWord v = 0; v < corpus.V; v++) {
                float prob = (float) ((count[l](v, k) + beta[l]) * inv_sum);
                phi[l](v, k) = prob;
                log_phi[l](v, k) = prob;
            }
        }

        for (TWord v = 0; v < corpus.V; v++)
            vsLn(K, &log_phi[l](v, 0), &log_phi[l](v, 0));
    }
}

void InstantiatedWeightSampling::InitializeTreeWeight() {

}

void InstantiatedWeightSampling::Estimate() {
    PartiallyCollapsedSampling::Estimate();
    tree.Instantiate(tree.GetRoot(), 0);
}
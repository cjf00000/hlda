//
// Created by jianfei on 16-10-20.
//

#include "corpus.h"
#include "InstantiatedWeightSampling.h"
#include <iostream>

using namespace std;

InstantiatedWeightSampling::InstantiatedWeightSampling(Corpus &corpus, int L,
                                                       std::vector<TProb> alpha, std::vector<TProb> beta,
                                                       std::vector<TProb> gamma,
                                                       int num_iters, int mc_samples, int mc_iters,
                                                       size_t minibatch_size, int topic_limit,
                                                       int threshold, int branching_factor, bool sample_phi) :
        PartiallyCollapsedSampling(corpus, L, alpha, beta, gamma, num_iters, mc_samples, mc_iters,
                                   minibatch_size, topic_limit, threshold, sample_phi),
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
    }
    ComputePhi();

    // Permute
    num_nontrivial_nodes.resize((size_t) L);
    for (TLen l = 0; l < L; l++) {
        auto perm = tree.Compress(l);

        phi[l].SetC(tree.NumNodes(l));
        log_phi[l].SetC(tree.NumNodes(l));

        count[l].PermuteColumns(perm);
        phi[l].PermuteColumns(perm);
        log_phi[l].PermuteColumns(perm);

        Permute(ck[l], perm);

        int i;
        for (i = 0; i < (int) perm.size() && ck[l][i] > 0; i++);

        num_nontrivial_nodes[l] = i;
    }
}

void InstantiatedWeightSampling::InitializeTreeWeight() {

}

void InstantiatedWeightSampling::Estimate() {
    PartiallyCollapsedSampling::Estimate();
    tree.Instantiate(tree.GetRoot(), 0);
}

std::vector<TProb> InstantiatedWeightSampling::WordScore(Document &doc, int l,
                                                         int num_instantiated, int num_collapsed) {
    int K = num_instantiated + num_collapsed;
    std::vector<TProb> result((size_t) K + 1);

    int K2 = num_nontrivial_nodes[l];
    auto result2 = CollapsedSampling::WordScore(doc, l, K2, 0);

    result.back() = result2.back();
    for (int k = 0; k < K2; k++)
        result[k] = result2[k];

    // Trivial nodes: log_phi = log(1 / V)
    auto begin = doc.BeginLevel(l);
    auto end = doc.EndLevel(l);
    TProb log_p = log(1. / corpus.V) * (end - begin);
    for (int k = K2; k < K; k++)
        result[k] = log_p;

    return result;
}
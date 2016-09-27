//
// Created by jianfei on 9/19/16.
//

#include "PartiallyCollapsedSampling.h"
#include "Clock.h"
#include "corpus.h"

using namespace std;

PartiallyCollapsedSampling::PartiallyCollapsedSampling(Corpus &corpus, int L, TProb alpha, TProb beta, TProb gamma,
                                                       int num_iters) :
        CollapsedSampling(corpus, L, alpha, beta, gamma, num_iters) {

}

void PartiallyCollapsedSampling::Initialize() {
    CollapsedSampling::Initialize();
    SamplePhi();
}

void PartiallyCollapsedSampling::Estimate() {
    for (int it = 0; it < num_iters; it++) {
        Clock clk;

        for (auto &doc: docs) {
            SampleC(doc, true);
            SampleZ(doc, true);
        }

        SamplePhi();

        double time = clk.toc();
        double throughput = corpus.T / time / 1048576;
        double perplexity = Perplexity();
        auto nodes = tree.GetAllNodes();
        printf("Iteration %d, %lu topics, %.2f seconds (%.2fMtoken/s), perplexity = %.2f\n",
               it, nodes.size(), time, throughput, perplexity);
    }
}

void PartiallyCollapsedSampling::SampleC(Document &doc, bool decrease_count) {
    if (decrease_count) {
        UpdateDocCount(doc, -1);
        tree.UpdateNumDocs(doc.c.back(), -1);
    }

    InitializeTreeWeight();

    DFSSample(doc);

    UpdateDocCount(doc, 1);
    tree.UpdateNumDocs(doc.c.back(), 1);
}

void PartiallyCollapsedSampling::SampleZ(Document &doc, bool decrease_count) {
    std::vector<TCount> cdl((size_t) L);
    std::vector<TProb> prob((size_t) L);
    for (auto k: doc.z) cdl[k]++;

    auto ids = doc.GetIDs();
    std::vector<bool> is_collapsed((size_t) L);
    for (int l = 0; l < L; l++) is_collapsed[l] = doc.c[l]->is_collapsed;

    TProb beta_bar = beta.Concentration();

    for (size_t n = 0; n < doc.z.size(); n++) {
        TWord v = doc.w[n];
        TTopic l = doc.z[n];
        if (decrease_count) {
            --cdl[l];
            --count(ids[l], v);
            --ck[ids[l]];
        }

        for (TLen i = 0; i < L; i++)
            if (is_collapsed[i])
                prob[i] = (alpha + cdl[i]) *
                          (beta(v) + count(ids[i], v)) / (beta_bar + ck[ids[i]]);
            else
                prob[i] = (alpha + cdl[i]) * phi(ids[i], v);

        l = DiscreteSample(prob.begin(), prob.end(), generator);
        doc.z[n] = l;

        ++cdl[l];
        ++count(ids[l], v);
        ++ck[ids[l]];
    }
}

TProb PartiallyCollapsedSampling::WordScore(Document &doc,
                                            int l, int topic, Tree::Node *node) {
    // Collapsed
    if (topic == -1 || node->is_collapsed)
        return CollapsedSampling::WordScore(doc, l, topic, node);

    // Not collapsed
    auto *b = doc.BeginLevel(l);
    auto *e = doc.EndLevel(l);

    double phi_score = 0;
    for (auto *w = b; w != e; w++)
        phi_score += log_phi(topic, *w);

    return phi_score;
}


void PartiallyCollapsedSampling::SamplePhi() {
    TTopic K = tree.GetMaxID();
    auto nodes = tree.GetAllNodes();
    int threshold = 20;
    for (auto *node: nodes)
        if (ck[node->id] > threshold) {
            node->is_collapsed = false;
        }

    log_phi.SetR(K);
    phi.SetR(K);
    for (TTopic k = 0; k < K; k++) {
        double inv_sum = 1. / (beta.Concentration() + ck[k]);

        for (TWord v = 0; v < corpus.V; v++) {
            double prob = (count(k, v) + beta(v)) * inv_sum;
            phi(k, v) = prob;
            log_phi(k, v) = log(prob);
        }
    }
}
//
// Created by jianfei on 9/19/16.
//

#include "PartiallyCollapsedSampling.h"
#include "Clock.h"
#include "corpus.h"
#include <iostream>

using namespace std;

PartiallyCollapsedSampling::PartiallyCollapsedSampling(Corpus &corpus, int L, vector<TProb> alpha, vector<TProb> beta,
                                                       vector<TProb> gamma,
                                                       int num_iters, int mc_samples, int mc_iters,
                                                       size_t minibatch_size,
                                                       int remove_iters, int remove_paths) :
        CollapsedSampling(corpus, L, alpha, beta, gamma, num_iters, mc_samples, mc_iters,
                          remove_iters, remove_paths),
        minibatch_size(minibatch_size) {
    current_it = -1;
}

void PartiallyCollapsedSampling::Initialize() {
    //CollapsedSampling::Initialize();
    ck.resize(1);
    count.SetR(1);
    ck[0] = 0;
    current_it = -1;

    cout << "Start initialize..." << endl;
    if (minibatch_size == 0)
        minibatch_size = docs.size();

    for (size_t d_start = 0; d_start < docs.size(); d_start += minibatch_size) {
        size_t d_end = min(docs.size(), d_start + minibatch_size);
        for (size_t d = d_start; d < d_end; d++) {
            auto &doc = docs[d];

            for (auto &k: doc.z)
                k = generator() % L;

            SampleC(doc, false, true);
            SampleZ(doc, true, true);
        }
        SamplePhi();

        printf("Processed %lu documents\n", d_end);
    }
    cout << "Initialized with " << tree.GetMaxID() << " topics." << endl;

    SamplePhi();
}

void PartiallyCollapsedSampling::Estimate() {
    for (int it = 0; it < num_iters; it++) {
        current_it = it;
        Clock clk;

        /*if (it == pow(floor(sqrt(it)), 2) && it >= 3) {
            printf("Resetting...\n");
            for (auto &doc: docs)
                ResetZ(doc);
        }*/
        if (current_it >= mc_iters)
            mc_samples = -1;

        SamplePhi();

        for (auto &doc: docs) {
            if (doc.initialized) {
                SampleC(doc, true, true);
                SampleZ(doc, true, true);
            }
        }

        SamplePhi();

        for (auto &doc: docs) {
            if (!doc.initialized) {
                doc.initialized = true;
                SampleC(doc, false, true);
                SampleZ(doc, true, true);
            }
        }

        //SamplePhi();

        auto nodes = tree.GetAllNodes();
        int num_big_nodes = 0;
        int num_docs_big = 0;
        for (auto *node: nodes)
            if (node->num_docs > 5) {
                num_big_nodes++;
                if (node->depth + 1 == L)
                    num_docs_big += node->num_docs;
            }

        double time = clk.toc();
        double throughput = corpus.T / time / 1048576;
        double perplexity = Perplexity();
        //printf("Iteration %d, %lu topics, %.2f seconds (%.2fMtoken/s), perplexity = %.2f\n",
        //       it, nodes.size(), time, throughput, perplexity);
        printf("Iteration %d, %lu topics (%d, %d), %.2f seconds (%.2fMtoken/s), perplexity = %.2f\n",
               it, nodes.size(), num_big_nodes, num_docs_big, time, throughput, perplexity);
    }
}

void PartiallyCollapsedSampling::SampleC(Document &doc, bool decrease_count, bool increase_count) {
    if (decrease_count) {
        UpdateDocCount(doc, -1);
        tree.UpdateNumDocs(doc.c.back(), -1);
        for (auto *node: doc.c)
            if (!node->is_collapsed && node->num_docs < 50) {
                // Become collapsed
                node->is_collapsed = true;
            }
    }

    InitializeTreeWeight();

    DFSSample(doc);

    UpdateDocCount(doc, 1);
    tree.UpdateNumDocs(doc.c.back(), 1);
}

void PartiallyCollapsedSampling::SampleZ(Document &doc, bool decrease_count, bool increase_count) {
    std::vector<TCount> cdl((size_t) L);
    std::vector<TProb> prob((size_t) L);
    for (auto k: doc.z) cdl[k]++;

    auto ids = doc.GetIDs();
    std::vector<bool> is_collapsed((size_t) L);
    for (int l = 0; l < L; l++) is_collapsed[l] = doc.c[l]->is_collapsed;

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
                prob[i] = (alpha[i] + cdl[i]) *
                          (beta[i] + count(ids[i], v)) / (beta[i] * corpus.V + ck[ids[i]]);
            else {
                //if (current_it < 5)
                //puts("What?1");
                prob[i] = (alpha[i] + cdl[i]) * phi(ids[i], v);
            }

        l = DiscreteSample(prob.begin(), prob.end(), generator);
        doc.z[n] = l;

        ++cdl[l];
        ++count(ids[l], v);
        ++ck[ids[l]];
    }
    /*double sum = 0;
    for (TLen l = 0; l < L; l++)
        sum += (doc.theta[l] = cdl[l] + alpha[l]);
    for (TLen l = 0; l < L; l++)
        doc.theta[l] /= sum;*/
}

TProb PartiallyCollapsedSampling::WordScore(Document &doc,
                                            int l, int topic, Tree::Node *node) {
    // Collapsed
    if (topic == -1 || node->is_collapsed)
        return CollapsedSampling::WordScore(doc, l, topic, node);

    //if (current_it < 5)
    //puts("What?2");

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
    int threshold = 50;
    // TODO: Is it really correct to process collapsed and uncollapsed simutanaeously?
    for (auto *node: nodes)
        node->is_collapsed = node->num_docs < threshold;

    for (auto *node: nodes)
        cout << node->is_collapsed;
    cout << endl;

    log_phi.SetR(K);
    phi.SetR(K);

    for (auto *node: nodes) {
        TTopic k = node->id;
        TProb b = beta[node->depth];

        double inv_sum = 1. / (b * corpus.V + ck[k]);

        for (TWord v = 0; v < corpus.V; v++) {
            double prob = (count(k, v) + b) * inv_sum;
            phi(k, v) = prob;
            log_phi(k, v) = log(prob);
        }
    }
}
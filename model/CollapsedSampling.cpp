//
// Created by jianfei on 8/30/16.
//

#include <iostream>
#include <cmath>
#include "CollapsedSampling.h"
#include "Clock.h"
#include "corpus.h"

using namespace std;

CollapsedSampling::CollapsedSampling(Corpus &corpus, int L,
                                     std::vector<TProb> alpha, std::vector<TProb> beta, vector<TProb> gamma,
                                     int num_iters, int mc_samples, int mc_iters,
                                     int topic_limit) :
        BaseHLDA(corpus, L, alpha, beta, gamma, num_iters, mc_samples), mc_iters(mc_iters),
        topic_limit(topic_limit) {}

void CollapsedSampling::Initialize() {
    ck.resize(1);
    count.SetR(1);
    ck[0] = 0;
    current_it = -1;

    cout << "Start initialize..." << endl;
    for (auto &doc: docs) {
        for (auto &k: doc.z)
            k = generator() % L;

        SampleC(doc, false, true);
        SampleZ(doc, true, true);

        if (tree.GetMaxID() > topic_limit)
            throw runtime_error("There are too many topics");
    }
    cout << "Initialized with " << tree.GetMaxID() << " topics." << endl;
}

void CollapsedSampling::Estimate() {
    for (int it = 0; it < num_iters; it++) {
        current_it = it;
        Clock clk;
        Check();
        if (current_it >= mc_iters)
            mc_samples = -1;

        for (auto &doc: docs) {
            SampleC(doc, true, true);
            SampleZ(doc, true, true);
        }

        double time = clk.toc();
        double throughput = corpus.T / time / 1048576;
        double perplexity = Perplexity();
        auto nodes = tree.GetAllNodes();

        int num_big_nodes = 0;
        int num_docs_big = 0;
        for (auto *node: nodes)
            if (node->num_docs > 5) {
                num_big_nodes++;
                if (node->depth + 1 == L)
                    num_docs_big += node->num_docs;
            }

        printf("Iteration %d, %lu topics (%d, %d), %.2f seconds (%.2fMtoken/s), perplexity = %.2f\n",
               it, nodes.size(), num_big_nodes, num_docs_big, time, throughput, perplexity);
    }
}

void CollapsedSampling::SampleZ(Document &doc, bool decrease_count, bool increase_count) {
    TLen N = (TLen) doc.z.size();
    auto ids = doc.GetIDs();
    std::vector<TProb> prob((size_t) L);
    std::vector<TCount> cdl((size_t) L);
    for (auto l: doc.z) cdl[l]++;

    for (TLen n = 0; n < N; n++) {
        TWord v = doc.w[n];
        TTopic l = doc.z[n];

        if (decrease_count) {
            --count(ids[l], v);
            --ck[ids[l]];
            --cdl[l];
        }

        for (TTopic i = 0; i < L; i++)
            prob[i] = (cdl[i] + alpha[i]) *
                      (count(ids[i], v) + beta[i]) / (ck[ids[i]] + beta[i] * corpus.V);

        l = DiscreteSample(prob.begin(), prob.end(), generator);

        if (increase_count) {
            ++count(ids[l], v);
            ++ck[ids[l]];
            ++cdl[l];
        }
        doc.z[n] = l;
    }

    /*double sum = 0;
    for (TLen l = 0; l < L; l++)
        sum += (doc.theta[l] = cdl[l] + alpha[l]);
    for (TLen l = 0; l < L; l++)
        doc.theta[l] /= sum;*/
}

void CollapsedSampling::SampleC(Document &doc, bool decrease_count, bool increase_count) {
    // Try delayed update for SampleC
    if (decrease_count) {
        UpdateDocCount(doc, -1);
        tree.UpdateNumDocs(doc.c.back(), -1);
    }

    // Compute NCRP probability
    InitializeTreeWeight();

    // Sample
    DFSSample(doc);

    // Increase num_docs
    if (increase_count) {
        UpdateDocCount(doc, 1);
        tree.UpdateNumDocs(doc.c.back(), 1);
    }
}

TProb CollapsedSampling::WordScore(Document &doc, int l, int topic, Tree::Node *node) {
    UNUSED(node);

    auto *b = doc.BeginLevel(l);
    auto *e = doc.EndLevel(l);

    decltype(b) w_next = nullptr;
    double result = 0;
    for (auto *w = b; w != e; w = w_next) {
        for (w_next = w; w_next != e && *w_next == *w; w_next++);
        int w_count = (int) (w_next - w);

        int cnt = topic == -1 ? 0 : count(topic, *w);
        result += LogGammaDifference(cnt + beta[l], w_count);
    }

    int w_count = (int) (e - b);
    int cnt = topic == -1 ? 0 : ck[topic];
    result -= lgamma(cnt + beta[l] * corpus.V + w_count) - lgamma(cnt + beta[l] * corpus.V);

    return result;
}

double CollapsedSampling::Perplexity() {
    doc_avg_likelihood.resize(docs.size());
    decltype(doc_avg_likelihood) new_dal;

    double log_likelihood = 0;
    std::vector<TProb> theta((size_t) L);

    size_t T = 0;
    for (auto &doc: docs) {
        double old_log_likelihood = log_likelihood;

        T += doc.z.size();
        // Compute theta
        for (auto k: doc.z) theta[k]++;
        double inv_sum = 1. / (doc.z.size() + alpha_bar);
        for (TLen l = 0; l < L; l++)
            theta[l] = (theta[l] + alpha[l]) * inv_sum;

        auto ids = doc.GetIDs();

        for (size_t n = 0; n < doc.z.size(); n++) {
            double prob = 0;
            TWord v = doc.w[n];
            for (int l = 0; l < L; l++) {
                double phi = (count(ids[l], v) + beta[l]) /
                             (ck[ids[l]] + beta[l] * corpus.V);
                prob += theta[l] * phi;
            }
            log_likelihood += log(prob);
        }

        double new_doc_avg_likelihood = (log_likelihood - old_log_likelihood) / doc.z.size();
        new_dal.push_back(new_doc_avg_likelihood);
    }

    return exp(-log_likelihood / T);
}

void CollapsedSampling::Check() {
    int sum = 0;
    for (TTopic k = 0; k < tree.GetMaxID(); k++)
        for (TWord v = 0; v < corpus.V; v++) {
            if (count(k, v) < 0)
                throw runtime_error("Error!");
            sum += count(k, v);
        }
    if (sum != corpus.T)
        throw runtime_error("Total token error!");
}

void CollapsedSampling::DFSSample(Document &doc) {
    auto nodes = tree.GetAllNodes();
    vector<TProb> prob(nodes.size(), -1e9);

    // Warning: this is not thread safe
    for (int s = 0; s < max(mc_samples, 1); s++) {
        // Resample Z
        discrete_distribution<int> mult(doc.theta.begin(), doc.theta.end());
        if (mc_samples != -1) {
            for (auto &l: doc.z) l = mult(generator);
        }
        doc.PartitionWByZ(L);

        // Compute empty probability
        vector<TProb> emptyProbability((size_t) L);
        for (int l = 0; l < L; l++)
            emptyProbability[l] = WordScore(doc, l, -1, nullptr);
        for (int l = L - 2; l >= 0; l--)
            emptyProbability[l] += emptyProbability[l + 1];

        for (size_t i = 0; i < nodes.size(); i++) {
            auto *node = nodes[i];

            if (node->depth == 0)
                node->sum_log_prob = WordScore(doc, node->depth, node->id, node);
            else
                node->sum_log_prob = node->parent->sum_log_prob +
                                     WordScore(doc, node->depth, node->id, node);

            if (node->depth + 1 == L) {
                prob[i] = LogSum(prob[i], node->sum_log_prob + node->sum_log_weight);
            } else {
                prob[i] = LogSum(prob[i], node->sum_log_prob + node->sum_log_weight +
                                          emptyProbability[node->depth + 1]);
            }
        }
    }

    // Sample
    Softmax(prob.begin(), prob.end());
    int node_number = DiscreteSample(prob.begin(), prob.end(), generator);
    if (node_number < 0 || node_number >= (int) prob.size())
        throw runtime_error("Invalid node number");
    auto *current = nodes[node_number];

    while (current->depth + 1 < L)
        current = tree.AddChildren(current);

    tree.GetPath(current, doc.c);
}

void CollapsedSampling::UpdateDocCount(Document &doc, int delta) {
    TTopic K = tree.GetMaxID();
    count.SetR(K);
    while (ck.size() < (size_t) K) ck.push_back(0);

    auto ids = doc.GetIDs();
    TLen N = (TLen) doc.z.size();
    for (TLen n = 0; n < N; n++) {
        TTopic k = ids[doc.z[n]];
        TWord v = doc.w[n];
        count(k, v) += delta;
        ck[k] += delta;
    }
}

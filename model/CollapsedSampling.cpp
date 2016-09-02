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
                                     TProb alpha, TProb beta, TProb gamma, int num_iters) :
        BaseHLDA(corpus, L, alpha, beta, gamma, num_iters) {}

void CollapsedSampling::Initialize() {
    BaseHLDA::Initialize();
    TTopic K = tree.GetMaxID();
    while ((TTopic) ck.size() < K) ck.push_back(0);
    for (TTopic k = 0; k < K; k++)
        for (TWord v = 0; v < corpus.V; v++)
            ck[k] += count(k, v);
}

void CollapsedSampling::Estimate() {
    for (int it = 0; it < num_iters; it++) {
        Clock clk;
        Check();
        SampleC();

        for (auto &doc: docs)
            SampleZ(doc);

        double time = clk.toc();
        double throughput = corpus.T / time / 1048576;
        double perplexity = Perplexity();
        auto nodes = tree.GetAllNodes();
        printf("Iteration %d, %lu topics, %.2f seconds (%.2fMtoken/s), perplexity = %.2f\n",
               it, nodes.size(), time, throughput, perplexity);
    }
}

void CollapsedSampling::SampleZ(Document &doc) {
    TLen N = (TLen) doc.z.size();
    auto ids = doc.GetIDs();
    std::vector<TProb> prob((size_t) L);
    std::vector<TCount> cdl((size_t) L);
    for (auto l: doc.z) cdl[l]++;
    TProb beta_bar = beta.Concentration();

    for (TLen n = 0; n < N; n++) {
        TWord v = doc.w[n];
        TTopic l = doc.z[n];

        --count(ids[l], v);
        --ck[ids[l]];
        --cdl[l];

        for (TTopic i = 0; i < L; i++)
            prob[i] = (cdl[i] + alpha) *
                      (count(ids[i], v) + beta(v)) / (ck[ids[i]] + beta_bar);

        l = DiscreteSample(prob.begin(), prob.end(), generator);

        ++count(ids[l], v);
        ++ck[ids[l]];
        ++cdl[l];
        doc.z[n] = l;
    }
}

void CollapsedSampling::SampleC() {
    for (auto &doc: docs) {
        UpdateDocCount(doc, -1);
        tree.UpdateNumDocs(doc.c.back(), -1);

        // Compute NCRP probability
        InitializeTreeWeight();

        // Sample
        DFSSample(doc);

        // Increase num_docs
        UpdateDocCount(doc, 1);
        tree.UpdateNumDocs(doc.c.back(), 1);
    }
    /*
    // Count
    std::vector<int> counts(tree.GetMaxID());
    for (auto &doc: docs)
        for (auto *node: doc.c)
            counts[node->id]++;

    cout << "C" << endl;
    auto nodes = tree.GetAllNodes();
    for (auto *node: nodes)
        if (counts[node->id] != node->num_docs)
            throw runtime_error("Error");*/
}

TProb CollapsedSampling::WordScore(Document &doc, int l, int topic) {
    auto *b = doc.BeginLevel(l);
    auto *e = doc.EndLevel(l);
    double beta_bar = beta.Concentration();

    decltype(b) w_next = nullptr;
    double result = 0;
    for (auto *w = b; w != e; w = w_next) {
        for (w_next = w; w_next != e && *w_next == *w; w_next++);
        int w_count = (int) (w_next - w);

        int cnt = topic == -1 ? 0 : count(topic, *w);
        result += LogGammaDifference(cnt + beta(*w), w_count);
    }

    int w_count = (int) (e - b);
    int cnt = topic == -1 ? 0 : ck[topic];
    result -= lgamma(cnt + beta_bar + w_count) - lgamma(cnt + beta_bar);

    return result;
}

double CollapsedSampling::Perplexity() {
    double log_likelihood = 0;
    std::vector<TProb> theta((size_t) L);
    double beta_bar = beta.Concentration();
    for (auto &doc: docs) {
        // Compute theta
        for (auto k: doc.z) theta[k]++;
        double inv_sum = 1. / (doc.z.size() + alpha * L);
        for (auto &t: theta) t = (t + alpha) * inv_sum;

        auto ids = doc.GetIDs();

        for (size_t n = 0; n < doc.z.size(); n++) {
            double prob = 0;
            for (int l = 0; l < L; l++) {
                double phi = (count(ids[doc.z[n]], doc.w[n]) + beta(doc.w[n])) /
                             (ck[ids[doc.z[n]]] + beta_bar);
                prob += theta[l] * phi;
            }
            log_likelihood += log(prob);
        }
    }
    return exp(-log_likelihood / corpus.T);
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

void CollapsedSampling::InitializeTreeWeight() {
    auto nodes = tree.GetAllNodes();
    nodes[0]->sum_log_weight = 0;

    for (auto *node: nodes)
        if (!node->children.empty()) {
            // Propagate
            double sum_weight = gamma;
            for (auto *child: node->children)
                sum_weight += child->num_docs;

            for (auto *child: node->children)
                child->sum_log_weight = node->sum_log_weight +
                                        log((child->num_docs + 1e-10) / sum_weight);

            node->sum_log_weight += log(gamma / sum_weight);
        }
}

void CollapsedSampling::DFSSample(Document &doc) {
    auto nodes = tree.GetAllNodes();
    vector<TProb> prob;
    prob.reserve(nodes.size());

    doc.PartitionWByZ(L);

    // Compute empty probability
    vector<TProb> emptyProbability((size_t) L);
    for (int l = 0; l < L; l++)
        emptyProbability[l] = WordScore(doc, l, -1);
    for (int l = L - 2; l >= 0; l--)
        emptyProbability[l] += emptyProbability[l + 1];

    // Warning: this is not thread safe
    for (auto *node: nodes) {
        if (node->depth == 0)
            node->sum_log_prob = WordScore(doc, node->depth, node->id);
        else
            node->sum_log_prob = node->parent->sum_log_prob +
                                 WordScore(doc, node->depth, node->id);

        if (node->depth + 1 == L) {
            prob.push_back(node->sum_log_prob + node->sum_log_weight);
        } else {
            prob.push_back(node->sum_log_prob + node->sum_log_weight +
                           emptyProbability[node->depth + 1]);
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
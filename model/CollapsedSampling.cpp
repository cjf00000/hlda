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
                                     int num_iters, int mc_samples, int anneal_iters) :
        BaseHLDA(corpus, L, alpha, beta, gamma, num_iters, mc_samples),
        anneal_iters(anneal_iters) {}

void CollapsedSampling::Initialize() {
    ck.resize(1);
    count.SetR(1);
    ck[0] = 0;
    current_it = -1;

    ProgressivelyOnlineInitialize();

    //printf("%lf %lf\n", beta(0), beta.Concentration());

    //auto bak = tree.gamma;
    //tree.gamma = 1e-9;
    //BaseHLDA::Initialize();
    //tree.gamma = bak;

    // Build Ck
    //TTopic K = tree.GetMaxID();
    //while ((TTopic) ck.size() < K) ck.push_back(0);
    //for (TTopic k = 0; k < K; k++)
    //    for (TWord v = 0; v < corpus.V; v++)
    //        ck[k] += count(k, v);
}

void CollapsedSampling::ProgressivelyOnlineInitialize() {
    cout << "Start initialize..." << endl;
    for (auto &doc: docs) {
        for (auto &k: doc.z)
            k = generator() % L;

        SampleC(doc, false, true);
        SampleZ(doc, true, true);
    }
    cout << "Initialized with " << tree.GetMaxID() << " topics." << endl;
}

void CollapsedSampling::Estimate() {
    for (int it = 0; it < num_iters; it++) {
        current_it = it;
        Clock clk;
        Check();
        if (current_it >= 20)
            mc_samples = -1;

        /*if (it % 5 == 4 && it <= 20) {
            printf("Resetting...\n");
            for (auto &doc: docs)
                ResetZ(doc);
        }*/
        if (current_it <= anneal_iters && current_it >= 5)
            RemovePath();

        for (auto &doc: docs) {
            if (doc.initialized) {
                SampleC(doc, true, true);
                SampleZ(doc, true, true);
            } else {
                doc.initialized = true;
                SampleC(doc, false, true);
                SampleZ(doc, true, true);
            }
        }
        //Recount();

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

    double anneal_start = 0.8;
    double anneal_rate = 1;
    if (anneal_iters != -1)
        anneal_rate = min(anneal_start + (1 - anneal_start) / (anneal_iters + 1) * (current_it + 1), 1.0);

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
    double sum = 0;
    for (TLen l = 0; l < L; l++)
        sum += (doc.theta[l] = cdl[l] + alpha[l]);
    for (TLen l = 0; l < L; l++)
        doc.theta[l] /= sum;
}

void CollapsedSampling::ResetZ(Document &doc) {
    auto ids = doc.GetIDs();
    for (size_t n = 0; n < doc.z.size(); n++) {
        TWord v = doc.w[n];
        TTopic l = doc.z[n];

        --count(ids[l], v);
        --ck[ids[l]];

        l = doc.z[n] = generator() % L;

        ++count(ids[l], v);
        ++ck[ids[l]];
    }
}

void CollapsedSampling::Recount() {
    int K = tree.GetMaxID();
    count.SetR(K);
    count.Clear();
    ck.resize(K);
    fill(ck.begin(), ck.end(), 0);
    auto nodes = tree.GetAllNodes();
    for (auto *node: nodes) node->num_docs = 0;

    for (auto &doc: docs) {
        auto ids = doc.GetIDs();
        for (size_t n = 0; n < doc.w.size(); n++) {
            ck[ids[doc.z[n]]]++;
            count(ids[doc.z[n]], doc.w[n])++;
        }
        tree.UpdateNumDocs(doc.c.back(), 1);
    }
}

void CollapsedSampling::SampleC(Document &doc, bool decrease_count, bool increase_count) {
    // Try delayed update for SampleC
    bool delayed_update = false;
    auto old_doc = doc;
    if (decrease_count && !delayed_update) {
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

    if (decrease_count && delayed_update) {
        UpdateDocCount(old_doc, -1);
        tree.UpdateNumDocs(old_doc.c.back(), -1);
    }
}

TProb CollapsedSampling::WordScore(Document &doc, int l, int topic, Tree::Node *node) {
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
                //prob += phi;
            }
            log_likelihood += log(prob);
        }

        double new_doc_avg_likelihood = (log_likelihood - old_log_likelihood) / doc.z.size();
        new_dal.push_back(new_doc_avg_likelihood);
    }
    /*
   // Compare new_dal with doc_avg_likelihood
   std::vector<pair<double, size_t>> amt_increase;
   for (size_t d = 0; d < docs.size(); d++)
       amt_increase.push_back(make_pair(new_dal[d] - doc_avg_likelihood[d], d));

   sort(amt_increase.begin(), amt_increase.end());
   amt_increase.resize(10);
   if (current_it > 0) {
       ofstream fout(("log_" + to_string(current_it)).c_str());
       for (int i = 0; i < 100; i++) {
           auto d = amt_increase[i].second;
           fout << d << ' ' << doc_avg_likelihood[d] << " -> " << new_dal[d]
                << " Old path: ";
           for (int l = 0; l < L; l++)
               fout << old_doc_ids[d][l] << ':' << old_doc_sizes[d][l] << ' ';

           fout << " New path: ";
           for (int l = 0; l < L; l++)
               fout << docs[d].c[l]->id << ':' << docs[d].c[l]->num_docs << ' ';
           fout << endl;
       }
   }

   doc_avg_likelihood = new_dal;
   old_doc_ids.resize(docs.size());
   old_doc_sizes.resize(docs.size());
   for (size_t d = 0; d < docs.size(); d++) {
       old_doc_ids[d] = docs[d].GetIDs();
       old_doc_sizes[d].resize((size_t) L);
       for (int l = 0; l < L; l++)
           old_doc_sizes[d][l] = docs[d].c[l]->num_docs;
   }*/

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

    //std::vector<TProb> myalpha{100, 100, 100, 100};
    //dirichlet_distribution<TProb> dir(alpha);
    for (int s = 0; s < max(mc_samples, 1); s++) {
        // Resample Z
        // Random Dirichlet

        discrete_distribution<int> mult(doc.theta.begin(), doc.theta.end());
        //discrete_distribution<int> mult(alpha.begin(), alpha.end());

        // Random multinomial
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
                /*if (current_it > 15)
                    prob.push_back(-1e9);
                else*/
                prob[i] = LogSum(prob[i], node->sum_log_prob + node->sum_log_weight +
                                          emptyProbability[node->depth + 1]);
            }
        }
    }

    // -1: 0.5, iter: 1
    /*
    double anneal_start = 0.5;
    double anneal_rate = 1;
    if (anneal_iters!=-1)
        anneal_rate = min(anneal_start + (1-anneal_start)/(anneal_iters+1)*(current_it+1), 1.0);

    for (auto &p: prob) p *= anneal_rate;*/

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

void CollapsedSampling::RemovePath() {
    // Randomly delete 3 paths
    auto nodes = tree.GetAllNodes();
    vector<bool> selected(nodes.size(), false);

    for (int i = 0; i < 2; i++) {
        // Select a path to remove
        int index;
        do {
            index = generator() % (int) nodes.size();
        } while (selected[index] || nodes[index]->depth + 1 != L);
        selected[index] = true;

        auto *node = nodes[index];

        // Reset corresponding documents
        for (auto &doc: docs) {
            if (doc.c.back()->id == node->id) {
                doc.initialized = false;
                UpdateDocCount(doc, -1);
                tree.UpdateNumDocs(doc.c.back(), -1);
                for (auto &l: doc.z) l = generator() % L;
            }
        }
    }
}
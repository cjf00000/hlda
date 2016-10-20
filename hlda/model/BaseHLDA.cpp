//
// Created by jianfei on 8/29/16.
//

#include <iostream>
#include <sstream>
#include "BaseHLDA.h"
#include "corpus.h"

using namespace std;

BaseHLDA::BaseHLDA(Corpus &corpus, int L,
                   std::vector<TProb> alpha, std::vector<TProb> beta, vector<TProb> gamma,
                   int num_iters, int mc_samples) :
        tree(L, gamma),
        corpus(corpus), L(L), alpha(alpha), beta(beta), gamma(gamma),
        num_iters(num_iters), mc_samples(mc_samples), phi((size_t) L), log_phi((size_t) L),
        count((size_t) L), log_normalization(L, 1000), new_topic(true) {

    TDoc D = corpus.D;
    docs.resize((size_t) D);
    for (int d = 0; d < D; d++)
        docs[d].w = corpus.w[d];

    for (auto &doc: docs) {
        doc.z.resize(doc.w.size());
        doc.c.resize((size_t) L);
        doc.theta.resize((size_t) L);
        fill(doc.theta.begin(), doc.theta.end(), 1. / L);
    }

    alpha_bar = accumulate(alpha.begin(), alpha.end(), 0.0);

    for (auto &m: phi) m.SetR(corpus.V);
    for (auto &m: log_phi) m.SetR(corpus.V);
    for (auto &m: count) m.SetR(corpus.V);

    for (TLen l = 0; l < L; l++)
        for (int i = 0; i < 1000; i++)
            log_normalization(l, i) = log(beta[l] + i);
}

void BaseHLDA::Visualize(std::string fileName, int threshold) {
    string dotFileName = fileName + ".dot";

    ofstream fout(dotFileName.c_str());
    fout << "graph tree {";
    // Output nodes
    auto nodes = tree.GetAllNodes();
    for (auto *node: nodes)
        if (node->num_docs > threshold)
            fout << "Node" << node->id << " [label=\"" << node->id << '\n'
                 << node->num_docs << "\n"
                 << node->weight << "\n"
                 << TopWords(node->depth, node->pos) << "\"]\n";

    // Output edges
    for (auto *node: nodes)
        if (node->depth != 0)
            if (node->num_docs > threshold && node->parent->num_docs > threshold)
                fout << "Node" << node->parent->id << " -- Node" << node->id << "\n";

    fout << "}";
}

std::string BaseHLDA::TopWords(int l, int id) {
    TWord V = corpus.V;
    vector<pair<int, int>> rank((size_t) V);
    long long sum = 0;
    for (int v = 0; v < V; v++) {
        int c = count[l](v, id);
        rank[v] = make_pair(-c, v);
        sum += c;
    }
    sort(rank.begin(), rank.end());

    ostringstream out;
    out << sum << "\n";
    for (int v = 0; v < 5; v++)
        out << -rank[v].first << ' ' << corpus.vocab[rank[v].second] << "\n";

    return out.str();
}

void BaseHLDA::InitializeTreeWeight() {
    auto nodes = tree.GetAllNodes();
    nodes[0]->sum_log_weight = 0;

    for (auto *node: nodes)
        if (!node->children.empty()) {
            // Propagate
            double sum_weight = gamma[node->depth];

            for (auto *child: node->children)
                sum_weight += child->num_docs;

            for (auto *child: node->children)
                child->sum_log_weight = node->sum_log_weight +
                                        log((child->num_docs + 1e-10) / sum_weight);

            node->sum_log_weight += log(gamma[node->depth] / sum_weight);
        }
}

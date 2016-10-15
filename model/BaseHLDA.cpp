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
        tree(L, gamma.back()),
        corpus(corpus), L(L), alpha(alpha), beta(beta), gamma(gamma),
        num_iters(num_iters), mc_samples(mc_samples),
        phi(0, corpus.V), log_phi(0, corpus.V), count(0, corpus.V) {
    TDoc D = corpus.D;
    docs.resize((size_t) D);
    for (int d = 0; d < D; d++)
        docs[d].w = corpus.w[d];
    for (auto &doc: docs) {
        doc.z.resize(doc.w.size());
        doc.c.resize((size_t) L);
        doc.theta.resize((size_t) L);
        fill(doc.theta.begin(), doc.theta.end(), 1. / L);
        doc.initialized = true;
    }
    alpha_bar = accumulate(alpha.begin(), alpha.end(), 0.0);
}

void BaseHLDA::Initialize() {
    for (auto &doc: docs) {
        // Sample c
        tree.Sample(doc.c, generator);
        tree.UpdateNumDocs(doc.c.back(), 1);

        // Sample z
        for (auto &k: doc.z)
            k = generator() % L;
    }
    UpdateCount();
    printf("Initialized with %d topics.\n", tree.GetMaxID());
}

void BaseHLDA::UpdateCount(size_t end) {
    count.SetR(tree.GetMaxID());
    count.Clear();
    size_t e = end == (size_t) -1 ? docs.size() : end;
    for (size_t d = 0; d < e; d++) {
        auto &doc = docs[d];
        TLen N = (TLen) doc.w.size();

        for (TLen n = 0; n < N; n++)
            count(doc.c[doc.z[n]]->id, doc.w[n])++;
    }
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
                 << TopWords(node->id) << "\"]\n";

    // Output edges
    for (auto *node: nodes)
        if (node->depth != 0)
            if (node->num_docs > threshold && node->parent->num_docs > threshold)
                fout << "Node" << node->parent->id << " -- Node" << node->id << "\n";

    fout << "}";
}

std::string BaseHLDA::TopWords(int id) {
    TWord V = corpus.V;
    vector<pair<int, int>> rank((size_t) V);
    long long sum = 0;
    for (int v = 0; v < V; v++) {
        int c = count(id, v);
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

void BaseHLDA::DFSSample(Document &doc) {
    auto nodes = tree.GetAllNodes();
    decltype(nodes) leaves;
    leaves.reserve(nodes.size());
    vector<TProb> prob;
    prob.reserve(nodes.size());

    doc.PartitionWByZ(L);

    // Warning: this is not thread safe
    for (auto *node: nodes) {
        if (node->depth == 0)
            node->sum_log_prob = WordScore(doc, node->depth, node->id, node);
        else
            node->sum_log_prob = node->parent->sum_log_prob +
                                 WordScore(doc, node->depth, node->id, node);

        if (node->depth + 1 == L) {
            leaves.push_back(node);
            prob.push_back(node->sum_log_prob + node->sum_log_weight);
        }
    }

    // Sample
    Softmax(prob.begin(), prob.end());
    int leaf_index = DiscreteSample(prob.begin(), prob.end(), generator);

    tree.GetPath(leaves[leaf_index], doc.c);
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

//
// Created by jianfei on 8/29/16.
//

#include <iostream>
#include <sstream>
#include "BaseHLDA.h"
#include "corpus.h"

using namespace std;

BaseHLDA::BaseHLDA(Corpus &corpus, int L,
                   TProb alpha, TProb beta, TProb gamma,
                   int num_iters) :
        tree(L, gamma),
        corpus(corpus), L(L), alpha(alpha), beta(corpus.V, beta), gamma(gamma),
        num_iters(num_iters),
        phi(corpus.V), log_phi(corpus.V), count(corpus.V) {
    TDoc D = corpus.D;
    docs.resize((size_t) D);
    for (int d = 0; d < D; d++)
        docs[d].w = corpus.w[d];
    for (auto &doc: docs) {
        doc.z.resize(doc.w.size());
        doc.c.resize((size_t) L);
    }
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
}

void BaseHLDA::UpdateCount(size_t end) {
    count.Resize(tree.GetMaxID());
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

void BaseHLDA::SampleC(bool clear_doc_count, size_t d_start, size_t d_end) {
    auto nodes = tree.GetAllNodes();
    vector<Tree::Node *> leaves;
    for (auto *node: nodes)
        if (node->depth + 1 == L)
            leaves.push_back(node);

    InitializeTreeWeight();

    vector<TProb> leaf_probability;
    leaf_probability.reserve(nodes.size());

    // Sample path
    if (clear_doc_count)
        for (auto *node: nodes)
            node->num_docs = 0;

    if (d_start == (size_t) -1) d_start = 0;
    if (d_end == (size_t) -1) d_end = docs.size();
    for (size_t d = d_start; d < d_end; d++) {
        auto &doc = docs[d];
        doc.PartitionWByZ(L);
        leaf_probability.clear();
        for (auto *node: nodes) {
            if (node->depth == 0)
                node->sum_log_prob = 0;
            else
                node->sum_log_prob = node->parent->sum_log_prob +
                                     WordScore(doc, node->depth, node->id);

            if (node->depth + 1 == L)
                leaf_probability.push_back(node->sum_log_prob + node->sum_log_weight);
        }

        // Sample
        Softmax(leaf_probability.begin(), leaf_probability.end());
        int leaf_index = DiscreteSample(leaf_probability.begin(),
                                        leaf_probability.end(), generator);

        tree.GetPath(leaves[leaf_index], doc.c);

        // Update counts
        for (auto *node: doc.c)
            node->num_docs += 1;
    }
}

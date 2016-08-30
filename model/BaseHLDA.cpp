//
// Created by jianfei on 8/29/16.
//

#include <iostream>
#include <sstream>
#include "BaseHLDA.h"
#include "corpus.h"

using namespace std;

vector<TTopic> Document::GetIDs() {
    std::vector<TTopic> result(c.size());
    for (size_t l = 0; l < c.size(); l++)
        result[l] = c[l]->id;
    return std::move(result);
}

void Document::PartitionWByZ(int L) {
    offsets.resize((size_t) L + 1);
    fill(offsets.begin(), offsets.end(), 0);
    reordered_w.resize(w.size());

    TLen N = (TLen) z.size();

    // Counting sort
    // Count
    for (auto k: z) offsets[k + 1]++;
    for (int l = 1; l <= L; l++) offsets[l] += offsets[l - 1];

    // Scatter
    for (int n = 0; n < N; n++)
        reordered_w[offsets[z[n]]++] = w[n];

    // Correct offset
    for (int l = L; l > 0; l--) offsets[l] = offsets[l - 1];
    offsets[0] = 0;
}

BaseHLDA::BaseHLDA(Corpus &corpus, int L,
                   TProb alpha, TProb beta, TProb gamma,
                   int num_iters) :
        tree(L, gamma),
        corpus(corpus), L(L), alpha(alpha), beta(beta), gamma(gamma),
        num_iters(num_iters),
        phi(corpus.V), log_phi(corpus.V), count(corpus.V) {

}

void BaseHLDA::Initialize() {
    TDoc D = corpus.D;
    docs.resize((size_t) D);
    for (int d = 0; d < D; d++)
        docs[d].w = corpus.w[d];

    for (auto &doc: docs) {
        doc.z.resize(doc.w.size());
        doc.c.resize((size_t) L);

        // Sample c
        tree.Sample(doc.c, generator);
        tree.UpdateNumDocs(doc.c.back(), 1);

        // Sample z
        for (auto &k: doc.z)
            k = generator() % L;
    }
    UpdateCount();
}

void BaseHLDA::UpdateCount() {
    count.Resize(tree.GetMaxID());
    count.Clear();
    for (auto &doc: docs) {
        TLen N = (TLen) doc.w.size();

        for (TLen n = 0; n < N; n++)
            count(doc.c[doc.z[n]]->id, doc.w[n])++;
    }
}

void BaseHLDA::Visualize(std::string fileName) {
    int threshold = -1;
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
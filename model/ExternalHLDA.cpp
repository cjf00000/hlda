//
// Created by jianfei on 10/14/16.
//

#include <fstream>
#include <iostream>
#include <sstream>
#include "corpus.h"
#include "ExternalHLDA.h"

using namespace std;

ExternalHLDA::ExternalHLDA(Corpus &corpus, int L,
                           TProb alpha, vector<TProb> beta, std::vector<TProb> gamma,
                           string prefix)
        : CollapsedSampling(corpus, L, alpha, beta, gamma, -1, -1),
          prefix(prefix) {

}

void ExternalHLDA::Initialize() {
    ReadTree();
    ReadPath();
    ReadLevel();
}

int GetID(map<int, int> &node_id_map, int x) {
    if (node_id_map.find(x) != node_id_map.end())
        return node_id_map[x];

    int s = (int) node_id_map.size();
    return node_id_map[x] = s;
}

void ExternalHLDA::ReadTree() {
    // Build tree and read count
    ifstream fin((prefix + "/mode").c_str());

    // Skip 9 lines
    string line;
    for (int i = 0; i < 9; i++)
        getline(fin, line);

    size_t total_count = 0;
    nodes.push_back(tree.GetRoot());
    while (getline(fin, line)) {
        istringstream sin(line);
        int node_id, parent_id, ndocs;
        double dummy;
        int c;
        sin >> node_id >> parent_id >> ndocs >> dummy >> dummy;

        node_id = GetID(node_id_map, node_id);
        if (parent_id != -1) parent_id = GetID(node_id_map, parent_id);

        if (parent_id != -1) {
            // Add link
            nodes.push_back(tree.AddChildren(nodes[parent_id]));
        }
        nodes.back()->num_docs = ndocs;

        // Read Count matrix
        TTopic k = (TTopic) nodes.size();
        count.SetR(k);
        ck.push_back(0);
        for (TWord v = 0; v < corpus.V; v++) {
            sin >> c;
            count(k - 1, v) = c;
            ck[k - 1] += c;
            total_count += c;
        }
    }
    cout << "Read " << tree.GetMaxID() << " nodes, total count = " << total_count << endl;
}

void ExternalHLDA::ReadPath() {
    ifstream fin((prefix + "/mode.assign").c_str());
    int doc_id;
    int node_id;
    double dummy;

    for (TDoc d = 0; d < corpus.D; d++) {
        fin >> doc_id >> dummy;
        doc_id_map.push_back(doc_id);

        auto &doc = docs[doc_id];
        for (TLen l = 0; l < L; l++) {
            fin >> node_id;
            doc.c[l] = nodes[node_id_map[node_id]];
        }
    }
    cout << "Finished reading path" << endl;
}

void ExternalHLDA::ReadLevel() {
    ifstream fin((prefix + "/mode.levels").c_str());
    TWord v;
    TLen l;
    string line;
    for (TDoc d = 0; d < corpus.D; d++) {
        getline(fin, line);
        for (auto &c: line) if (c == ':') c = ' ';
        istringstream sin(line);

        auto &doc = docs[doc_id_map[d]];
        size_t old_size = doc.w.size();
        doc.w.clear();
        doc.z.clear();
        while (sin >> v >> l) {
            doc.w.push_back(v);
            doc.z.push_back(l);
        }
        if (doc.w.size() != old_size)
            throw runtime_error("Document length do not match " + to_string(d) + " " +
                                to_string(old_size) + " " + to_string(doc.w.size()));
    }
    cout << "Finished reading levels" << endl;
}

void ExternalHLDA::Estimate() {
    current_it = -1;
    double perplexity = Perplexity();
    printf("Perplexity = %.2f\n", perplexity);
}




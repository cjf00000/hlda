//
// Created by jianfei on 8/29/16.
//

#ifndef HLDA_TREE_H
#define HLDA_TREE_H

#include <vector>
#include "utils.h"

#define Path std::vector<Tree::Node*>

class Tree {
public:
    struct Node {
        Node *parent;
        std::vector<Node *> children;
        int id, depth;

        int num_docs;
        double weight;
        double sum_log_weight;
        double sum_log_prob;

        bool is_collapsed;
    };

    void Copy(const Tree &from);

    Tree(int L, double gamma);

    ~Tree();

    Node *AddChildren(Node *parent);

    void Remove(Node *node);

    void UpdateNumDocs(Node *leaf, int delta);

    template<class TGenerator>
    void Sample(Path &path, TGenerator &generator) {
        auto *current = root;
        path.resize(L);
        path[0] = current;
        for (int l = 1; l < L; l++) {
            int N = (int) current->children.size();
            std::vector<double> prob((size_t) N + 1);
            for (int n = 0; n < N; n++)
                prob[n] = current->children[n]->num_docs;
            prob[N] = gamma;

            int nch = DiscreteSample(prob.begin(), prob.end(), generator);
            Node *next = nullptr;
            if (nch == N)
                next = AddChildren(current);
            else
                next = current->children[nch];

            path[l] = next;
            current = next;
        }
    }

    void SetL(int L) { this->L = L; }

    int GetMaxID() { return max_id; }

    std::vector<Node *> GetAllNodes() const;

    void GetPath(Node *leaf, Path &path);

    int L;
    double gamma;

private:
    int GetFreeID();

    void AddFreeID(int id);

    void getAllNodes(Node *root, std::vector<Node *> &result) const;

    Node *root;
    std::vector<int> unallocated_ids;
    int max_id;
};

#endif //HLDA_TREE_H

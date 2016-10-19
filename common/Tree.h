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

    Tree(int L, double gamma);

    ~Tree();

    Node *AddChildren(Node *parent);

    void Remove(Node *node);

    void UpdateNumDocs(Node *leaf, int delta);

    int GetMaxID() { return max_id; }

    std::vector<Node *> GetAllNodes() const;

    void GetPath(Node *leaf, Path &path);

    Node *GetRoot() { return root; }

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

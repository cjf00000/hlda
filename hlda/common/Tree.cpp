//
// Created by jianfei on 8/29/16.
//

#include <map>
#include <iostream>
#include "Tree.h"

using namespace std;

using Node = Tree::Node;

Tree::Tree(int L, double gamma) : L(L), gamma(gamma), idpool(L), max_id(1) {
    root = new Node();
    root->id = 0;
    root->pos = idpool[0].Allocate();
    root->depth = 0;
    root->parent = nullptr;
    root->is_collapsed = true;
}

Tree::~Tree() {
    auto nodes = GetAllNodes();
    for (auto *node: nodes)
        delete node;
}

Node *Tree::AddChildren(Node *parent) {
    Node *node = new Node();
    node->parent = parent;
    node->depth = parent->depth + 1;
    node->id = max_id++;
    node->pos = idpool[node->depth].Allocate();
    node->is_collapsed = true;
    parent->children.push_back(node);
    return node;
}

void Tree::Remove(Node *node) {
    auto *parent = node->parent;
    auto child = std::find(parent->children.begin(), parent->children.end(), node);
    parent->children.erase(child);

    idpool[node->depth].Free(node->pos);
    delete node;
}

void Tree::UpdateNumDocs(Node *leaf, int delta) {
    while (leaf != nullptr) {
        leaf->num_docs += delta;
        auto *next_leaf = leaf->parent;
        if (leaf->num_docs == 0)
            Remove(leaf);

        leaf = next_leaf;
    }
}

std::vector<Node *> Tree::GetAllNodes() const {
    std::vector<Node *> result;
    getAllNodes(root, result);
    return std::move(result);
}

void Tree::getAllNodes(Node *root, std::vector<Node *> &result) const {
    result.push_back(root);
    for (auto *child: root->children)
        getAllNodes(child, result);
}

void Tree::GetPath(Node *leaf, Path &path) {
    path.resize((size_t) L);
    for (int l = L - 1; l >= 0; l--, leaf = leaf->parent) path[l] = leaf;
}

std::vector<int> Tree::Compress(int l) {
    auto nodes = GetAllNodes();
    std::vector<int> result((size_t) NumNodes(l), -1);

    // Sort according to 1. is_collapsed, 2. num_docs
    vector<pair<size_t, int>> rank;
    for (auto *node: nodes)
        if (node->depth == l)
            rank.push_back(make_pair(node->is_collapsed ? 0 : 1e9 + node->num_docs,
                                     node->pos));

    sort(rank.begin(), rank.end());
    reverse(rank.begin(), rank.end());

    // Map the numbers
    for (int i = 0; i < (int) rank.size(); i++)
        result[rank[i].second] = i;

    // Change pos
    for (auto *node: nodes)
        if (node->depth == l)
            node->pos = result[node->pos];

    // Reset idpool
    idpool[l].Clear();
    for (size_t i = 0; i < rank.size(); i++)
        idpool[l].Allocate();

    return result;
}
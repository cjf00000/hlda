//
// Created by jianfei on 8/29/16.
//

#include <map>
#include <iostream>
#include "Tree.h"

using namespace std;

using Node = Tree::Node;

void Tree::Copy(const Tree &from) {
    std::cout << "Copying Tree" << std::endl;
    L = from.L;
    gamma = from.gamma;

    root = new Node();
    root->id = 0;
    root->depth = 0;
    root->parent = nullptr;

    auto from_nodes = from.GetAllNodes();
    std::map<int, Node *> my_id_to_nodes;
    my_id_to_nodes[root->id] = root;

    for (size_t i = 1; i < from_nodes.size(); i++) {
        auto *from_node = from_nodes[i];
        auto *parent = my_id_to_nodes[from_node->parent->id];
        auto *new_node = AddChildren(parent);
        new_node->id = from_node->id;
        new_node->depth = from_node->depth;
        my_id_to_nodes[new_node->id] = new_node;
    }
    unallocated_ids = from.unallocated_ids;
    max_id = from.max_id;
}

Tree::Tree(int L, double gamma) : L(L), gamma(gamma), max_id(1) {
    root = new Node();
    root->id = 0;
    root->depth = 0;
    root->parent = nullptr;
}

Tree::~Tree() {
    auto nodes = GetAllNodes();
    for (auto *node: nodes)
        delete node;
}

Node *Tree::AddChildren(Node *parent) {
    Node *node = new Node();
    node->parent = parent;
    node->id = GetFreeID();
    node->depth = parent->depth + 1;
    parent->children.push_back(node);
    return node;
}

void Tree::Remove(Node *node) {
    auto *parent = node->parent;
    auto child = std::find(parent->children.begin(), parent->children.end(), node);
    parent->children.erase(child);

    AddFreeID(node->id);
    delete node;
}

int Tree::GetFreeID() {
    if (unallocated_ids.empty())
        return max_id++;
    else {
        int id = unallocated_ids.back();
        unallocated_ids.pop_back();
        return id;
    }
}

void Tree::AddFreeID(int id) {
    unallocated_ids.push_back(id);
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
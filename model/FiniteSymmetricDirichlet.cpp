//
// Created by jianfei on 8/29/16.
//

#include <iostream>
#include <corpus.h>
#include <cmath>
#include "FiniteSymmetricDirichlet.h"

using namespace std;

FiniteSymmetricDirichlet::FiniteSymmetricDirichlet(Corpus &corpus, int L,
                                                   TProb alpha, TProb beta, TProb gamma,
                                                   int branching_factor, int num_iters) :
        BaseHLDA(corpus, L, alpha, beta, gamma, num_iters), branching_factor(branching_factor) {

}

void FiniteSymmetricDirichlet::Initialize() {
    BaseHLDA::Initialize();

    // Create virtual nodes
    auto nodes = tree.GetAllNodes();
    for (auto *node: nodes) {
        AddVirtualTree(node);
    }
    count.Resize(tree.GetMaxID());

    for (auto &doc: docs)
        SampleTheta(doc);

    SamplePhi();
    SamplePi();
    cout << "Initialization finished. " << tree.GetMaxID() << " nodes." << endl;
}

void FiniteSymmetricDirichlet::Estimate() {
    for (int it = 0; it < num_iters; it++) {
        for (auto &doc: docs)
            SampleZ(doc);

        SampleC();

        UpdateCount();
        SamplePhi();

        SamplePi();
        
        printf("Iteration %d\n", it);
    }
}

void FiniteSymmetricDirichlet::AddVirtualTree(Tree::Node *node) {
    if (node->depth + 1 >= L)
        return;

    for (int b = 0; b < branching_factor; b++) {
        auto *subtree = tree.AddChildren(node);
        AddVirtualTree(subtree);
    }
}

void FiniteSymmetricDirichlet::SampleZ(Document &doc) {
    auto ids = doc.GetIDs();
    std::vector<TProb> prob((size_t) L);
    for (size_t n = 0; n < doc.z.size(); n++) {
        TWord v = doc.w[n];
        for (int l = 0; l < L; l++)
            prob[l] = doc.theta[l] * phi(ids[l], v);

        doc.z[n] = DiscreteSample(prob.begin(), prob.end(), generator);
    }

    SampleTheta(doc);
}

void FiniteSymmetricDirichlet::SampleTheta(Document &doc) {
    doc.theta.resize((size_t) L);
    fill(doc.theta.begin(), doc.theta.end(), 0);
    TLen N = (TLen) doc.z.size();
    for (auto k: doc.z)
        doc.theta[k]++;

    double normalizing_constant = 1. / (N + alpha * L);
    for (int l = 0; l < L; l++)
        doc.theta[l] = (doc.theta[l] + alpha) * normalizing_constant;
}

void FiniteSymmetricDirichlet::SamplePhi() {
    TTopic K = tree.GetMaxID();
    log_phi.Resize(tree.GetMaxID());
    phi.Resize(tree.GetMaxID());
    for (TTopic k = 0; k < K; k++) {
        double sum = beta * corpus.V;
        for (TWord v = 0; v < corpus.V; v++)
            sum += count(k, v);

        sum = 1. / sum;
        for (TWord v = 0; v < corpus.V; v++) {
            double prob = (count(k, v) + beta) * sum;
            phi(k, v) = prob;
            log_phi(k, v) = log(prob);
        }
    }
}

void FiniteSymmetricDirichlet::SamplePi() {
    auto nodes = tree.GetAllNodes();
    for (auto *node: nodes) {
        int num_children = (int) node->children.size();
        TProb weight_sum = gamma;
        for (auto *child: node->children)
            weight_sum += child->num_docs;

        TProb inv_weight_sum = 1. / weight_sum;
        TProb smoothing = gamma / num_children;
        for (auto *child: node->children)
            child->weight = (child->num_docs + smoothing) * inv_weight_sum;
    }
}

TProb FiniteSymmetricDirichlet::WordScore(Document &doc, int l, int topic) {
    auto *b = doc.BeginLevel(l);
    auto *e = doc.EndLevel(l);

    double theta_score = (e - b) * log(doc.theta[l]);
    double phi_score = 0;
    for (auto *w = b; w != e; w++)
        phi_score += log_phi(topic, *w);

    return theta_score + phi_score;
}

void FiniteSymmetricDirichlet::InitializeTreeWeight() {
    auto nodes = tree.GetAllNodes();
    nodes[0]->sum_log_weight = 0;
    for (size_t i = 1; i < nodes.size(); i++)
        nodes[i]->sum_log_weight = log(nodes[i]->weight) + nodes[i]->parent->weight;
}
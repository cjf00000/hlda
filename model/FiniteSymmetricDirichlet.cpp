//
// Created by jianfei on 8/29/16.
//

#include <iostream>
#include <corpus.h>
#include <cmath>
#include "Clock.h"
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
    count.SetR(tree.GetMaxID());

    for (auto &doc: docs)
        SampleTheta(doc);

    SamplePhi();
    SamplePi();

    //ProgressivelyOnlineInitialize();

    cout << "Initialization finished. " << tree.GetMaxID() << " nodes." << endl;
}

void FiniteSymmetricDirichlet::Estimate() {
    for (int it = 0; it < num_iters; it++) {
        Clock clk;
        SampleC(true);

        for (auto &doc: docs)
            SampleZ(doc);

        //SampleCCollapseZ(true);
        //for (auto &doc: docs)
        //    SampleZ(doc);

        UpdateCount();
        SamplePhi();

        SamplePi();

        double time = clk.toc();
        double throughput = corpus.T / time / 1048576;
        double perplexity = Perplexity();
        auto nodes = tree.GetAllNodes();
        printf("Iteration %d, %lu topics, %.2f seconds (%.2fMtoken/s), perplexity = %.2f\n",
               it, nodes.size(), time, throughput, perplexity);
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

void FiniteSymmetricDirichlet::SampleC(bool clear_doc_count, size_t d_start, size_t d_end) {
    auto nodes = tree.GetAllNodes();

    InitializeTreeWeight();

    // Sample path
    if (clear_doc_count)
        for (auto *node: nodes)
            node->num_docs = 0;

    if (d_start == (size_t) -1) d_start = 0;
    if (d_end == (size_t) -1) d_end = docs.size();

    for (size_t d = d_start; d < d_end; d++) {
        auto &doc = docs[d];

        DFSSample(doc);

        // Update counts
        for (auto *node: doc.c)
            node->num_docs += 1;
    }
}

void FiniteSymmetricDirichlet::SampleCCollapseZ(bool clear_doc_count, size_t d_start, size_t d_end) {
    auto nodes = tree.GetAllNodes();
    decltype(nodes) leaves;
    for (auto *node: nodes) if (node->depth+1==L) leaves.push_back(node);
    vector<double> log_prob(leaves.size());

    InitializeTreeWeight();

    // Sample path
    if (clear_doc_count)
        for (auto *node: nodes)
            node->num_docs = 0;

    if (d_start == (size_t) -1) d_start = 0;
    if (d_end == (size_t) -1) d_end = docs.size();

    for (size_t d = d_start; d < d_end; d++) {
        auto &doc = docs[d];
        log_prob.clear();

        for (auto *leaf: leaves) {
            tree.GetPath(leaf, doc.c);
            auto ids = doc.GetIDs();

            double log_likelihood = 0;
            for (int n=0; n<(int)doc.z.size(); n++) {
                double p = 0;
                for (int l=0; l<L; l++)
                    p += doc.theta[l] * phi(ids[l], doc.w[n]);
                log_likelihood += log(p);
            }
            log_prob.push_back(log_likelihood + leaf->sum_log_weight);
        }

        Softmax(log_prob.begin(), log_prob.end());
        int leaf_index = DiscreteSample(log_prob.begin(), log_prob.end(), generator);

        tree.GetPath(leaves[leaf_index], doc.c);

        // Update counts
        for (auto *node: doc.c)
            node->num_docs += 1;
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
    log_phi.SetR(tree.GetMaxID());
    phi.SetR(tree.GetMaxID());
    for (TTopic k = 0; k < K; k++) {
        double sum = beta.Concentration();
        for (TWord v = 0; v < corpus.V; v++)
            sum += count(k, v);

        sum = 1. / sum;
        for (TWord v = 0; v < corpus.V; v++) {
            double prob = (count(k, v) + beta(v)) * sum;
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

TProb FiniteSymmetricDirichlet::WordScore(Document &doc, int l, int topic, Tree::Node *node) {
    auto *b = doc.BeginLevel(l);
    auto *e = doc.EndLevel(l);

    double phi_score = 0;
    for (auto *w = b; w != e; w++)
        phi_score += log_phi(topic, *w);

    return phi_score;
}

void FiniteSymmetricDirichlet::InitializeTreeWeight() {
    auto nodes = tree.GetAllNodes();
    nodes[0]->sum_log_weight = 0;
    for (size_t i = 1; i < nodes.size(); i++)
        nodes[i]->sum_log_weight = log(nodes[i]->weight) + nodes[i]->parent->weight;
}

double FiniteSymmetricDirichlet::Perplexity() {
    double log_likelihood = 0;
    for (auto &doc: docs) {
        auto ids = doc.GetIDs();

        for (auto w: doc.w) {
            double prob = 0;
            for (int l = 0; l < L; l++)
                prob += doc.theta[l] * phi(ids[l], w);

            log_likelihood += log(prob);
        }
    }
    return exp(-log_likelihood / corpus.T);
}

void FiniteSymmetricDirichlet::ProgressivelyOnlineInitialize() {
    // Don't use symmetric beta
    //InitializeBeta();

    UpdateCount(0);
    SamplePhi();

    // Perform progressively online initialization
    int batch_size = 200;

    for (size_t d_start = 0; d_start < docs.size(); d_start += batch_size) {
        size_t d_end = std::min(docs.size(), d_start + batch_size);
        printf("Doc %lu - %lu\n", d_start, d_end);

        for (size_t d = d_start; d < d_end; d++)
            for (auto &k: docs[d].z)
                k = generator() % L;

        for (int it = 0; it < 2; it++) {
            SampleC(false, d_start, d_end);

            for (size_t d = d_start; d < d_end; d++)
                SampleZ(docs[d]);
        }

        UpdateCount(d_end);
        SamplePhi();
        SamplePi();
    }
}

void FiniteSymmetricDirichlet::InitializeBeta() {
    std::vector<double> tf((size_t) corpus.V, 10);
    for (auto &doc: docs)
        for (auto w: doc.w)
            tf[w]++;

    beta.Set(tf, beta.Concentration());
}

void FiniteSymmetricDirichlet::LayerwiseInitialize(FiniteSymmetricDirichlet &model) {
    // Copy tree
    tree.Copy(model.tree);
    tree.SetL(L);

    // Add virtual nodes
    auto nodes = tree.GetAllNodes();
    for (auto *node: nodes)
        if (node->depth + 2 == L)
            AddVirtualTree(node);

    // Estimate count
    count.SetR(tree.GetMaxID());
    count.Clear();
    for (TTopic k = 0; k < model.tree.GetMaxID(); k++)
        for (TWord v = 0; v < corpus.V; v++)
            count(k, v) = model.count(k, v);

    SamplePhi();
    SamplePi();

    // Perform progressively online initialization
    int batch_size = 500;

    for (size_t d_start = 0; d_start < docs.size(); d_start += batch_size) {
        size_t d_end = std::min(docs.size(), d_start + batch_size);
        printf("Doc %lu - %lu\n", d_start, d_end);

        for (size_t d = d_start; d < d_end; d++) {
            for (auto &k: docs[d].z)
                k = generator() % L;
            SampleTheta(docs[d]);
        }

        for (int it = 0; it < 2; it++) {
            SampleC(false, d_start, d_end);

            for (size_t d = d_start; d < d_end; d++)
                SampleZ(docs[d]);
        }

        UpdateCount(d_end);
        SamplePhi();
        SamplePi();
    }
}
//
// Created by jianfei on 9/2/16.
//

#include <iostream>
#include "DirectSamplingHDP.h"
#include "corpus.h"
#include "utils.h"

using namespace std;

DirectSamplingHDP::DirectSamplingHDP(Corpus &corpus, int n_iter,
                                     double gamma, double alpha, double eta)
        : corpus(corpus), n_iter(n_iter), gamma(gamma),
          alpha(alpha), eta(eta), cwk(corpus.V, 0), log_stirling(0, 0) {
    size_t max_len = 0;
    docs.resize(corpus.w.size());

    for (size_t d = 0; d < corpus.w.size(); d++) {
        docs[d].w = corpus.w[d];
        docs[d].z.resize(docs[d].w.size());
        max_len = max(corpus.w[d].size(), max_len);
    }

    log_stirling(0, 0) = 0;
    for (int i = 1; i <= max_len; i++) log_stirling(i, 0) = -1e9;
    for (int n = 1; n <= max_len; n++) {
        log_stirling(n, n) = 0;

        for (int k = 1; k < n; k++)
            log_stirling(n, k) = LogSum(log(n - 1) + log_stirling(n - 1, k),
                                        log_stirling(n - 1, k - 1));
    }
    std::cout << "Finished initialization" << endl;
}

void DirectSamplingHDP::Initialize() {

}

void DirectSamplingHDP::Estimate() {

}

void DirectSamplingHDP::SampleBeta(int n_end) {
    // TODO Just approximate m with n...
    std::vector<double> prob;
    std::vector<double> sample;
    std::vector<int> topics;

    for (TTopic k = 0; k < pool.Size(); k++)
        if (pool.Has(k)) {
            prob.push_back(ck[k]);
            topics.push_back(k);
        }
    prob.push_back(gamma);

    sample.resize(prob.size());
    dirichlet_distribution<double> dir(prob);
    sample = dir(generator);

    fill(beta.begin(), beta.end(), 0);
    for (size_t i = 0; i < topics.size(); i++)
        beta[topics[i]] = sample[i];

    beta_u = sample.back();
}

void DirectSamplingHDP::SampleDoc(Document &doc) {
    TLen N = (TLen) doc.w.size();
    TTopic K = pool.Size();
    std::vector<TCount> cdk((size_t) K);
    std::vector<double> prob((size_t) K + 1);
    for (auto k: doc.z) cdk[k]++;

    double eta_bar = eta * corpus.V;
    beta_distribution<double> rbeta(1, gamma);

    for (TLen n = 0; n < N; n++) {
        TWord v = doc.w[n];
        TTopic k = doc.z[n];
        --cdk[k];
        --ck[k];
        --cwk(v, k);

        if (ck[k] == 0) {
            // Topic death
            beta_u += beta[k];
            beta[k] = 0;
            pool.Free(k);
        }

        // Compute distribution
        for (TTopic i = 0; i < K; i++)
            prob[i] = (cdk[i] + alpha * beta[i]) * (cwk(v, i) + eta) / (ck[i] + eta_bar);

        prob[K] = alpha * beta_u / corpus.V;
        TTopic sample = DiscreteSample(prob.begin(), prob.begin() + K + 1, generator);

        if (sample == K) {
            // Topic birth
            k = pool.Allocate();
            K = pool.Size();
            cwk.SetC(K);
            if (ck.size() < K) ck.push_back(0);
            if (beta.size() < K) beta.push_back(0);
            if (prob.size() < K) prob.push_back(0);
            if (cdk.size() < K) cdk.push_back(0);

            // Stick breaking
            double stick_weight = rbeta(generator);
            beta[k] = beta_u * stick_weight;
            beta_u = beta_u * (1 - stick_weight);
        } else {
            k = sample;
        }

        doc.z[n] = k;
        ++cdk[k];
        ++ck[k];
        ++cwk(v, k);
    }
}

double DirectSamplingHDP::Perplexity() {

}
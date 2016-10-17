//
// Created by jianfei on 8/30/16.
//

#ifndef HLDA_COLLAPSEDSAMPLING_H
#define HLDA_COLLAPSEDSAMPLING_H

#include "BaseHLDA.h"

class CollapsedSampling : public BaseHLDA {
public:
    CollapsedSampling(Corpus &corpus, int L,
                      std::vector<TProb> alpha, std::vector<TProb> beta, std::vector<TProb> gamma,
                      int num_iters,
                      int mc_samples, int mc_iters,
                      int remove_iters, int remove_paths);

    void Initialize() override;

    void ProgressivelyOnlineInitialize();

    virtual void Estimate();

protected:
    virtual void SampleZ(Document &doc, bool decrease_count, bool increase_count);

    virtual void SampleC(Document &doc, bool decrease_count, bool increase_count);

    virtual void ResetZ(Document &doc);

    virtual void Recount();

    void DFSSample(Document &doc) override;

    void RemovePath();

    double Perplexity();

    void Check();

    void UpdateDocCount(Document &doc, int delta);

    virtual TProb WordScore(Document &doc, int l, int topic, Tree::Node *node);

    std::vector<TCount> ck;

    int current_it, mc_iters, remove_iters, remove_paths;

    std::vector<double> doc_avg_likelihood;
    std::vector<std::vector<int>> old_doc_ids;
    std::vector<std::vector<int>> old_doc_sizes;
};


#endif //HLDA_COLLAPSEDSAMPLING_H

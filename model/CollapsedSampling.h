//
// Created by jianfei on 8/30/16.
//

#ifndef HLDA_COLLAPSEDSAMPLING_H
#define HLDA_COLLAPSEDSAMPLING_H

#include "BaseHLDA.h"

class CollapsedSampling : public BaseHLDA {
public:
    CollapsedSampling(Corpus &corpus, int L,
                      TProb alpha, std::vector<TProb> beta, std::vector<TProb> gamma,
                      int num_iters, int mc_samples);

    void Initialize() override;

    void ProgressivelyOnlineInitialize();

    virtual void Estimate();

protected:
    virtual void SampleZ(Document &doc, bool decrease_count, bool increase_count);

    virtual void SampleC(Document &doc, bool decrease_count, bool increase_count);

    virtual void ResetZ(Document &doc);

    virtual void Recount();

    void DFSSample(Document &doc) override;

    double Perplexity();

    void Check();

    void UpdateDocCount(Document &doc, int delta);

    virtual TProb WordScore(Document &doc, int l, int topic, Tree::Node *node);

    std::vector<TCount> ck;

    int current_it;

    std::vector<double> doc_avg_likelihood;
    std::vector<std::vector<int>> old_doc_ids;
    std::vector<std::vector<int>> old_doc_sizes;
};


#endif //HLDA_COLLAPSEDSAMPLING_H

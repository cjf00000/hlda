#include <iostream>
#include "corpus.h"
#include "FiniteSymmetricDirichlet.h"
#include "CollapsedSampling.h"

using namespace std;

int main() {
    // Read corpus

    // NIPS
    /*Corpus corpus("data/nips.vocab", "data/nips.libsvm.train");
    cout << "Corpus read. " << corpus.T << " tokens " << endl;
    int L = 5;
    double alpha = 0.5;
    double beta = 0.2;                                                                                                                                                                                   ;
    //double gamma = 4.0;
    //double gamma = 1e-100;
    std::vector<double> gamma{1e-280, 1e-150, 1e-100, 1e-50};
    int branching_factor = 0;
    int num_iters = 30;*/

    // NYT
    Corpus corpus("data/nysmaller.vocab", "data/nysmaller.libsvm.train");
    cout << "Corpus read. " << corpus.T << " tokens " << endl;
    int L = 4;
    double alpha = 0.5;
    double beta = 0.2;
    //double gamma = 4.0;
    //double gamma = 0.001;
    std::vector<double> gamma{1e-90, 1e-60, 1e-30};
    int branching_factor = 0;
    int num_iters = 30;

    // Problem 1: there are too much topics during initialization
    // Problem 2: it cannot produce new topics...

    // FSD
    //auto *model1 = new FiniteSymmetricDirichlet(corpus, L, alpha, beta, gamma, branching_factor, num_iters);
    auto *model1 = new CollapsedSampling(corpus, L, alpha, beta, gamma, num_iters);
    //auto *model1 = new PartiallyCollapsedSampling(corpus, L, alpha, beta, gamma, num_iters);
    model1->Initialize();
    model1->Estimate();

    model1->Visualize("tree1", 10);


    /*auto *last_model = model1;

    for (int l = 2; l <= L; l++) {
        auto *model = new FiniteSymmetricDirichlet(corpus, l,
                                                   alpha, beta, gamma,
                                                   l == 2 ? 6 : branching_factor, num_iters);
        model->LayerwiseInitialize(*last_model);
        model->Estimate();
        model->Visualize("tree" + to_string(l), 30);

        last_model = model;
    }*/

    return 0;
}
#include <iostream>
#include "corpus.h"
#include "FiniteSymmetricDirichlet.h"
#include "CollapsedSampling.h"

using namespace std;

int main() {
    // Read corpus
    Corpus corpus("data/nips.vocab", "data/nips.libsvm.train");
    cout << "Corpus read. " << corpus.T << " tokens " << endl;

    // Initialize model
    int L = 4;
    double alpha = 0.5;
    double beta = 0.5;
    double gamma = 3.0;
    //double gamma = 0.001;
    int branching_factor = 0;
    int num_iters = 30;

    // Problem 1: there are too much topics during initialization
    // Problem 2: it cannot produce new topics...

    // FSD
    auto *model1 = new FiniteSymmetricDirichlet(corpus, L,
                                   alpha, beta, gamma,
                                   branching_factor, num_iters);

    //auto model1 = new CollapsedSampling(corpus, L, alpha, beta, gamma, num_iters);
    model1->Initialize();
    model1->Estimate();

    model1->Visualize("tree1", 5);


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
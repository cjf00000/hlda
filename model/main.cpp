#include <iostream>
#include "corpus.h"
#include "FiniteSymmetricDirichlet.h"

using namespace std;

int main() {
    // Read corpus
    Corpus corpus("data/nips.vocab", "data/nips.libsvm.train");
    cout << "Corpus read. " << corpus.T << " tokens " << endl;

    // Initialize model
    int L = 4;
    double alpha = 0.5;
    double beta = 0.01;
    double gamma = 1.2;
    //double gamma = 0.01;
    int branching_factor = 3;
    int num_iters = 30;

    auto *model1 = new FiniteSymmetricDirichlet(corpus, 1,
                                   alpha, beta, gamma,
                                   branching_factor, num_iters);
    model1->Initialize();
    model1->Estimate();

    model1->Visualize("tree1", 10);

    auto *last_model = model1;

    for (int l = 2; l <= L; l++) {
        auto *model = new FiniteSymmetricDirichlet(corpus, l,
                                                   alpha, beta, gamma,
                                                   l == 2 ? 6 : branching_factor, num_iters);
        model->LayerwiseInitialize(*last_model);
        model->Estimate();
        model->Visualize("tree" + to_string(l), 10);

        last_model = model;
    }

    return 0;
}
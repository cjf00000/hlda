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
    int branching_factor = 0;
    int num_iters = 30;
    FiniteSymmetricDirichlet model(corpus, L,
                                   alpha, beta, gamma,
                                   branching_factor, num_iters);
    model.Initialize();

    model.Estimate();

    model.Visualize("tree", 10);

    return 0;
}
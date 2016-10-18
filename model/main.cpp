#include <iostream>
#include <sstream>
#include <exception>
#include <gflags/gflags.h>
#include "corpus.h"
#include "FiniteSymmetricDirichlet.h"
#include "PartiallyCollapsedSampling.h"
#include "ExternalHLDA.h"

using namespace std;

DEFINE_string(prefix, "data/nysmaller", "prefix of the corpus");
DEFINE_string(algo, "pcs", "Algorithm, cs, pcs, or es");
DEFINE_int32(L, 4, "number of levels");
DEFINE_string(alpha, "0.5,0.5,0.5,0.5", "Prior on level assignment, delimited by comma");
DEFINE_string(beta, "1,0.4,0.3,0.2", "Prior on topics, delimited by comma");
DEFINE_string(gamma, "1e-40,1e-30,1e-20", "Parameter of nCRP, delimited by comma");
DEFINE_int32(n_iters, 70, "Number of iterations");
DEFINE_int32(n_mc_samples, 5, "Number of Monte-Carlo samples, -1 for none.");
DEFINE_int32(n_mc_iters, 30, "Number of Monte-Carl iterations, -1 for none.");
DEFINE_int32(n_remove_trees, -1, "Number of subtrees to remove for each iteration, -1 for none.");
DEFINE_int32(n_remove_iters, -1, "Number of iterations to remove subtrees, -1 for none.");
DEFINE_int32(minibatch_size, 1000, "Minibatch size for initialization (for pcs)");
DEFINE_int32(topic_limit, 100, "Upper bound of number of topics to terminate.");
DEFINE_string(model_path, "out/run014", "Path of model for es");
DEFINE_string(vis_prefix, "vis_result/tree", "Path of visualization");

vector<double> Parse(string src, int L, string name) {
    for (auto &ch: src) if (ch==',') ch = ' ';
    istringstream sin(src);

    vector<double> result;
    double p;
    while (sin >> p) result.push_back(p);
    if (result.size() == 1)
        result = vector<double>(L, p);
    if (result.size() != (size_t)L)
        throw runtime_error("The length of " + name + 
                " is incorrect, must be 1 or " + to_string(L));
    return result;
}

template <class T>
void Run(T& model)
{
    model.Initialize();
    model.Estimate();
    model.Visualize(FLAGS_vis_prefix, -1);
}

int main(int argc, char **argv) {
    gflags::SetUsageMessage("Usage : ./warplda [ flags... ]");
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    // Parse alpha, beta and gamma
    auto alpha = Parse(FLAGS_alpha, FLAGS_L, "alpha");
    auto beta = Parse(FLAGS_beta, FLAGS_L, "beta");
    auto gamma = Parse(FLAGS_gamma, FLAGS_L-1, "gamma");

    if (FLAGS_algo != "pcs" && FLAGS_algo != "cs" && FLAGS_algo != "es")
        throw runtime_error("Invalid algorithm");

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
    */

    // NYT
    // Read corpus
    Corpus corpus((FLAGS_prefix + ".vocab").c_str(), 
            (FLAGS_prefix + ".libsvm.train").c_str());
    cout << "Corpus read. " << corpus.T << " tokens " << endl;

    if (FLAGS_algo == "cs") {
        CollapsedSampling model(corpus, 
                FLAGS_L, alpha, beta, gamma, 
                FLAGS_n_iters, FLAGS_n_mc_samples, FLAGS_n_mc_iters, 
                FLAGS_n_remove_iters, FLAGS_n_remove_trees, FLAGS_topic_limit);
        Run(model);
    }
    else if (FLAGS_algo == "pcs") {
        PartiallyCollapsedSampling model(corpus, 
                FLAGS_L, alpha, beta, gamma, 
                FLAGS_n_iters, FLAGS_n_mc_samples, FLAGS_n_mc_iters, FLAGS_minibatch_size,
                FLAGS_n_remove_iters, FLAGS_n_remove_trees, FLAGS_topic_limit);
        Run(model);
    }
    else {
        ExternalHLDA model(corpus, 
                FLAGS_L, alpha, beta, gamma, FLAGS_model_path);
        Run(model);
    }

    return 0;
}

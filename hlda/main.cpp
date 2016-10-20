#include <iostream>
#include <sstream>
#include <gflags/gflags.h>
#include "corpus.h"
#include "CollapsedSampling.h"
#include "mkl_vml.h"
#include "PartiallyCollapsedSampling.h"
//#include "ExternalHLDA.h"

using namespace std;

DEFINE_string(prefix, "data/nysmaller", "prefix of the corpus");
DEFINE_string(algo, "pcs", "Algorithm, cs, pcs, or es");
DEFINE_int32(L, 4, "number of levels");
DEFINE_string(alpha, "0.5,0.5,0.5,0.5", "Prior on level assignment, delimited by comma");
DEFINE_string(beta, "1,0.4,0.3,0.2", "Prior on topics, delimited by comma");
DEFINE_string(gamma, "1e-40,1e-30,1e-20", "Parameter of nCRP, delimited by comma");
DEFINE_int32(n_iters, 30, "Number of iterations");
DEFINE_int32(n_mc_samples, 5, "Number of Monte-Carlo samples, -1 for none.");
DEFINE_int32(n_mc_iters, 20, "Number of Monte-Carl iterations, -1 for none.");
DEFINE_int32(minibatch_size, 1000, "Minibatch size for initialization (for pcs)");
DEFINE_int32(topic_limit, 100, "Upper bound of number of topics to terminate.");
DEFINE_string(model_path, "out/run014", "Path of model for es");
DEFINE_string(vis_prefix, "vis_result/tree", "Path of visualization");
DEFINE_int32(threshold, 50, "Threshold for a topic to be instantiated.");

vector<double> Parse(string src, int L, string name) {
    for (auto &ch: src) if (ch==',') ch = ' ';
    istringstream sin(src);

    vector<double> result;
    double p;
    while (sin >> p) result.push_back(p);
    if (result.size() == 1)
        result = vector<double>((size_t) L, p);
    if (result.size() != (size_t)L)
        throw runtime_error("The length of " + name + 
                " is incorrect, must be 1 or " + to_string(L));
    return result;
}

int main(int argc, char **argv) {
    gflags::SetUsageMessage("Usage : ./hlda [ flags... ]");
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    vmlSetMode(VML_EP); // VML_HA VML_LA

    // Parse alpha, beta and gamma
    auto alpha = Parse(FLAGS_alpha, FLAGS_L, "alpha");
    auto beta = Parse(FLAGS_beta, FLAGS_L, "beta");
    auto gamma = Parse(FLAGS_gamma, FLAGS_L-1, "gamma");

    if (FLAGS_algo != "pcs" && FLAGS_algo != "cs" && FLAGS_algo != "es")
        throw runtime_error("Invalid algorithm");

    // NIPS
    /*Corpus corpus("data/nips.vocab", "data/nips.libsvm.train");
    int L = 5;
    double alpha = 0.5;
    double beta = 0.2;
    std::vector<double> gamma{1e-280, 1e-150, 1e-100, 1e-50};
    */

    // NYT
    // Read corpus
    Corpus corpus((FLAGS_prefix + ".vocab").c_str(), 
            (FLAGS_prefix + ".libsvm.train").c_str());
    cout << "Corpus read. " << corpus.T << " tokens " << endl;

    BaseHLDA *model = nullptr;
    if (FLAGS_algo == "cs") {
        model = new CollapsedSampling(corpus,
                                      FLAGS_L, alpha, beta, gamma,
                                      FLAGS_n_iters, FLAGS_n_mc_samples, FLAGS_n_mc_iters,
                                      FLAGS_topic_limit);
    }
    else if (FLAGS_algo == "pcs") {
        model = new PartiallyCollapsedSampling(corpus,
                                               FLAGS_L, alpha, beta, gamma,
                                               FLAGS_n_iters, FLAGS_n_mc_samples, FLAGS_n_mc_iters,
                                               (size_t) FLAGS_minibatch_size,
                                               FLAGS_topic_limit, FLAGS_threshold);
    }
    else {
        /*model = new ExternalHLDA(corpus,
                                 FLAGS_L, alpha, beta, gamma, FLAGS_model_path);*/
    }

    model->Initialize();
    model->Estimate();
    model->Visualize(FLAGS_vis_prefix, -1);

    return 0;
}

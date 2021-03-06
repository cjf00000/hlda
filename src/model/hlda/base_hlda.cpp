//
// Created by jianfei on 8/29/16.
//

#include <iostream>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <random>
#include <omp.h>
#include <thread>
#include <map>
#include "base_hlda.h"
#include "clock.h"
#include "hlda_corpus.h"
#include "utils.h"
#include "mkl_vml.h"
#include "global_lock.h"
#include "statistics.h"

using namespace std;

int calc_font_size(int max_font_size, int min_font_size, int max_size, int min_size, int my_size) {
    double rate = (log(my_size) - log(min_size)) / (log(max_size) - log(min_size));
    double size = (max_font_size - min_font_size) * rate + min_font_size;
    return int(size);
};

BaseHLDA::BaseHLDA(HLDACorpus &corpus, HLDACorpus &to_corpus, HLDACorpus &th_corpus, int L,
                   std::vector<TProb> alpha, std::vector<TProb> beta, vector<double> log_gamma,
                   int num_iters, int mc_samples, int mc_iters, size_t minibatch_size,
                   int topic_limit, bool sample_phi,
                   int process_id, int process_size, bool check, bool random_start) :
        process_id(process_id), process_size(process_size),
        tree(L, log_gamma),
        corpus(corpus), to_corpus(to_corpus), th_corpus(th_corpus),
        L(L), alpha(alpha), beta(beta), log_gamma(log_gamma),
        num_iters(num_iters), mc_samples(mc_samples), 
        current_it(-1), mc_iters(mc_iters), minibatch_size(minibatch_size),
        topic_limit(topic_limit), sample_phi(sample_phi),
        phi((size_t) L), log_phi((size_t) L),
        count(L, corpus.V, omp_get_max_threads()),
        icount(1, process_size, corpus.V, 1/*K*/, row_partition,
               process_size, process_id),
        new_topic(true), check(check), allow_new_topic(true) {

    MPI_Comm_dup(MPI_COMM_WORLD, &comm);

    generators.resize(omp_get_max_threads());
    if (random_start) {
        std::random_device rd;
        for (auto &gen: generators)
            gen.seed(rd(), rd());
    } else {
        std::mt19937_64 rd;
        for (auto &gen: generators)
            gen.seed(rd(), rd());
    }

    TDoc D = corpus.D;
    docs.resize((size_t) D);
    for (int d = 0; d < D; d++)
        docs[d].w = corpus.w[d];

    for (auto &doc: docs) {
        doc.z.resize(doc.w.size());
        doc.c.resize((size_t) L);
        doc.theta.resize((size_t) L);
        fill(doc.theta.begin(), doc.theta.end(), 1. / L);
        doc.initialized = false;
    }
    LOG_IF(FATAL, to_corpus.D != th_corpus.D) 
        << "The size of to and th corpus are different";
    //LOG(INFO) << "Doc";

    to_docs.resize(to_corpus.D);
    th_docs.resize(th_corpus.D);
    for (int d = 0; d < to_corpus.D; d++) {
        to_docs[d].w = to_corpus.w[d];
        to_docs[d].z.resize(to_docs[d].w.size());
        to_docs[d].theta.resize(L);

        th_docs[d].w = th_corpus.w[d];
        th_docs[d].z.resize(th_docs[d].w.size());
        th_docs[d].theta.resize(L);
    }
    //shuffle(docs.begin(), docs.end(), generator);
    //LOG(INFO) << "TODoc";

    alpha_bar = accumulate(alpha.begin(), alpha.end(), 0.0);

    for (auto &m: phi) {
        m.SetR(corpus.V, true);
        m.SetC(1, true);
    }
    for (auto &m: log_phi) {
        m.SetR(corpus.V, true);
        m.SetC(1, true);
    }
    //LOG(INFO) << "M";

    MPI_Comm_rank(comm, &process_id);
    MPI_Comm_size(comm, &process_size);
    //LOG(INFO) << "Comm";

    for (int l = 0; l < L; l++) 
        topic_mutexes.emplace_back(new std::mutex[MAX_NUM_TOPICS]);

    log_work = decltype(log_work)(omp_get_max_threads(), std::vector<TProb>(VECTOR_LENGTH));
    //LOG(INFO) << "Done";
}

void BaseHLDA::SetOutfile(std::string outfile) {
    this->outfile = outfile;
}

void BaseHLDA::Initialize() {
    //LOG(INFO) << "Initialize";
    //CollapsedSampling::Initialize();
    current_it = -1;

    if (minibatch_size == 0)
        minibatch_size = docs.size();

    int num_threads = omp_get_max_threads();
    LOG(INFO) << "OpenMP: using " << num_threads << " threads";
    auto &generator = GetGenerator();
    int mb_count = 0;
    omp_set_dynamic(0);
    Clock clk;

    // Compute the minibatch size for this node
    size_t local_mb_size = std::min(static_cast<size_t>(minibatch_size),
                                    docs.size() / num_threads);

    size_t local_num_mbs, num_mbs;
    local_num_mbs = (docs.size() - 1) / local_mb_size + 1;
    MPI_Allreduce(&local_num_mbs, &num_mbs, 1, MPI_UNSIGNED_LONG_LONG,
                  MPI_MAX, comm);
    LOG_IF(INFO, process_id == 0) << "Each node has " << num_mbs << " minibatches.";

    int processed_node = 0, next_processed_node;
    for (int degree_of_parallelism = 1; processed_node < process_size;
         degree_of_parallelism++, processed_node = next_processed_node) {
        next_processed_node = std::min(process_size,
                                       processed_node + degree_of_parallelism);
        LOG_IF(INFO, process_id == 0)
        << "Initializing node " << processed_node
        << " to node " << next_processed_node;

        size_t minibatch_size = docs.size() / num_mbs + 1;
        if (processed_node <= process_id && process_id < next_processed_node) {
            for (size_t d_start = 0; d_start < docs.size();
                 d_start += minibatch_size) {
                auto d_end = min(docs.size(), d_start + minibatch_size);
                if (degree_of_parallelism == 1)
                    omp_set_num_threads(min(++mb_count, num_threads));

                for (int it = 0; it < init_iiter; it++) {
                    num_instantiated = tree.GetNumInstantiated();
#pragma omp parallel for
                    for (size_t d = d_start; d < d_end; d++) {
                        auto &doc = docs[d];

                        if (it == 0)
                            for (auto &k: doc.z)
                                k = generator() % L;
    
                        doc.initialized = true;
                        SampleC(doc, (bool)it, true, allow_new_topic);
                        SampleZ(doc, true, true, allow_new_topic);
                    }
                    AllBarrier();
                    omp_set_num_threads(num_threads);
                    SamplePhi();
                    AllBarrier();
                    //Check();
                }

                auto ret = tree.GetTree();
                LOG(INFO) << "Node: " << process_id
                          << " Processed document [" << d_start << ", " << d_end
                          << ") documents, " << ret.nodes.size()
                          << " topics.";
                if ((int)ret.nodes.size() > (size_t) topic_limit) {
                    allow_new_topic = false;
//                    throw runtime_error("There are too many topics");
                }
            }
        } else {
            for (size_t i = 0; i < num_mbs; i++) {
                for (int it = 0; it < init_iiter; it++) {
                    AllBarrier();
                    SamplePhi();
                    AllBarrier();
                    //Check();
                }
            }
        }
        MPI_Barrier(comm);
        auto ret = tree.GetTree();
        LOG_IF(INFO, process_id==0) << ANSI_YELLOW << "Num nodes: " << ret.num_nodes
                                    << "    Num instantiated: " << num_instantiated << ANSI_NOCOLOR;
        OutputSizes();
    }
    omp_set_num_threads(num_threads);

    SamplePhi();
    LOG_IF(INFO, process_id == 0) << "Initialized in " << clk.toc() << " seconds";
}

void BaseHLDA::Estimate() {
    double total_time = 0;
    for (int it = 0; it < num_iters; it++) {
        //shuffle(docs.begin(), docs.end(), GetGenerator());
        current_it = it;
        Clock clk;

        if (current_it >= mc_iters)
            mc_samples = -1;

        Clock clk2;
        Statistics<double> c_time, z_time;
        lockdoc_time.Reset();
        s1_time.Reset();
        s2_time.Reset();
        s3_time.Reset();
        s4_time.Reset();
        wsc_time.Reset();
        t1_time.Reset();
        t2_time.Reset();
        t3_time.Reset();
        t4_time.Reset();
        num_c.Reset();
        num_i.Reset();
        #pragma omp parallel for schedule(dynamic, 10)
        for (int d = 0; d < corpus.D; d++) {
            Clock clk;
            auto &doc = docs[d];
            SampleC(doc, true, true, allow_new_topic);
            c_time.Add(clk.toc()); clk.tic();
            SampleZ(doc, true, true, allow_new_topic);
            z_time.Add(clk.toc()); clk.tic();
        }
        int num_syncs = count.GetNumSyncs();
        auto bytes_communicated = count.GetBytesCommunicated();
        auto sample_time = clk2.toc();

        clk2.tic();
        AllBarrier();
        auto barrier_time = clk2.toc();

        clk2.tic();
        SamplePhi();
        auto phi_time = clk2.toc();

        clk2.tic();
        AllBarrier();
        barrier_time += clk2.toc();

        auto ret = tree.GetTree();
        int num_big_nodes = 0;
        int num_docs_big = 0;
        for (auto &node: ret.nodes)
            if (node.num_docs > 50) {
                num_big_nodes++;
                if (node.depth + 1 == L)
                    num_docs_big += node.num_docs;
            }

        size_t num_topics = 0;
        if (process_id == 0) {
            // Calculate num nodes and num instantiated
            std::vector<int> num_nodes(L), num_i(L);
            for (auto &node: ret.nodes)
                if (node.num_docs) {
                    num_nodes[node.depth]++;
                    if (node.pos < num_instantiated[node.depth]) 
                        num_i[node.depth]++;
                    num_topics++;
                }

            LOG(INFO) << ANSI_YELLOW << "Num nodes: " << num_nodes
                                 << "    Num instantiated: " << num_i << ANSI_NOCOLOR;
            auto ret = tree.GetTree();
            LOG(INFO) << ANSI_YELLOW << "Num nodes: " << ret.num_nodes
                      << "    Num instantiated: " << num_instantiated << ANSI_NOCOLOR;
            if (num_topics > topic_limit)
                allow_new_topic = false;
        }
        double time = clk.toc();
        total_time += time;

        double throughput = corpus.T / time / 1048576;
        clk2.tic();
        double perplexity = Perplexity();
        auto perplexity_time = clk2.toc();
        LOG_IF(INFO, process_id == 0) 
            << std::fixed << std::setprecision(2)
            << ANSI_GREEN << "Iteration " << it 
            << ", " << num_topics << " topics (" 
            << num_big_nodes << ", " << num_docs_big << "), "
            << time << " seconds (" << throughput << " Mtoken/s), perplexity = "
            << perplexity 
            << " (" << num_syncs/sample_time << " syncs/s, " << bytes_communicated/1048576 << " MB communicated.)"
            << ANSI_NOCOLOR;

        double check_time = 0;
        if (check) {
            clk2.tic();
            Check();
            check_time = clk2.toc();
            
            tree.Check();
        }
        LOG_IF(INFO, process_id == 0) << "Time usage: "
                  << std::fixed << std::setprecision(2)
                  << " sample:" << sample_time
                  << " phi:" << phi_time
                  << " barrier: " << barrier_time
                  << " perplexity:" << perplexity_time 
                  << " check:" << check_time 
                  << " c:" << c_time.Sum()
                  << " z:" << z_time.Sum()
                  << " l:" << lockdoc_time.Sum()
                  << " 1:" << s1_time.Sum()
                  << " 2:" << s2_time.Sum()
                  << " 3:" << s3_time.Sum()
                  << " 4:" << s4_time.Sum()
                  << " wsc:" << wsc_time.Mean()
                  << " cphi:" << compute_phi_time
                  << " cnt:" << count_time
                  << " sync:" << sync_time
                  << " set:" << set_time;
        OutputSizes();

        LOG_IF(INFO, process_id == 0)
            << t1_time.Sum() << ' '
            << t2_time.Sum() << ' '
            << t3_time.Sum() << ' '
            << t4_time.Sum() << ' '
            << num_c.Sum() << ' '
            << num_i.Sum() << ' ';
    }
    OutputTheta();
    LOG_IF(INFO, process_id == 0) << "Finished in " << total_time << " seconds.";
}

void BaseHLDA::Visualize(std::string fileName, int threshold) {
    string dotFileName = fileName + ".dot";
    int max_font_size = 30, min_font_size = 6;

    auto ret = tree.GetTree();
    int min_node_size = 1;
    int max_node_size = 0;
    for (auto &node: ret.nodes)
        max_node_size = std::max(max_node_size, node.num_docs);

    vector<int> counts(L);
    for (int l = 0; l < L; l++)
        for (int k = 0; k < ret.num_nodes[l]; k++)
            counts[l] += ck_dense[k + icount_offset[l]];
    LOG(INFO) << "Layer weight: " << counts;

    ofstream fout(dotFileName.c_str());
    fout << "graph tree {\nnode[shape=rectangle]\n";
    // Output nodes
    for (size_t i = 0; i < ret.nodes.size(); i++) {
        auto &node = ret.nodes[i];
        if (node.num_docs > threshold) {
            auto font_size = calc_font_size(max_font_size, min_font_size, 
                    max_node_size, min_node_size, node.num_docs);
            fout << "Node" << i << " [fontsize=" 
                 << font_size
                 << ",label=<<FONT POINT-SIZE=\"6\">"
                 << i << "  " << node.pos << "  " 
                 << node.num_docs << "<BR/></FONT>\n"
                 << TopWords(node.depth, node.pos, font_size, min_font_size) << ">]\n";
        }
    }

    // Output edges
    for (size_t i = 0; i < ret.nodes.size(); i++) {
        auto &node = ret.nodes[i];
        if (node.depth != 0)
            if (node.num_docs > threshold &&
                ret.nodes[node.parent_id].num_docs > threshold)
                fout << "Node" << node.parent_id
                     << " -- Node" << i << "\n";
    }

    fout << "}";

    std::string metaJsonFileName = fileName + ".meta.json";
    ofstream fmetaJson(metaJsonFileName.c_str());
    fmetaJson << "{\"vocab\": [";
    for (int v = 0; v < corpus.V; v++) {
        auto wd = corpus.vocab[v];
        if (wd[0] == 'z' && wd[1] == 'z' && wd[2] == 'z') {
            std::string new_str;
            for (size_t i = 4; i < wd.size(); ) {
                size_t next_ = i;
                while (next_ < wd.size() && wd[next_] != '_') next_++;
                new_str += toupper(wd[i]);
                for (int j = i+1; j < next_; j++) new_str += wd[j];
                if (next_ < wd.size()) new_str += ' ';
                i = next_ + 1;
            }
            wd = new_str;
        }
        fmetaJson << "\"" << wd << "\"";
        if (v + 1 < corpus.V)
            fmetaJson << ", ";
    }
    fmetaJson << "],\n\"nodes\": [";
    for (size_t i = 0; i < ret.nodes.size(); i++) {
        auto &node = ret.nodes[i];
        fmetaJson << "{\"id\": " << i 
                  << ", \"parent\": " << node.parent_id
                  << ", \"frequency\": " << node.num_docs << "}";
        if (i + 1 < ret.nodes.size())
            fmetaJson << ",\n";
    }
    fmetaJson << "]}";

    std::string cntFileName = fileName + ".count";
    ofstream fcount(cntFileName.c_str());
    //std::string distFileName = fileName + ".dist";
    //ofstream fdist(distFileName.c_str());
    //fdist.precision(5);
    //fdist << std::scientific;
    for (size_t k = 0; k < ret.nodes.size(); k++) {
        auto &node = ret.nodes[k];
        int kk = icount_offset[node.depth] + node.pos;
        size_t sum = 0;
        for (int v = 0; v < corpus.V; v++) {
            fcount << icount(v, kk) << '\t';
            sum += icount(v, kk);
        }
        double b = beta[node.depth];
        double inv_normalization = 1. / (sum + b * corpus.V);
        //for (int v = 0; v < corpus.V; v++)
        //    fdist << (icount(v, kk) + b) * inv_normalization << '\t';
        fcount << '\n';
        //fdist << '\n';
    }
}

std::string BaseHLDA::TopWords(int l, int id, int max_font_size, int min_font_size) {
    TWord V = corpus.V;
    vector<pair<int, int>> rank((size_t) V);
    long long sum = 0;
    int max_cnt = 0;
    for (int v = 0; v < V; v++) {
        auto c = icount(v, id+icount_offset[l]);
        rank[v] = make_pair(-c, v);
        sum += c;
        max_cnt = std::max(max_cnt, int(c));
    }
    sort(rank.begin(), rank.end());

    ostringstream out;
    int min_cnt = std::min(int(sum / V) + 1, -rank[5].first + 1);
    //out << sum << "\n";
    out << "<FONT POINT-SIZE=\"6\">";
    for (int v = 0; v < 5; v++)
        if (-rank[v].first > min_cnt)
            out << -rank[v].first << ' ';
    out << "<BR/></FONT>";
    for (int v = 0; v < 5; v++) {
        auto wd = corpus.vocab[rank[v].second];
        if (wd[0] == 'z' && wd[1] == 'z' && wd[2] == 'z') {
            std::string new_str;
            for (size_t i = 4; i < wd.size(); ) {
                size_t next_ = i;
                while (next_ < wd.size() && wd[next_] != '_') next_++;
                new_str += toupper(wd[i]);
                for (int j = i+1; j < next_; j++) new_str += wd[j];
                if (next_ < wd.size()) new_str += ' ';
                i = next_ + 1;
            }
            wd = new_str;
        }
        if (-rank[v].first > min_cnt) {
            auto font_size = calc_font_size(max_font_size, min_font_size, max_cnt, min_cnt, -rank[v].first);
            out << "<FONT POINT-SIZE=\"" << font_size << "\">"
                << wd << "</FONT><BR/>";
        }
    }

    return out.str();
}

void BaseHLDA::PermuteC(std::vector<std::vector<int>> &perm) {
    std::vector<std::vector<int>> inv_perm(L);
    for (int l=0; l<L; l++) {
        inv_perm[l].resize((size_t)*std::max_element(perm[l].begin(), perm[l].end())+1);
        for (size_t i=0; i<perm[l].size(); i++)
            inv_perm[l][perm[l][i]] = (int)i;
    }
    for (auto &doc: docs)
        for (int l = 0; l < L; l++)
            doc.c[l] = inv_perm[l][doc.c[l]];
}

void BaseHLDA::LockDoc(Document &doc) {
    Clock clk;
//    for (int l = 0; l < L; l++)
//        if (doc.c[l] >= num_instantiated[l])
//            topic_mutexes[l][doc.c[l]].lock();
    lockdoc_time.Add(clk.toc());
}

void BaseHLDA::UnlockDoc(Document &doc) {
//    for (int l = 0; l < L; l++)
//        if (doc.c[l] >= num_instantiated[l])
//            topic_mutexes[l][doc.c[l]].unlock();
}

xorshift& BaseHLDA::GetGenerator() {
    return generators[omp_get_thread_num()];
}

void BaseHLDA::AllBarrier() {
    std::thread count_thread([&](){
//        Clock clk;
        count.Compress();
//        LOG_IF(INFO, process_id==0) << "Compress" << clk.toc();
    });
    std::thread tree_thread([&](){
//        Clock clk;
        tree.Barrier();
//        LOG_IF(INFO, process_id==0) << "Tree Barrier" << clk.toc();
    });
    count_thread.join();
    tree_thread.join();
}

void BaseHLDA::UpdateICount() {
    // Compute icount_offset
    Clock clk;
    icount_offset.resize(static_cast<int>(L+1));
    icount_offset[0] = 0;
    auto ret = tree.GetTree();
    for (int l = 0; l < L; l++)
        icount_offset[l+1] = icount_offset[l] + ret.num_nodes[l];

    icount.resize(corpus.V, icount_offset.back());

    // Count
    std::atomic<size_t> total_count(0);
#pragma omp parallel for
    for (size_t d = 0; d < docs.size(); d++) 
        if (docs[d].initialized) {
            auto &doc = docs[d];
            auto tid = omp_get_thread_num();
            for (size_t n = 0; n < doc.w.size(); n++) {
                TLen l = doc.z[n];
                TTopic k = (TTopic)doc.c[l];
                TWord v = doc.w[n];
                icount.increase(v, k + icount_offset[l]);
            }
            total_count += doc.w.size();
        }
    count_time = clk.toc(); clk.tic();
    //LOG(INFO) << "Total count = " << total_count;

    // Sync
    icount.sync();
    sync_time = clk.toc(); clk.tic();

    ck_dense = icount.rowMarginal();

    for (int l = 0; l < L; l++) {
#pragma omp parallel for
        for (int r = 0; r < corpus.V; r++)
            for (int c = num_instantiated[l]; c < ret.num_nodes[l]; c++)
                count.Set(l, r, c, icount(r, c+icount_offset[l]));
        for (int c = num_instantiated[l]; c < ret.num_nodes[l]; c++)
            count.SetSum(l, c, ck_dense[c+icount_offset[l]]);
    }
    set_time = clk.toc(); clk.tic();
}

void BaseHLDA::SampleZ(Document &doc,
                       bool decrease_count, bool increase_count,
                       bool allow_new_topic) {
    //std::lock_guard<std::mutex> lock(model_mutex);
    std::vector<TCount> cdl((size_t) L);
    std::vector<TProb> prob((size_t) L);
    for (auto k: doc.z) cdl[k]++;

    auto &pos = doc.c;
    std::vector<bool> is_collapsed((size_t) L);
    for (int l = 0; l < L; l++) {
        is_collapsed[l] = !allow_new_topic ? false :
                             doc.c[l] >= num_instantiated[l];
    }

    // TODO: the first few topics will have a huge impact...
    // Read out all the required data
    auto tid = omp_get_thread_num();
    LockDoc(doc);

    auto &generator = GetGenerator();
    for (size_t n = 0; n < doc.z.size(); n++) {
        TWord v = doc.w[n];
        TTopic l = doc.z[n];
        if (decrease_count) {
            if (pos[l] >= num_instantiated[l])
                count.Dec(tid, l, v, pos[l]);
            --cdl[l];
        }

        for (TLen i = 0; i < L; i++)
            if (is_collapsed[i])
                prob[i] = (cdl[i] + alpha[i]) *
                          (count.Get(i, v, pos[i]) + beta[i]) /
                          (count.GetSum(i, pos[i]) + beta[i] * corpus.V);
            else {
                prob[i] = (alpha[i] + cdl[i]) * phi[i](v, pos[i]);
            }

        l = (TTopic) DiscreteSample(prob.begin(), prob.end(), generator);
        doc.z[n] = l;

        if (increase_count) {
            if (pos[l] >= num_instantiated[l])
                count.Inc(tid, l, v, pos[l]);
            ++cdl[l];
        }
    }
    UnlockDoc(doc);
    /*double sum = 0;
    for (TLen l = 0; l < L; l++)
        sum += (doc.theta[l] = cdl[l] + alpha[l]);
    for (TLen l = 0; l < L; l++)
        doc.theta[l] /= sum;*/
    count.Publish(tid);
}

void BaseHLDA::SampleC(Document &doc, bool decrease_count,
                                bool increase_count,
                                bool allow_new_topic) {
    Clock clk;
    // Sample
    int S = max(mc_samples, 1);
    std::vector<decltype(doc.z)> zs(S);
    vector<vector<vector<TProb>>> all_scores((size_t) S); // S * L * nodes per layer
    auto z_bak = doc.z;

    auto &generator = GetGenerator();
    // Stage 1: In the first mc_iters iterations, resample z and compute score for instantiated docs
    // Otherwise just compute score for instantiated docs
    for (int s = 0; s < S; s++) {
        // Resample Z
        Clock clk3;
        if (mc_samples != -1) {
            for (auto &l: doc.z)
                l = (TTopic)(((unsigned long long) generator() * L) >> 32);
        }
        zs[s] = doc.z;

        doc.PartitionWByZ(L, false);
        t3_time.Add(clk3.toc());

        auto &scores = all_scores[s]; scores.resize(L);
        for (TLen l = 0; l < L; l++) {
            TTopic num_i = (TTopic) num_instantiated[l];
            scores[l].resize(num_i);
#pragma forceinline
            WordScoreInstantiated(doc, l, num_i, scores[l].data());
        }
    }
    s1_time.Add(clk.toc()); clk.tic();

    //std::lock_guard<std::mutex> lock(model_mutex);
    if (decrease_count) {
        doc.z = z_bak;
        UpdateDocCount(doc, -1);
        tree.DecNumDocs(doc.leaf_id);
    }
    auto ret = tree.GetTree();
    auto &nodes = ret.nodes;
    vector<TProb> prob(nodes.size() * S, -1e9f);
    std::vector<TProb> sum_log_prob(nodes.size());
    s2_time.Add(clk.toc()); clk.tic();

    // Stage 2: compute score for collapsed topics
    for (int s = 0; s < S; s++) {
        Clock clk4;
        doc.z = zs[s];
        doc.PartitionWByZ(L);
        t4_time.Add(clk4.toc());

        auto &scores = all_scores[s];
        for (TLen l = 0; l < L; l++) {
            TTopic num_i = (TTopic)num_instantiated[l];
            TTopic num_collapsed = (TTopic)(ret.num_nodes[l] - num_i);
            LOG_IF(FATAL, (int)num_collapsed < 0)
                << "Num collapsed < 0";

            scores[l].resize(num_i + num_collapsed + 1);
            if (allow_new_topic) {
#pragma forceinline
                scores[l].back() = WordScoreCollapsed(doc, l,
                                                      num_i, num_collapsed,
                                                      scores[l].data()+num_i);
            } else {
#pragma forceinline
                WordScoreInstantiated(doc, l, num_i + num_collapsed, 
                        scores[l].data());
                scores[l].back() = -1e20f;
            }
        }
#ifdef BAD_LAYOUT
        // Brute-Force computing the score for each path
        for (size_t i = 0; i < nodes.size(); i++) {
            if (nodes[i].depth + 1 == L) {
                std::vector<int> c(L);
                int current_node = i;
                for (int l = L-1; l >=0; l--) {
                    c[l] = nodes[current_node].pos;
                    current_node = nodes[current_node].parent_id;
                }
                auto result = WordScoreCollapsedPath(doc, c);
                LOG_IF(FATAL, result>1e100) << "Just to spend time...";
            }
        }
#endif
    }

    s3_time.Add(clk.toc()); clk.tic();
    // Stage 3
    for (int s = 0; s < S; s++) {
        auto &scores = all_scores[s];

        vector<TProb> emptyProbability((size_t) L, 0);
        for (int l = L - 2; l >= 0; l--)
            emptyProbability[l] = emptyProbability[l + 1] + scores[l + 1].back();

        // Propagate the score
        for (size_t i = 0; i < nodes.size(); i++) {
            auto &node = nodes[i];

            if (node.depth == 0)
                sum_log_prob[i] = scores[node.depth][node.pos];
            else
                sum_log_prob[i] = scores[node.depth][node.pos]
                                  + sum_log_prob[node.parent_id];

            if (node.depth + 1 == L) {
                prob[i*S+s] = (TProb)(sum_log_prob[i] + node.log_path_weight);
            } else {
                if (new_topic)
                    prob[i * S + s] = (TProb)(sum_log_prob[i] +
                                              node.log_path_weight + emptyProbability[node.depth]);
            }
        }
    }
    // Sample
    Softmax(prob.begin(), prob.end());
    int node_number = DiscreteSample(prob.begin(), prob.end(), generator) / S;
    if (node_number < 0 || node_number >= (int) nodes.size())
        throw runtime_error("Invalid node number");

    auto leaf_id = node_number;

    // Increase num_docs
    if (increase_count) {
        auto ret = tree.IncNumDocs(leaf_id);
        doc.leaf_id = ret.id;
        doc.c = ret.pos;
        UpdateDocCount(doc, 1);
    } if (!allow_new_topic) {
        doc.c = tree.GetPath(leaf_id).pos;
    }
    s4_time.Add(clk.toc());
}

TProb BaseHLDA::WordScoreCollapsed(Document &doc, int l, int offset, int num, TProb *result) {
    Clock clk;
    num_c.Add(num);
    auto b = beta[l];
    auto b_bar = b * corpus.V;

    memset(result, 0, num*sizeof(TProb));
    TProb empty_result = 0;
    auto &work = log_work[omp_get_thread_num()];

    auto begin = doc.BeginLevel(l);
    auto end = doc.EndLevel(l);

    const auto &local_count = count.GetMatrix(l);

    // Make sure that we do not access outside the boundary
    int actual_num = std::min(num, static_cast<int>(local_count.GetC()) - offset);
    for (int k = actual_num; k < num; k++) 
        result[k] = -1e20f;

    for (auto i = begin; i < end; i++) {
        auto c_offset = doc.c_offsets[i];
        auto v = doc.reordered_w[i];
        float my_empty_result = logf(c_offset + b);
        for (TTopic k = 0; k < actual_num; k++) {
            auto cnt = local_count.Get(v, offset + k);
            if (cnt == 0)
                result[k] += my_empty_result;
            else
                result[k] += logf(cnt + c_offset + b);
        }
        empty_result += my_empty_result;
    }

    auto w_count = end - begin;
    for (TTopic k = 0; k < actual_num; k++)
        result[k] -= lgamma(local_count.GetSum(offset+k) + b_bar + w_count) -
                lgamma(local_count.GetSum(offset+k) + b_bar);

    empty_result -= lgamma(b_bar + w_count) - lgamma(b_bar);
    wsc_time.Add(actual_num);
    t2_time.Add(clk.toc());
    return empty_result;
}

TProb BaseHLDA::WordScoreCollapsedPath(Document &doc, std::vector<int> c) {
    // Compute the score for the path c
    // TODO: this is still optimized because of the sort
    float result = 0;
    for (int l = 0; l < L; l++) {
        auto begin = doc.BeginLevel(l);
        auto end = doc.EndLevel(l);
        const auto &local_count = count.GetMatrix(l);
        auto b = beta[l];
        auto b_bar = b * corpus.V;
        // Contains unknown topic
        if (c[l] >= local_count.GetC())
            return -1e20f;

        for (auto i = begin; i < end; i++) {
            auto c_offset = doc.c_offsets[i];
            auto v        = doc.reordered_w[i];
            auto cnt      = max((int)local_count.Get(v, c[l]), 0);
            if (cnt == 0)
                result += logf(c_offset + b);
            else
                result += logf(cnt + c_offset + b);
        }

        auto w_count = end - begin;
        result -= lgamma(local_count.GetSum(c[l]) + b_bar + w_count) -
                  lgamma(local_count.GetSum(c[l]) + b_bar);
    }
    return result;
}


TProb BaseHLDA::WordScoreInstantiated(Document &doc, int l, int num, TProb *result) {
    num_i.Add(num);
    Clock clk;
    memset(result, 0, num*sizeof(TProb));

    auto begin = doc.BeginLevel(l);
    auto end = doc.EndLevel(l);

    auto &local_log_phi = log_phi[l];

    for (auto i = begin; i < end; i++) {
        auto v = doc.reordered_w[i];

        auto *p = &local_log_phi(v, 0);
        for (TTopic k = 0; k < num; k++)
            result[k] += p[k];
    }
    
    TProb empty_result = logf(1./corpus.V) * (end - begin);
    t1_time.Add(clk.toc());
    return empty_result;
}

double BaseHLDA::Perplexity() {
    double log_likelihood = 0;

    size_t T = 0;

#pragma omp parallel for
    for (int d = 0; d < corpus.D; d++) {
        std::vector<double> theta((size_t) L);
        auto &doc = docs[d];
        double doc_log_likelihood = 0;

        // Compute theta
        for (auto k: doc.z) theta[k]++;
        double inv_sum = 1. / (doc.z.size() + alpha_bar);
        for (TLen l = 0; l < L; l++)
            theta[l] = (theta[l] + alpha[l]) * inv_sum;

        for (size_t n = 0; n < doc.z.size(); n++) {
            double prob = 0;
            TWord v = doc.w[n];
            for (int l = 0; l < L; l++) {
                double phi = (icount(v, doc.c[l]+icount_offset[l]) + beta[l]) /
                             (ck_dense[doc.c[l]+icount_offset[l]] + beta[l] * corpus.V);

                prob += theta[l] * phi;
            }
            doc_log_likelihood += log(prob);
        }
#pragma omp critical
        {
            T += doc.z.size();
            log_likelihood += doc_log_likelihood;
        }
    }

    double global_log_likelihood;
    size_t global_T;
    MPI_Allreduce(&log_likelihood, &global_log_likelihood, 1, MPI_DOUBLE, MPI_SUM, comm);
    MPI_Allreduce(&T, &global_T, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, comm);
    return exp(-global_log_likelihood / global_T);
}

void BaseHLDA::Check() {
    int sum = 0;
    for (TLen l = 0; l < L; l++) {
        const auto &local_count = count.GetMatrix(l);
        for (TTopic k = 0; k < local_count.GetC(); k++)
            for (TWord v = 0; v < corpus.V; v++) {
                if (local_count.Get(v, k) < 0) // TODO
                    throw runtime_error("Error!");
                sum += local_count.Get(v, k);
            }
    }
    int local_size = 0;
    for (auto &doc: docs) if (doc.initialized) local_size += doc.w.size();
    int global_size;
    MPI_Allreduce(&local_size, &global_size, 1, MPI_INT,
            MPI_SUM, comm);
    //if (sum != global_size)
    //    throw runtime_error("Total token error! expected " +
    //                        to_string(corpus.T) + ", got " + to_string(sum));

    // Check the tree
    std::vector<int> num_docs(10000), total_num_docs(10000);
    auto ret = tree.GetTree();
    auto &nodes = ret.nodes;
    for (auto &doc: docs) if (doc.initialized) {
        for (int l = 0; l < L; l++) {
            auto pos = doc.c[l];
            // Find node by pos
            auto it = find_if(nodes.begin(), nodes.end(), 
                    [&](const ConcurrentTree::RetNode& node) {
                        return node.depth == l && node.pos == pos; });
            LOG_IF(FATAL, it == nodes.end()) << "Check error: pos not found";

            num_docs[it - nodes.begin()]++;
        }
    }
    MPI_Allreduce(num_docs.data(), total_num_docs.data(), 10000,
            MPI_INT, MPI_SUM, comm);
    for (int id = 0; id < ret.nodes.size(); id++)
        LOG_IF(FATAL, total_num_docs[id] != nodes[id].num_docs) 
            << "Num docs error at " << id 
            << " expected " << total_num_docs[id] 
            << " got " << nodes[id].num_docs
            << " tree \n" << ret;

    // Check the count matrix
    std::vector<Matrix<int>> count2(L);
    std::vector<std::vector<int>> ck2(L);
    std::vector<Matrix<int>> global_count2(L);
    std::vector<std::vector<int>> global_ck2(L);
    for (int l=0; l<L; l++) {
        const auto &local_count = count.GetMatrix(l);
        count2[l].SetR(corpus.V);
        count2[l].SetC(local_count.GetC());
        ck2[l].resize(local_count.GetC());
        global_count2[l].SetR(corpus.V);
        global_count2[l].SetC(local_count.GetC());
        global_ck2[l].resize(local_count.GetC());
    }
    for (auto &doc: docs) if (doc.initialized) {
        for (size_t n = 0; n < doc.z.size(); n++) {
            auto z = doc.z[n];
            auto v = doc.w[n];
            auto c = doc.c[z];
            if (c >= count2[z].GetC())
                throw std::runtime_error("Range error");
            if (v >= count2[z].GetR())
                throw std::runtime_error("R error " + std::to_string(v));
            count2[z](v, c)++;
            ck2[z][c]++;
        }
    }
    // Reduce count2 and ck2
    for (int l=0; l<L; l++) {
        MPI_Allreduce(count2[l].Data(), global_count2[l].Data(), 
                      count2[l].GetR() * count2[l].GetC(), 
                      MPI_INT, MPI_SUM, comm);
        MPI_Allreduce(ck2[l].data(), global_ck2[l].data(),
                      ck2[l].size(), 
                      MPI_INT, MPI_SUM, comm);
    }

    size_t sum_2 = std::accumulate(ck_dense, ck_dense+icount_offset.back(), 0);
    if (sum_2 != global_size)
        throw runtime_error("Total token error! expected " +
                            to_string(corpus.T) + ", got " + to_string(sum_2));

    bool if_error = false;
    for (int l=0; l<L; l++) {
        const auto &local_count = count.GetMatrix(l);
        for (int r = 0; r < corpus.V; r++)
            for (int c = num_instantiated[l]; c < ret.num_nodes[l]; c++)
                if (local_count.Get(r, c) != global_count2[l](r, c)) {
                    LOG(WARNING) << "Count error at " 
                              << l << "," << r << "," << c
                              << " expected " << global_count2[l](r, c) 
                              << " get " << local_count.Get(r, c);
                    if_error = true;
                }

        for (int r = 0; r < corpus.V; r++)
            for (int c = 0; c < ret.num_nodes[l]; c++) 
                if (icount(r, c+icount_offset[l]) != global_count2[l](r, c)) {
                    LOG(FATAL) << "ICount error at " 
                              << l << "," << r << "," << c
                              << " expected " << global_count2[l](r, c) 
                              << " get " << icount(r, c+icount_offset[l]);
                    if_error = true;
                }

        for (int c = num_instantiated[l]; c < ret.num_nodes[l]; c++)
            if (local_count.GetSum(c) != global_ck2[l][c]) {
                LOG(WARNING) << "Ck error at " 
                          << l << "," << c
                          << " expected " << global_ck2[l][c]
                          << " get " << local_count.GetSum(c);
                if_error = true;
            }
    }


    MPI_Barrier(comm);
    if (if_error)
        throw std::runtime_error("Check error");
}

void BaseHLDA::UpdateDocCount(Document &doc, int delta) {
    // Update number of topics
    auto tid = omp_get_thread_num();
    for (TLen l = 0; l < L; l++)
        count.Grow(tid, l, doc.c[l] + 1);

    LockDoc(doc);
    TLen N = (TLen) doc.z.size();
    if (delta == 1)
        for (TLen n = 0; n < N; n++) {
            TLen l = doc.z[n];
            TTopic k = (TTopic)doc.c[l];
            TWord v = doc.w[n];
            if (k >= num_instantiated[l]) {
                count.Inc(tid, l, v, k);
            }
        }
    else if (delta == -1)
        for (TLen n = 0; n < N; n++) {
            TLen l = doc.z[n];
            TTopic k = (TTopic)doc.c[l];
            TWord v = doc.w[n];
            if (k >= num_instantiated[l]) {
                count.Dec(tid, l, v, k);
            }
        }
    else
        throw std::runtime_error("Invalid delta");
    UnlockDoc(doc);

    count.Publish(tid);
}

double BaseHLDA::PredictivePerplexity() {
    int num_test_c_samples = 20;
    int num_test_z_burnin = 20;
    int num_test_z_samples = 20;

    double local_log_likelihood = 0;
    size_t local_T = 0;

    auto ret = tree.GetTree();
    auto K   = ret.num_nodes.size();

#pragma omp parallel for
    for (int d = 0; d < to_corpus.D; d++) {
        auto &generator = GetGenerator();
        auto &to_doc = to_docs[d];
        auto &th_doc = th_docs[d];

        double doc_log_likelihood = -1e20;
        std::vector<double> theta(L);
        for (int ncs = 0; ncs < num_test_c_samples; ncs++) {
            double sample_log_likelihood = 0;
            // Reset z
            for (auto &l: to_doc.z) l = generator() % L;
            fill(theta.begin(), theta.end(), 0);

            // Sample c and z
            for (int nz = 0; nz < num_test_z_burnin+num_test_z_samples; nz++) {
                SampleC(to_doc, false, false, false);
                //LOG(INFO) << "SampleC " << to_doc.c;
                SampleZ(to_doc, false, false, false);
                //LOG(INFO) << "SampleZ";
                if (nz >= num_test_z_burnin) {
                    for (auto l: to_doc.z)
                        theta[l]++;
                }
            }

            // Compute likelihood
            double inv_normalization = 1. / (to_doc.w.size() + alpha_bar) 
                / num_test_z_samples;
            for (int l = 0; l < L; l++)
                theta[l] = (theta[l] + alpha[l]*num_test_z_samples) 
                    * inv_normalization;

            for (int n = 0; n < th_doc.w.size(); n++) {
                auto v = th_doc.w[n];
                double ll = 0;
                for (int l = 0; l < L; l++) {
                    auto p = phi[l](v, to_doc.c[l]);
                    ll += theta[l] * p;
                }
                //LOG(INFO) << v << " " << corpus.vocab[v] << ' ' << ll;
                sample_log_likelihood += log(ll);
            }
            doc_log_likelihood = LogSum(doc_log_likelihood, sample_log_likelihood);
        }
#pragma omp critical
        {
            local_log_likelihood += doc_log_likelihood - log(num_test_c_samples);
            local_T += th_doc.w.size();
        }
        //LOG(INFO) << local_log_likelihood << ' ' << exp(-local_log_likelihood / local_T);
        //exit(0);
    }

    double global_log_likelihood;
    size_t global_T;
    MPI_Allreduce(&local_log_likelihood, &global_log_likelihood, 
            1, MPI_DOUBLE, MPI_SUM, comm);
    MPI_Allreduce(&local_T, &global_T, 
            1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, comm);

    //LOG(INFO) <<  local_log_likelihood << " " << local_T <<  " " << 
    //    global_log_likelihood << " " << global_T;

    return exp(-global_log_likelihood / global_T);
}


void BaseHLDA::OutputTheta() {
    int num_test_c_samples = 10;
    int num_test_z_burnin = 10;
    int num_test_z_samples = 10;

    std::vector<std::map<int, int>> theta(corpus.D);
#pragma omp parallel for
    for (int d = 0; d < corpus.D; d++) {
        auto &generator = GetGenerator();
        auto &doc = docs[d];

        for (int ncs = 0; ncs < num_test_c_samples; ncs++) {
            // Reset z
            for (auto &l: doc.z) l = generator() % L;

            // Sample c and z
            for (int nz = 0; nz < num_test_z_burnin+num_test_z_samples; nz++) {
                SampleC(doc, false, false, false);
                SampleZ(doc, false, false, false);
                if (nz >= num_test_z_burnin) {
                    for (auto l: doc.z)
                        theta[d][l*10000+doc.c[l]]++;
                }
            }
        }
    }

    LOG(INFO) << "Writing results";
    if (!outfile.empty()) {
        ofstream ftheta(outfile.c_str());
        for (size_t d = 0; d < corpus.D; d++) {
            auto &t = theta[d];
            for (auto kv: t)
                ftheta << kv.first << ':' << kv.second << ' ';
            ftheta << '\n';
        }
    }
}


void BaseHLDA::ComputePhi() {
    auto ret = tree.GetTree();
    auto &generator = GetGenerator();

    if (!sample_phi) {
        for (TLen l = 0; l < L; l++) {
            TTopic K = (TTopic) ret.num_nodes[l];
            auto offset = icount_offset[l];

            vector<float> inv_normalization(K);
            for (TTopic k = 0; k < K; k++)
                inv_normalization[k] = 1.f / (beta[l] * corpus.V + ck_dense[k+offset]);
#pragma omp parallel for
            for (TWord v = 0; v < corpus.V; v++) {
                for (TTopic k = 0; k < K; k++) {
                    TProb prob = (icount(v, k+offset) + beta[l]) 
                                 * inv_normalization[k];
                    phi[l](v, k) = prob;
                    log_phi[l](v, k) = prob;
                }
                vsLn(K, &log_phi[l](v, 0), &log_phi[l](v, 0));
            }
        }
    } else {
        for (TLen l = 0; l < L; l++) {
            TTopic K = (TTopic) ret.num_nodes[l];
            auto offset = icount_offset[l];

            for (TTopic k = 0; k < K; k++) {
                TProb sum = 0;
                for (TWord v = 0; v < corpus.V; v++) {
                    TProb concentration = (TProb)(icount(v, k+offset) + beta[l]);
                    gamma_distribution<TProb> gammarnd(concentration);
                    TProb p = gammarnd(generator);
                    phi[l](v, k) = p;
                    sum += p;
                }
                TProb inv_sum = 1.0f / sum;
                for (TWord v = 0; v < corpus.V; v++) {
                    phi[l](v, k) *= inv_sum;
                    log_phi[l](v, k) = phi[l](v, k);
                }
            }

            for (TWord v = 0; v < corpus.V; v++)
                vsLn(K, &log_phi[l](v, 0), &log_phi[l](v, 0));
        }
    }
}

// TODO output the size of corpus, docs, phi + log_phi, icount, pub_sub
void BaseHLDA::OutputSizes() {
    size_t corpus1_size = 0;
    for (auto &d: corpus.w) 
        corpus1_size += d.capacity();

    size_t corpus2_size = 0;
    for (auto &d: docs) 
        corpus2_size += d.z.size() + d.w.size() + d.reordered_w.size() + d.c_offsets.size();

    size_t phi_size = 0;
    for (auto &p: phi) phi_size += p.GetR() * p.GetC();
    for (auto &p: log_phi) phi_size += p.GetR() * p.GetC();

    size_t count_size = count.Capacity();

    size_t icount_size = icount.capacity();

    //size_t pub_sub_size = count.pub_sub.Capacity();
    size_t pub_sub_size = 0;

    size_t G = (1<<30);

    LOG_IF(INFO, process_id == 0)
        << " C1: " << (double)corpus1_size / G 
        << " C2: " << (double)corpus2_size / G 
        << " Phi: " << (double)phi_size / G 
        << " Count: " << (double)count_size / G 
        << " ICount: " << (double)icount_size / G 
        << " PubSub: " << (double)pub_sub_size / G;
}


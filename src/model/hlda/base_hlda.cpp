//
// Created by jianfei on 8/29/16.
//

#include <iostream>
#include <sstream>
#include <random>
#include <omp.h>
#include <thread>
#include "base_hlda.h"
#include "corpus.h"
#include "clock.h"

using namespace std;

int calc_font_size(int max_font_size, int min_font_size, int max_size, int min_size, int my_size) {
    double rate = (log(my_size) - log(min_size)) / (log(max_size) - log(min_size));
    double size = (max_font_size - min_font_size) * rate + min_font_size;
    return int(size);
};

BaseHLDA::BaseHLDA(Corpus &corpus, Corpus &to_corpus, Corpus &th_corpus, int L,
                   std::vector<TProb> alpha, std::vector<TProb> beta, vector<double> gamma,
                   int num_iters, int mc_samples, int process_id, int process_size, bool check) :
        process_id(process_id), process_size(process_size),
        tree(L, gamma),
        corpus(corpus), to_corpus(to_corpus), th_corpus(th_corpus),
        L(L), alpha(alpha), beta(beta), gamma(gamma),
        num_iters(num_iters), mc_samples(mc_samples), phi((size_t) L), log_phi((size_t) L),
        count(L, corpus.V, omp_get_max_threads()),
        icount(1, process_size, corpus.V, 1/*K*/, row_partition,
               process_size, process_id),
        new_topic(true), check(check) {

    std::mt19937_64 rd;
    generators.resize(omp_get_max_threads());
    for (auto &gen: generators)
        gen.seed(rd(), rd());

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

    alpha_bar = accumulate(alpha.begin(), alpha.end(), 0.0);

    for (auto &m: phi) {
        m.SetR(corpus.V, true);
        m.SetC(1, true);
    }
    for (auto &m: log_phi) {
        m.SetR(corpus.V, true);
        m.SetC(1, true);
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &process_id);
    MPI_Comm_size(MPI_COMM_WORLD, &process_size);

    for (int l = 0; l < L; l++) 
        topic_mutexes.emplace_back(new std::mutex[MAX_NUM_TOPICS]);
}

void BaseHLDA::Visualize(std::string fileName, int threshold) {
    string dotFileName = fileName + ".dot";
    int max_font_size = 30, min_font_size = 6;

    auto ret = tree.GetTree();
    int min_node_size = 1;
    int max_node_size = 0;
    for (auto &node: ret.nodes)
        max_node_size = std::max(max_node_size, node.num_docs);

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
    for (int l = 0; l < L; l++)
        if (doc.c[l] >= num_instantiated[l])
            topic_mutexes[l][doc.c[l]].lock();
    lockdoc_time.Add(clk.toc());
}

void BaseHLDA::UnlockDoc(Document &doc) {
    for (int l = 0; l < L; l++)
        if (doc.c[l] >= num_instantiated[l])
            topic_mutexes[l][doc.c[l]].unlock();
}

xorshift& BaseHLDA::GetGenerator() {
    return generators[omp_get_thread_num()];
}

void BaseHLDA::AllBarrier() {
    std::thread count_thread([&](){count.Compress();});
    std::thread tree_thread([&](){tree.Barrier();});
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

//
// Created by jianfei on 16-11-11.
//

#include <iostream>
#include <thread>
#include <memory>
#include <mpi.h>
#include <publisher_subscriber.h>
#include "corpus.h"
#include "clock.h"
#include <chrono>
#include "glog/logging.h"
#include <sstream>
#include "dcm_dense.h"
#include "concurrent_matrix.h"
#include "concurrent_tree.h"
#include "matrix.h"
#include "adlm_dense.h"
#include "adlm_sparse.h"

using namespace std;
using namespace std::chrono;

struct Operation {
    int pos, delta;
};

int main(int argc, char **argv) {
    google::InitGoogleLogging(argv[0]);
    // output all logs to stderr
    FLAGS_stderrthreshold=google::INFO;
    FLAGS_colorlogtostderr=true;

    int provided;
    MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &provided);
    LOG_IF(FATAL, provided != MPI_THREAD_MULTIPLE) << "MPI_THREAD_MULTIPLE is not supported";
    int process_id, process_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &process_id);
    MPI_Comm_size(MPI_COMM_WORLD, &process_size);
    LOG(INFO) << process_id << ' ' << process_size;

//    {
//        bool is_publisher = process_id < 2;
//        bool is_subscriber = process_id >= 1;
//
//        auto on_recv = [&](char *data, int length){
//            LOG(INFO) << process_id << " received " << std::string(data, data+length);
//        };
//
//        PublisherSubscriber<decltype(on_recv)> pubsub(is_subscriber, on_recv);
//        LOG(INFO) << "PubSub started";
//
//        std::this_thread::sleep_for(1s);
//        if (process_id == 0) {
//            string message = "Message from node 0";
//            pubsub.Publish(message.data(), message.size());
//        }
//
//        std::this_thread::sleep_for(1s);
//        if (process_id == 1) {
//            string message = "Message from node 1";
//            pubsub.Publish(message.data(), message.size());
//        }
//
//        pubsub.Barrier();
//    }

//    {
//        // Generate some data
//        int num_docs = 10000;
//        float avg_doc_length = 1000;
//        int vocab_size = 10000;
//        auto corpus = Corpus::Generate(num_docs, avg_doc_length, vocab_size);
//        LOG(INFO) << "Corpus have " << corpus.T << " tokens";
//
//        // Pubsub for cv
//        std::vector<int> cv((size_t)vocab_size);
//        auto on_recv = [&](const char *msg, size_t length) {
//            cv[*((const int*)msg)]++;
//        };
//        PublisherSubscriber<decltype(on_recv)> pubsub(true, on_recv);
//
//        // Another pubsub for cv
//        std::vector<int> cv2((size_t)vocab_size);
//        auto on_recv2 = [&](const char *msg, size_t length) {
//            cv2[*((const int*)msg)]++;
//        };
//        PublisherSubscriber<decltype(on_recv2)> pubsub2(true, on_recv2);
//
//        // Compute via allreduce
//        std::vector<int> local_cv((size_t)vocab_size);
//        std::vector<int> global_cv((size_t)vocab_size);
//        for (auto &doc: corpus.w)
//            for (auto v: doc)
//                local_cv[v]++;
//        MPI_Allreduce(local_cv.data(), global_cv.data(), vocab_size,
//                      MPI_INT, MPI_SUM, MPI_COMM_WORLD);
//
//        // Compute via pubsub
//        Clock clk;
//        for (auto &doc: corpus.w) {
//            for (auto v: doc) {
//                pubsub.Publish((char*)&v, sizeof(v));
//                pubsub2.Publish((char*)&v, sizeof(v));
//            }
//        }
//        pubsub.Barrier();
//        pubsub2.Barrier();
//        LOG(INFO) << "Finished in " << clk.toc() << " seconds. (" << pubsub.GetNumSyncs() << " syncs)";
//
//        // Compare
//        LOG_IF(FATAL, global_cv != cv) << "Incorrect CV";
//        LOG_IF(FATAL, global_cv != cv2) << "Incorrect CV2";
//    }

//    {
//        AtomicVector<int> v;
//
//        auto PrintVector = [&]() {
//            for (int i = 0; i < process_size; i++) {
//                if (i == process_id) {
//                    auto sess = v.GetSession();
//                    auto size = sess.Size();
//                    std::ostringstream sout;
//                    sout << "Node " << i << " size = " << size;
//                    for (int i = 0; i < size; i++)
//                        sout << " " << sess.Get(i);
//                    LOG(INFO) << sout.str();
//                }
//                MPI_Barrier(MPI_COMM_WORLD);
//            }
//        };
//
//        if (process_id == 0) {
//            v.IncreaseSize(5);
//            auto sess = v.GetSession();
//            sess.Inc(3);
//            sess.Inc(2);
//            sess.Dec(1);
//        }
//        v.Barrier();
//        PrintVector();
//
//        if (process_id == 1) {
//            auto sess = v.GetSession();
//            sess.Inc(4);
//        }
//        v.Barrier();
//        PrintVector();
//    }

//    {
//        AtomicVector<int> v;
//
//        int vector_size = 100000;
//        int num_operations = 1000000;
//
//        std::mt19937 generator;
//        std::vector<Operation> operations(static_cast<size_t>(num_operations));
//        for (auto &op: operations) {
//            op.pos = static_cast<int>(generator() % vector_size);
//            op.delta = generator() % 2 == 0 ? 1 : -1;
//        }
//        std::vector<int> oracle(static_cast<size_t>(vector_size));
//        std::vector<int> global_oracle(static_cast<size_t>(vector_size));
//        for (auto &op: operations)
//            oracle[op.pos] += op.delta;
//        MPI_Allreduce(oracle.data(), global_oracle.data(), vector_size,
//                      MPI_INT, MPI_SUM,
//                      MPI_COMM_WORLD);
//        LOG(INFO) << "Generated oracle";
//
//        // Resize on node 0
//        if (process_id == 0)
//            v.IncreaseSize(vector_size);
//        v.Barrier();
//
//        {
//            auto sess = v.GetSession();
//            for (auto &op: operations)
//                if (op.delta == 1)
//                    sess.Inc(op.pos);
//                else
//                    sess.Dec(op.pos);
//        }
//        v.Barrier();
//
//        {
//            auto sess = v.GetSession();
//            for (int i = 0; i < vector_size; i++)
//                LOG_IF(FATAL, sess.Get(i) != global_oracle[i])
//                  << "Incorrect result. Expect " << global_oracle[i]
//                  << " got " << sess.Get(i);
//        }
//    }

//    {
//        AtomicMatrix<int> m;
//
//        auto PrintMatrix = [&]() {
//            for (int i = 0; i < process_size; i++) {
//                if (i == process_id) {
//                    auto sess = m.GetSession();
//                    auto R = sess.GetR();
//                    auto C = sess.GetC();
//                    std::ostringstream sout;
//                    sout << "Node " << i << " R = " << R << " C = " << C << "\n";
//                    for (int r=0; r<R; r++) {
//                        for (int c=0; c<C; c++)
//                            sout << sess.Get(r, c) << " ";
//                        sout << "\n";
//                    }
//                    LOG(INFO) << sout.str();
//                }
//                MPI_Barrier(MPI_COMM_WORLD);
//            }
//        };
//
//        m.SetR(3);
//
//        if (process_id == 0) {
//            m.IncreaseC(5);
//            auto sess = m.GetSession();
//            sess.Inc(2, 4);
//            sess.Inc(1, 3);
//            sess.Dec(1, 2);
//        }
//        m.Barrier();
//        PrintMatrix();
//
//        if (process_id == 1) {
//            auto sess = m.GetSession();
//            sess.Inc(2, 2);
//        }
//        m.Barrier();
//        PrintMatrix();
//    }

//    {
//        AtomicVector<int> v;
//        auto sess = v.GetSession();
//        auto sess2 = std::move(sess);
//    }
//    DCMDense<int> dcm(1, process_size, 5, 1, row_partition, process_size, process_id);
//    dcm.resize(5, 3);
//    if (process_id == 0) {
//        dcm.increase(3, 1);
//        dcm.increase(2, 0);
//    } else {
//        dcm.increase(1, 1);
//    }
//    dcm.sync();
//    for (int p = 0; p < process_size; p++) {
//        if (p == process_id) {
//            cout << "Node " << p << endl;
//            for (int r = 0; r < 5; r++) {
//                auto *row = dcm.row(r);
//                for (int c = 0; c < 3; c++)
//                    cout << row[c] << ' ';
//                cout << endl;
//            }
//            cout << "Marginal" << endl;
//            auto *m = dcm.rowMarginal();
//            for (int c = 0; c < 3; c++)
//                cout << m[c] << ' ';
//            cout << endl;
//        }
//        MPI_Barrier(MPI_COMM_WORLD);
//    }

    //{
    //    ADLMDense m(1, 2, 1, 1);
    //    auto PrintMatrix = [&]() {
    //        m.Barrier();
    //        LOG(INFO) << "Barrier";
    //        for (int p = 0; p < process_size; p++) {
    //            if (p == process_id) {
    //                cout << "Node " << p << " Matrix of " 
    //                    << m.GetC(0) << " columns." << endl;
    //                for (int r = 0; r < 2; r++) {
    //                    for (int c = 0; c < m.GetC(0); c++)
    //                        cout << m.Get(0, r, c) << ' ';
    //                    cout << endl;
    //                }
    //                cout << "Sum"  << endl;
    //                for (int c = 0; c < m.GetC(0); c++)
    //                    cout << m.GetSum(0, c) << ' ';
    //                cout << endl;
    //            }
    //            MPI_Barrier(MPI_COMM_WORLD);
    //        }
    //    };

    //    m.Grow(0, 0, 3);
    //    LOG(INFO) << "Grown";
    //    m.Inc(0, 0, 0, 1); 
    //    m.Inc(0, 0, 1, 2);
    //    m.Inc(0, 0, 0, 1);
    //    m.Publish(0);
    //    LOG(INFO) << "Published";
    //    PrintMatrix();

    //    m.Grow(0, 0, 7);
    //    m.Inc(0, 0, 0, 4);
    //    m.Inc(0, 0, 0, 6);
    //    m.Publish(0);
    //    PrintMatrix();

    //    m.Compress();
    //    PrintMatrix();

    //    m.Grow(0, 0, 10);
    //    m.Inc(0, 0, 1, 9);
    //    m.Publish(0);
    //    PrintMatrix();

    //    m.Grow(0, 0, 20);
    //    m.Inc(0, 0, 1, 19);
    //    m.Publish(0);
    //    PrintMatrix();

    //    m.Compress();
    //    PrintMatrix();
    //}

    //{
    //    int num_rows = 100;
    //    int num_cols = 1;
    //    int num_ops = 10000000;
    //    float grow_prob = 0.00001;
    //    float compress_prob = 0.000005;
    //    float inc_prob = 0.8;

    //    Matrix<int> mat(num_rows, num_cols);
    //    ADLMDense con_mat(1, num_rows, 1);
    //    if (process_id == 0)
    //        con_mat.Grow(0, 0, num_cols);

    //    std::mt19937 generator;
    //    std::uniform_real_distribution<float> u01;
    //    int C = num_cols;
    //    for (int i = 0; i < num_ops; i++) {
    //        float u = u01(generator);
    //        if (u < compress_prob) {
    //            con_mat.Publish(0);
    //        }
    //        else if (u < grow_prob) {
    //            ++C;
    //            if (process_id == 0)
    //                con_mat.Grow(0, 0, C);
    //            mat.SetC(C);
    //        } else {
    //            auto r = generator() % num_rows;
    //            auto c = generator() % C;
    //            if (u < inc_prob) {
    //                mat(r, c)++;
    //                if (process_id == 0) 
    //                    con_mat.Inc(0, 0, r, c);
    //            } else {
    //                mat(r, c)--;
    //                if (process_id == 0)
    //                    con_mat.Dec(0, 0, r, c);
    //            }
    //        }
    //    }
    //    con_mat.Publish(0);
    //    con_mat.Barrier();

    //    std::vector<int> sum(C);
    //    // Check that con_mat = mat
    //    for (int r = 0; r < num_rows; r++)
    //        for (int c = 0; c < C; c++) {
    //            LOG_IF(FATAL, con_mat.Get(0, r, c) != mat(r, c)) 
    //                << "Incorrect value at (" << r << ", " << c
    //                << ") expected " << mat(r, c) << " got " << con_mat.Get(0, r, c);
    //            sum[c] += mat(r, c);
    //        }
    //    for (int c = 0; c < C; c++)
    //        LOG_IF(FATAL, con_mat.GetSum(0, c) != sum[c]) << "Incorrect sum";
    //}

    //{
    //    ConcurrentTree tree(3, std::vector<double>{1, 2});
    //    cout << tree.GetTree() << endl;

    //    cout << tree.AddNodes(0) << endl;
    //    cout << tree.GetTree() << endl;

    //    cout << tree.IncNumDocs(2) << endl;
    //    cout << tree.GetTree() << endl;

    //    cout << tree.AddNodes(1) << endl;
    //    cout << tree.IncNumDocs(3) << endl;
    //    cout << tree.GetTree() << endl;

    //    cout << tree.AddNodes(0) << endl;
    //    cout << tree.IncNumDocs(5) << endl;
    //    cout << tree.GetTree() << endl;

    //    tree.IncNumDocs(2);
    //    tree.IncNumDocs(3);
    //    tree.IncNumDocs(3);
    //    tree.IncNumDocs(3);
    //    tree.IncNumDocs(5);
    //    tree.IncNumDocs(5);

    //    ConcurrentTree::IDPos idpos[2] = {{4, 2}, {7, 5}};
    //    tree.AddNodes(idpos, 2);
    //    cout << tree.IncNumDocs(7) << endl;
    //    cout << tree.GetTree() << endl;
    //    tree.SetThreshold(2);

    //    cout << tree.Compress() << endl;
    //    cout << tree.GetNumInstantiated() << endl;
    //    cout << tree.GetTree() << endl;

    //    tree.SetBranchingFactor(1);
    //    tree.Instantiate();
    //    cout << tree.GetTree() << endl;
    //}

    //{
    //    int N = 2;
    //    int R = 3;
    //    std::vector<int> msgs;
    //    if (process_id == 0) {
    //        msgs.insert(msgs.end(), {0, 0, 0, 1});
    //        msgs.insert(msgs.end(), {1, 2, 2, 1});
    //        msgs.insert(msgs.end(), {1, 2, 2, -1});
    //        msgs.insert(msgs.end(), {1, 2, 2, 1});
    //        msgs.insert(msgs.end(), {1, 2, 0, 1});
    //        msgs.insert(msgs.end(), {0, 1, 1, 0});
    //    } else {
    //        msgs.insert(msgs.end(), {1, 0, 1, 1});
    //        msgs.insert(msgs.end(), {1, 2, 1, 1});
    //    }
    //    CVA<SpEntry> cva(N * R);
    //    auto sizes = ADLMSparse::ComputeDelta(N, R, 
    //            MPI_COMM_WORLD, process_id, process_size, msgs, cva);

    //    auto PrintMatrices = [&]() {
    //        for (int p = 0; p < process_size; p++) {
    //            if (p == process_id) {
    //                cout << "Node " << p << endl;
    //                for (int r = 0; r < cva.R; r++) {
    //                    auto row = cva.Get(r);
    //                    for (auto &entry: row)
    //                        cout << entry.k << ':' << entry.v << ' ';
    //                    cout << endl;
    //                }
    //            }
    //            MPI_Barrier(MPI_COMM_WORLD);
    //        }
    //    };

    //    PrintMatrices();
    //    LOG(INFO) << process_id << " " << sizes;
    //}

    //{
    //    // Large scale test of ADLMSparse::ComputeDelta
    //    std::vector<int> msgs;
    //    int N = 1;
    //    int R = 100000;
    //    int C = 100;
    //    int ops = 1e7;
    //    msgs.reserve(ops * 4);

    //    std::vector<Matrix<int>> oracle(N);
    //    std::vector<Matrix<int>> result(N);
    //    for (auto &m: oracle)
    //        m.Resize(R, C);
    //    for (auto &m: result)
    //        m.Resize(R, C);

    //    //ADLMSparse adlm(N, R, omp_get_max_threads());

    //    LOG(INFO) << "Start generating data";
    //    xorshift generator;
    //    for (int p = 0; p < process_size; p++) {
    //        generator.seed(p, p);
    //        for (int o = 0; o < ops; o++) {
    //            int n = generator() % N;
    //            int r = generator() % R;
    //            int c = generator() % C;
    //            int delta = generator() % 3 - 1;

    //            if (p == process_id)
    //                msgs.insert(msgs.end(), {n, r, c, delta});
    //            else 
    //                oracle[n](r, c) += delta;
    //        }
    //    }
    //    MPI_Barrier(MPI_COMM_WORLD);
    //    //std::this_thread::sleep_for(10s);

    //    LOG(INFO) << "Start communicating";
    //    CVA<SpEntry> delta(N * R);
    //    Clock clk;
    //    ADLMSparse::ComputeDelta(N, R, 
    //            MPI_COMM_WORLD, process_id, process_size, msgs, delta);
    //    LOG(INFO) << "Elapsed " << clk.toc() << " seconds";

    //    for (int n = 0; n < N; n ++)
    //        for (int r = 0; r < R; r++) {
    //            auto row = delta.Get(n * R + r);
    //            for (auto &entry: row)
    //                result[n](r, entry.k) += entry.v;
    //        }

    //    for (int n = 0; n < N; n ++)
    //        for (int r = 0; r < R; r++) {
    //            auto row = delta.Get(n * R + r);
    //            for (int c = 0; c < C; c++)
    //                if (result[n](r, c) != oracle[n](r, c))
    //                    LOG(FATAL) <<
    //                        "The result is incorrect at (" 
    //  s                     << process_id << ", " << n << ", " << r << ", " <<
    //                        c << ") expected " << oracle[n](r, c) << " get "
    //                        << result[n](r, c);
    //        }

    //    //std::this_thread::sleep_for(10s);
    //}

    //{
    //    int N = 2;
    //    int R = 3;
    //    int C = 3;
    //    LOG_IF(FATAL, process_size != 2) << "Process size must be 2";

    //    //vector<int> msgs;
    //    //if (process_id == 0) msgs.insert(msgs.end(), {1, 1, 1, 1});
    //    //CVA<SpEntry> cva(6);
    //    //ADLMSparse::ComputeDelta(N, R, MPI_COMM_WORLD, process_id, process_size,
    //    //        msgs, cva);

    //    //auto PrintMatrices = [&]() {
    //    //    for (int p = 0; p < process_size; p++) {
    //    //        if (p == process_id) {
    //    //            cout << "Node " << p << endl;
    //    //            for (int r = 0; r < cva.R; r++) {
    //    //                auto row = cva.Get(r);
    //    //                for (auto &entry: row)
    //    //                    cout << entry.k << ':' << entry.v << ' ';
    //    //                cout << endl;
    //    //            }
    //    //        }
    //    //        MPI_Barrier(MPI_COMM_WORLD);
    //    //    }
    //    //};

    //    //PrintMatrices();
    //    

    //    ADLMSparse adlm(N, R, omp_get_max_threads());
    //    
    //    auto PrintMatrices = [&]() {
    //        for (int p = 0; p < process_size; p++) {
    //            if (p == process_id) {
    //                cout << "Node " << p << endl;
    //                for (int n = 0; n <N; n++) {
    //                    for (int r = 0; r < R; r++) {
    //                        for (int c = 0; c < adlm.GetC(n); c++)
    //                            cout << adlm.Get(n, r, c) << ' ';
    //                        cout << endl;
    //                    }
    //                    cout << endl;
    //                }
    //            }
    //           MPI_Barrier(MPI_COMM_WORLD);
    //        }
    //    };

    //    adlm.Grow(0, process_id, C);
    //    adlm.Publish(0);
    //    adlm.Barrier();
    //    //std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    //    if (process_id == 0) {
    //        adlm.Inc(0, 1, 1, 1);
    //    } else {
    //        adlm.Inc(0, 0, 2, 1);
    //    }
    //    adlm.Publish(0);
    //    adlm.Barrier();
    //    //std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    //    PrintMatrices();
    //}
    
    {
        // Large scale test of ADLMSparse::ComputeDelta
        int N = 2;
        int R = 100000;
        int C = 100;
        int ops = 1e8;
        double prob = 0.01;
        std::uniform_real_distribution<double> u01;

        std::vector<Matrix<int>> oracle(N);
        for (auto &m: oracle)
            m.Resize(R, C);

        ADLMSparse adlm(N, R, omp_get_max_threads());
        for (int i = 0; i < N; i++)
            adlm.Grow(0, i, C);
        adlm.Publish(0);
        adlm.Barrier();

        LOG(INFO) << "Start generating data";
        xorshift generator;
        xorshift generator2;
        for (int p = 0; p < process_size; p++) {
            generator.seed(p+1, p+1);
            for (int o = 0; o < ops; o++) {
                int n = generator() % N;
                int r = generator() % R;
                int c = generator() % C;

                if (p == process_id) {
                    adlm.Inc(0, n, r, c);
                    if (u01(generator2) < prob) {
                        adlm.Publish(0);
                    }
                }
                oracle[n](r, c) ++;
            }
        }
        adlm.Publish(0);
        MPI_Barrier(MPI_COMM_WORLD);
        //std::this_thread::sleep_for(10s);
        adlm.Barrier();

        //if (process_id == 0) {
        //    for (int n = 0; n < N; n++) {
        //        for (int r = 0; r < R; r++) {
        //            for (int c = 0; c < C; c++)
        //                cout << oracle[n](r, c) << ' ';
        //            cout << endl;
        //        }
        //        cout << endl;
        //    }
        //}

        //for (int p = 0; p < process_size; p++) {
        //    if (p == process_id) {
        //        cout << "Node " << p << endl;
        //        for (int n = 0; n < N; n++) {
        //            for (int r = 0; r < R; r++) {
        //                for (int c = 0; c < C; c++)
        //                    cout << adlm.Get(n, r, c) << ' ';
        //                cout << endl;
        //            }
        //            cout << endl;
        //        }
        //    }
        //    MPI_Barrier(MPI_COMM_WORLD);
        //}

        for (int n = 0; n < N; n ++) {
            LOG_IF(FATAL, C != adlm.GetC(n))
                << "C is incorrect";
            for (int r = 0; r < R; r++)
                for (int c = 0; c < C; c++)
                    LOG_IF(FATAL, oracle[n](r, c) != adlm.Get(n, r, c))
                            << "The result is incorrect at (" 
                            << process_id << ", " << n << ", " << r << ", " <<
                            c << ") expected " << oracle[n](r, c) << " get "
                            << adlm.Get(n, r, c);
        }
    }

    MPI_Finalize();
    google::ShutdownGoogleLogging();
}

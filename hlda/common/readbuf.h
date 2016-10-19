#ifndef __READBUF
#define __READBUF

#include <vector>
#include <iostream>
#include <fstream>
#include <exception>
#include <stdexcept>
#include <omp.h>

using namespace std;

class ReadBuf {
public:
    ReadBuf(const char *fileName, size_t capacity) : capacity(capacity), buff(capacity), fin(fileName),
                                                     size_processed(0) {
        if (!fin) {
            cerr << "Failed opening file: " << fileName << endl;
            exit(-1);
        }
    }

    template<class T>
    void Scan(T operation) {
        size_t used = capacity;
        while (Fill(used)) {
            used = Process(operation);
        }
        Process(operation);
    }

private:
    bool Fill(size_t used) {
        buff.erase(buff.begin(), buff.begin() + used);
        buff.resize(capacity);
        size_t reserve = capacity - used;
        bool success = !(fin.read(buff.data() + reserve, used).fail());
        size = reserve + fin.gcount();
        buff.resize(size);
        return success;
    }

    template<class T>
    size_t Process(T operation) {
        // Find last '\n'
        //	char lf = '\n';
        long long pos = -1;
        for (long long n = (long long) buff.size() - 1; n >= 0; n--)
            if (buff[n] == '\n') {
                pos = n;
                break;
            }
        if (pos == -1)
            throw runtime_error("input line too line");

        long long used = pos + 1;

        // Find all occurence of '\n'
        int N = omp_get_max_threads();
        vector<vector<size_t>> crs_h((size_t) N);
        vector<long long> allcr;
        allcr.push_back(-1);
#pragma omp parallel
        {
            int id = omp_get_thread_num();
            size_t part = buff.size() / N;
            vector<size_t> &crs = crs_h[id];
            size_t beg = id * part;
            size_t end = (id + 1 == N ? buff.size() : (id + 1) * part);
            for (size_t n = beg; n < end; n++)
                if (buff[n] == '\n')
                    crs.push_back(n);
        }

        for (int i = 0; i < N; i++)
            allcr.insert(allcr.end(), crs_h[i].begin(), crs_h[i].end());

        int ndocs = (int) allcr.size() - 1;
#pragma omp parallel
        {
#pragma omp for
            for (int i = 0; i < ndocs; i++) {
                string doc = string(buff.begin() + allcr[i] + 1, buff.begin() + allcr[i + 1]);
                operation(doc);
            }
        }

        return (size_t) used;
    }

public:
    size_t capacity;
    size_t size;
    vector<char> buff;
    ifstream fin;
    size_t size_processed;
};

#endif

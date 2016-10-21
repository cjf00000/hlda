#!/usr/bin/env bash
rm list.txt
touch list.txt
for algo in cs pcs
do
    for n_mc_iters in -1 30
    do
        for beta in 0.125 0.25 0.5 1 2 4
        do
            for alpha in 0.3 0.4 0.5 0.6 0.7
            do
                for gamma in 1e-60 1e-80 1e-100 1e-120 #1e-3 1e-5 1e-10 1e-15 1e-20 1e-25 1e-30 1e-40 1e-50 1e-60 1e-80 1e-100 1e-120
                do
                    id=${algo}_mc${n_mc_iters}_beta${beta}_alpha${alpha}_gamma${gamma}
                    echo $id $algo $n_mc_iters $beta $alpha $gamma >> list.txt
                done
            done
        done
    done
done

cat list.txt | xargs -n6 -P24 ./worker.sh

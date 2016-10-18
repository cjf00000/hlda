#!/usr/bin/env bash
id=$1
algo=$2
n_mc_iters=$3
beta=$4
alpha=$5
gamma=$6

bin=../release/model/hlda

beta1=$beta
beta2=`echo "$beta1 * 0.5" | bc`
beta3=`echo "$beta1 * 0.25" | bc`
beta4=`echo "$beta1 * 0.25" | bc`
beta_param="$beta1,$beta2,$beta3,$beta4"

command="$bin --algo $algo --n_mc_iters $n_mc_iters --alpha $alpha --beta $beta_param --gamma $gamma --prefix ../data/nysmaller --vis_prefix ../vis_result/$id.dot"

echo $1 $2 $3 $4 $5 $6 $command

$command >../result/$id.log 2>&1

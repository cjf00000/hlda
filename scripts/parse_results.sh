#!/usr/bin/env bash
# Parse results for my implementation
beta=0.5
for algo in cs_mc-1 cs_mc20 pcs_mc-1 pcs_mc20
do
    rm ${algo}.summary
    for i in `ls ../result/${algo}*beta${beta}*.log`
    do
        beta=`echo $i | sed -e 's/_/ /g' -e 's/beta/ /g' -e 's/alpha/ /g' -e 's/gamma/ /g' | awk '{print $3}'`
        alpha=`echo $i | sed -e 's/_/ /g' -e 's/beta/ /g' -e 's/alpha/ /g' -e 's/gamma/ /g' | awk '{print $4}'`
        gamma=`echo $i | sed -e 's/_/ /g' -e 's/beta/ /g' -e 's/alpha/ /g' -e 's/gamma/ /g' -e 's/\.log/ /g' | awk '{print $5}'`
        info=`grep Iteration $i | awk '{print $3, $12}' | tail -n 1` 
        if [ -n "$info" ]; then
            echo $info $alpha $beta $gamma >> ${algo}.summary
        fi
    done
done

# Parse results for Blei's implementation
rm hlda-c.summary
for name in `ls -d ../../hlda-c/beta${beta}*`
do
    beta=`echo $name | sed -e 's/_/ /g' -e 's/beta/ /g' -e 's/alpha/ /g' -e 's/gamma/ /g' | awk '{print $2}'`
    alpha=`echo $name | sed -e 's/_/ /g' -e 's/beta/ /g' -e 's/alpha/ /g' -e 's/gamma/ /g' | awk '{print $3}'`
    gamma=`echo $name | sed -e 's/_/ /g' -e 's/beta/ /g' -e 's/alpha/ /g' -e 's/gamma/ /g' | awk '{print $4}'`
    beta2=`echo "$beta * 0.5" | bc`
    beta3=`echo "$beta * 0.25" | bc`
    echo $beta,$alpha,$gamma,$beta2,$beta3
    #echo "../release/model/hlda --prefix ../data/nysmaller -L 4 --algo es --alpha $alpha --gamma $gamma --beta ${beta},${beta2},${beta3},${beta3} --model_path $name/run000"
    ../release/model/hlda --prefix ../data/nysmaller -L 4 --algo es --alpha $alpha --gamma $gamma --beta ${beta},${beta2},${beta3},${beta3} --model_path $name/run000 >result 2>&1
    
    perplexity=`grep Perplexity result | awk '{print $3}'`
    topics=`grep Read result | awk '{print $2}'`
    
    if [ -n "$perplexity" ]; then
        echo $topics $perplexity $alpha $beta $gamma >> hlda-c.summary
    fi
done

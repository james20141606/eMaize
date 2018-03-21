#! /bin/bash
[ -d final_predictions ] || mkdir final_predictions
# predict trait1 and trait2
for trait in trait1 trait2;do
    for i in 1 2 3;do
        if [ $i = 1 ];then gamma=0.05
        elif [ $i = 2 ];then gamma=0.10
        elif [ $i = 3 ];then gamma=0.15
        else echo "invalid i=$i"; exit 1
        fi
        bin/run_mixed_model.py mixed_ridge \
            --genotype-file best_subsets/snps:/$trait/$i/X \
            --transpose-genotype \
            --gsm-file output/gsm/random_select/100000/1 \
            --phenotype-file data/phenotypes/all:$trait \
            --parent-table-file data/parent_table \
            --train-index-file data/train_test_indices:/train \
            --test-index-file data/train_test_indices:/test \
            --cv-type s1f --gammas $gamma --alphas 0.001 \
            -o final_predictions/${trait}.${i}
    done
done
# predict trait3
for i in 0 1;do
    bin/run_mixed_model.py mixed_ridge \
        --genotype-file best_subsets/snps:/trait3/$i/X \
        --transpose-genotype \
        --gsm-file output/gsm/random_select/100000/1 \
        --phenotype-file data/phenotypes/all:$trait \
        --parent-table-file data/parent_table \
        --train-index-file data/train_test_indices:/train \
        --test-index-file data/train_test_indices:/test \
        --cv-type s1f --gammas 0.15 --alphas 0.001 \
        -o final_predictions/trait3.${i}
done
# combine predictions
bin/combine_predictions.py -i best_subsets -o best_subsets
## Run FastLMM
```bash
bin/create_cv_folds.py -i emaize_data/phenotype/pheno_emaize.txt \
    --k-male 5 --k-female 20 --max-size 20 -m cross \
    -o output/fastlmm/cv_index.cross
    
    
bin/run_fastlmm.py single_snp --snp-file output/random_select/100000:0 \
    --phenotype-file emaize_data/phenotype/pheno_emaize.txt \
    --k0-file output/random_select/10000:0 \
    --sample-indices-file output/fastlmm/cv_index.cross:/0/train \
    -o output/fastlmm
    


{
cvfolds=$(seq 0 20) 
inds=$(seq 0 20)
for cvfold in $cvfolds;do
    for ind in $inds;do
            echo bin/run_fastlmm.py single_snp \
                --snp-file output/random_select/10000*100snp:$ind \
                --phenotype-file emaize_data/phenotype/pheno_emaize.txt \
                --k0-file output/random_select/10000:0 \
                --sample-indices-file output/fastlmm/cv_index.cross:/$cvfold/train\
                -o output/fastlmm/pvalue10000*100/$cvfold/$ind
    done
done
} | parallel -P 10

{
cvfolds=$(seq 4 5) 
inds=$(seq 0 20)
for cvfold in $cvfolds;do
    for ind in $inds;do
            echo bin/run_fastlmm.py single_snp \
                --snp-file output/random_select/10000*100snp:$ind \
                --phenotype-file emaize_data/phenotype/pheno_emaize.txt \
                --k0-file output/random_select/10000:0 \
                --sample-indices-file output/fastlmm/cv_index.cross:/$cvfold/train\
                -o output/fastlmm/pvalue10000*100/$cvfold/$ind
    done
done
} > Jobs/single_snp.txt
qsubgen -n single_snp -q Z-LU -a 1-2 -j 5 --bsub --task-file Jobs/single_snp.txt
bsub < Jobs/single_snp.sh

final_predict p value
{
inds=$(seq 0 20)
for ind in $inds;do
            echo bin/run_fastlmm.py single_snp \
                --snp-file output/random_select/10000*100snp:$ind \
                --phenotype-file emaize_data/phenotype/pheno_emaize.txt \
                --k0-file output/random_select/10000:0 \
                --sample-indices-file output/fastlmm/predict_index:/0/train\
                -o output/fastlmm/pvaluefinal_predict/$ind
done
} | parallel -P 4  




{
cvfolds=$(seq 4 5) 
for cvfold in $cvfolds;do
        echo bin/run_fastlmm.py selected_snp \
            --input-result-file output/fastlmm/pvalue10000*100/$cvfold/ \
            --sample-indices-file output/fastlmm/cv_index.cross:/$cvfold/ \
            -o output/fastlmm/4000select/$cvfold/
done
} | parallel -P 2


{
cvfolds=$(seq 6 8) 
for cvfold in $cvfolds;do
        echo bin/run_fastlmm.py selected_snp \
            --input-result-file output/fastlmm/pvalue10000*100/$cvfold/ \
            --sample-indices-file output/fastlmm/cv_index.cross:/$cvfold/ \
            -o output/fastlmm/4000select/$cvfold/
done
} > Jobs/selected_snp.txt
qsubgen -n selected_snp -q Z-LU -a 1-4 -j 5 --bsub --task-file Jobs/selected_snp.txt
bsub < Jobs/selected_snp.sh


# feature selection findal_predict
{
bin/run_fastlmm.py selected_snp \
            --input-result-file output/fastlmm/pvaluefinal_predict/ \
            --sample-indices-file output/fastlmm/predict_index:/0/ \
            -o output/fastlmm/final4000select/
} | parallel -P 1

    

！！random projection在ibme上
--k0-file output/random_projection/2bit/normalized_matrix/r=10000:/X \
--snp-file output/random_select/10000:0 \

{
cvfolds=$(seq 0 19)  
for cvfold in $cvfolds;do   
        echo bin/run_fastlmm.py fastlmm \
            --snp-file output/fastlmm/4000select/$cvfold/selected. \
            --k0-file output/k0.file:0 \
            --phenotype-file emaize_data/phenotype/pheno_emaize.txt \
            --cvindex-file output/fastlmm/cv_index.cross:/$cvfold/ \
            --penalty 1000 \
            -o output/fastlmm/predict_4000select/$cvfold/
done
} | parallel -P 4

-- bin/run_fastlmm.py fastlmm --snp-file output/fastlmm/selected/0/selected. --k0-file output/random_select/10000:0 --phenotype-file emaize_data/phenotype/pheno_emaize.txt --cvindex-file output/fastlmm/cv_index.cross:/0/ --penalty 1000 -o output/fastlmm/predict/0
    
    
    
%run -d -b /dev/shm/shibinbin/anaconda2/lib/python2.7/site-packages/fastlmm/inference/lmm.py:482 bin/run_fastlmm.py fastlmm \
    --snp-file output/random_select/10000:1 \
    --k0-file output/random_projection/2bit/normalized_matrix/r=10000:/X \
    --transpose-k0 \
    --phenotype-file emaize_data/phenotype/pheno_emaize.txt \
    --cvindex-file output/fastlmm/cv_index.cross:0 \
    --n-snps 1000 --penalty 1000.0 \
    -o output/fastlmm

```
## Modify FastLMM code to add regularization to weights of fixed effects
* In function `fastlmm.inference.lmm.predict`
Add argument: `penalty=0.0`.
Add `penalty` argument to `LMM.findH2` and `lmm.nLLeval`:
```python
if h2raw is None:
    res = lmm.findH2(penalty=penalty) #!!!why is REML true in the return???
else:
    res = lmm.nLLeval(h2=h2raw, penalty=penalty)
```
* In function `fastlmm.inference.lmm.findH2`, remove `**kwargs`.
```python
def f(x,resmin=resmin):
```
## Predict using GSMs and get residuals
```bash
bin/create_cv_folds.py -i emaize_data/phenotype/pheno_emaize.txt \
    --k-male 5 --k-female 20 --max-size 20 -m cross \
    -o output/mixed_model/cv_index.cross
model_name=ridge
model_name=gpr
traits="trait1 trait2 trait3"
cvfolds=$(seq 0 9)
gsm_genotype_file=output/random_select/10000:/0/X
gsm_genotype_file=output/random_projection/2bit/normalized_matrix/r=10000:/X
{
for trait in $traits;do
    for cvfold in $cvfolds;do
        echo bin/run_mixed_model.py single_model \
            --genotype-file $gsm_genotype_file \
            --phenotype-file data/phenotypes/all:${trait} \
            --parent-table-file data/parent_table \
            --sample-indices-file output/fastlmm/cv_index.cross:/$cvfold/train \
            --model-name $model_name --normalize-x --output-residuals \
            -o output/mixed_model/residuals/$model_name/$trait/$cvfold
    done
done
} | parallel -P 4
```
## Evaluate prediction using GSMs
```bash
{
for trait in $traits;do
    for cvfold in $cvfolds;do
        echo bin/run_mixed_model.py evaluate \
            -i output/mixed_model/residuals/$model_name/$trait/$cvfold/predictions \
            --sample-indices-file output/fastlmm/cv_index.cross:/$cvfold/test \
            -o output/mixed_model/residuals/$model_name/$trait/$cvfold/metrics.txt
    done
done
} | parallel -P 4
```
## Plot predictions using GSMs
```bash
for trait in $traits;do
    for cvfold in 0;do
        bin/run_mixed_model.py plot_predictions \
            -i output/mixed_model/residuals/$model_name/$trait/$cvfold/predictions \
            --parent-table-file data/parent_table \
            --train-indices-file output/fastlmm/cv_index.cross:/$cvfold/train \
            --test-indices-file output/fastlmm/cv_index.cross:/$cvfold/test \
            -o output/mixed_model/residuals/$model_name/$trait/$cvfold/predictions.pdf
    done
done
```
## Predict using GSMs and Ridge CV to get residuals
```bash
model_name=ridge_cv
{
for trait in $traits;do
    for cvfold in $cvfolds;do
        echo bin/run_mixed_model.py single_model \
            --genotype-file $gsm_genotype_file \
            --phenotype-file data/phenotypes/all:${trait} \
            --parent-table-file data/parent_table \
            --sample-indices-file output/fastlmm/cv_index.cross:/$cvfold/train \
            --model-name $model_name --normalize-x --output-residuals \
            -o output/mixed_model/residuals/$model_name/$trait/$cvfold
    done
done
} | parallel -P 4
```
## Feature selection by ANOVA
```bash
genotype_file=output/random_select/100000:/1/X
genotype_file=data/genotype_minor/chr1:data
{
for trait in $traits;do
    for cvfold in $cvfolds;do
        echo bin/filter_features.py anova_linregress \
            --genotype-file $genotype_file \
            --phenotype-file output/mixed_model/residuals/$model_name/$trait/$cvfold/predictions:residual \
            --sample-indices-file output/fastlmm/cv_index.cross:/$cvfold/train \
            --batch-size 100000 \
            -o output/mixed_model/anova_linregress/$model_name/$trait/$cvfold
    done
done
} | parallel -P 4
```
## Convert phenotypes to HDF5 format
```bash
bin/preprocess.py phenotypes_to_hdf5 -i emaize_data/phenotype/pheno_emaize.txt -o data/phenotypes/all
```
## Predict residuals
```bash
{
for trait in $traits;do
    for cvfold in $cvfolds;do
        echo bin/run_mixed_model.py single_model \
            --genotype-file output/random_select/100000:/1/X \
            --parent-table-file data/parent_table \
            --phenotype-file output/mixed_model/residuals/$model_name/$trait/$cvfold/predictions:residual \
            --sample-indices-file output/fastlmm/cv_index.cross:/$cvfold/train \
            --feature-indices-file output/mixed_model/anova_linregress/$model_name/$trait/${cvfold}:reject \
            --model-name ridge_cv --transpose-x --normalize-x \
            -o output/mixed_model/predict_residuals/$model_name/$trait/$cvfold
    done
done
} | parallel -P 4
```
## Evaluate prediction of residuals
```bash
for trait in trait1 trait2 trait3;do
    for cvfold in 0;do
        bin/run_mixed_model.py evaluate \
            -i output/mixed_model/predict_residuals/$model_name/$trait/$cvfold/predictions \
            --sample-indices-file output/fastlmm/cv_index.cross:/$cvfold/test \
            -o output/mixed_model/predict_residuals/$model_name/$trait/$cvfold/metrics.txt
    done
done
```
## Predict phenotypes using mixed model
```bash
{
for trait in $traits;do
    for cvfold in $cvfolds;do
        echo bin/run_mixed_model.py mixed_model \
            -a output/mixed_model/predict_residuals/$model_name/$trait/$cvfold/predictions \
            -b output/mixed_model/residuals/$model_name/$trait/$cvfold/predictions \
            --phenotype-file data/phenotypes/all:${trait} \
            --parent-table-file data/parent_table \
            --sample-indices-file output/fastlmm/cv_index.cross:/$cvfold/train \
            --model-name linear \
            -o output/mixed_model/mixed_model/$model_name/$trait/$cvfold
    done
done
} | parallel -P 4
```
## Evaluate predictions of the mixed model
```bash
for trait in $traits;do
    for cvfold in $cvfolds;do
        bin/run_mixed_model.py evaluate \
            -i output/mixed_model/mixed_model/$model_name/$trait/$cvfold/predictions \
            --sample-indices-file output/fastlmm/cv_index.cross:/$cvfold/test \
            -o output/mixed_model/mixed_model/$model_name/$trait/$cvfold/metrics.txt
    done
done
```
## Plot predictions of the mixed model
```bash
for trait in $traits;do
    for cvfold in $cvfolds;do
        bin/run_mixed_model.py plot_predictions \
            -i output/mixed_model/mixed_model/$model_name/$trait/$cvfold/predictions \
            --parent-table-file data/parent_table \
            --train-indices-file output/fastlmm/cv_index.cross:/$cvfold/train \
            --test-indices-file output/fastlmm/cv_index.cross:/$cvfold/test \
            -o output/mixed_model/mixed_model/$model_name/$trait/$cvfold/predictions.pdf
    done
done
```

## Predict phenotypes using mixed model (with CV)
```bash
model_name=ridge
{
for trait in $traits;do
    for cvfold in $cvfolds;do
        echo bin/run_mixed_model.py mixed_model \
            -a output/mixed_model/predict_residuals/$model_name/$trait/$cvfold/predictions \
            -b output/mixed_model/residuals/$model_name/$trait/$cvfold/predictions \
            --phenotype-file data/phenotypes/all:${trait} \
            --sample-indices-file output/fastlmm/cv_index.cross:/$cvfold/train \
            --parent-table-file data/parent_table \
            --model-name linear_cv \
            -o output/mixed_model/mixed_model_cv/$model_name/$trait/$cvfold
    done
done
} | parallel -P 4
```
## Evaluate predictions of the mixed model (with CV)
```bash
for trait in $traits;do
    for cvfold in $cvfolds;do
        bin/run_mixed_model.py evaluate \
            -i output/mixed_model/mixed_model_cv/$model_name/$trait/$cvfold/predictions \
            --sample-indices-file output/fastlmm/cv_index.cross:/$cvfold/test \
            -o output/mixed_model/mixed_model_cv/$model_name/$trait/$cvfold/metrics.txt
    done
done
```



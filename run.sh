# CMNIST
## Filter training
python train.py --experiment CMNIST_filter --name NAME --biased_var -1 \
    --mi 10.0 --gc 100.0 --dc 100.0 --gr 100.0 

## Filter evaluation
### Ours
python eval.py --experiment CMNIST_downstream_our --name NAME --biased_var 0 \
    --filter_train_mode universal
python eval.py --experiment CMNIST_downstream_our --name NAME --biased_var -1 \
    --filter_train_mode universal

### Baseline
python eval.py --experiment CMNIST_downstream_baseline --name NAME --biased_var 0
python eval.py --experiment CMNIST_downstream_baseline --name NAME --biased_var -1


# CelebA
## Filter training
python train.py --experiment CelebA_filter --name NAME --CelebA_train_mode CelebA_FFHQ \
    --mi 50.0 --gc 50.0 --dc 50.0 --gr 100.0 \
    --shortcut_layers 1 --inject_layers 1 --enc_layers 5 --dec_layers 5 --dis_layers 5

## Filter evaluation
python eval.py --experiment CelebA_downstream_our --name NAME --CelebA_train_mode CelebA_train_ex \
    --attributes Blond_Hair --CelebA_test_mode unbiased_ex --filter_train_mode universal
python eval.py --experiment CelebA_downstream_our --name NAME --CelebA_train_mode CelebA_train_ex \
    --attributes Blond_Hair --CelebA_test_mode conflict_ex --filter_train_mode universal

### Baseline
python eval.py --experiment CelebA_downstream_baseline --name NAME \
    --attributes Blond_Hair --CelebA_test_mode unbiased_ex
python eval.py --experiment CelebA_downstream_baseline --name NAME \
    --attributes Blond_Hair --CelebA_test_mode conflict_ex


# Adult
## Filter training
python train.py --experiment Adult_filter --name NAME --Adult_train_mode all \
    --mi 5.0 --gc 5.0 --dc 5.0 --gr 10.0 --epochs 100

## Filter evaluation
### Ours
python eval.py --experiment Adult_downstream_our --name NAME --Adult_train_mode eb1_balanced --Adult_test_mode eb2_balanced --filter_train_mode universal
python eval.py --experiment Adult_downstream_our --name NAME --Adult_train_mode eb1_balanced --Adult_test_mode balanced --filter_train_mode universal
python eval.py --experiment Adult_downstream_our --name NAME --Adult_train_mode eb2_balanced --Adult_test_mode eb1_balanced --filter_train_mode universal
python eval.py --experiment Adult_downstream_our --name NAME --Adult_train_mode eb2_balanced --Adult_test_mode balanced --filter_train_mode universal

### Baseline
python eval.py --experiment Adult_downstream_baseline --name NAME --Adult_train_mode eb1_balanced --Adult_test_mode eb2_balanced 
python eval.py --experiment Adult_downstream_baseline --name NAME --Adult_train_mode eb1_balanced --Adult_test_mode balanced 
python eval.py --experiment Adult_downstream_baseline --name NAME --Adult_train_mode eb2_balanced --Adult_test_mode eb1_balanced 
python eval.py --experiment Adult_downstream_baseline --name NAME --Adult_train_mode eb2_balanced --Adult_test_mode balanced 


# IMDB
## Filter training
python train.py --experiment IMDB_filter --name NAME --IMDB_train_mode all \
    --mi 50.0 --gc 50.0 --dc 50.0 --gr 100.0 \
    --shortcut_layers 1 --inject_layers 1 --enc_layers 5 --dec_layers 5 --dis_layers 5

## Filter evaluation
### Ours
python eval.py --experiment IMDB_downstream_our --name NAME --IMDB_train_mode eb1 --IMDB_test_mode eb2 --filter_train_mode universal
python eval.py --experiment IMDB_downstream_our --name NAME --IMDB_train_mode eb1 --IMDB_test_mode unbiased --filter_train_mode universal
python eval.py --experiment IMDB_downstream_our --name NAME --IMDB_train_mode eb2 --IMDB_test_mode eb1 --filter_train_mode universal
python eval.py --experiment IMDB_downstream_our --name NAME --IMDB_train_mode eb2 --IMDB_test_mode unbiased --filter_train_mode universal

### Baseline
python eval.py --experiment IMDB_downstream_baseline --name NAME --IMDB_train_mode eb1 --IMDB_test_mode eb2
python eval.py --experiment IMDB_downstream_baseline --name NAME --IMDB_train_mode eb1 --IMDB_test_mode unbiased
python eval.py --experiment IMDB_downstream_baseline --name NAME --IMDB_train_mode eb2 --IMDB_test_mode eb1
python eval.py --experiment IMDB_downstream_baseline --name NAME --IMDB_train_mode eb2 --IMDB_test_mode unbiased








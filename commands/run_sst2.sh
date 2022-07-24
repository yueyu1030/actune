task=SST-2
######## Experiment Setups (GPU, Random Seed etc.) 
seed=0
gpu=0
n_gpu=1
method=active_selftrain
######## Training Setups (bsz, learning rate etc.) 
max_seq_len=64
batch_size=8
self_training_batch_size=32
eval_batch_size=512
labels=50
rounds=10
dev_labels=1000
steps=5
tsb_logging_steps=20
logging_steps=50
st_logging_steps=30
self_training_max_steps=1500
epochs=10
lr=2e-5
self_training_weight=0.5
gce_loss=0
gce_loss_q=0.7

if [ $method == 'active_selftrain' ]; then
	eps=0.7
	pool=0.4
else  # not used
	eps=0
	pool=0
fi
soft_label=1
model_type=roberta-base

pool_scheduler=1
gamma_scheduler=1
pool=3000
pool_min=6000
self_training_weight=0.5
eps=0.6
gamma=1
n_centroids=25 # number of clusters for region-aware sampling
beta=0.5	   # weight for region-aware sampling
prob=1
gamma=1
weight_embedding=1
al_method=region_entropy
if [ ${method} == "active_selftrain" ]; then
    expname=${task}_${model_type}_${method}_lr${lr}_pool${pool}_smoothprob${prob}_gamma${gamma}_weight${self_training_weight}_eps${eps}_soft${soft_label}_${al_method}_seed${seed}
elif [ ${method} == "finetune" ]; then
	expname=${task}_${model_type}_${method}_lr${lr}_${al_method}_seed${seed}
fi

if [ ${al_method} == "region_entropy" ] || [ ${al_method} == "region_cal" ] ; then 
	sample_per_group=15
	expname="${expname}_beta${beta}_w${weight_embedding}_k${n_centroids}_sample${sample_per_group}"
	region_command="--region_beta=${beta} --sample_per_group=${sample_per_group}"
else 
	region_command=""
fi
if [ ${pool_scheduler} == 1 ]; then
	expname="${expname}_poolmin${pool_min}"
	pool_command="--pool_scheduler=1 --pool_min=${pool_min}"
else
	pool_command=""
fi
if [ ${gamma_scheduler} == 1 ]; then
	expname="${expname}_gammamin${gamma_min}"
	gamma_command="--gamma_scheduler=1 --gamma_min=${gamma_min}"
else
	gamma_command=""
fi
######### path for saving the results
tsb_dir=../exp/active_self_training/tsb/${expname}
output_dir=../datasets/${task}-${labels}-${seed}

rm -r ${tsb_dir}
mkdir -p ${tsb_dir}
mkdir -p ${output_dir}
echo ${method}
train_cmd="CUDA_VISIBLE_DEVICES=${gpu} python3 main.py --do_train --do_eval --task=${task} \
	--train_file=train.json --dev_file=valid.json --test_file=test.json \
	--unlabel_file=unlabeled.json --data_dir="../datasets/${task}-${labels}-${seed}" --seed=${seed} \
	--output_dir=${output_dir} --tsb_dir=${tsb_dir} \
	--logging_steps=${logging_steps} --self_train_logging_steps=${st_logging_steps} --tsb_logging_steps=${tsb_logging_steps} \
	--sample_labels=${labels} --rounds=${rounds} --dev_labels=${dev_labels} \
	--gpu=${gpu} --n_gpu=${n_gpu} --num_train_epochs=${epochs} --weight_decay=1e-8 \
	--learning_rate=${lr} --model_type=${model_type} \
	--method=${method} --batch_size=${batch_size} --eval_batch_size=${eval_batch_size} --self_training_batch_size=${self_training_batch_size} \
	--max_seq_len=${max_seq_len} --auto_load=1 --pool=${pool} \
	--self_training_eps=${eps} --self_training_weight=${self_training_weight} \
	--self_training_max_step=${self_training_max_steps} --al_method=${al_method} \
	--gce_loss=${gce_loss} --gce_loss_q=${gce_loss_q} \
	--gamma=${gamma} --smooth_prob=${prob}  ${region_command} ${pool_command} ${gamma_command}"
echo $train_cmd
eval $train_cmd

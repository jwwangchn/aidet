#!/bin/bash
#------------------------------config-----------------------------------
model='dota_v014_centermap_net_r50_v1_trainval'
evaluation_set='test'
epoch=12
dataset='dota'

#------------------------------train-----------------------------------

if [ $1 == 1 ]
then
    # train but not debug
    echo "==== start no debug training, mode name = ${model} ===="
    ./tools/dist_train.sh configs/${dataset}/${model}.py 4
elif [ $1 == 2 ]
then
    # train and debug
    echo "==== start debug training, mode name = ${model} ===="
    export CUDA_VISIBLE_DEVICES=0
    python tools/train.py configs/${dataset}/${model}.py --gpus 1
elif [ $1 == 3 ]
then
    echo "==== start training + validation, mode name = ${model} ===="
    ./tools/dist_train.sh configs/${dataset}/${model}.py 4 --validate
elif [ $1 == 0 ]
then
    # skip training
    echo "==== skip training ===="
fi


#------------------------------inference and eval-----------------------------------
if [ $2 == 1 ]
then
    echo "==== start 4 GPU coco test, mode name = ${model} ===="
    mkdir -p results/${dataset}/${model}

    ./tools/dota/dist_dota_test.sh configs/${dataset}/${model}.py work_dirs/${model}/epoch_${epoch}.pth 4 --out results/${dataset}/${model}/coco_results.pkl --eval hbb obb --options submit_path=$(pwd)/results/dota/${model} annopath=$(pwd)/data/dota/v0/${evaluation_set}/labelTxt-v1.0/{:s}.txt imageset_file=$(pwd)/data/dota/v0/${evaluation_set}/${evaluation_set}set.txt excel=$(pwd)/results/dota/${model}/${model}_results.xlsx jsonfile_prefix=$(pwd)/results/${dataset}/${model}/${model}
elif [ $2 == 2 ]
then
    echo "==== start 1 GPU coco test, mode name = ${model} ===="
    export CUDA_VISIBLE_DEVICES=0
    mkdir -p results/${dataset}/${model}

    python tools/dota/dota_test.py configs/${dataset}/${model}.py work_dirs/${model}/epoch_${epoch}.pth --out results/${dataset}/${model}/coco_results.pkl --eval hbb obb --options submit_path=$(pwd)/results/dota/${model} annopath=$(pwd)/data/dota/v0/${evaluation_set}/labelTxt-v1.0/{:s}.txt imageset_file=$(pwd)/data/dota/v0/${evaluation_set}/${evaluation_set}set.txt excel=$(pwd)/results/dota/${model}/${model}_results.xlsx jsonfile_prefix=$(pwd)/results/${dataset}/${model}/${model}
elif [ $2 == 3 ]
then
    echo "==== start 1 GPU (evaluation sample) test, mode name = ${model} ===="
    export CUDA_VISIBLE_DEVICES=0
    mkdir -p results/${dataset}/${model}

    python tools/dota/dota_test.py configs/${dataset}/${model}.py work_dirs/${model}/epoch_${epoch}.pth --out results/${dataset}/${model}/coco_results.pkl --eval hbb obb --options submit_path=$(pwd)/results/dota/${model} annopath=$(pwd)/data/dota/v0/evaluation_sample/labelTxt-v1.0/{:s}.txt imageset_file=$(pwd)/data/dota/v0/evaluation_sample/evaluation_sample.txt excel=$(pwd)/results/dota/${model}/${model}_results.xlsx jsonfile_prefix=$(pwd)/results/${dataset}/${model}/${model}
elif [ $2 == 0 ]
then
    # read the results file
    echo "==== skip inference ===="
fi

#------------------------------load result file and evaluation-----------------------------------
if [ $3 == 1 ]
then
    echo "==== start loading results files and evaluation, mode name = ${model} ===="

    python tools/dota/dota_eval.py configs/${dataset}/${model}.py --results results/${dataset}/${model}/coco_results.pkl --eval hbb obb --options submit_path=$(pwd)/results/dota/${model} annopath=$(pwd)/data/dota/v0/${evaluation_set}/labelTxt-v1.0/{:s}.txt imageset_file=$(pwd)/data/dota/v0/${evaluation_set}/${evaluation_set}set.txt excel=$(pwd)/results/dota/${model}/${model}_results.xlsx PR_path=$(pwd)/results/${dataset}/${model}/PR jsonfile_prefix=$(pwd)/results/${dataset}/${model}/${model}
elif [ $3 == 2 ]
then
    python tools/dota/dota_eval.py configs/${dataset}/${model}.py --results results/${dataset}/${model}/coco_results.pkl --eval hbb obb --options submit_path=$(pwd)/results/dota/${model} annopath=$(pwd)/data/dota/v0/${evaluation_set}/labelTxt-v1.0/{:s}.txt imageset_file=$(pwd)/data/dota/v0/${evaluation_set}/${evaluation_set}set.txt excel=$(pwd)/results/dota/${model}/${model}_results.xlsx PR_path=$(pwd)/results/${dataset}/${model}/PR jsonfile_prefix=$(pwd)/results/${dataset}/${model}/${model} skip_format=True
    echo "==== skip evaluation ===="
elif [ $3 == 0 ]
then
    # read the results file
    echo "==== skip evaluation ===="
fi

echo "finish!!! mode name = ${model}"
# send the notification email
# cd ../wwtool
# python tools/utils/send_email.py

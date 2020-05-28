#!/usr/bin/env bash
for model in 'bc_v004_mask_rcnn_r50_v2_shanghai_roof' 'bc_v005_mask_rcnn_r50_v2_beijing_roof' 'bc_v006_mask_rcnn_r50_v2_haerbin_roof'
do
    for city in 'jinan' 'chengdu' 'beijing' 'shanghai' 'haerbin' 
    do
        echo "==== start evaluate on city: ${city}, model: ${model} ===="
        #------------------------------config-----------------------------------
        # model='bc_v003_mask_rcnn_r50_v2_chengdu_roof'
        epoch=12
        dataset='buildchange'

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

            ./tools/buildchange/dist_buildchange_test.sh configs/${dataset}/${model}.py work_dirs/${model}/epoch_${epoch}.pth 4 --out results/${dataset}/${model}/coco_results.pkl --eval bbox segm --options jsonfile_prefix=$(pwd)/results/${dataset}/${model}/${model}_${city} --evaluation_city ${city}
        elif [ $2 == 2 ]
        then
            echo "==== start 1 GPU coco test, mode name = ${model} ===="
            export CUDA_VISIBLE_DEVICES=0
            mkdir -p results/${dataset}/${model}

            python tools/buildchange/buildchange_test.py configs/${dataset}/${model}.py work_dirs/${model}/epoch_${epoch}.pth --out results/${dataset}/${model}/coco_results.pkl --eval bbox segm --options jsonfile_prefix=$(pwd)/results/${dataset}/${model}/${model}_${city} --evaluation_city ${city}
        elif [ $2 == 0 ]
        then
            # read the results file
            echo "==== skip inference, mode name = ${model} ===="
        fi


        # send the notification email
        # cd ../wwtool
        # python tools/utils/send_email.py
        echo "==== finish evaluate on city: ${city} ===="
    done
done
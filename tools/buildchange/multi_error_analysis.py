from coco_error_analysis import analyze_results


if __name__ == "__main__":
    models = ['bc_v002_mask_rcnn_r50_v2_jinan_roof'
              'bc_v003_mask_rcnn_r50_v2_chengdu_roof', 
              'bc_v004_mask_rcnn_r50_v2_shanghai_roof', 
              'bc_v005_mask_rcnn_r50_v2_beijing_roof', 
              'bc_v006_mask_rcnn_r50_v2_haerbin_roof']

    for model in models:
        city = model.split('_')[-2]

        result = f'./results/buildchange/{model}/{model}.segm.json'
        ann = f'data/buildchange/v2/coco/annotations/buildchange_v2_val_{city}.json'
        out_dir = f'results/buildchange/{model}/analysis'

        print(f"start processing {model}")
        analyze_results(result, ann, ['segm'], out_dir)
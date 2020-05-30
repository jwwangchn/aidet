from coco_error_analysis import analyze_results


if __name__ == "__main__":
    models = ['dota_v002_theta_obb_r50_v1_train', 'dota_v003_point_obb_r50_v1_train', 'dota_v004_h_obb_r50_v1_train']
    titles = ['$\theta$-based OBB', 'Point-based OBB', '$h$-based OBB']
    for title, model in zip(titles, models):
        city = model.split('_')[-2]

        result = f'./results/dota/{model}/{model}.dota.json'
        ann = f'data/dota/v1/coco/annotations/dota_test_v1_1.0_best_keypoint.json'
        out_dir = f'results/dota/{model}/analysis'

        print(f"start processing {model}")



        analyze_results(result, ann, ['segm'], out_dir, title=title)
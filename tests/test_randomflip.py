from mmdet.datasets.pipelines.transforms import RandomFlip
import wwtool
import numpy as np

if __name__ == '__main__':
    thetaobbs = [[200, 200, 300, 150, 45*np.pi/180], 
                [700, 800, 300, 200, 135*np.pi/180]]
    pointobbs = [wwtool.thetaobb2pointobb(thetaobb) for thetaobb in thetaobbs]

    img = wwtool.generate_image(1024, 1024)
    img_origin = img.copy()
    wwtool.imshow_rbboxes(img, thetaobbs, win_name='origin')

    print("origin: ", np.round(pointobbs, 0))

    pointobbs = np.array(pointobbs)
    random_flip = RandomFlip()
    pointobbs = random_flip.rbbox_flip(np.array(pointobbs), img.shape, 'horizontal').tolist()

    print("flipped: ", np.round(pointobbs, 0))

    flipped_thetaobbs = [wwtool.pointobb2thetaobb(pointobb) for pointobb in pointobbs]
    wwtool.imshow_rbboxes(img_origin, flipped_thetaobbs, win_name='flipped')
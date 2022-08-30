import cv2
import numpy as np
import glob
import argparse
import tqdm
import os


CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
           'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
           'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
           'train', 'tvmonitor')


def main(args):
    truth_ann = glob.glob(f'{args.truth_masks_path}/*.png')

    union_list = [0 for _ in range(len(CLASSES))]
    intersection_list = [0 for _ in range(len(CLASSES))]
    
    for ta in tqdm.tqdm(truth_ann):
        truth_mask = cv2.imread(ta)
        generated_mask = cv2.imread(ta.replace(args.truth_masks_path, args.bcos_masks_path))

        for ic, c in enumerate(CLASSES):
            truth = np.where(truth_mask == ic+1, 1, 0)
            gen = np.where(generated_mask == ic+1, 1, 0)
            if truth.sum() == 0:
                continue
            union = ((truth + gen) > 0).sum()
            intersection = ((truth + gen) > 1).sum()
            union_list[ic] += union
            intersection_list[ic] += intersection
            
#            print(CLASSES[ic], 'IoU =', intersection / union)
#        print('')

    # print overall class IoUs
    for ic, c in enumerate(CLASSES):
        iou = intersection_list[ic] / union_list[ic]
        print(f'{CLASSES[ic]}   IoU = {iou*100:.1f}')
        

            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth-masks-path", required=True, help="Directory containing reference .png annotations")
    parser.add_argument("--bcos-masks-path", required=True, help="Directory containing generated .png annotations")
    args = parser.parse_args()
        
    main(args)

        
            

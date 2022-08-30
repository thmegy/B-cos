import argparse
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import cv2
import tqdm
import imagesize



PALETTE = [(106, 0, 228), (119, 11, 32), (165, 42, 42), (0, 0, 192),
               (197, 226, 255), (0, 60, 100), (0, 0, 142), (255, 77, 255),
               (153, 69, 1), (120, 166, 157), (0, 182, 199), (0, 226, 252),
               (182, 182, 255), (0, 0, 230), (220, 20, 60), (163, 255, 0),
               (0, 82, 0), (3, 95, 161), (0, 80, 100), (183, 130, 88)]

cmaps = []
for ic in range(len(PALETTE)):
    colors = [(PALETTE[ic][0], PALETTE[ic][1], PALETTE[ic][2], c) for c in np.linspace(0,1,100)]
    cmaps.append( mcolors.LinearSegmentedColormap.from_list(f'mycmap{ic}', colors, N=5) )
    


def main(args):
    '''
    Visualise segmentation maps on original images
    '''
    
    # get names of images from detection dataset
    images_detection = glob.glob(f'{args.impath_detection}/*.jpg')[:50]

    for imd in tqdm.tqdm(images_detection):
        im_name = imd.split('/')[-1].replace('.jpg', '')
        seg_annot = cv2.imread(f'{args.segpath}/{im_name}.png')
        seg = seg_annot[:,:,0]
        
        # print mask on image and save
        image = cv2.imread(imd)
        plt.figure(figsize=(12, 12))
        plt.imshow(image[:, :, ::-1])
        
        for ic in np.unique(seg):
            if ic == 0:
                continue
            seg_tmp = np.where(seg==ic, True, False)
#            if seg_tmp.sum()>0:
#                print(ic)
            plt.imshow(seg_tmp, cmap=cmaps[ic-1], alpha=0.3)
    
        plt.axis('off')
        plt.savefig(imd.replace(args.impath_detection, f'{args.output}/'))

        


                
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--impath-detection", required=True, help="path to original detection dataset")
    parser.add_argument("--segpath", required=True, help="path to segmentation maps")
    parser.add_argument("--output", required=True, help="path to output images")
    args = parser.parse_args()

    os.makedirs(f'{args.output}/', exist_ok=True)
    
    main(args)


import argparse
from PIL import Image
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import cv2
import torch
from torch.autograd import Variable
from torchvision import transforms
import tqdm
import imagesize

#from experiments.Cracks.bcos.model import get_model
#from experiments.Cracks.bcos.experiment_parameters import exps
from experiments.voc.bcos.model import get_model
from experiments.voc.bcos.experiment_parameters import exps
from data.data_handler import Data
from data.data_transforms import AddInverse
from interpretability.utils import grad_to_img, explanation_mode
from project_utils import to_numpy, to_numpy_img
from data.data_transforms import MyToTensor



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
    Make segmentation masks of images extracted from a detection dataset (cropped bboxes), using the bcos interpretability method,
    then combine the masks to create an overall segmentation mask for each image in the detection dataset. 
    '''

    # load model and dataset (to get transforms)
    exp_params = exps["inception_v3"]
    exp_params['load_pretrained'] = True
#    data = Data("Cracks", only_test_loader=False, **exp_params)
    data = Data("voc", only_test_loader=False, **exp_params)
    data_loader = data.get_test_loader()

    model = get_model(exp_params).cuda()
    explanation_mode(model, True)
    t = transforms.Compose([transforms.Resize(320), MyToTensor()])
    
    # there is a mismatch in class indices between dataset used in the bcos training and what is expected in mmsegmentation --> two separate dicts
#    class_to_idx_mask = {'Arrachement_pelade':0,
#                         'Faiencage':1,
#                         'Nid_de_poule':2,
#                         'Transversale':3,
#                         'Longitudinale':4,
#                         'Reparation':5}
#
#    class_to_idx_bcos = {'Arrachement_pelade':0,
#                         'Faiencage':1,
#                         'Longitudinale':2,
#                         'Nid_de_poule':3,
#                         'Reparation':4,
#                         'Transversale':5}

    class_to_idx_mask = {'aeroplane':0,
                         'bicycle':1,
                         'bird':2,
                         'boat':3,
                         'bottle':4,
                         'bus':5,
                         'car':6,
                         'cat':7,
                         'chair':8,
                         'cow':9,
                         'diningtable':10,
                         'dog':11,
                         'horse':12,
                         'motorbike':13,
                         'person':14,
                         'pottedplant':15,
                         'sheep':16,
                         'sofa':17,
                         'train':18,
                         'tvmonitor':19}

    class_to_idx_bcos = {'aeroplane':0,
                         'bicycle':1,
                         'bird':2,
                         'boat':3,
                         'bottle':4,
                         'bus':5,
                         'car':6,
                         'cat':7,
                         'chair':8,
                         'cow':9,
                         'diningtable':10,
                         'dog':11,
                         'horse':12,
                         'motorbike':13,
                         'person':14,
                         'pottedplant':15,
                         'sheep':16,
                         'sofa':17,
                         'train':18,
                         'tvmonitor':19}

    # get names of images from detection dataset
    images_detection = glob.glob(f'{args.impath_detection}/*.jpg')

    for imd in tqdm.tqdm(images_detection):
        # get cropped images from a given detection image
        imd_name = imd.split('/')[-1]

        # prepare overall segmentation mask
        imd_size = imagesize.get(imd)
        segmentation_mask = np.zeros((imd_size[1], imd_size[0]))
        
        # loop on classes
        n_bbox = 0
        for c in class_to_idx_mask.keys():
            target_class = class_to_idx_bcos[c] # class id from bcos dataset
            output_class = class_to_idx_mask[c] # class id expected by dataset in mmsegmentation
            
            images_cropped = glob.glob(f'{args.impath_cropped}/{c}/{imd_name.replace(".jpg","")}*.jpg')
            n_bbox += len(images_cropped)

            # sort cropped images by area (descending order)
            crop_area = []
            for imc in images_cropped:
                size = imagesize.get(imc)
                crop_area.append(size[0]*size[1])
            crop_area = np.array(crop_area)
            idx_sort = np.argsort(crop_area)[::-1]
            sorted_crops = np.array(images_cropped)[idx_sort]

            segmentation_mask_class = np.zeros((imd_size[1], imd_size[0]))

            # extract segmentation mask for cropped image, in descending crop size
            for imc in sorted_crops:
                
                im = Image.open(imc)
                im_w, im_h = im.size
                im = t(im)[None,:,:,:].cuda()

                model.zero_grad()
                _im = Variable(AddInverse()(im), requires_grad=True)
                pred = model(_im)[0, :]
                pred[target_class].backward()
                w = _im.grad[0]

                mask = grad_to_img(_im[0], w, smooth=55, alpha_percentile=99.5) # get activation map
                mask = np.where(mask[:,:,3]>0.1, output_class+1, 0).astype(np.uint8) # keep only most activated pixels
                mask = cv2.resize(mask,
                                  None,
                                  fx=im_w / mask.shape[1],
                                  fy=im_h / mask.shape[0],
                                  interpolation=cv2.INTER_NEAREST)
                
                with open(imc.replace('.jpg', '.txt'), 'r') as f:
                    position = f.readlines()[0]
                    x1, y1, x2, y2 = position.split(' ')

                segmentation_mask_class[int(y1):int(y2), int(x1):int(x2)] += mask
                torch.cuda.empty_cache()
                

            segmentation_mask = np.where(segmentation_mask_class>0, output_class+1, segmentation_mask)

        if n_bbox == 0:
            print(f'\nWARNING: no cropped bbox found for {imd_name}')
        cv2.imwrite(imd.replace(args.impath_detection, f'{args.output}/').replace('.jpg', '.png'), segmentation_mask)
        


                
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--impath-detection", required=True, help="path to original detection dataset")
    parser.add_argument("--impath-cropped", required=True, help="path to classification dataset containing .jpg and corresponding .txt files with the coordinates of images")
    parser.add_argument("--output", required=True, help="path to output annotations and images")
    args = parser.parse_args()

    os.makedirs(f'{args.output}/', exist_ok=True)
    
    main(args)


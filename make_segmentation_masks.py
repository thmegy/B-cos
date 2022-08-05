import argparse
from PIL import Image
import glob
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import torch
from torch.autograd import Variable
from torchvision import transforms
import tqdm
import imagesize

from experiments.Cracks.bcos.model import get_model
from experiments.Cracks.bcos.experiment_parameters import exps
from data.data_handler import Data
from data.data_transforms import AddInverse
from interpretability.utils import grad_to_img, explanation_mode
from project_utils import to_numpy, to_numpy_img



def main(args):
    '''
    Make segmentation masks of images extracted from a detection dataset (cropped bboxes), using the bcos interpretability method,
    then combine the masks to create an overall segmentation mask for each image in the detection dataset. 
    '''

    # load model and dataset (to get transforms)
    exp_params = exps["inception_v3"]
    exp_params['load_pretrained'] = True
    data = Data("Cracks", only_test_loader=False, **exp_params)
    data_loader = data.get_test_loader()

    model = get_model(exp_params).cuda()

    class_to_idx = {'Arrachement_pelade':0,
                    'Faiencage':1,
                    'Nid_de_poule':2,
                    'Transversale':3,
                    'Longitudinale':4,
                    'Reparation':5}
    
    # get names of images from detection dataset
    images_detection = glob.glob(f'{args.impath_detection}/*.jpg')

    for imd in tqdm.tqdm(images_detection):
        # get cropped images from a given detection image
        imd_name = imd.split('/')[-1]
        images_cropped = glob.glob(f'{args.impath_cropped}/*/{imd_name.replace(".jpg","")}*.jpg')

        # prepare overall segmentation mask
        imd_size = imagesize.get(imd)
        segmentation_mask = np.zeros((imd_size[1], imd_size[0]))
        
        # extract segmentation mask for cropped image
        for imc in images_cropped:
            # get class index based on the name of the directory where is image is
            class_name = imc.split('/')[-2]
            target_class = class_to_idx[class_name]
            
            im = Image.open(imc)
            im_w, im_h = im.size
            im = data_loader.dataset.transform(im)[None,:,:,:].cuda()

            model.zero_grad()
            _im = Variable(AddInverse()(im), requires_grad=True)
            pred = model(_im)[0, :]
            pred[target_class].backward()
            w = _im.grad[0]

            mask = grad_to_img(_im[0], w, smooth=35, alpha_percentile=90) # get activation map
            mask = np.where(mask[:,:,3]>0.2, target_class+1, 0).astype(np.uint8) # keep only most activated pixels
            mask = cv2.resize(mask,
                               None,
                               fx=im_w / mask.shape[1],
                               fy=im_h / mask.shape[0],
                               interpolation=cv2.INTER_NEAREST)

            with open(imc.replace('.jpg', '.txt'), 'r') as f:
                position = f.readlines()[0]
                x1, y1, x2, y2 = position.split(' ')

            segmentation_mask[int(y1):int(y2), int(x1):int(x2)] = mask

            cv2.imwrite(imd.replace(args.impath_detection, args.output).replace('.jpg', '.png'), segmentation_mask)

            torch.cuda.empty_cache()

                
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--impath-detection", required=True, help="path to original detection dataset")
    parser.add_argument("--impath-cropped", required=True, help="path to classification dataset containing .jpg and corresponding .txt files with the coordinates of images")
    parser.add_argument("--output", required=True, help="path to output annotations and images")
    args = parser.parse_args()

    os.makedirs(f'{args.output}/', exist_ok=True)
    
    main(args)


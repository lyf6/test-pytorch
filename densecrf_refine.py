import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral, create_pairwise_gaussian
import os
from PIL import Image
from scipy.special import softmax
from tifffile import imread
from tifffile import imsave
np.set_printoptions(threshold=np.inf)

crose_dir = '/home/yf/disk/output/myann/seg_hrnet_w48_trainval_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484x2/test_results'
img_dir = '/home/yf/disk/myannotated_cd/rgbp_div'
save_dir = '/home/yf/disk/myannotated_cd/rgbp-res'
crose_ls = os.listdir(crose_dir)
for path in crose_ls:
    if '2017' in path:
        img_path = os.path.join(crose_dir, path)
        img = np.array(imread(img_path))
        rgb_path =  os.path.join(img_dir,path)
        # rgb = np.array(Image.open(rgb_path))
        rgb = np.array(imread(rgb_path))

        #print(rgb.shape)
        c, w, h = img.shape
        d =  dcrf.DenseCRF2D(w, h, c)
        # print(img.shape)
        #img = img.transpose(2, 1, 0)
        prob = softmax(img, axis = 0)
        #print(prob.shape)
        U = unary_from_softmax(prob)
        d.setUnaryEnergy(U)
        # This adds the color-independent term, features are the locations only
        # d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
        #                       normalization=dcrf.NORMALIZE_SYMMETRIC)
        feats_guassian = create_pairwise_gaussian(sdims=(10, 10), shape=(w, h))
        d.addPairwiseEnergy(feats_guassian, compat=3,kernel=dcrf.DIAG_KERNEL,
                               normalization=dcrf.NORMALIZE_SYMMETRIC)
        # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
        # d.addPairwiseBilateral(sxy=(20, 20), srgb=(13, 13, 13), rgbim=rgb,
        #                        compat=10,
        #                        kernel=dcrf.DIAG_KERNEL,
        #                        normalization=dcrf.NORMALIZE_SYMMETRIC)
        pairwise_energy = create_pairwise_bilateral(sdims=(10, 10), schan=(0.01,), img=rgb, chdim=2)
        d.addPairwiseEnergy(pairwise_energy, compat=10, kernel=dcrf.DIAG_KERNEL,normalization=dcrf.NORMALIZE_SYMMETRIC)
        # Run five inference steps.
        Q = d.inference(40)

        # Find out the most probable class for each pixel.
        MAP = np.argmax(Q, axis=0)

        # Convert the MAP (labels) back to the corresponding colors and save the image.
        # Note that there is no "unknown" here anymore, no matter what we had at first.
        #
        #imwrite(fn_output, MAP.reshape(img.shape))
        #print(MAP.shape)
        MAP = MAP.reshape((w, h))
        #print(MAP.shape)
        target_path = os.path.join(save_dir, path)
        # MAP = Image.fromarray(MAP)
        print(target_path)
        imsave(target_path, MAP)
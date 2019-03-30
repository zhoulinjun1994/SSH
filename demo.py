# ------------------------------------------
# SSH: Single Stage Headless Face Detector
# Demo
# by Mahyar Najibi
# ------------------------------------------

from __future__ import print_function
from SSH.test import detect
from argparse import ArgumentParser
import os
from utils.get_config import cfg_from_file, cfg, cfg_print
from utils.test_utils import visusalize_detections
from nms.nms_wrapper import nms
import caffe
import cv2
import os
import numpy as np
import time

def delete_overlap(dets, thresh):
    keep = nms(dets, thresh)
    return dets[keep, :]

def parser():
    parser = ArgumentParser('SSH Demo!')
    parser.add_argument('--im',dest='im_path',help='Path to the image',
                        default='data/demo/test.jpg',type=str)
    parser.add_argument('--gpu',dest='gpu_id',help='The GPU ide to be used',
                        default=5,type=int)
    parser.add_argument('--proto',dest='prototxt',help='SSH caffe test prototxt',
                        default='SSH/models/test_ssh.prototxt',type=str)
    parser.add_argument('--model',dest='model',help='SSH trained caffemodel',
                        default='output/ssh/wider_train/SSH_iter_21000.caffemodel',type=str)
    parser.add_argument('--out_path',dest='out_path',help='Output path for saving the figure',
                        default='data/demo',type=str)
    parser.add_argument('--cfg',dest='cfg',help='Config file to overwrite the default configs',
                        default='SSH/configs/wider_pyramid.yml',type=str)
    parser.add_argument('--probthresh',dest='probthresh',help='Output probability threshold',
                        default=0.55,type=float)
    return parser.parse_args()

if __name__ == "__main__":

    # Parse arguments
    args = parser()

    # Load the external config
    if args.cfg is not None:
        cfg_from_file(args.cfg)
    # Print config file
    cfg_print(cfg)

    # Loading the network
    cfg.GPU_ID = args.gpu_id
    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)
    assert os.path.isfile(args.prototxt),'Please provide a valid path for the prototxt!'
    assert os.path.isfile(args.model),'Please provide a valid path for the caffemodel!'

    print('Loading the network...', end="")
    net = caffe.Net(args.prototxt, args.model, caffe.TEST)
    net.name = 'SSH'
    print('Done!')

    # Read image
    assert os.path.isfile(args.im_path),'Please provide a path to an existing image!'
    pyramid = True if len(cfg.TEST.SCALES)>1 else False

    start = time.clock()
    # Perform detection
    cls_dets,_ = detect(net,args.im_path,visualization_folder=args.out_path,visualize=False,pyramid=pyramid)
    im = cv2.imread(args.im_path)
    L_block = 3
    W_block = 3
    cls_dets_small = []
    if im.shape[0] > 1600 or im.shape[1] > 1600:
        for l_ind in range(L_block + 1):
            for w_ind in range(W_block + 1):
                L_size = im.shape[0] / L_block
                W_size = im.shape[1] / W_block
                L_left = l_ind * (im.shape[0] - L_size) / L_block
                W_left = w_ind * (im.shape[1] - W_size) / W_block
                L_right = L_left + L_size
                W_right = W_left + W_size
                if l_ind == L_block:
                    L_right = im.shape[0]
                if w_ind == W_block:
                    W_right = im.shape[1]

                cv2.imwrite('tmp.jpg', im[L_left:L_right, W_left:W_right, :])
                cls_dets_tmp, _ = detect(net, 'tmp.jpg', visualization_folder=args.out_path, visualize=False, pyramid=pyramid)
                tmp = cls_dets_tmp[:, 0:4]
                tmp = tmp + np.array([W_left, L_left, W_left, L_left])
                cls_dets_small.append(np.concatenate((tmp, cls_dets_tmp[:, 4].reshape((-1, 1))), axis=1))
                os.system('rm tmp.jpg')
        for i in range((L_block + 1) * (W_block + 1)):
            cls_dets = np.concatenate((cls_dets, cls_dets_small[i]), axis=0)

    cls_dets = np.array(cls_dets, np.float32)
    cls_dets = delete_overlap(cls_dets, cfg.TEST.NMS_THRESH)
    inds = np.where(cls_dets[:, -1] >= args.probthresh)[0]
    end = time.clock()
    print('Execute Time: %s seconds' % (end - start))
    print('Face Num: %d' % inds.shape[0])
    imfname = os.path.basename(args.im_path)
    plt_name = os.path.splitext(imfname)[0] + '_detections_{}'.format(net.name)
    visusalize_detections(im, cls_dets, plt_name=plt_name, visualization_folder=args.out_path, thresh=args.probthresh)








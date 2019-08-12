import argparse
import os
import sys
import warnings


from matplotlib import pyplot as plt
plt.rcParams['image.origin'] = 'lower'
plt.rcParams['figure.figsize'] = (6,6)
plt.rcParams['figure.dpi'] = 300


from tensorflow.python import keras
from tensorflow.python.keras.preprocessing import image
import tensorflow as tf

import numpy as np
from tqdm import tqdm
import itertools
import pickle
import cv2

# Change these to absolute imports if you copy this script outside the keras_retinanet package.
from keras_retinanet import layers  # noqa: F401
from keras_retinanet import losses
from keras_retinanet import models
from keras_retinanet.callbacks import RedirectModel
from keras_retinanet.callbacks.eval import Evaluate
from keras_retinanet.models.retinanet import retinanet_bbox
from keras_retinanet.preprocessing.csv_generator import CSVGenerator
from keras_retinanet.preprocessing.sail_generator import SAIL_Generator
from keras_retinanet.preprocessing.sail_val_generator import SAIL_val_Generator
from keras_retinanet.preprocessing.kitti import KittiGenerator
from keras_retinanet.preprocessing.open_images import OpenImagesGenerator
from keras_retinanet.preprocessing.pascal_voc import PascalVocGenerator
from keras_retinanet.utils.anchors import make_shapes_callback
from keras_retinanet.utils.config import read_config_file, parse_anchor_parameters
from keras_retinanet.utils.keras_version import check_keras_version
from keras_retinanet.utils.model import freeze as freeze_model
from keras_retinanet.utils.transform import random_transform_generator
from keras_retinanet.utils.sail_eval_data_generator import SAIL_EvalDataGenerator
from keras_retinanet.utils.anchors import anchors_for_shape, compute_gt_annotations
from keras_retinanet.utils.visualization import draw_boxes, draw_annotations
from keras_retinanet.utils.eval import _get_annotations, _get_detections
from utils.service_defs import iou, rect_area





def exclude_redundant_labelbbox_pair(detected_bboxes, scores, iou_threshold=0.5):
    # bboxes here in x1,y1,x2,y2 format
    detected_bboxes_xywh_format = np.array([[x1, y1, x2 - x1, y2 - y1] for (x1, y1, x2, y2) in detected_bboxes])
    item_to_exclude = None

    rects1 = []
    scores1 = []
    rects2 = []
    scores2 = []
    i_values = []
    j_values = []
    for i, j in itertools.combinations(np.arange(detected_bboxes_xywh_format.shape[0]), 2):
        i_values.append(i)
        j_values.append(j)
        rects1.append(detected_bboxes_xywh_format[i])
        scores1.append(scores[i])
        rects2.append(detected_bboxes_xywh_format[j])
        scores2.append(scores[j])

    iou_values = iou(rects1, rects2)

    excluding_pair_idx = np.argmax(iou_values)
    iou_excluding = iou_values[excluding_pair_idx]
    if iou_excluding >= iou_threshold:
        # exclude one of them
        item_to_exclude = i_values[excluding_pair_idx] if scores1[excluding_pair_idx] >= scores2[excluding_pair_idx] else j

    return item_to_exclude



def makedirs(path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


def model_with_weights(model, weights, skip_mismatch):
    if weights is not None:
        # model.load_weights(weights, by_name=True, skip_mismatch=skip_mismatch)
        model.load_weights(weights, by_name=True)
    return model





def create_test_generator(args):
    common_args = {
        'batch_size'       : args.batch_size,
        'config'           : args.config,
        'image_min_side'   : args.image_min_side,
        'image_max_side'   : args.image_max_side,
    }

    if args.annotations:
        if 'val_steps' in dir(args):
            validation_generator = SAIL_val_Generator(args.val_annotations,
                                                      args.classes,
                                                      transform_generator=None,
                                                      val_steps=args.val_steps,
                                                      **common_args)
        else:
            validation_generator = SAIL_val_Generator(args.val_annotations,
                                                      args.classes,
                                                      transform_generator=None,
                                                      **common_args)
    return validation_generator





def create_models(backbone_retinanet, num_classes, weights, multi_gpu=0,
                  freeze_backbone=False, lr=1e-5, config=None):
    """ Creates three models (model, training_model, prediction_model).

    Args
        backbone_retinanet : A function to call to create a retinanet model with a given backbone.
        num_classes        : The number of classes to train.
        weights            : The weights to load into the model.
        multi_gpu          : The number of GPUs to use for training.
        freeze_backbone    : If True, disables learning for the backbone.
        config             : Config parameters, None indicates the default configuration.

    Returns
        model            : The base model. This is also the model that is saved in snapshots.
        training_model   : The training model. If multi_gpu=0, this is identical to model.
        prediction_model : The model wrapped with utility functions to perform object detection (applies regression values and performs NMS).
    """

    modifier = freeze_model if freeze_backbone else None

    # load anchor parameters, or pass None (so that defaults will be used)
    anchor_params = None
    num_anchors   = None
    if config and 'anchor_parameters' in config:
        anchor_params = parse_anchor_parameters(config)
        num_anchors   = anchor_params.num_anchors()

    # Keras recommends initialising a multi-gpu model on the CPU to ease weight sharing, and to prevent OOM errors.
    # optionally wrap in a parallel model
    if multi_gpu > 1:
        from tensorflow._api.v1.keras.utils import multi_gpu_model
        with tf.device('/cpu:0'):
            model = model_with_weights(backbone_retinanet(num_classes, num_anchors=num_anchors, modifier=modifier), weights=weights, skip_mismatch=True)
        training_model = multi_gpu_model(model, gpus=multi_gpu)
    else:
        model = model_with_weights(backbone_retinanet(num_classes, num_anchors=num_anchors, modifier=modifier), weights=weights, skip_mismatch=True)
        training_model = model

    # make prediction model
    prediction_model = retinanet_bbox(model=model, anchor_params=anchor_params)

    # compile model
    training_model.compile(
        loss={
            'regression'    : losses.smooth_l1(),
            'classification': losses.focal()
        },
        optimizer=keras.optimizers.Adam(lr=lr, clipnorm=0.001)
    )

    return model, training_model, prediction_model






def parse_args(args):
    """ Parse the arguments.
    """
    parser     = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
    subparsers = parser.add_subparsers(help='Arguments for specific dataset types.', dest='dataset_type')
    subparsers.required = True

    def csv_list(string):
        return string.split(',')

    csv_parser = subparsers.add_parser('csv')
    csv_parser.add_argument('annotations', help='Path to CSV file containing annotations for training.')
    csv_parser.add_argument('classes', help='Path to a CSV file containing class label mapping.')
    csv_parser.add_argument('--val-annotations', help='Path to CSV file containing annotations for validation (optional).')

    #region +mk@sail
    sail_parser = subparsers.add_parser('sail')
    sail_parser.add_argument('annotations', help='Path to pickle-file containing annotations for training.')
    sail_parser.add_argument('classes', help='Path to a CSV file containing class label mapping.')
    sail_parser.add_argument('--val-annotations', help='Path to pickle-file containing annotations for validation (optional).')
    #endregion +mk@sail

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--snapshot',          help='Resume training from a snapshot.')
    group.add_argument('--imagenet-weights',  help='Initialize the model with pretrained imagenet weights. This is the default behaviour.', action='store_const', const=True, default=True)
    group.add_argument('--weights',           help='Initialize the model with weights from a file.')
    group.add_argument('--no-weights',        help='Don\'t initialize the model with any weights.', dest='imagenet_weights', action='store_const', const=False)

    parser.add_argument('--backbone',         help='Backbone model used by retinanet.', default='resnet50', type=str)
    parser.add_argument('--batch-size',       help='Size of the batches.', default=1, type=int)
    parser.add_argument('--val-batch-size',   help='Size of the batches for evaluation.', default=32, type=int)
    parser.add_argument('--gpu',              help='Id of the GPU to use (as reported by nvidia-smi).')
    parser.add_argument('--multi-gpu',        help='Number of GPUs to use for parallel processing.', type=int, default=0)
    parser.add_argument('--multi-gpu-force',  help='Extra flag needed to enable (experimental) multi-gpu support.', action='store_true')
    parser.add_argument('--epochs',           help='Number of epochs to train.', type=int, default=50)
    parser.add_argument('--steps',            help='Number of steps per epoch.', type=int, default=10000)
    parser.add_argument('--val-steps',        help='Number of steps per validation run.', type=int, default=100)
    parser.add_argument('--lr',               help='Learning rate.', type=float, default=1e-5)
    parser.add_argument('--snapshot-path',    help='Path to store snapshots of models during training (defaults to \'./snapshots\')', default='./snapshots')
    parser.add_argument('--tensorboard-dir',  help='Log directory for Tensorboard output', default='./logs')
    parser.add_argument('--no-snapshots',     help='Disable saving snapshots.', dest='snapshots', action='store_false')
    parser.add_argument('--no-evaluation',    help='Disable per epoch evaluation.', dest='evaluation', action='store_false')
    parser.add_argument('--freeze-backbone',  help='Freeze training of backbone layers.', action='store_true')
    parser.add_argument('--random-transform', help='Randomly transform image and annotations.', action='store_true')
    parser.add_argument('--image-min-side',   help='Rescale the image so the smallest side is min_side.', type=int, default=800)
    parser.add_argument('--image-max-side',   help='Rescale the image if the largest side is larger than max_side.', type=int, default=1333)
    parser.add_argument('--config',           help='Path to a configuration parameters .ini file.')
    parser.add_argument('--weighted-average', help='Compute the mAP using the weighted average of precisions among classes.', action='store_true')
    parser.add_argument('--proba-threshold', help='threshold on the probability of detected objects', type=float)
    # parser.add_argument('--shrinking-iou-threshold', help='threshold on the iou of two bboxes at the stage of reducing redundant detectedions', default=0.5, type=float)
    parser.add_argument('--shrinking-proba-perc-threshold', help='threshold on the probability in terms of percentile of scores of current image detections', default=90, type=float)

    # Fit generator arguments
    parser.add_argument('--workers', help='Number of multiprocessing workers. To disable multiprocessing, set workers to 0', type=int, default=1)
    parser.add_argument('--max-queue-size', help='Queue length for multiprocessing workers in fit generator.', type=int, default=10)

    return parser.parse_args(args)





def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    # optionally choose specific GPU
    # if args.gpu:
    #     os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # create the generators
    test_generator = create_test_generator(args)

    if os.path.exists('/app/Ant_mcs_detection/output/test_set_all_detections.pkl'):
        with open('/app/Ant_mcs_detection/output/test_set_all_detections.pkl', 'rb') as f:
            all_detections = pickle.load(f)
    else:
        with tf.device('/device:GPU:3'):
            config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, device_count={'CPU': 1, 'GPU': 1})
            config.gpu_options.allow_growth = True
            session = tf.Session(config=config)
            keras.backend.set_session(session)

        print('Creating model, this may take a second...')
        model = models.load_model(os.path.abspath(args.weights), backbone_name=args.backbone)
        model = models.convert_model(model)

        # print model summary
        print(model.summary())

        score_threshold = 0.05
        max_detections = 0
        save_path = None
        eval_batch_size = 16

        all_detections = _get_detections(test_generator, model,
                                         score_threshold=score_threshold,
                                         max_detections=max_detections,
                                         save_path=save_path,
                                         eval_batch_size=eval_batch_size)
        with open('/app/Ant_mcs_detection/output/test_set_all_detections.pkl', 'wb') as f:
            pickle.dump(all_detections, f)


    all_annotations = _get_annotations(test_generator, predicted_items_count=len(all_detections))


    proba_percentiles = np.linspace(10, 99, 90)
    iou_values_per_example = []
    for example_idx in tqdm(range(len(all_annotations)), total=len(all_annotations)):
        image       = np.copy(test_generator.load_image(example_idx))
        _, example_mask, _, _ = test_generator.data_manager.get_all_data(test_generator.image_names[example_idx])
        example_mask = np.copy(example_mask)

        image = np.copy(image)

        image[:, :, 0] = 255 - image[:, :, 0]
        image[:, :, 1] = 255 - image[:, :, 1]
        # image = np.ma.array(image)
        image_mask = np.tile(1-example_mask, (1,1,3))
        image = image * image_mask
        annotations = test_generator.load_annotations(example_idx)
        anchors = anchors_for_shape(image.shape, anchor_params=None)
        positive_indices, _, max_indices = compute_gt_annotations(anchors, annotations['bboxes'])

        curr_detections = all_detections[example_idx][0]
        # selected_detections = curr_detections[curr_detections[:, -1] >= args.proba_threshold, :]
        # bboxes = selected_detections[:, :4]
        # probas = selected_detections[:, -1]
        bboxes = curr_detections[:, :4]
        probas = curr_detections[:, -1]

        # region filter detections


        bboxes_filtered = np.copy(bboxes)
        scores_filtered = np.copy(probas)
        # proba_threshold = np.percentile(scores_filtered, args.shrinking_proba_perc_threshold)
        proba_threshold = args.proba_threshold
        filter_indices = np.where(scores_filtered >= proba_threshold)[0]
        bboxes_filtered = bboxes_filtered[filter_indices, :]
        scores_filtered = scores_filtered[filter_indices]


        #region IR_only
        f = plt.figure(figsize=(6,6), dpi=300)
        plt.imshow(image[:,:,0], cmap='gray', vmin=0, vmax=255)
        for i in range(annotations['bboxes'].shape[0]):
            label = annotations['labels'][i]
            box = annotations['bboxes'][i]
            x1, y1, x2, y2 = np.array(box).astype(int)
            plt.plot([x1, x1, x2, x2, x1], [y1, y2, y2, y1, y1], color='red', linewidth=2)

        for bbox, proba in zip(bboxes_filtered, scores_filtered):
            x1, y1, x2, y2 = bbox.astype(int)
            plt.plot([x1, x1, x2, x2, x1], [y1, y2, y2, y1, y1], color='lightgreen', linewidth=2, alpha = proba/probas.max())
            plt.text(x1, y1, '%.3f' % proba, fontsize = 10)
        plt.axis('off')
        plt.savefig(os.path.join('/app/Ant_mcs_detection/output/images_IRonly/',
                                 'test_img_IR_withLabels_%05d.png' % example_idx))
        plt.close()
        #endregion IR_only

        # region IR_WV_SLP
        f = plt.figure(figsize=(6, 6), dpi=300)
        plt.imshow(image)
        for i in range(annotations['bboxes'].shape[0]):
            label = annotations['labels'][i]
            box = annotations['bboxes'][i]
            x1, y1, x2, y2 = np.array(box).astype(int)
            plt.plot([x1, x1, x2, x2, x1], [y1, y2, y2, y1, y1], color='red', linewidth=2)

        for bbox, proba in zip(bboxes_filtered, scores_filtered):
            x1, y1, x2, y2 = bbox.astype(int)
            plt.plot([x1, x1, x2, x2, x1], [y1, y2, y2, y1, y1], color='lightgreen', linewidth=2, alpha=proba / probas.max())
            plt.text(x1, y1, '%.3f' % proba, fontsize=10)
        plt.axis('off')
        plt.savefig(os.path.join('/app/Ant_mcs_detection/output/images_IR_WV_SLP/',
                                 'test_img_IR_WV_SLP_withLabels_%05d.png' % example_idx))
        plt.close()
        #endregion IR_WV_SLP

        # region IR_WV_SLP
        # f = plt.figure(figsize=(10, 10), dpi=300)
        #
        # _ = plt.subplot(3,3,1)
        # _ = plt.imshow(image[:,:,0], cmap='gray', vmin=0, vmax=255)
        # _ = plt.title('IR')
        # _ = plt.axis('off')
        #
        # _ = plt.subplot(3, 3, 2)
        # _ = plt.imshow(image[:, :, 1], cmap='gray', vmin=0, vmax=255)
        # _ = plt.title('WV')
        # _ = plt.axis('off')
        #
        # _ = plt.subplot(3, 3, 3)
        # _ = plt.imshow(image[:, :, 2], cmap='gray', vmin=0, vmax=255)
        # _ = plt.title('SLP')
        # _ = plt.axis('off')
        #
        # _ = plt.subplot(3, 3, 4)
        # _ = plt.imshow(image)
        # _ = plt.title('\"RGB\"-like')
        # _ = plt.axis('off')
        #
        # _ = plt.subplot(3, 3, 5)
        # _ = plt.imshow(image[:,:,0], cmap='gray', vmin=0, vmax=255)
        # for i in range(annotations['bboxes'].shape[0]):
        #     label = annotations['labels'][i]
        #     box = annotations['bboxes'][i]
        #     x1, y1, x2, y2 = np.array(box).astype(int)
        #     plt.plot([x1, x1, x2, x2, x1], [y1, y2, y2, y1, y1], color='red', linewidth=2)
        # _ = plt.axis('off')
        #
        # _ = plt.subplot(3, 3, 6)
        # _ = plt.imshow(image)
        # for i in range(annotations['bboxes'].shape[0]):
        #     label = annotations['labels'][i]
        #     box = annotations['bboxes'][i]
        #     x1, y1, x2, y2 = np.array(box).astype(int)
        #     plt.plot([x1, x1, x2, x2, x1], [y1, y2, y2, y1, y1], color='red', linewidth=2)
        # _ = plt.axis('off')
        #
        # _ = plt.subplot(3, 3, 7)
        # _ = plt.imshow(image)
        # for bbox, proba in zip(bboxes_filtered, scores_filtered):
        #     x1, y1, x2, y2 = bbox.astype(int)
        #     plt.plot([x1, x1, x2, x2, x1], [y1, y2, y2, y1, y1], color='lightgreen', linewidth=2, alpha=proba / probas.max())
        #     plt.text(x1, y1, '%.3f' % proba, fontsize=10)
        # _ = plt.axis('off')
        #
        # _ = plt.subplot(3, 3, 8)
        # _ = plt.imshow(image[:, :, 0], cmap='gray', vmin=0, vmax=255)
        # for i in range(annotations['bboxes'].shape[0]):
        #     label = annotations['labels'][i]
        #     box = annotations['bboxes'][i]
        #     x1, y1, x2, y2 = np.array(box).astype(int)
        #     plt.plot([x1, x1, x2, x2, x1], [y1, y2, y2, y1, y1], color='red', linewidth=2)
        # for bbox, proba in zip(bboxes_filtered, scores_filtered):
        #     x1, y1, x2, y2 = bbox.astype(int)
        #     plt.plot([x1, x1, x2, x2, x1], [y1, y2, y2, y1, y1], color='lightgreen', linewidth=2, alpha=proba / probas.max())
        #     plt.text(x1, y1, '%.3f' % proba, fontsize=10)
        # _ = plt.axis('off')
        #
        # _ = plt.subplot(3, 3, 9)
        # _ = plt.imshow(image)
        # for i in range(annotations['bboxes'].shape[0]):
        #     label = annotations['labels'][i]
        #     box = annotations['bboxes'][i]
        #     x1, y1, x2, y2 = np.array(box).astype(int)
        #     plt.plot([x1, x1, x2, x2, x1], [y1, y2, y2, y1, y1], color='red', linewidth=2)
        # for bbox, proba in zip(bboxes_filtered, scores_filtered):
        #     x1, y1, x2, y2 = bbox.astype(int)
        #     plt.plot([x1, x1, x2, x2, x1], [y1, y2, y2, y1, y1], color='lightgreen', linewidth=2, alpha=proba / probas.max())
        #     plt.text(x1, y1, '%.3f' % proba, fontsize=10)
        # _ = plt.axis('off')
        #
        # plt.savefig(os.path.join('/app/Ant_mcs_detection/output/all_in_one/',
        #                          'test_img_%05d.png' % example_idx))
        # plt.close()
        # endregion IR_WV_SLP

        # if example_idx>=31:
        #     break














if __name__ == '__main__':
    main()
    print('finished')
    quit()

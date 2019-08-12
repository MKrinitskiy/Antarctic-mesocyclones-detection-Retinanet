import argparse
import os
import sys
import warnings

from tensorflow._api.v1 import keras
from tensorflow._api.v1.keras.preprocessing import image
import tensorflow as tf


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


def makedirs(path):
    # Intended behavior: try to create the directory,
    # pass if the directory exists already, fails otherwise.
    # Meant for Python 2.7/3.n compatibility.
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


# def get_session():
#     """ Construct a modified tf session.
#     """
#     config = tf.ConfigProto()
#     config.gpu_options.allow_growth = True
#     return tf.Session(config=config)


def model_with_weights(model, weights, skip_mismatch):
    """ Load weights for model.

    Args
        model         : The model to load weights for.
        weights       : The weights to load.
        skip_mismatch : If True, skips layers whose shape of weights doesn't match with the model.
    """
    if weights is not None:
        # model.load_weights(weights, by_name=True, skip_mismatch=skip_mismatch)
        model.load_weights(weights, by_name=True)
    return model





def create_val_generator(args):
    common_args = {
        'batch_size'       : args.batch_size,
        'config'           : args.config,
        'image_min_side'   : args.image_min_side,
        'image_max_side'   : args.image_max_side,
    }

    if args.val_annotations:
        if 'val_steps' in dir(args):
            validation_generator = SAIL_val_Generator(args.val_annotations,
                                                  args.classes,
                                                  val_steps=args.val_steps,
                                                  **common_args)
        else:
            validation_generator = SAIL_val_Generator(args.val_annotations,
                                                  args.classes,
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




def check_args(parsed_args):
    """ Function to check for inherent contradictions within parsed arguments.
    For example, batch_size < num_gpus
    Intended to raise errors prior to backend initialisation.

    Args
        parsed_args: parser.parse_args()

    Returns
        parsed_args
    """

    # if parsed_args.multi_gpu > 1 and parsed_args.batch_size < parsed_args.multi_gpu:
    #     raise ValueError(
    #         "Batch size ({}) must be equal to or higher than the number of GPUs ({})".format(parsed_args.batch_size,
    #                                                                                          parsed_args.multi_gpu))
    #
    # if parsed_args.multi_gpu > 1 and parsed_args.snapshot:
    #     raise ValueError(
    #         "Multi GPU training ({}) and resuming from snapshots ({}) is not supported.".format(parsed_args.multi_gpu,
    #                                                                                             parsed_args.snapshot))
    #
    # if parsed_args.multi_gpu > 1 and not parsed_args.multi_gpu_force:
    #     raise ValueError("Multi-GPU support is experimental, use at own risk! Run with --multi-gpu-force if you wish to continue.")
    #
    # if 'resnet' not in parsed_args.backbone:
    #     warnings.warn('Using experimental backbone {}. Only resnet50 has been properly tested.'.format(parsed_args.backbone))

    return parsed_args




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

    # Fit generator arguments
    parser.add_argument('--workers', help='Number of multiprocessing workers. To disable multiprocessing, set workers to 0', type=int, default=1)
    parser.add_argument('--max-queue-size', help='Queue length for multiprocessing workers in fit generator.', type=int, default=10)

    return check_args(parser.parse_args(args))





def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    # optionally choose specific GPU
    # if args.gpu:
    #     os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with tf.device('/device:GPU:3'):
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, device_count={'CPU': 1, 'GPU': 1})
        config.gpu_options.allow_growth = True
        session = tf.Session(config=config)
        keras.backend.set_session(session)

    # create the generators
    val_generator = create_val_generator(args)


    weights = args.weights
    # default to imagenet if nothing else is specified

    backbone = models.backbone(args.backbone)

    print('Creating model, this may take a second...')
    # model = model_with_weights(backbone(val_generator.num_classes(), num_anchors=None, modifier=freeze_model), weights=weights, skip_mismatch=True)
    model, training_model, prediction_model = create_models(
        backbone_retinanet=backbone.retinanet,
        num_classes=val_generator.num_classes(),
        weights=weights,
        multi_gpu=args.multi_gpu,
        freeze_backbone=args.freeze_backbone,
        lr=args.lr,
        config=args.config
    )

    # print model summary
    print(model.summary())

    evaluation = Evaluate(val_generator, weighted_average=args.weighted_average, eval_batch_size=args.val_batch_size)
    evaluation = RedirectModel(evaluation, prediction_model)

    generator = val_generator
    iou_threshold = 0.5
    score_threshold = 0.05
    max_detections = 100
    save_path = None
    weighted_average = False
    verbose = True
    eval_batch_size = 16

    from keras_retinanet.utils.sail_eval_data_generator import SAIL_EvalDataGenerator
    eval_batch_datagen = SAIL_EvalDataGenerator(generator, batch_size=eval_batch_size)
    boxes, scores, labels = model.predict_generator(eval_batch_datagen, steps=len(eval_batch_datagen), verbose=True,
                                                    workers=0, use_multiprocessing=False)[:3]









if __name__ == '__main__':
    main()

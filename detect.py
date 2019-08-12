from inference_data_generator import *
from batches_data_generator import *
from args_parser import parse_args
import tensorflow as tf
from tensorflow.python import keras
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys
from utils import *
from tqdm import tqdm
import time


plt.rcParams['image.origin'] = 'lower'


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def exclude_redundant_labelbbox_pair(detected_bboxes, iou_threshold=0.5):
    # bboxes here in x1,y1,x2,y2 format
    detected_bboxes_xywh_format = np.array([[x1, y1, x2 - x1, y2 - y1] for (x1, y1, x2, y2) in detected_bboxes])
    item_to_exclude = None

    rects1 = []
    rects2 = []
    i_values = []
    j_values = []
    for i, j in itertools.combinations(np.arange(detected_bboxes_xywh_format.shape[0]), 2):
        i_values.append(i)
        j_values.append(j)
        rects1.append(detected_bboxes_xywh_format[i])
        rects2.append(detected_bboxes_xywh_format[j])

    iou_values = iou(rects1, rects2)

    excluding_pair_idx = np.argmax(iou_values)
    iou_excluding = iou_values[excluding_pair_idx]
    if iou_excluding >= iou_threshold:
        # exclude one of them
        item_to_exclude = i_values[excluding_pair_idx] if rect_area(rects1[excluding_pair_idx]) < rect_area(rects2[excluding_pair_idx]) else j

    return item_to_exclude




def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    generator = SAIL_inference_datagenerator(base_data_path = args.base_data_directory,
                                             interpolation_constants_directory=args.interpolation_constants_directory)

    # examples = []
    # data_fnames = []
    # sat_labels = []
    # datetimes = []
    # for i in range(args.batch_size):

    keras.backend.set_session(get_session())
    model = models.load_model(os.path.abspath(args.model_snapshot), backbone_name=args.backbone)
    model = models.convert_model(model)

    snapshots_processing_delta_t = np.zeros(len(generator), dtype=np.float64)

    for datafile_idx in range(len(generator)):
        start_time = time.time()

        if 'prev_start_time' in locals():
            delta_t = start_time - prev_start_time
            snapshots_processing_delta_t[datafile_idx-1] = delta_t
            print('prev. snapshot processed in %f s; estimated processing time: %f s' % (delta_t, ((len(generator)-(datafile_idx+1)) * np.mean(snapshots_processing_delta_t[snapshots_processing_delta_t > 0.1]))))
            prev_start_time = start_time
        else:
            prev_start_time = start_time

        curr_fname = generator.data_fnames[datafile_idx]
        curr_fname_basename = os.path.basename(curr_fname)
        reex = '.+(MSG\d).+(\d{14})\.nc'
        match = re.match(reex, curr_fname_basename)
        sat_label = match.groups()[0]
        if sat_label == 'MSG1':
            continue
        dt_str = match.groups()[1]
        dt = datetime.strptime(dt_str, "%Y%m%d%H%M%S")
        curr_snapshot_results_filename = os.path.join(args.output_directory,
                                                      datetime.strftime(dt, "%Y%m%d"),
                                                      '%s_%s_p%s.pkl' % (sat_label, datetime.strftime(dt, "%Y%m%d%H%M%S"), ('%.5f' % args.proba_threshold).replace('.', '_')))
        curr_snapshot_vis_plot_filename = os.path.join(args.output_directory,
                                                       datetime.strftime(dt, "%Y%m%d"),
                                                       '%s_%s_p%s.png' % (sat_label, datetime.strftime(dt, "%Y%m%d%H%M%S"), ('%.5f' % args.proba_threshold).replace('.', '_')))
        if os.path.exists(curr_snapshot_results_filename):
            generator.current += 1
            print('this file has been already processed earlier. Skipping.')
            continue


        example,shared_mask,crops,masks,data_fname,dt,crop_bboxes,sat_label = next(generator)
        print('%s : processing file %d of %d: %s' % (str(start_time), datafile_idx+1, len(generator), data_fname))

        # examples.append(crops)
        # data_fnames.append(data_fname)
        # sat_labels.append(sat_label)
        # datetimes.append(dt)
        # examples = np.concatenate(examples, axis=0)

        #region debug_plot
        # crop_ch5_normed = example[0, :, :, 0]
        #
        # f = plt.figure(figsize=(6,6), dpi=300)
        # im = plt.imshow(scale_ch5_back(crop_ch5_normed), cmap=cmap_ch5, vmin=200., vmax=320.)
        # for idx in range(len(crop_bboxes)):
        #     x1,y1,x2,y2 = crop_bboxes[idx]
        #     # p = plt.subplot(3, 3, idx+1)
        #     # ax = plt.gca()
        #     _ = plt.plot([x1,x1,x2,x2,x1], [y1,y2,y2,y1,y1], color='green')
        #
        # _ = plt.axis('off')
        # # _ = plt.title(str(datetimes[idx]))
        # plt.show()
        #endregion debug_plot

        curr_example_batch_generator = SAIL_batches_generator(crops, batch_size=args.batch_size)
        detected_boxes_per_crop = []
        scores_per_crop = []
        for batch_idx in range(len(curr_example_batch_generator)):
            images_batch, scales = next(curr_example_batch_generator)

            # prediction!
            boxes, scores, pred_labels = model.predict_on_batch(images_batch)
            boxes = [np.array([box for box in curr_boxes if np.square(box - np.array([-1., -1., -1., -1.])).sum() > 0.]) for curr_boxes in boxes]
            scores = [np.array([sc for sc in curr_scores if sc > -1.]) for curr_scores in scores]
            detected_boxes_per_crop = detected_boxes_per_crop + boxes
            scores_per_crop = scores_per_crop + scores

        if len(detected_boxes_per_crop) == 0:
            continue

        # translate these labels bboxes
        translated_detected_boxes_per_crop = [[box + np.array([l, b, l, b]) for box in curr_boxes] for (curr_boxes, (l, b, r, t)) in zip(detected_boxes_per_crop, crop_bboxes)]
        # flat this list
        translated_detected_boxes_per_crop = [box[np.newaxis, :] for boxes_of_crop in translated_detected_boxes_per_crop for box in boxes_of_crop]
        if len(translated_detected_boxes_per_crop) == 0:
            continue
        # concat to one array
        translated_detected_boxes_per_crop_flat = np.concatenate(translated_detected_boxes_per_crop, axis=0)
        # concat scores to one array
        scores_per_crop_flat = np.concatenate(scores_per_crop)

        # indices1 = np.where(scores_per_crop_flat<1.)[0]
        # translated_detected_boxes_per_crop_flat = translated_detected_boxes_per_crop_flat[indices1]
        # scores_per_crop_flat = scores_per_crop_flat[indices1]

        selected_indices = np.where((scores_per_crop_flat >= args.proba_threshold) & (scores_per_crop_flat<1.))[0]

        if len(selected_indices) > 30:
            print('adjusting proba_thresh...')
            curr_thresh = args.proba_threshold
            failed_searching_suitable_threshold = False
            while np.sum((scores_per_crop_flat >= curr_thresh)&(scores_per_crop_flat<1.)) > 30:
                curr_thresh = (1.- 0.98*(1-curr_thresh))
                print('%f : %d bboxes' % (curr_thresh, np.sum((scores_per_crop_flat >= curr_thresh)&(scores_per_crop_flat<1.))))
                if ((np.abs(curr_thresh-1.)<1.e-3) & (np.sum((scores_per_crop_flat >= curr_thresh)&(scores_per_crop_flat<1.)) > 30)):
                    failed_searching_suitable_threshold = True
                    break
            if failed_searching_suitable_threshold:
                print('failed searching suitable threshold. !!! Skipping this example !!!')
                continue
            selected_indices = np.where((scores_per_crop_flat >= curr_thresh)&(scores_per_crop_flat<1.))[0]
        translated_detected_boxes_per_crop_flat_filtered = translated_detected_boxes_per_crop_flat[selected_indices]
        scores_per_crop_flat_filtered = scores_per_crop_flat[selected_indices]

        translated_detected_boxes_shrinked = np.copy(translated_detected_boxes_per_crop_flat_filtered)
        scores_shrinked = np.copy(scores_per_crop_flat_filtered)
        with tqdm(np.arange(len(translated_detected_boxes_per_crop_flat_filtered))) as prbr:
            while True:
                if translated_detected_boxes_shrinked.shape[0] < 2:
                    break
                item_to_exclude = exclude_redundant_labelbbox_pair(translated_detected_boxes_shrinked, iou_threshold=args.shrinking_iou_threshold)
                if item_to_exclude is None:
                    break
                translated_detected_boxes_shrinked = np.array([translated_detected_boxes_shrinked[i] for i in range(translated_detected_boxes_shrinked.shape[0]) if i != item_to_exclude])
                scores_shrinked = np.array([scores_shrinked[i] for i in range(scores_shrinked.shape[0]) if i != item_to_exclude])
                prbr.update(1)


        curr_snapshot_detected_data_dict = {'data_fname': data_fname,
                                            'sat_label': sat_label,
                                            'dt': dt,
                                            'proba_threshold': args.proba_threshold,
                                            'shrinking_iou_threshold': args.shrinking_iou_threshold,
                                            'scores_shrinked': scores_shrinked,
                                            'translated_detected_boxes_shrinked': translated_detected_boxes_shrinked,
                                            'projection_shape': example.shape,
                                            'retinanet_backbone': args.backbone,
                                            'retinanet_snapshot_file': args.model_snapshot}
        EnsureDirectoryExists(os.path.dirname(curr_snapshot_results_filename))
        with open(curr_snapshot_results_filename, 'wb') as f:
            pickle.dump(curr_snapshot_detected_data_dict, f)



        crop_ch5_normed = example[0, :, :, 0]
        crop_ch5_normed = np.ma.asarray(crop_ch5_normed)
        crop_ch5_normed.mask = shared_mask.astype(np.bool)
        crop_ch9_normed = example[0, :, :, 1]
        crop_ch9_normed = np.ma.asarray(crop_ch9_normed)
        crop_ch9_normed.mask = shared_mask.astype(np.bool)
        crop_btd_normed = example[0, :, :, 2]
        crop_btd_normed = np.ma.asarray(crop_btd_normed)
        crop_btd_normed.mask = shared_mask.astype(np.bool)

        #region debug_plot
        f = plt.figure(figsize=(8, 8), dpi=300)
        p = plt.subplot(2, 2, 1)
        ax = plt.gca()
        im = plt.imshow(scale_ch5_back(crop_ch5_normed), cmap=cmap_ch5, vmin=200., vmax=320.)
        for box, score in zip(translated_detected_boxes_shrinked, scores_shrinked):
            (bbox_x1, bbox_y1, bbox_x2, bbox_y2) = box.astype(int)
            plt.plot([bbox_x1, bbox_x1, bbox_x2, bbox_x2, bbox_x1], [bbox_y1, bbox_y2, bbox_y2, bbox_y1, bbox_y1], color='green', linewidth=0.5)
            plt.text(bbox_x2, bbox_y2 + 4, '%.3f' % score, fontsize=6, color='magenta')
        _ = plt.axis('off')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)
        ax.set_title('ch5, K')

        p = plt.subplot(2, 2, 2)
        ax = plt.gca()
        im = plt.imshow(scale_ch9_back(crop_ch9_normed), cmap=cmap_ch9, vmin=200., vmax=320.)
        for box, score in zip(translated_detected_boxes_shrinked, scores_shrinked):
            (bbox_x1, bbox_y1, bbox_x2, bbox_y2) = box.astype(int)
            plt.plot([bbox_x1, bbox_x1, bbox_x2, bbox_x2, bbox_x1], [bbox_y1, bbox_y2, bbox_y2, bbox_y1, bbox_y1], color='green', linewidth=0.5)
            plt.text(bbox_x1, bbox_y1, '%.3f' % score, fontsize=6, color='magenta')
        _ = plt.axis('off')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)
        # cbar.set_label('ch5, K', rotation=270)
        ax.set_title('ch9, K')

        p = plt.subplot(2, 2, 3)
        ax = plt.gca()
        # im = plt.imshow(scale_btd_back(crop_btd_normed), cmap='jet', vmin=scale_btd_back(btd_thresh))
        im = plt.imshow(scale_btd_back(crop_btd_normed), cmap=cmap_btd, vmin=-80., vmax=3.3)
        for box, score in zip(translated_detected_boxes_shrinked, scores_shrinked):
            (bbox_x1, bbox_y1, bbox_x2, bbox_y2) = box.astype(int)
            plt.plot([bbox_x1, bbox_x1, bbox_x2, bbox_x2, bbox_x1], [bbox_y1, bbox_y2, bbox_y2, bbox_y1, bbox_y1], color='green', linewidth=0.5)
            plt.text(bbox_x1, bbox_y1, '%.3f' % score, fontsize=6, color='magenta')
        _ = plt.axis('off')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)
        # cbar.set_label('ch5, K', rotation=270)
        ax.set_title('BTD, K')

        # _ = plt.show()
        plt.tight_layout()
        plt.savefig(curr_snapshot_vis_plot_filename, dpi=300, pad_inches=0)
        plt.close()

        #endregion debug_plot
        # if datafile_idx > 2:
        #     break



if __name__ == '__main__':
    main()
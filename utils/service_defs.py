import os, sys, fnmatch
# from .labelFile import *
# from .xml_io import ArbitraryXMLReader
import numpy as np
from netCDF4 import Dataset
# from .sat_operations import *
import cv2
from lxml import etree
import hashlib, json
import re
import numba
from numba import jit



def find_files(directory, pattern):
    import os, fnmatch
    flist = []
    for root, dirs, files in os.walk(directory):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                filename = os.path.join(root, basename)
                filename = filename.replace('\\', '/')
                flist.append(filename)
    return flist




# def loadArbitraryXMLByFilename(xmlPath):
#     if os.path.isfile(xmlPath) is False:
#         return
#
#     MCCxmlParseReader = ArbitraryXMLReader(xmlPath)
#     shapes = MCCxmlParseReader.getShapes()
#     return shapes




def orthodrome(pt1, pt2):
    return np.sqrt(np.sum(((pt2-pt1)*(np.asarray([np.cos(np.pi*pt1[1]/180), 1.])) * 111.3)**2))




# def read_ncfile_data(fname):
#     with Dataset(fname, 'r') as ds:
#         wmo_filname = ds.wmo_filename
#         reex = '.+(MSG\d).+(\d{14})\.nc'
#         match = re.match(reex, wmo_filname)
#         sat_label = match.groups()[0]
#         dt_str = match.groups()[1]
#         sat_constants = sat_values(sat_label)
#
#         lats = ds.variables['lat'][:]
#         lons = ds.variables['lon'][:]
#         mask = lats.mask
#
#         ch5 = ds.variables['ch5'][:]
#         ch5.mask = mask
#         ch5 = t_brightness_calculate(ch5, sat_constants, 'ch5')
#
#         ch9 = ds.variables['ch9'][:]
#         ch9.mask = mask
#         ch9 = t_brightness_calculate(ch9, sat_constants, 'ch9')
#
#         btd = ch5 - ch9
#         btd.mask = mask
#         btd.mask[btd > 50.] = True
#     return lats, lons, ch5, ch9, btd, sat_label, dt_str



def point_inside_contour(contour, point):
    return bool(cv2.pointPolygonTest(contour, point, False)==1.)


def label_xmlfile_contents(fname):
    tree = etree.parse(fname)
    filename_element = tree.find('filename')
    curr_nc_fname = filename_element.text

    objects = []
    for object_elem in tree.findall('object'):
        t2 = etree.ElementTree(element=object_elem)
        curr_object = {'type': t2.find('name').text, 'nc_fname': curr_nc_fname}
        # for object_elem in t2.getiterator():
        curr_object['lat0'] = np.float64(t2.find('ellipse/lat0').text)
        curr_object['lon0'] = np.float64(t2.find('ellipse/lon0').text)
        curr_object['lat1'] = np.float64(t2.find('ellipse/lat1').text)
        curr_object['lon1'] = np.float64(t2.find('ellipse/lon1').text)
        curr_object['lat2'] = np.float64(t2.find('ellipse/lat2').text)
        curr_object['lon2'] = np.float64(t2.find('ellipse/lon2').text)
        hash_value = dict_hash(curr_object)
        curr_object['id'] = hash_value

        objects.append(curr_object)

    return objects



def dict_hash(dict_to_hash):
    hashvalue = hashlib.sha256(json.dumps(dict_to_hash, sort_keys=True).encode())
    return hashvalue.hexdigest()



def DoesPathExistAndIsDirectory(pathStr):
    if os.path.exists(pathStr) and os.path.isdir(pathStr):
        return True
    else:
        return False


def EnsureDirectoryExists(pathStr):
    import traceback
    if not DoesPathExistAndIsDirectory(pathStr):
        try:
            os.mkdir(pathStr)
        except Exception as ex:
            err_fname = './errors.log'
            exc_type, exc_value, exc_traceback = sys.exc_info()
            with open(err_fname, 'a') as errf:
                traceback.print_tb(exc_traceback, limit=None, file=errf)
                traceback.print_exception(exc_type, exc_value, exc_traceback, limit=None, file=errf)
            print(str(ex))
            print('the directory you are trying to place a file to doesn\'t exist and cannot be created:\n%s' % pathStr)
            raise FileNotFoundError('the directory you are trying to place a file to doesn\'t exist and cannot be created:')



def closest_xypoint(latlonpoint, lats_proj, lons_proj):
    dlat = lats_proj - latlonpoint[1]
    dlon = lons_proj - latlonpoint[0]
    darc_sqr = dlat ** 2 + dlon ** 2
    closest_pt_idx = np.unravel_index(np.argmin(darc_sqr), darc_sqr.shape)  # row and column indices!
    closest_pt_y = closest_pt_idx[0]  # row number - so y value
    closest_pt_x = closest_pt_idx[1]  # column number - so x value
    return np.array([closest_pt_x, closest_pt_y])

def intersection_area(img_with_ellipse, contour):
    cont_image = cv2.drawContours(np.zeros_like(img_with_ellipse, dtype=np.uint8), [contour], 0, 255, -1)
    intersection = cv2.bitwise_and(cont_image, img_with_ellipse)
    return (intersection / 255).sum()














#region rectangles operations
# rectangle is: x,y,w,h
# x,y - coordinates of the lower-left corner
# w - width (pixels)
# h - height (pixels)
def rect_union(a,b):
    # xywh rects here

    x = min(a[0], b[0])
    y = min(a[1], b[1])
    w = max(a[0]+a[2], b[0]+b[2]) - x
    h = max(a[1]+a[3], b[1]+b[3]) - y
    return (x, y, w, h)


@jit
def rect_intersection(a,b):
    # xywh rects here

    x = max(a[0], b[0])
    y = max(a[1], b[1])
    w = min(a[0]+a[2], b[0]+b[2]) - x
    h = min(a[1]+a[3], b[1]+b[3]) - y
    if w<=0 or h<=0: return (0,0,0,0)
    return (x, y, w, h)


@jit
def rect_area(rect):
    # xywh rects here
    return rect[2]*rect[3]




def check_if_intersects(rect1, rect2):
    # xywh rects here
    return (rect_area(rect_intersection(rect1, rect2)) > 0.)



@jit
def iou(rects1, rects2):
    # xywh rects here
    iou_values = np.zeros(len(rects1))
    for idx,(rect1,rect2) in enumerate(zip(rects1, rects2)):
        intersection_rect = rect_intersection(rect1, rect2)
        if rect_area(intersection_rect) > 0:
            intersection_area = rect_area(intersection_rect)
            union_area = rect1[2]*rect1[3] + rect2[2]*rect2[3] - intersection_area
            iou = intersection_area/union_area
            iou_values[idx] = iou
        else:
            iou_values[idx] = 0
    return iou_values



def check_if_none_intersects(test_rect, other_rects):
    for rect in other_rects:
        if check_if_intersects(test_rect, rect):
            return False
    return True

def CheckIfNoneOfLabelsBreaks(labelsBBoxes, subimgRect):
    final = True
    for labelBBox in labelsBBoxes:
        # check if there is no partial intersection - either subRect is completely inside or completely outside the MSCsRects
        final = final & ((rect_intersection(labelBBox, subimgRect) == labelBBox) | (
                    rect_intersection(labelBBox, subimgRect) is None))
        if not final:
            break
    return final


def CheckIfAtLeastOneLabelRect(LabelsRects, subimgRect):
    final = False
    for labelRect in LabelsRects:
        final = final | (rect_intersection(labelRect, subimgRect) == labelRect)
        if final:
            break
    return final


def cut_sample_bboxes(labels_outer_bboxes, orig_img_size=(1716, 2168), sample_size=(512, 512), samples_per_snapshot=1):
    selected_rects = []

    for smpl_idx in range(samples_per_snapshot):
        accepted = False
        while not accepted:
            tl_x = np.random.randint(0, orig_img_size[1] - sample_size[1])
            tl_y = np.random.randint(0, orig_img_size[0] - sample_size[0])
            subimg_rect = (tl_x, tl_y, sample_size[0], sample_size[1])

            accepted = CheckIfNoneOfLabelsBreaks(labels_outer_bboxes, subimg_rect) & CheckIfAtLeastOneLabelRect(
                labels_outer_bboxes, subimg_rect)

        selected_rects.append(subimg_rect)

    return selected_rects


#endregion rectangles operations
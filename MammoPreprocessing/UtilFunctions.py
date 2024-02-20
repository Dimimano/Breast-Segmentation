import pydicom
import os
import shutil
import pandas as pd
from pathlib import Path
import matplotlib.pylab as plt
import numpy as np
import random
import cv2

'''File Formating'''

def new_name_dcm(dcm_path):

    """
    This function takes the absolute path of a .dcm file
    and renames it according to the convention below:

    1. Full mammograms:
        - Mass-Training_P_00001_LEFT_CC_FULL.dcm
    2. Cropped image:
        - Mass-Training_P_00001_LEFT_CC_CROP_1.dcm
        - Mass-Training_P_00001_LEFT_CC_CROP_2.dcm
        - ...
    3. Mask image:
        - Mass-Training_P_00001_LEFT_CC_MASK_1.dcm
        - Mass-Training_P_00001_LEFT_CC_MASK_2.dcm
        - ...


    Parameters
    ----------
    dcm_path : {str}
        The relative (or absolute) path of the .dcm file
        to rename, including the .dcm filename.
        e.g. "source_folder/Mass-Training_P_00001_LEFT_CC/1-1.dcm"

    Returns
    -------
    new_name : {str}
        The new name that the .dcm file should have
        WITH the ".dcm" extention WITHOUT its relative
        (or absolute) path.
        e.g. "Mass-Training_P_00001_LEFT_CC_FULL.dcm"
    False : {boolean}
        False is returned if the new name of the .dcm
        file cannot be determined.
    """

    try:
        # Read dicom.
        ds = pydicom.dcmread(dcm_path)

    except Exception as ex:
        print(ex)
        return None

    else:
        # Get information.
        patient_id = ds.PatientID
        img_type = ds.SeriesDescription

        # === FULL ===
        if "full" in img_type:
            new_name = patient_id + "_FULL" + ".dcm"
            print(f"FULL --- {new_name}")
            return new_name

        # === CROP ===
        elif "crop" in img_type:

            # Double check if suffix is integer.
            suffix = patient_id.split("_")[-1]

            if suffix.isdigit():
                new_patient_id = patient_id.split("_" + suffix)[0]
                new_name = new_patient_id + "_CROP" + "_" + suffix + ".dcm"
                print(f"CROP --- {new_name}")
                return new_name

            elif not suffix.isdigit():
                print(f"CROP ERROR, {patient_id}")
                return False

        # === MASK ===
        elif "mask" in img_type:

            # Double check if suffix is integer.
            suffix = patient_id.split("_")[-1]

            if suffix.isdigit():
                new_patient_id = patient_id.split("_" + suffix)[0]
                new_name = new_patient_id + "_MASK" + "_" + suffix + ".dcm"
                print(f"MASK --- {new_name}")
                return new_name


            elif not suffix.isdigit():
                print(f"MASK ERROR, {patient_id}")
                return False

        # === img_type NOT RECOGNISED ===
        else:
            print(f"img_type CANNOT BE IDENTIFIED, {img_type}")
            return False

def move_dcm_up(dest_dir, source_dir, dcm_filename):

    """
    This function move a .dcm file from its given source
    directory into the given destination directory. It also
    handles conflicting filenames by adding "___a" to the
    end of a filename if the filename already exists in the
    destination directory.

    Parameters
    ----------
    dest_dir : {str}
        The relative (or absolute) path of the folder that
        the .dcm file needs to be moved to.
    source_dir : {str}
        The relative (or absolute) path where the .dcm file
        needs to be moved from, including the filename.
        e.g. "source_folder/Mass-Training_P_00001_LEFT_CC_FULL.dcm"
    dcm_filename : {str}
        The name of the .dcm file WITH the ".dcm" extension
        but WITHOUT its (relative or absolute) path.
        e.g. "Mass-Training_P_00001_LEFT_CC_FULL.dcm".

    Returns
    -------
    None
    """

    dest_dir_with_new_name = os.path.join(dest_dir, dcm_filename)

    # If the destination path does not exist yet...
    if not os.path.exists(dest_dir_with_new_name):
        shutil.move(source_dir, dest_dir)

    # If the destination path already exists...
    elif os.path.exists(dest_dir_with_new_name):
        # Add "_a" to the end of `new_name` generated above.
        new_name_2 = dcm_filename.strip(".dcm") + "___a.dcm"
        # This moves the file into the destination while giving the file its new name.
        shutil.move(source_dir, os.path.join(dest_dir, new_name_2))

def delete_empty_folders(top, error_dir):

    """
    This function recursively walks through a given directory
    (`top`) using depth-first search (bottom up) and deletes
    any directory that is empty (ignoring hidden files).
    If there are directories that are not empty (except hidden
    files), it will save the absolute directory in a Pandas
    dataframe and export it as a `not-empty-folders.csv` to
    `error_dir`.

    Parameters
    ----------
    top : {str}
        The directory to iterate through.
    error_dir : {str}
        The directory to save the `not-empty-folders.csv` to.

    Returns
    -------
    None
    """

    curdir_list = []
    files_list = []

    for (curdir, dirs, files) in os.walk(top=top, topdown=False):

        if curdir != str(top):

            dirs.sort()
            files.sort()

            print(f"WE ARE AT: {curdir}")
            print("=" * 10)

            print("List dir:")

            directories_list = [f for f in os.listdir(curdir) if not f.startswith('.')]
            print(directories_list)

            if len(directories_list) == 0:
                print("DELETE")
                shutil.rmtree(curdir, ignore_errors=True)

            elif len(directories_list) > 0:
                print("DON'T DELETE")
                curdir_list.append(curdir)
                files_list.append(directories_list)

            print()
            print("Moving one folder up...")
            print("-" * 40)
            print()

    if len(curdir_list) > 0:
        not_empty_df = pd.DataFrame(list(zip(curdir_list, files_list)),
                                    columns =["curdir", "files"])
        to_save_path = os.path.join(error_dir, "not-empty-folders.csv")
        not_empty_df.to_csv(to_save_path, index=False)

def sumMasks(mask_list):

    summed_mask = np.zeros(mask_list[0].shape)

    for arr in mask_list:
        summed_mask = np.add(summed_mask, arr)

    # Binarise (there might be some overlap, resulting in pixels with
    # values of 510, 765, etc...)
    _, summed_mask_bw = cv2.threshold(
        src=summed_mask, thresh=1, maxval=255, type=cv2.THRESH_BINARY
    )

    return summed_mask_bw

'''Mask to Bounding box conversion'''

def draw_rect(im, cords, color = None):
    """Draw the rectangle on the image

    Parameters
    ----------

    im : numpy.ndarray
        numpy image

    cords: numpy.ndarray
        Numpy array containing bounding boxes of shape `N X 4` where N is the
        number of bounding boxes and the bounding boxes are represented in the
        format `x1 y1 x2 y2`

    Returns
    -------

    numpy.ndarray
        numpy image with bounding boxes drawn on it

    """

    im = im.copy()

    cords = cords.reshape(-1,4)
    if not color:
        color = [255,255,255]
    for cord in cords:

        pt1, pt2 = (cord[0], cord[1]) , (cord[2], cord[3])

        pt1 = int(pt1[0]), int(pt1[1])
        pt2 = int(pt2[0]), int(pt2[1])

        im = cv2.rectangle(im.copy(), pt1, pt2, color, int(max(im.shape[:2])/200))
    return im

from skimage.measure import label, regionprops, find_contours

def mask_to_border(mask):
    h, w = mask.shape
    border = np.zeros((h, w))

    contours = find_contours(mask, 0.5)
    print(len(contours))
    for contour in contours:
        for c in contour:
            x = int(c[0])
            y = int(c[1])
            border[x][y] = 1

    return border

""" Mask to bounding boxes """
def mask_to_bbox(mask):
    bboxes = []

    mask = mask_to_border(mask)
    lbl = label(mask)
    props = regionprops(lbl)
    print(len(props))
    for prop in props:
        x1 = prop.bbox[1]
        y1 = prop.bbox[0]

        x2 = prop.bbox[3]
        y2 = prop.bbox[2]

        bboxes.append([x1, y1, x2, y2])

    return bboxes

def bbox2yolo(x1, y1, x2, y2, image_w, image_h):
    return [((x2 + x1)/(2*image_w)), ((y2 + y1)/(2*image_h)), (x2 - x1)/image_w, (y2 - y1)/image_h]

def yolo2bbox(x_center, y_center, w, h,  image_w, image_h):
    w = w * image_w
    h = h * image_h
    x1 = ((2 * x_center * image_w) - w)/2
    y1 = ((2 * y_center * image_h) - h)/2
    x2 = x1 + w
    y2 = y1 + h
    return [x1, y1, x2, y2]

def mask_to_box(mask):

    np_seg = np.array(mask)

    segmentation = np.where(np_seg == 1)

    if len(segmentation) != 0 and len(segmentation[1]) != 0 and len(segmentation[0]) != 0:
        x_min = int(np.min(segmentation[1]))
        x_max = int(np.max(segmentation[1]))
        y_min = int(np.min(segmentation[0]))
        y_max = int(np.max(segmentation[0]))

    return [x_min, x_max, y_min, y_max]
    
def load_data(input_path_mammos, input_path_masks):
  '''Load the mammograms and the correspoding masks.'''

  mammograms, masks = [], []

  # Mammograms
  for mammo in os.listdir(input_path_mammos):

    mammogram = cv2.imread(input_path_mammos + mammo)/255
    mammograms.append(mammogram)

  # Masks
  for mask in os.listdir(input_path_masks):

    mask = cv2.imread(input_path_masks + mask, cv2.IMREAD_GRAYSCALE)/255
    mask = np.expand_dims(mask, axis=-1) # convert shape [320,320] -> [320,320,1]
    masks.append(mask)

  # Store dataset in np.arrays.
  mammograms = np.array(mammograms)
  masks = np.array(masks)

  return mammograms, masks
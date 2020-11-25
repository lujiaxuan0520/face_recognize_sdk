"""
delete the dirty data in the datasets ,using before train.py
"""

import os
import os.path as osp
from imutils import paths

def generate_list(images_directory, saved_name=None):
    """
    Args:
        images_directory: the directory, including WebFace and LFW format
    Returns:
        data_list: [<path> <label>]
    """
    subdirs = os.listdir(images_directory)
    num_ids = len(subdirs)
    data_list = []
    for i in range(num_ids):
        subdir = osp.join(images_directory, subdirs[i])
        files = os.listdir(subdir)
        paths = [osp.join(subdir, file) for file in files]
        paths_with_Id = ["{} {}\n".format(p,i) for p in paths]
        data_list.extend(paths_with_Id)
    
    if saved_name:
        with open(saved_name, 'w', encoding='utf-8') as f:
            f.writelines(data_list)
    return data_list

def transform_clean_list(webface_directory, cleaned_list_path):
    """
    Args:
        webface_directory: WebFace directory
        cleaned_list_path: the path of cleaned_list.txt
    Returns:
        cleaned_list: the list after transform
    """
    with open(cleaned_list_path, encoding='utf-8') as f:
        cleaned_list = f.readlines()
    cleaned_list = [p.replace('\\', '/') for p in cleaned_list]
    cleaned_list = [osp.join(webface_directory, p) for p in cleaned_list]
    return cleaned_list

def remove_dirty_image(webface_directory, cleaned_list):
    cleaned_list = set([c.split()[0] for c in cleaned_list])
    for p in paths.list_images(webface_directory):
        if p.replace('\\', '/') not in cleaned_list:
            print("remove {}".format(p))
            os.remove(p)

if __name__ == '__main__':
    data = '../data/CASIA-WebFace/'
    lst = '../data/cleaned list.txt'
    cleaned_list = transform_clean_list(data, lst)
    remove_dirty_image(data, cleaned_list)
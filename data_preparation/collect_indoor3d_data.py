import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
import indoor3d_util

anno_paths = [line.rstrip() for line in open(os.path.join(BASE_DIR, 'meta/area5_anno_paths.txt'))]
anno_paths = [os.path.join(indoor3d_util.DATA_PATH, p) for p in anno_paths]

output_folder = os.path.join(BASE_DIR, 's3dis_npy')
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

# Note: there is an extra character in the v1.2 data in Area_5/hallway_6. It's fixed manually.
for anno_path in anno_paths:
    print(anno_path)
    try:
        elements = anno_path.split('/')
        out_filename = elements[-3]+'_'+elements[-2]+'.npy' # Area_1_hallway_1.npy
        indoor3d_util.collect_point_label(anno_path, os.path.join(output_folder, out_filename), 'numpy')
    except:
        print(anno_path, 'ERROR!!')

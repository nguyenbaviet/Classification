from attr import field
from git import base
from matplotlib import image
import pandas as pd
import os
import shutil

# base_dir = '/home/huyphan1/viet/liveness/added_data'
# base_dir = '/home/huyphan1/viet/liveness/trained_data_v0.1/idcard_liveness'
# LABEL_BASE = [ 'NORMAL', 'SCREEN_CAPTURE', 'PHOTOCOPY']
# def get_file(base_dir):
#   data = []
#   labels = []
#   for root, dirs, files in os.walk(base_dir):
#     if len(dirs) > 0:
#       for d in dirs:
#         get_file(os.path.join(base_dir, d))
#     else:
#       for f in files:
#         label = root.split('/')[-1]
#         if '.jpg' in f and label in LABEL_BASE:
#           data.append(os.path.join(root, f))
#           labels.append(label)
#   return data, labels
# img_pathes, labels = get_file(base_dir)

# data = {'img_name': img_pathes, 'label': labels}

# df = pd.DataFrame(data)

# df.to_csv(os.path.join(base_dir, 'label.csv'))

# base_dir = '/home/huyphan1/nhan/liveness'
# img_dir = os.path.join(base_dir, 'corner_cut')
# annos_file = os.path.join(base_dir, 'added_corner_cut.csv')
# save_dir = '/home/huyphan1/nhan/corner_cut/corner_cut_200'
# data = pd.read_csv(annos_file)
# data = data[data['choice'].str.contains('corner_cut', na=False)]
# images = data['image'].apply(lambda x: os.path.join('/home/huyphan1', x.split('=')[-1]))
# for image in images:
#   shutil.copy(image, save_dir)

# img_dirs = ['/home/huyphan1/nhan/corner_cut/corner_cut_200', '/home/huyphan1/nhan/corner_cut/corner_cut_800', '/home/huyphan1/nhan/corner_cut/FAKE_CUT']
# files = []
# for img_dir in img_dirs:
#   imgs = [os.path.join(img_dir, f) for f in os.listdir(img_dir)]
#   if 'FAKE_CUT' in img_dir:
#     # files.extend(imgs[:1000])
#     files.extend(imgs)
#   else:
#     files.extend(imgs)

# anns = pd.DataFrame({'img_name': files, 'label': ['corner_cut' for _ in range(len(files))]})
# anns.to_csv('/home/huyphan1/nhan/corner_cut/corner_cut_1K.csv')

anns_file = '/home/huyphan1/tan.tran1/data/DARKLAUNCH_IDCARD_CHECKER_VS_VNPT_20220209/corner_cut_liveness.csv'
data = pd.read_csv(anns_file)
print(len(data[data['corner cut'] == 'CORNER_CUT']))
data['liveness'][data['corner cut'] == 'CORNER_CUT'] = 'CORNER_CUT'
data.dropna(subset=['liveness'], inplace=True)
print(len(data[data['liveness'] == 'CORNER_CUT']))
data.to_csv('/home/huyphan1/tan.tran1/data/DARKLAUNCH_IDCARD_CHECKER_VS_VNPT_20220209/corner_cut_liveness_norm.csv')
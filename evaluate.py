from cProfile import label
from email.mime import image
from tabnanny import filename_only
from typing import final
import numpy as np
import pandas as pd
import os
import cv2
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn import metrics
import matplotlib.pyplot as plt

from inference import LivenessChecker, LABEL

LABEL = [i.replace('label_', '').upper() for i in LABEL]
NOT_NORMAL = ['PHOTOCOPY', 'SCREEN_CAPTURE', 'PRINTED_COLOR', 'CORNER_CUT']

def infer(checkpoint_path, config_file, img_path, anns_path, save_path):
  liveness_model = LivenessChecker(checkpoint_path, config_file)

  df = pd.read_csv(anns_path)

  liveness_new = []
  score_new = []
  for idx, row in df.iterrows():
    print(f'{idx}/{len(df)}')
    request_id = row['request_id']
    surface = row['surface_id']
    surface_idx = 0 if surface == 'F' else 1
    _img_path = os.path.join(img_path, '10k_vnpt_vs_ai_' + surface, request_id + f'_img_{surface_idx}.jpg')
    img = cv2.imread(_img_path)
    if img is None:
      liveness_new.append('IMG_IS_NULL')
      score_new.append(0)
      continue
    label, score = liveness_model([img])
    liveness_new.append(LABEL[label[0]])
    score_new.append(score[0])
  df['LIVENESS_NEW'] = liveness_new
  df['SCORE_LIVENESS_NEW'] = score_new
  df.to_csv(save_path)

def infer_1(checkpoint_path, config_file, img_path, anns_path, save_path):
  liveness_model = LivenessChecker(checkpoint_path, config_file)

  df = pd.read_csv(anns_path)

  liveness_new = []
  score_new = []
  for idx, row in df.iterrows():
    print(f'{idx}/{len(df)}')
    request_id = row['image'].split('=')[-1]
    _img_path = os.path.join('/home/huyphan1', request_id)
    img = cv2.imread(_img_path)
    if img is None:
      liveness_new.append('IMG_IS_NULL')
      score_new.append(0)
      continue
    label, score = liveness_model([img])
    liveness_new.append(LABEL[label[0]])
    score_new.append(score[0])
  df['LIVENESS_NEW'] = liveness_new
  df['SCORE_LIVENESS_NEW'] = score_new
  df.to_csv(save_path)

def infer2(checkpoint_path, config_file, save_path):
  img_path = '/home/huyphan1/tan.tran1/data/DARKLAUNCH_IDCARD_CHECKER_VS_VNPT_20220209'
  anns_path = os.path.join(img_path, 'final_data_1.csv')
  # anns_path = os.path.join(img_path, 'corner_cut_liveness_norm.csv')
  liveness_model = LivenessChecker(checkpoint_path, config_file)

  df = pd.read_csv(anns_path)

  liveness_new = []
  score_new = []
  for idx, row in df.iterrows():
    print(f'{idx}/{len(df)}')
    image_name = row['image']
    if '_0' in image_name:
      image_path = os.path.join(img_path, 'DARKLAUNCH_IDCARD_CHECKER_VS_VNPT_20220209_F', image_name.replace('_0', '_img_0.jpg'))
    else:
      image_path = os.path.join(img_path, 'DARKLAUNCH_IDCARD_CHECKER_VS_VNPT_20220209_B', image_name.replace('_1', '_img_1.jpg'))
    # image_path = os.path.join('/home/huyphan1', image_name.split('=')[-1])
    img = cv2.imread(image_path)
    if img is None:
      liveness_new.append('IMG_IS_NULL')
      print(image_path)
      score_new.append(0)
      continue
    label, score = liveness_model([img])
    liveness_new.append(LABEL[label[0]])
    score_new.append(score[0])
  df['LIVENESS_NEW'] = liveness_new
  df['SCORE_LIVENESS_NEW'] = score_new
  df.to_csv(save_path)

def eval(anns_path, class_id=0):
  df = pd.read_csv(anns_path)
  gt = df['QC_LIVENESS_BI']
  gt = [1 if i == 'NORMAL' else 0 for i in gt]

  vnpt_pred = df['VNPT_LIVENESS_BI']
  vnpt_pred = [1 if i == 'NORMAL' else 0 for i in vnpt_pred]

  momo_pred_v0 = df['MOMO_LIVENESS_BI']
  momo_pred_v0 = [1 if i == 'NORMAL' else 0 for i in momo_pred_v0]

  pred = df['LIVENESS_NEW']
  pred = [1 if i == 'NORMAL' else 0 for i in pred]

  darklaunch_v1 = metrics.precision_recall_fscore_support(gt, momo_pred_v0)
  print('darklaunch v1: ', print_score(darklaunch_v1, class_id))

  vnpt = metrics.precision_recall_fscore_support(gt, vnpt_pred)
  print('VNPT: ', print_score(vnpt, class_id))

  new_result = metrics.precision_recall_fscore_support(gt, pred)
  print('new result: ', print_score(new_result, class_id))

def print_score(score, class_id=0):
  return f'\n Precision: {score[0][class_id]} \n Recall: {score[1][class_id]} \n F1 score: {score[2][class_id]}'


def convert_ms_code_to_string():
  mapping_list = {0: 'NORMAL', 640: 'SCREEN_CAPTURE', 642: 'PHOTOCOPY', 619: 'QUALIY', 651: 'PUNCHED', 622: 'CORNER_CUT'}
  unused_list = [608, 643, 649, -470, 641, -495, 634, 637]
  file_path = '/home/huyphan1/viet/liveness/bq-results-20220125-155937-hcdd6f6c5c79.csv'
  data = pd.read_csv(file_path)
  data = data.astype({'MS_CODE': int})
  data = data[~(data['MS_CODE'].isin(unused_list))]
  def convert(ms_code):
    code_string = mapping_list[ms_code]
    if code_string in NOT_NORMAL:
      return 'NOT_NORMAL'
    return 'NORMAL'
  data['MS_CODE'] = data['MS_CODE'].apply(convert)
  return data

def merge_prediction():
  ctv_label = pd.read_csv('/home/huyphan1/viet/liveness/10K_on_prod_ctv_label_r2.csv')
  prediction = pd.read_csv('/home/huyphan1/viet/liveness/10K_on_prod_prediction.csv')
  prediction = prediction.drop(columns=['choice'])

  final_data = pd.merge(ctv_label, prediction, on=['image'])
  final_data = final_data[['image', 'choice', 'LIVENESS_NEW', 'SCORE_LIVENESS_NEW']]
  final_data = final_data.to_csv('10K_prediction_r2.csv')

def eval_10K_binary_class(class_id=0):
  vnpt_data = convert_ms_code_to_string()

  ctv_label = pd.read_csv('/home/huyphan1/viet/liveness/10K_prediction_r2.csv')
  def convert_ctv_label(label):
    if label in NOT_NORMAL:
      return 'NOT_NORMAL'
    return 'NORMAL'
  ctv_label['choice'] = ctv_label['choice'].apply(convert_ctv_label)

  ctv_label = ctv_label[ctv_label['image'].str.contains('_img_0', na=False)]

  def get_request_id(img_path):
    return img_path.split('/')[-1].split('_')[0]

  ctv_label['RESULT_REFERENCE'] = ctv_label['image'].apply(get_request_id)
  
  df = pd.merge(vnpt_data, ctv_label, on=['RESULT_REFERENCE'])

  gt = df['choice']
  gt = [1 if i == 'NORMAL' else 0 for i in gt]

  vnpt_pred = df['MS_CODE']
  vnpt_pred = [1 if i == 'NORMAL' else 0 for i in vnpt_pred]


  pred = df['LIVENESS_NEW']
  pred = [1 if i == 'NORMAL' else 0 for i in pred]

  vnpt = metrics.precision_recall_fscore_support(gt, vnpt_pred)
  print('VNPT: ', print_score(vnpt, class_id))

  new_result = metrics.precision_recall_fscore_support(gt, pred)
  print('new result: ', print_score(new_result, class_id))
  
def eval_10K_multi_class():
  print(LABEL)
  ctv_label = pd.read_csv('/home/huyphan1/viet/liveness/10K_prediction_r2.csv')
  def convert_ctv_label(label):
    if label in NOT_NORMAL:
      return LABEL.index(label)
    return LABEL.index('NORMAL')
  for c in ctv_label['choice']:
    if isinstance(c, list):
      print(c)
      exit()
  ctv_label['choice'] = ctv_label['choice'].apply(convert_ctv_label)

  gt = ctv_label['choice']

  pred = ctv_label['LIVENESS_NEW'].apply(lambda x: LABEL.index(x))
  pred = list(pred)
  LABEL.remove('NOT_NORMAL')
  cm = confusion_matrix(gt, pred)
  disp = ConfusionMatrixDisplay(cm, display_labels=LABEL)
  disp.plot()

  print(metrics.precision_recall_fscore_support(gt, pred))
  # plt.savefig('confusion_matrix.jpg')
  plt.show()

def eval_darklaunch_r2():
  # mapping_dict = {'LIVENESS': 0, 'NORMAL': 0, 'PRINTED_COLOR': 1, 'PHOTOCOPY':3, 'SCREEN_CAPTURE':2, 'CORNER_CUT': 4}
  mapping_dict = {'LIVENESS': 0, 'NORMAL': 0, 'PRINTED_COLOR': 1, 'PHOTOCOPY':1, 'SCREEN_CAPTURE':1, 'CORNER_CUT': 1}
  # anns_file = f'{base_dir}/darklaunch_r2_new_weight.csv'
  anns_file = f'{base_dir}/darklaunch_r2_new_weight_added_corner_cut_new_weight.csv'
  # anns_file = f'{base_dir}/corner_cut_liveness_full_prediction.csv'
  
  data = pd.read_csv(anns_file)
  data = data[data['LIVENESS_NEW'] != 'NOT_NORMAL']
  gt = [mapping_dict[d] for d in data['liveness']]
  pred = [mapping_dict[d] for d in data['LIVENESS_NEW']]
  print(metrics.precision_recall_fscore_support(gt, pred))
  cm = confusion_matrix(gt, pred)
  # label_name = ['LIVENESS', 'PRINTED_COLOR', 'SCREEN_CAPTURE', 'PHOTOCOPY', 'CORNER_CUT']
  # # label_name = ['LIVENESS', 'PRINTED_COLOR', 'SCREEN_CAPTURE', 'CORNER_CUT']
  # disp = ConfusionMatrixDisplay(cm,display_labels=label_name)
  # disp.plot()
  # plt.savefig('confusion_matrix.jpg')
  # plt.show()


def v_test():
  gt_anns = '/home/huyphan1/viet/liveness/final_data_1.csv'
  pred_anns = '/home/huyphan1/viet/liveness/darklaunch_r2_new_weight_added_corner_cut.csv'
  gt = pd.read_csv(gt_anns)
  pred = pd.read_csv(pred_anns)

  gt['LIVENESS_NEW'] = pred['LIVENESS_NEW']
  gt.dropna(subset=['LIVENESS_NEW'], inplace=True)
  gt.to_csv('/home/huyphan1/viet/liveness/darklaunch_r2_new_weight_added_corner_cut_normalize.csv')

if __name__ == '__main__':
  checkpoint_path = 'weights/best_b3.pth'
  config_file = 'configs/b3.yaml'
  base_dir = '/home/huyphan1/viet/liveness'
  # img_path = '/home/huyphan1/cv_team/darklaunch_OCR/11262021/20211127-vnpt-vs-ai-ocr'
  # # anns_path = f'{base_dir}/tan_liveness_9.2K_normalize.csv'
  # anns_path = f'{base_dir}/10K_on_prod_round_1.csv'
  # save_path = f'{base_dir}/eval_9.2K.csv'
  # # save_path = f'{base_dir}/10K_on_prod_prediction.csv'
  
  # # infer(checkpoint_path, config_file, img_path, anns_path, save_path)
  # # infer_1(checkpoint_path, config_file, '', anns_path, save_path)

  # eval(save_path, 0)
  # eval_10K_binary_class()
  # eval_10K_multi_class()

  # infer2(checkpoint_path, config_file, save_path=f'{base_dir}/darklaunch_r2_new_weight_added_corner_cut_new_weight.csv')
  eval_darklaunch_r2()


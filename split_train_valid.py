from cv2 import add
import numpy as np
import pandas as pd
import os
from skmultilearn.model_selection import IterativeStratification
import shutil
from collections import Counter
from skmultilearn.model_selection.measures import get_combination_wise_output_matrix
from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# LABELS = ["label_printed_color","label_corner_cut","label_info_tampered","label_face_tampered","label_photocopy","label_screen_capture","label_partial", "label_no_plastic_border", "label_special_plastic_laminating", "label_stapler_pin_border", "label_not_normal", "label_normal"]
# LABELS_WITH_IMAGE = ["image", "label_printed_color","label_printed_color","label_corner_cut","label_info_tampered","label_face_tampered","label_photocopy","label_screen_capture","label_partial", "label_no_plastic_border", "label_special_plastic_laminating", "label_stapler_pin_border", "label_not_normal", "label_normal"]

def format_dir(f):
  url_name = f.split('/')[-1]
  name = os.path.splitext(url_name)[0]
  extension = os.path.splitext(url_name)[1]
  filename = os.path.join(name+"_"+f.split('/')[-2]+extension)
  return filename

def save_csv(X, y, file_name):
  df = pd.DataFrame(data = y, columns = LABELS)
  df["image"] = X
  df = df[LABELS_WITH_IMAGE]
  df.to_csv(file_name, index=False)
  return 0

# print(prediction_df.head())
save_dir = '/home/huyphan1/viet/liveness'
img_dir = '/home/huyphan1/cv_team/labeling_data_ocrv2/idcard_fraud/print_color'
aug_df = pd.read_csv(f"{save_dir}/liveness.csv")

# drop corner cut, *_tampered, parital, no_plastic_border, special plastic laminating, stapler ping border
aug_df = aug_df[~aug_df["choice"].str.contains("NOT_NORMAL", na=False)]
aug_df = aug_df[~aug_df["choice"].str.contains("TAMPERED", na=False)]
aug_df = aug_df[~aug_df["choice"].str.contains("PARTIAL", na=False)]
aug_df = aug_df[~aug_df["choice"].str.contains("NO_PLASTIC_BORDER", na=False)]
aug_df = aug_df[~aug_df["choice"].str.contains("SPECIAL_PLASTIC_LAMINATING", na=False)]
aug_df = aug_df[~aug_df["choice"].str.contains("STAPLER_PIN_BORDER", na=False)]

# convert label
aug_df['label_printed_color'] = np.where(aug_df['choice'].str.contains("PRINTED_COLOR", na=False), 1, 0)
aug_df['label_corner_cut'] = np.where(aug_df['choice'].str.contains("CORNER_CUT", na=False), 1, 0)
# aug_df['label_info_tampered'] = np.where(aug_df['choice'].str.contains("INFO_TAMPERED", na=False), 1, 0)
# aug_df['label_face_tampered'] = np.where(aug_df['choice'].str.contains("FACE_TAMPERED", na=False), 1, 0)
aug_df['label_photocopy'] = np.where(aug_df['choice'].str.contains("PHOTOCOPY", na=False), 1, 0)
aug_df['label_screen_capture'] = np.where(aug_df['choice'].str.contains("SCREEN_CAPTURE", na=False), 1, 0)
# aug_df['label_partial'] = np.where(aug_df['choice'].str.contains("PARTIAL", na=False), 1, 0)
# aug_df["label_no_plastic_border"] = np.where(aug_df['choice'].str.contains("NO_PLASTIC_BORDER", na=False), 1, 0)
# aug_df["label_special_plastic_laminating"] = np.where(aug_df['choice'].str.contains("SPECIAL_PLASTIC_LAMINATING", na=False), 1, 0)
# aug_df["label_stapler_pin_border"] = np.where(aug_df['choice'].str.contains("STAPLER_PIN_BORDER", na=False), 1, 0)
# aug_df["label_not_normal"] = np.where(aug_df['choice'].str.contains("NOT_NORMAL", na=False), 1, 0)
aug_df["label_normal"] = np.where(aug_df['choice'].str.contains("NORMAL", na=False), 1, 0)
aug_df['img_name'] = img_dir + aug_df['image'].str.split('print_color').str[-1]

# recorrect anns
aug_df['label_normal'][(aug_df['label_printed_color'] == 1) | (aug_df['label_screen_capture'] == 1)| (aug_df['label_photocopy'] == 1)] = 0

full_df = aug_df.drop_duplicates(subset=['img_name'])

# print(len(full_df[(full_df['label_printed_color'] == 1) & (full_df['label_normal'] == 1)]))
# print(len(full_df[(full_df['label_screen_capture'] == 1) & (full_df['label_normal'] == 1)]))
# print(len(full_df[(full_df['label_corner_cut'] == 1) & (full_df['label_normal'] == 1)]))

LABELS = ["label_printed_color","label_photocopy","label_screen_capture", "label_corner_cut", "label_normal"]
LABELS_WITH_IMAGE = ["image","label_printed_color", "label_photocopy","label_screen_capture", "label_corner_cut", "label_normal"]


# merge with added corner cut data

added_data_path = '/home/huyphan1/nhan/corner_cut/corner_cut_1K.csv'
added_data = pd.read_csv(added_data_path)

added_data[LABELS] = 0
added_data['label_corner_cut'][added_data["label"] == 'corner_cut'] = 1

added_data = added_data[["img_name"] + LABELS]

full_df = pd.concat([full_df, added_data])

# merge with v0.1 data
data_v0_1_path = '/home/huyphan1/viet/liveness/trained_data_v0.1/idcard_liveness/label.csv'
data_v0_1 = pd.read_csv(data_v0_1_path)

data_v0_1[LABELS] = 0
data_v0_1['label_normal'][data_v0_1["label"] == 'NORMAL'] = 1
data_v0_1['label_screen_capture'][data_v0_1["label"] == 'SCREEN_CAPTURE'] = 1
data_v0_1['label_photocopy'][data_v0_1["label"] == 'PHOTOCOPY'] = 1

data_v0_1 = data_v0_1[["img_name"] + LABELS]

full_df = pd.concat([full_df, data_v0_1])

# print(full_df['label_printed_color'].value_counts())
# print(full_df['label_corner_cut'].value_counts())
# print(full_df['label_info_tampered'].value_counts())
# print(full_df['label_face_tampered'].value_counts())
# print(full_df['label_screen_capture'].value_counts())
# print(full_df['label_no_plastic_border'].value_counts())
# print(full_df['label_special_plastic_laminating'].value_counts())
# print(full_df['label_stapler_pin_border'].value_counts())
# print(full_df['label_not_normal'].value_counts())
# print(full_df['label_normal'].value_counts())
# print(full_df['label_partial'].value_counts())
# print(full_df['label_photocopy'].value_counts())

def iterative_train_test_split(X, y, train_size):
    """Custom iterative train test split which
    'maintains balanced representation with respect
    to order-th label combinations.'
    """
    stratifier = IterativeStratification(
        n_splits=2, order=1, sample_distribution_per_fold=[1.0-train_size, train_size, ])
    train_indices, test_indices = next(stratifier.split(X, y))
    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]
    return X_train, X_test, y_train, y_test

X = full_df["img_name"].to_numpy()
y = full_df[LABELS].to_numpy()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1) 
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.1, random_state=1)


X_train, X_val, y_train, y_val = iterative_train_test_split(
    X, y, train_size=0.9)
# X_val, X_test, y_val, y_test = iterative_train_test_split(
#     X_, y_, train_size=0.99)


print(f"train: {len(X_train)} ({len(X_train)/len(X):.2f})\n"
      f"val: {len(X_val)} ({len(X_val)/len(X):.2f})\n")
      # f"test: {len(X_test)} ({len(X_test)/len(X):.2f})")

save_csv(X_train, y_train, f"{save_dir}/train.csv")
save_csv(X_val, y_val, f"{save_dir}/valid.csv")
# save_csv(X_test, y_test, f"{save_dir}/test.csv")


counts = {}
counts["train_counts"] = Counter(str(combination) for row in get_combination_wise_output_matrix(
    y_train, order=1) for combination in row)
counts["val_counts"] = Counter(str(combination) for row in get_combination_wise_output_matrix(
    y_val, order=1) for combination in row)
counts["test_counts"] = Counter(str(combination) for row in get_combination_wise_output_matrix(
    y_test, order=1) for combination in row)


# View distributions
print(pd.DataFrame({
    "train": counts["train_counts"],
    "val": counts["val_counts"],
    # "test": counts["test_counts"]
}).T.fillna(0))

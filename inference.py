from argparse import ArgumentParser
import os
import warnings
from typing import List, Tuple

import numpy as np
from PIL import Image
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import torch
from torch.cuda.amp import autocast
from torchvision.transforms import Compose, Resize, Normalize, ToTensor

from cvcore.config import get_cfg
from models import build_cls_model

warnings.filterwarnings("ignore", category=UserWarning)

parser = ArgumentParser()
parser.add_argument(
    "--resume", help="path to checkpoint", default="weights/best_b0.pth"
)
parser.add_argument("--config", help="path to config. file", default="configs/b3.yaml")

# LABEL = ["label_printed_color","label_photocopy","label_screen_capture", "label_corner_cut", "label_not_normal", "label_normal"]
LABEL = ["label_printed_color","label_photocopy","label_screen_capture", "label_corner_cut", "label_normal"]

class LivenessChecker:
  def __init__(self, checkpoint_file: str, cfg_file: str):
    if torch.cuda.is_available():
        self.device = torch.device("cuda")
    else:
        self.device = torch.device("cpu")

    current_path = os.path.dirname(os.path.realpath(__file__))
    checkpoint_path = os.path.join(current_path, checkpoint_file)
    cfg_path = os.path.join(current_path, cfg_file)
    assert os.path.isfile(cfg_path), f"{cfg_path} is not a regular file"

    cfg = get_cfg()
    cfg.merge_from_file(cfg_path)
    cfg.MODEL.BACKBONE.PRETRAINED = "None"

    self.cfg = cfg.clone()
    self.transform = Compose(
        [
            Resize((cfg.DATA.HEIGHT, cfg.DATA.WIDTH), interpolation=3),
            ToTensor(),
            Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ]
    )
    self.model = build_cls_model(cfg, cfg.MODEL.NUM_CLASSES)
    self.model.eval()
    self.model.to(self.device)
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint.pop("model"))
    else:
        warnings.warn(
            f"{checkpoint_path} is not a regular file, model predictions might be incorrect"
        )

    self.threshold = 0.4

  def preprocess(self, img: np.ndarray) -> torch.tensor:
    img = Image.fromarray(img[:, :, ::-1])
    img = self.transform(img)
    img = img.unsqueeze(0)
    return img

  def preprocess_batch(self, imgs: List[np.ndarray]) -> torch.tensor:
    img_tensor = []
    for img in imgs:
        img = self.preprocess(img)
        img_tensor.append(img)
    img_tensor = torch.cat(img_tensor, 0)
    return img_tensor.to(self.device)

  # @torch.no_grad()
  # def __call__(self, imgs: List[np.ndarray]) -> Tuple[np.ndarray]:
  #   """
  #   Inference a list of image(s)
  #   """
  #   assert isinstance(imgs, list)
  #   imgs = self.preprocess_batch(imgs)

  #   with autocast():
  #       outputs = self.model(imgs)
  #   probs = torch.sigmoid(outputs).cpu()
  #   decisions = (probs > self.threshold).long()

  #   return probs.numpy(), decisions.numpy()

  @torch.no_grad()
  def __call__(self, imgs: List[np.ndarray], single_label=True) -> Tuple[np.ndarray]:
    """
    Inference a list of image(s)
    """
    assert isinstance(imgs, list)
    imgs = self.preprocess_batch(imgs)

    with autocast():
        outputs = self.model(imgs)
    probs = torch.sigmoid(outputs).cpu().numpy()
    if single_label:
      return np.argmax(probs, axis=1), np.max(probs, axis=1)
    decisions = (probs > self.threshold)
    
    return probs, np.array(decisions, dtype=int)

if __name__ == '__main__':
  import cv2
  # img_path = '/home/huyphan1/cv_team/labeling_data_ocrv2/idcard_fraud/print_color/20k_fraud_F/fffb555f-f83c-43d6-9bdd-29569c39a0ff_img_0.jpg'
  img_path = '/home/huyphan1/viet/classification/01862701425_FRONT_832.jpg'
  img = [cv2.imread(img_path)]

  liveness_model = LivenessChecker('weights/best_b3.pth', 'configs/b3.yaml')

  b = liveness_model(img)
  print(b)
  print(b[0][0])
from detectron2.utils.logger import setup_logger
from detectron2.data.datasets import register_coco_instances
from detectron2.engine.defaults import DefaultTrainer, DefaultPredictor
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from pycocotools.cocoeval import COCOeval
from detectron2.evaluation.coco_evaluation import COCOevalMaxDets
from detectron2 import model_zoo
from detectron2.config import get_cfg
from MyTrainer import *

import matplotlib.pyplot as plt
import tensorflow as tf
import os 
import pickle
import json

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

setup_logger()
# https://github.com/facebookresearch/detectron2/blob/main/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml

config_file_path = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
checkpoint_url = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"

# config_file_path = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
# checkpoint_url = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"

output_dir = "output"

device = "cuda"

train_dataset_name = "wall_train"
train_images_path = "trainingset/train_crop"
train_json_annot_path = "trainingset/train.json" 

val_dataset_name = "wall_val"
val_images_path = "trainingset/val_crop"
val_json_annot_path = "trainingset/val.json"

val_gt_dataset_name = "wall_gt_val"
val_gt_images_path = "trainingset/val_gt_crop"
val_gt_json_annot_path = "trainingset/val_gt.json"

test_gt_dataset_name = "wall_gt_test"
test_gt_images_path = "trainingset/test_gt_crop"
test_gt_json_annot_path = "trainingset/test_gt.json"

val_test_gt_dataset_name = "wall_gt_val_test"
val_test_gt_images_path = "trainingset/val_test_crop"
val_test_gt_json_annot_path = "trainingset/val_test_gt.json"

cfg_save_path = "output/wall_OD_cfg.pickle"

num_classes = 1

register_coco_instances(name = train_dataset_name, metadata={},
json_file=train_json_annot_path, image_root=train_images_path)

register_coco_instances(name = val_dataset_name, metadata={},
json_file=val_json_annot_path, image_root=val_images_path)

register_coco_instances(name = val_gt_dataset_name, metadata={},
json_file=val_gt_json_annot_path, image_root=val_gt_images_path)

register_coco_instances(name = test_gt_dataset_name, metadata={},
json_file=test_gt_json_annot_path, image_root=test_gt_images_path)

register_coco_instances(name = val_test_gt_dataset_name, metadata={},
json_file=val_test_gt_json_annot_path, image_root=val_test_gt_images_path)

def get_train_cfg(config_file_path, checkpoint_url, train_dataset_name, num_classes, device, output_dir):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config_file_path))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(checkpoint_url)
    cfg.DATASETS.TRAIN = (train_dataset_name,)
    # cfg.DATASETS.TEST = ()
    # cfg.DATASETS.TEST = (val_dataset_name,)

    cfg.DATALOADER.NUM_WORKERS = 4

    cfg.SOLVER.IMS_PER_BATCH = 8
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 5000
    cfg.SOLVER.STEPS = []

    # cfg.SOLVER.WARMUP_ITERS = 1000
    # cfg.SOLVER.MAX_ITER = 1500 #adjust up if val mAP is still rising, adjust down if overfit
    # cfg.SOLVER.STEPS = (1000, 1500)

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
    # cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.DEVICE = device
    cfg.OUTPUT_DIR = output_dir
    
    cfg.TEST.EVAL_PERIOD = 100
    return cfg

def main():
    cfg = get_train_cfg(config_file_path, checkpoint_url, train_dataset_name, num_classes, device, output_dir)

    os.makedirs(output_dir)
    with open(cfg_save_path, 'wb') as f:
        pickle.dump(cfg, f, protocol=pickle.HIGHEST_PROTOCOL)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    trainer = MyTrainer(cfg)
    # trainer = DefaultTrainer(cfg)
    # trainer.resume_or_load(resume=True)
    trainer.resume_or_load(resume=False)
    trainer.train()

def load_json_arr(json_path):
    lines = []
    with open(json_path, 'r') as f:
        for line in f:
            lines.append(json.loads(line))
    return lines

def train_val_loss_curve(experiment_folder):
    iter_loss, total_loss, iter_val, val_loss, lr_iter, lr = [], [], [], [], [], []
    experiment_metrics = load_json_arr(experiment_folder + '/metrics.json')
    for metric in experiment_metrics:
        if 'bbox/AP' and 'validation_loss' in metric:
            iter_val.append(((metric["iteration"])*64)/6000)
            val_loss.append(metric["validation_loss"])
        if "total_loss" in metric:
            iter_loss.append(((metric["iteration"])*64)/6000)
            total_loss.append(metric["total_loss"])
        if "lr" in metric:
            lr_iter.append(((metric["iteration"])*64)/6000)
            lr.append(metric["lr"])

    plt.plot(iter_loss, total_loss)
    plt.plot(iter_val, val_loss)
    plt.legend(['training loss', 'validation loss'], loc='upper right')
    plt.xlabel('epoch')
    plt.ylabel('total loss')
    plt.show()
    plt.plot(lr_iter, lr)
    plt.legend(['learning rate'], loc='lower right')
    plt.show()

def evaluator(model_path, dataset_name):
    cfg = get_train_cfg(config_file_path, checkpoint_url, train_dataset_name, num_classes, device, output_dir)
    # cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, model_path)
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.1
    # trainer = MyTrainer(cfg)
    # trainer.resume_or_load(resume=False)
    predictor = DefaultPredictor(cfg)
    evaluator = COCOEvaluator(dataset_name, ("bbox",), False, output_dir)
    val_loader = build_detection_test_loader(cfg, dataset_name)
    inference_on_dataset(predictor.model, val_loader, evaluator)


if __name__== '__main__':
    # main()

    # train_val_loss_curve(experiment_folder='output/')

    evaluator('output/model_0004999.pth', train_dataset_name)











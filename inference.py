from detectron2.engine import DefaultPredictor

import pickle
import csv
from OD_utils import *
import os
from preprocessing_utils import sequence_crop_image

# Step 1: sequence cropping
def sequence_cropping(w_crop, h_crop, inf_img_path, save_img_path):
    for inf_img in os.listdir(inf_img_path):
        inf_img_name, _ =  os.path.splitext(inf_img)
        os.mkdir(save_img_path + inf_img_name)
        save_dir = save_img_path + inf_img_name
        img_ori_path = os.path.join(inf_img_path, inf_img)
        sequence_crop_image(w_crop, h_crop, inf_img_name, img_ori_path, save_dir)

# Step 2: detectron2 detection
def classification_segmentation(pickle_file, weight_file, img_path):
    with open(pickle_file, 'rb') as f:
        cfg = pickle.load(f)
    cfg.MODEL.WEIGHTS = weight_file
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
    predictor = DefaultPredictor(cfg)
    img_name = img_path.split('/')[2]

    output_csv_path = 'inference/pred_csv_json/' + img_name
    final_img_path = 'inference/pred_crop/' + img_name
    # os.makedirs(final_img_path)

    with open(output_csv_path + '.csv', 'w', newline='') as inf:
        header = ['crop_idx', 'class', 'shape', 'left_top_x', 'left_top_y', 'left_bottom_x', 'left_bottom_y',\
                                                'right_top_x', 'right_top_y', 'right_bottom_x', 'right_bottom_y', \
                                                'abs_ltx', 'abs_lty', 'abs_lbx', 'abs_lby', 'abs_rtx', 'abs_rty', 'abs_rbx', 'abs_rby']
        writer = csv.writer(inf)
        writer.writerow(header)
        for img_crop_name in os.listdir(img_path):
            img_crop = os.path.join(img_path, img_crop_name)
            print(img_crop)
            pred_crop_result = detectron2_inference_lst_dic(pickle_file, cfg.MODEL.WEIGHTS, img_crop)
            crop_data = inference_csv_prep(pred_crop_result)
            writer.writerows(crop_data)
            show_save_image(img_crop, predictor, final_img_path + '/' + img_crop_name, False)

# Step 3: visuailzation for prediction on drawing
def prediction_visualization(ori_img, h_crop, w_crop, each_crop_path, inf_img_name, output_path):
    bg_img_path = 'inference/blank.jpg' 
    combine_sequence_crop(h_crop, w_crop, bg_img_path, ori_img, each_crop_path, inf_img_name, output_path)

# Step 4: final output for 3D reconstruction json file
def final_output_json(img_path):
    img_name = img_path.split('/')[2]
    output_csv_path = 'inference/pred_csv_json/' + img_name + '.csv'
    final_json(output_csv_path, 'inference/pred_csv_json/' + img_name + '.json')

# Step 5: modify json to connect images
def mod_json(json_file, e, n):
    with open(json_file, 'r') as f:
        data = json.load(f)
        for key, value in data.items():
            for obj in value:
                obj[2] += e
                obj[4] += e
                obj[3] += n
                obj[5] += n
    with open(json_file, "w") as jsonFile:
        json.dump(data, jsonFile, indent=4)

if __name__== '__main__':
    # Step 1: 
    # sequence_cropping(800, 800, 'inference/sheet_image', 'inference/crop_image/')
    # Step 2: 
    pickle_file = 'output\wall_OD_cfg.pickle'
    weight_file = 'output\model_0004999.pth'
    for img_folder in os.listdir('inference/crop_image'):
        img_path = 'inference/crop_image/'+ img_folder
        img_name = img_path.split('/')[2]
        classification_segmentation(pickle_file, weight_file, img_path)
    # Step 3:
    # for img in os.listdir('inference/sheet_image/'):
    #     name, _ =  os.path.splitext(img)
    #     prediction_visualization(ori_img='inference/sheet_image/'+img, h_crop=800, w_crop=800,\
    #                             each_crop_path='inference/pred_crop/'+name, inf_img_name=name, \
    #                             output_path='inference/pred_sheet/'+img)


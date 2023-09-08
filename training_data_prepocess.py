import numpy as np
import tensorflow as tf
import csv
import matplotlib.pyplot as plt
import cv2
import json
import open3d as o3d
import pandas as pd
import os
from preprocessing_utils import *
import shutil

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# # Step 1: subdivide mesh and create text file with vertex
# def subdivide_mesh(meshfile, txtfile):
#   mesh_ply = o3d.io.read_triangle_mesh(meshfile)
#   mesh = o3d.geometry.TriangleMesh.subdivide_midpoint(mesh_ply,1)
#   vert_arr = np.asarray(mesh.vertices)
#   with open(txtfile, 'w') as f:
#       for v in vert_arr:
#         f.writelines(str(v[0]) + ' ' + str(v[1]) + ' ' + str(v[2]) + '\n')

# # Step 2: downsample point cloud & visualize & get points array
# def downsample_pts(txtfile):
#   pcd = o3d.io.read_point_cloud(txtfile, format='xyz')
#   ds_pcd = o3d.geometry.PointCloud.uniform_down_sample(pcd,10)
#   o3d.visualization.draw_geometries([ds_pcd])
#   pcd_arr = np.asarray(ds_pcd.points)
#   return pcd_arr

# Step 1: read xyz to array -> create histogram from array
def floors_arr(xyzfile, down_thr, up_thr):
  first_floors_arr = []
  second_floors_arr = []
  df = pd.read_table(xyzfile, skiprows=0, delim_whitespace=True, names=['x','y','z'])
  pcd_arr = df.to_numpy()
  for v in pcd_arr:
    if down_thr <= v[1] <= up_thr:
      first_floors_arr.append(v)
    else:
      second_floors_arr.append(v)
  return first_floors_arr, second_floors_arr

def histogram_image(arr, bin, greyimage):
  x = []
  y = []
  for v in arr:
    x.append(v[0])
    y.append(v[2])
  H, xedges, yedges = np.histogram2d(x, y, [int((max(x)-min(x))*bin), int((max(y)-min(y))*bin)])
  H = H.T
  fig = plt.figure()
  ax = plt.axes()
  X, Y = np.meshgrid(xedges, yedges)
  ax.pcolormesh(X, Y, H, cmap='gist_yarg')
  plt.axis('off')
  plt.savefig(greyimage)

def rgb_to_hex(r, g, b):
  return ('{:02X}' * 3).format(r, g, b)

# Step 2: scale image for training
def scale_img(imgfile, resize_w, resize_h, outputimg):
  img = cv2.imread(imgfile)
  resized = cv2.resize(img, (resize_w, resize_h), interpolation=cv2.INTER_AREA)
  cv2.imwrite(outputimg, resized)

# Step 3: create color dictionary append to json file
# -> wall min max color object to array 
# -> clean wall anntotation
# -> change wall anntotation to pixel
# -> add annotation to json file
def hex(txt):
    with open(txt, 'r') as f:
        hex = []
        lines = f.readlines()[1:]
        for annos in lines:
            anno = annos.split(',')
            if anno[2][1:-1] == 'wall':
                hex.append(anno[1])
    return hex

def wall_array(txt, ply):
  wall_arr = []
  wall_hex = hex(txt)
  with open(ply, 'r') as fr:
      contents = fr.readlines()[14:]
      for i in contents:
          c = i.split(' ')
          if c[0] == '3' and c[1] == '0' and c[2] == '1':
              break
          h = rgb_to_hex(int(c[3]),int(c[4]),int(c[5]))
          for wall in wall_hex:
              if h == wall:
                  wall_arr.append([float(c[0]),float(c[1]),float(c[2]),h]) 
  return wall_arr

def dic_color(txt, ply, down_thr, up_thr):
  dic = {}
  wall_arr = wall_array(txt, ply)
  for line in wall_arr:
    if down_thr <= float(line[1]) <= up_thr:
      hex = line[3]
      if dic.__contains__(hex) == False:
        dic[hex] = []
      dic[hex].append([float(line[0]), float(line[2])])
  return dic

def min_max_array(txt, ply, down_thr, up_thr):
  arr = []
  dic = dic_color(txt, ply, down_thr, up_thr)
  for _, value in dic.items():
    min_value = np.amin(value, axis=0)
    max_value = np.amax(value, axis=0)
    arr.append([min_value[0], min_value[1], max_value[0], max_value[1]])
  return arr

def rescale_anno(txt, ply, down_thr, up_thr, area_thr, wh_ratio_thr):
  rescale_arr = []
  arr = min_max_array(txt, ply, down_thr, up_thr)
  for rec in arr:
    w = abs(rec[2]-rec[0])
    h = abs(rec[3]-rec[1])
    if w*h > area_thr and w*h < 1/area_thr:
      if w/h >= wh_ratio_thr or w/h <= 1/wh_ratio_thr:
        pt1 = [rec[0], rec[3]]
        pt2 = [rec[2], rec[3]]
        pt3 = [rec[2], rec[1]]
        pt4 = [rec[0], rec[1]]
        rescale_arr.append([pt1,pt2,pt3,pt4])
  return rescale_arr

def get_xy_lim(arr, bin):
  x = []
  y = []
  for v in arr:
    x.append(v[0])
    y.append(v[2])
  H, xedges, yedges = np.histogram2d(x, y, [int((max(x)-min(x))*bin), int((max(y)-min(y))*bin)])
  H = H.T
  fig = plt.figure()
  ax = plt.axes()
  X, Y = np.meshgrid(xedges, yedges)
  ax.pcolormesh(X, Y, H, cmap='gist_yarg')
  xaxmin, xaxmax = ax.get_xlim()
  yaxmin, yaxmax = ax.get_ylim()
  return xaxmin, xaxmax, yaxmin, yaxmax

def pixel_arr(txt, ply, xyzfile, bin, down_thr, up_thr, area_thr, wh_ratio_thr):
  first_arr, _ = floors_arr(xyzfile, down_thr, up_thr)
  xaxmin, xaxmax, yaxmin, yaxmax = get_xy_lim(first_arr, bin)
  rescale_arr = rescale_anno(txt, ply, down_thr, up_thr, area_thr, wh_ratio_thr)
  arr = []
  x = []
  y = []
  for rec in rescale_arr:
    for pt in rec:
      x.append(pt[0])
      y.append(pt[1])
  fig, ax = plt.subplots()
  points, = ax.plot(x,y, 'ro')
  ax.axis([xaxmin, xaxmax, yaxmin, yaxmax])
  x, y = points.get_data()
  xy_pixels = ax.transData.transform(np.vstack([x,y]).T)
  xpix, ypix = xy_pixels.T
  width, height = fig.canvas.get_width_height()
  ypix = height - ypix
  for xp, yp in zip(xpix, ypix):
    arr.append([xp*3400/640, yp*2200/480])
  return arr

def mani_arr(txt, ply, xyzfile, bin, down_thr, up_thr, area_thr, wh_ratio_thr):
  pts_arr = []
  arr = pixel_arr(txt, ply, xyzfile, bin, down_thr, up_thr, area_thr, wh_ratio_thr)
  for i, pts in enumerate(arr):
    if i%4 == 0:
      pt = []
      pt.append(pts)
      pt.append(arr[i+1])
      pt.append(arr[i+2])
      pt.append(arr[i+3])
    pts_arr.append(pt)
  return pts_arr

# clean up & merge annotation array
def merge_anno(txt, ply, xyzfile, bin, down_thr, up_thr, area_thr, wh_ratio_thr, gap_thr_ud, gap_thr_lr):
  pts_arr = mani_arr(txt, ply, xyzfile, bin, down_thr, up_thr, area_thr, wh_ratio_thr)
  res_bbox = []
  [res_bbox.append(bbox) for bbox in pts_arr if bbox not in res_bbox]
  arr = np.copy(res_bbox).tolist()
  del_lst = []
  for obj in arr:
    for i in range(len(arr)):
      if abs(float(obj[0][0]) - float(obj[1][0])) >= abs(float(obj[3][1]) - float(obj[0][1])) and \
        abs(float(arr[i][0][0]) - float(arr[i][1][0])) >= abs(float(arr[i][3][1]) - float(arr[i][0][1])) and \
        obj != arr[i]:    # horizonal obj
        if (abs(float(arr[i][0][1]) - float(obj[3][1])) <= 30 and \
        float(obj[0][0]) - gap_thr_ud <= float(arr[i][0][0]) <= float(obj[1][0]) + gap_thr_ud) or \
          (abs(float(arr[i][0][1]) - float(obj[3][1])) <= gap_thr_ud and \
        float(arr[i][0][0]) - gap_thr_ud <= float(obj[0][0]) <= float(arr[i][1][0]) + gap_thr_ud) or\
          (abs(float(arr[i][0][0]) - float(obj[1][0])) <= gap_thr_lr and \
            abs(float(arr[i][0][1]) - float(obj[1][1])) <= gap_thr_lr):
          arr[i][0][0] = min(float(obj[0][0]), float(arr[i][0][0]))
          arr[i][0][1] = min(float(obj[0][1]), float(arr[i][0][1]))
          arr[i][1][0] = max(float(obj[1][0]), float(arr[i][1][0]))
          arr[i][1][1] = min(float(obj[1][1]), float(arr[i][1][1]))
          arr[i][2][0] = max(float(obj[2][0]), float(arr[i][2][0]))
          arr[i][2][1] = max(float(obj[2][1]), float(arr[i][2][1]))
          arr[i][3][0] = min(float(obj[3][0]), float(arr[i][3][0]))
          arr[i][3][1] = max(float(obj[3][1]), float(arr[i][3][1]))
          del_lst.append(obj)
      if abs(float(obj[0][0]) - float(obj[1][0])) < abs(float(obj[3][1]) - float(obj[0][1])) and \
        abs(float(arr[i][0][0]) - float(arr[i][1][0])) < abs(float(arr[i][3][1]) - float(arr[i][0][1])) and \
        obj != arr[i]:    # vertical obj      
        if (abs(float(arr[i][0][0]) - float(obj[1][0])) <= 40 and \
        float(obj[1][1]) - gap_thr_ud <= float(arr[i][0][1]) <= float(obj[2][1]) + 40) or\
          (abs(float(arr[i][0][0]) - float(obj[3][0])) <= 40 and \
            abs(float(arr[i][0][1]) - float(obj[3][1])) <= 40):
          arr[i][0][0] = min(float(obj[0][0]), float(arr[i][0][0]))
          arr[i][0][1] = min(float(obj[0][1]), float(arr[i][0][1]))
          arr[i][1][0] = max(float(obj[1][0]), float(arr[i][1][0]))
          arr[i][1][1] = min(float(obj[1][1]), float(arr[i][1][1]))
          arr[i][2][0] = max(float(obj[2][0]), float(arr[i][2][0]))
          arr[i][2][1] = max(float(obj[2][1]), float(arr[i][2][1]))
          arr[i][3][0] = min(float(obj[3][0]), float(arr[i][3][0]))
          arr[i][3][1] = max(float(obj[3][1]), float(arr[i][3][1]))
          del_lst.append(obj)
    for del_obj in del_lst:
      if del_obj in arr:
        arr.remove(del_obj)
  return res_bbox, arr

def anno_json(jsonfile, txt, ply, xyzfile, bin, down_thr, up_thr, area_thr, wh_ratio_thr, gap_thr_ud, gap_thr_lr):
  with open(jsonfile) as j:
    data = json.load(j)
    res_bbox, arr = merge_anno(txt, ply, xyzfile, bin, down_thr, up_thr, area_thr, wh_ratio_thr, gap_thr_ud, gap_thr_lr)
    for poly in arr:
      data['shapes'].append({'label': 'wall', 'points': poly, 'group_id': None, 'shape_type': 'polygon', 'flags': {}})
    del_item = data['shapes'][0]
    data['shapes'].remove(del_item)
  with open(jsonfile, 'w') as j:
    json.dump(data, j, indent=4)

# Step 4: train val test split
def train_val_test():
  root_dir = 'trainingset/'
  val_ratio = 0.1
  test_ratio = 0.1
  os.makedirs(root_dir +'val/')
  os.makedirs(root_dir +'test/')
  allFileNames = os.listdir(root_dir +'train/')
  np.random.shuffle(allFileNames)
  train_FileNames, val_FileNames, test_FileNames = np.split(np.array(allFileNames),
                                                      [int(len(allFileNames)* (1 - (val_ratio + test_ratio))), 
                                                      int(len(allFileNames)* (1 - test_ratio))])
  val_FileNames = [root_dir +'val/' + name for name in val_FileNames.tolist()]
  test_FileNames = [root_dir +'test/'+ name for name in test_FileNames.tolist()]
  print('Total images: ', len(allFileNames))
  print('Training: ', len(train_FileNames))
  print('Validation: ', len(val_FileNames))
  print('Testing: ', len(test_FileNames))
  for name in val_FileNames:
      n = name.split('/')
      shutil.copy(root_dir + 'train/' + n[2], name)
  for name in test_FileNames:
      n = name.split('/')
      shutil.copy(root_dir + 'train/' + n[2], name)

# Step 5: Random cropping dataset
def random_crop(original_img_path, n_crop, crop_width, crop_height, drawing_json_root, save_contain_ele):
  for img in os.listdir(original_img_path):
      print(img)
      name, _ = os.path.splitext(img)
      for num_crop in range(100):
          (crop_img, x, y) = crop_image(original_img_path, name, crop_width, crop_height)
          drawing_json = drawing_json_root + name + '.json' 
          crop_json = save_contain_ele + name + '_' + str(num_crop) + '.json'
          shutil.copy(drawing_json, crop_json)
          with open(crop_json, "r") as file:
              data = json.load(file)
          crop_json, _ = crop_img_vert(data, crop_json, x, y, crop_width, crop_height)
          crop_img.save(save_contain_ele + name + '_' + str(num_crop) + ".jpg", 'JPEG')

if __name__== '__main__':
  # # Step 1:
  # first_arr, sec_arr = floors_arr('dataset/00758-HfMobPm86Xn/HfMobPm86Xn.xyz')
  # x1, x2, y1, y2 = histogram_image(first_arr, 5, '-2.63_-0.06_HfMobPm86Xn.jpg')
  # # Step 2:
  imgfile = 'trainingset/0.4_2.75_nACV8wLu1u5.jpg'
  outputimg = 'QN2dRqwd84J.jpg'
  # scale_img(imgfile, 3400, 2200, outputimg)
  # # Step 3:
  ply_file = 'dataset/00324-DoSbsoo4EAg/DoSbsoo4EAg.ply'
  label_file = 'dataset/00324-DoSbsoo4EAg/DoSbsoo4EAg.semantic.txt'
  jsonfile = 'trainingset\-3.05_-0.12_DoSbsoo4EAg.json'
  xyzfile = 'dataset/00324-DoSbsoo4EAg/DoSbsoo4EAg.xyz'
  bin = 5
  # anno_json(jsonfile, label_file, ply_file, xyzfile, bin,-3.05,-0.12, 0.05, 3, 40, 40)
  # # Step 4:
  # train_val_test()
  # # Step 5:
  original_img_path = 'trainingset/train'
  save_contain_ele = 'trainingset/temp/'
  drawing_json_root = 'trainingset/train/'
  crop_width = 1100
  crop_height = 1100
  n_crop = 100
  random_crop(original_img_path, n_crop, crop_width, crop_height, drawing_json_root, save_contain_ele)
  # # Step 6:
  # run in terminal: "python labelmetococo.py trainingset/test_gt_crop --output test_gt.json"




  
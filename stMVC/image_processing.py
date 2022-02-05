# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 10:17:53 2021

@author: chunman zuo
"""
from PIL.Image import NONE
import cv2
import json
from anndata import AnnData

import matplotlib.pyplot as plt
plt.switch_backend('Agg')

import numpy as np
import pandas as pd
import os
import glob2

from typing import Optional, Union
from anndata import AnnData
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from matplotlib.image import imread
from sklearn.metrics import pairwise_distances

from torchvision import transforms
from torch.utils.data import Dataset

class CustomDataset(Dataset):

	def __init__(self, imgs_path = None, sampling_index = None, sub_path = None, 
				 file_code = None, transform = None):

		file_list = None

		if file_code is not None:
			temp_file_list = []

			for z in list(range(len(file_code))):
				temp_files    =  glob2.glob( sub_path + file_code[z] + "/tmp/*.jpeg" )
				temp_file_list.extend(temp_files)
			file_list = temp_file_list

		else:
			file_list = glob2.glob( str(imgs_path) + "/*.jpeg" )

		self.data      = []
		self.barcode   = []

		if file_code is not None:
			if sampling_index is not None:
				for index in sampling_index:
					self.data.append( file_list[index] )
					#self.barcode.append( file_list[index].rpartition("/")[-1].rpartition("40.jpeg")[0] )
					temp_code1 = file_list[index].rpartition("/")[0].rpartition("/")[0].rpartition("/")[2]
					temp_code2 = file_list[index].rpartition("/")[-1].rpartition("40.jpeg")[0]
					self.barcode.append( temp_code1 + "_" + temp_code2 )
			else:
				for file in file_list:
					self.data.append( file )
					temp_code1 = file.rpartition("/")[0].rpartition("/")[0].rpartition("/")[2]
					temp_code2 = file.rpartition("/")[-1].rpartition("40.jpeg")[0]
					self.barcode.append( temp_code1 + "_" + temp_code2 )
		else:
			for file in file_list:
				self.data.append( file )
				self.barcode.append( file.rpartition("/")[-1].rpartition("40.jpeg")[0] )

		self.transform = transform

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		img_path   = self.data[idx]
		img        = Image.open( img_path )

		image_code = self.barcode[idx]

		if self.transform is not None:
			pos_1 = self.transform(img)
			pos_2 = self.transform(img)

		return pos_1, pos_2, image_code


train_transform_64 = transforms.Compose([
	transforms.RandomResizedCrop(64),
	transforms.RandomHorizontalFlip(p=0.5),
	transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
	transforms.RandomGrayscale(p=0.8),
	transforms.ToTensor(),
	transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

train_transform_32 = transforms.Compose([
	transforms.RandomResizedCrop(32),
	transforms.RandomHorizontalFlip(p=0.5),
	transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
	transforms.RandomGrayscale(p=0.8),
	transforms.ToTensor(),
	transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

test_transform = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])


def tiling(
	adata: AnnData,
	out_path: str = None,
	library_id: str = None,
	crop_size: int = 40,
	target_size: int = 32,
	verbose: bool = False,
	copy: bool = False,
) -> Optional[AnnData]:
	"""
	adopted from stLearn package
	Tiling H&E images to small tiles based on spot spatial location
	"""

	if library_id is None:
		library_id = list(adata.uns["spatial"].keys())[0]

	# Check the exist of out_path
	if not os.path.isdir(out_path):
		os.mkdir(out_path)

	image = adata.uns["spatial"][library_id]["images"][
		adata.uns["spatial"][library_id]["use_quality"]
	]
	if image.dtype == np.float32 or image.dtype == np.float64:
		image = (image * 255).astype(np.uint8)
	img_pillow = Image.fromarray(image)
	tile_names = []

	with tqdm(
		total=len(adata),
		desc="Tiling image",
		bar_format="{l_bar}{bar} [ time left: {remaining} ]",
	) as pbar:
		for barcode, imagerow, imagecol in zip(adata.obs.index, adata.obs["imagerow"], adata.obs["imagecol"]):
			imagerow_down  = imagerow - crop_size / 2
			imagerow_up    = imagerow + crop_size / 2
			imagecol_left  = imagecol - crop_size / 2
			imagecol_right = imagecol + crop_size / 2
			tile           = img_pillow.crop( (imagecol_left, imagerow_down,
											   imagecol_right, imagerow_up)
											)
			tile.thumbnail((target_size, target_size), Image.ANTIALIAS)
			tile.resize((target_size, target_size))

			tile_name = str(barcode) + str(crop_size)
			out_tile  = Path(out_path) / (tile_name + ".jpeg")

			tile_names.append(str(out_tile))

			if verbose:
				print(
					"generate tile at location ({}, {})".format(
						str(imagecol), str(imagerow)
					)
				)
			tile.save(out_tile, "JPEG")

			pbar.update(1)

	adata.obs["tile_path"] = tile_names
	return adata if copy else None


def image_segmentation_spots(adata: AnnData,
							 sub_dir: str     = None,
							 json_file: str   = None,
							 image_width:int  = 2000,
							 image_height:int = 2000,
							 crop_size: int   = 40,
							 region_pro:float = 0.5):

    ## processing the JSON file generated by labelme software

	sh_fname = os.path.join(sub_dir, json_file)
	with open(sh_fname, 'r') as f:
		sh_json = json.load(f)

	shape          = (image_width, image_height)
	check_or_not   = [False] * len(adata)
	cell_type_dict = {}
	
	for sh in sh_json['shapes']:
		polys  = []
		geom   = np.squeeze(np.array(sh['points']))
		pts    = geom.astype(int)
		polys.append(pts)
		mask   = np.zeros(shape)
		cv2.fillPoly(mask, polys, 1)
		mask   = mask.astype(int)
		temp_celltype = sh['label']
		count_status  = 0

		for barcode, imagerow, imagecol in zip(adata.obs.index, adata.obs["imagerow"], adata.obs["imagecol"]):
			imagerow_down  = imagerow - crop_size / 2
			imagerow_up    = imagerow + crop_size / 2
			imagecol_left  = imagecol - crop_size / 2
			imagecol_right = imagecol + crop_size / 2

			spot  = [np.array([[imagecol_left, imagerow_up],[imagecol_left, imagerow_down], [imagecol_right,imagerow_down], [imagecol_right,imagerow_up]])]
			mask1 = np.zeros(shape)
			cv2.fillPoly(mask1, spot, 1)
			mask1 = mask1.astype(int)
			mask2 = mask + mask1

			if np.sum(mask1>0)/np.sum(mask2>1) > region_pro:
				if check_or_not[count_status] is False:
					if cell_type_dict.__contains__(temp_celltype):
						temp_sopt_list = cell_type_dict[temp_celltype]
						temp_sopt_list.append(barcode)
					else:
						temp_spo_l = []
						temp_spo_l.append( barcode )
						cell_type_dict[temp_celltype] = temp_spo_l

					check_or_not[count_status] = True

			count_status = count_status + 1

	return cell_type_dict


#!/usr/bin/python
#-*- coding: utf-8 -*-

import time, pdb, argparse, subprocess
import glob
import os
from tqdm import tqdm
import json
import random

from SyncNetInstance_calc_scores import *

# ==================== LOAD PARAMS ====================


parser = argparse.ArgumentParser(description = "SyncNet");

parser.add_argument('--initial_model', type=str, default="data/syncnet_v2.model", help='');
parser.add_argument('--batch_size', type=int, default='128', help='');
parser.add_argument('--vshift', type=int, default='15', help='');
parser.add_argument('--data_root', type=str, default='', help='');
parser.add_argument('--tmp_dir', type=str, default="tmp", help='');
parser.add_argument('--reference', type=str, default="demo", help='');
parser.add_argument('--real', default=False, action="store_true");

opt = parser.parse_args();


# ==================== RUN EVALUATION ====================

s = SyncNetInstance();

s.loadParameters(opt.initial_model);

path = os.path.join(opt.data_root, "*.mp4")
all_videos = glob.glob(path)

prog_bar = tqdm(range(len(all_videos)))
avg_confidence = 0.
avg_min_distance = 0.

count = 1
for videofile_idx in prog_bar:
	videofile = all_videos[videofile_idx]
	try:
		offset, confidence, min_distance = s.evaluate(opt, videofile=videofile)
		avg_confidence += confidence
		avg_min_distance += min_distance
		prog_bar.set_description('Avg Confidence: {}, Avg Minimum Dist: {}'.format(round(avg_confidence / count, 3), round(avg_min_distance / count, 3)))
		count += 1
		prog_bar.refresh()
	except:
		continue

print ('Average Confidence: {}'.format(avg_confidence/count))
print ('Average Minimum Distance: {}'.format(avg_min_distance/count))



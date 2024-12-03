"""Modulo para o Download dos arquivos do Kaggle."""

import os
import shutil
import kagglehub
import kaggle

new_path = os.path.join(os.getcwd(), "BASE\\")

if not os.path.exists( new_path):
    os.mkdir( "BASE")

path = kagglehub.dataset_download("gorororororo23/plant-growth-data-classification")

list_dir = os.listdir( path)

for arq in list_dir:
    shutil.move( path + "\\" + arq , new_path)

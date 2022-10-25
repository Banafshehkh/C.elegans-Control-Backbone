from tkinter.tix import COLUMN


# Author: Banafsheh Khazali
# Date: Oct 10, 2022


# import packages
import os
from os import listdir
import pandas as pd
import numpy as np
import glob
import pickle
import fnmatch
from tkinter.tix import COLUMN

# Read neuron names and drop NAN, "", ', and u in the names
names_path = r"/Users/banafshehkhazali/Documents/Research/CElegans/"
os.chdir(names_path)
neuron_names =  pd.read_csv(names_path + "neuron_names.txt", names=["Neuron","NAN"])
df = pd.DataFrame(neuron_names)
df.drop("NAN", axis = 1, inplace = True)
df["Neuron"] = df["Neuron"].str.replace("u", "")
df["Neuron"] = df["Neuron"].str.replace("'", "")
df["Neuron"] = df["Neuron"].str.replace(" ", "")
neuronArray = df["Neuron"].values

# Add neuron names to the dataset .npy files and save them as csv files
path_no200 = r"/Users/banafshehkhazali/Documents/Research/Data/saved_dynamics/No200/"
os.chdir(path_no200)
np_yesdata = {}
x = 0
for np_name in glob.glob(path_no200 + '*.npy'):
    np_yesdata[np_name] = np.load(np_name)
    df_no200 = pd.DataFrame(np_yesdata[np_name])
    df_no200.columns = neuronArray
    

    pathn200 = "/Users/banafshehkhazali/Documents/Research/Data/saved_dynamics/No200_withnames/"
    for filename in os.listdir(pathn200):
        df_no200.to_csv(pathn200 + "200n" + str(x) + ".csv" )
        
    x += 1    
    




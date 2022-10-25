# Author: Banafsheh Khazali
# Date: Oct 10, 2022
# Data Preprocesing stage including:
# 1. cut the .npy file timng
# 2. switch rows and columns
# 3. Labeling data (Y/N)
# 4. Visualize a sample of each dataset
# import packages
import os
from os import listdir
import pandas as pd
import numpy as np
import glob
import pickle
import fnmatch

os.path.abspath(".")
allow_pickle = True


# Read all data from both "No" and "Yes" dataset
path_no = r"/Users/banafshehkhazali/Documents/Research/Data/saved_dynamics/No/"
os.chdir(path_no)
file_types = ['npy']
np_nodata = {dir_content: np.load(dir_content)
           for dir_content in listdir(path_no)
           if dir_content.split('.')[-1] in file_types}



path_yes = r"/Users/banafshehkhazali/Documents/Research/Data/saved_dynamics/Yes/"
os.chdir(path_yes)
np_yesdata = {}
for np_name in glob.glob(path_yes + '*.npy'):
    np_yesdata[np_name] = np.load(np_name)



# Counting the number of yes and no data
countNo = len(fnmatch.filter(os.listdir(path_no), '*.npy'))
countYes = len(fnmatch.filter(os.listdir(path_yes), '*.npy'))


def main():
    first_200_rows() 
    first_400_rows()
    first_600_rows()
    

# cut the data until second 200
# for every arrays in np_nodata save the first 200 elements in a new array
# Then show them in pandas format

def first_200_rows():
    """
    This function extract the first 200 rows of the yes and no data
    """
    no_list_200 = []
    yes_list_200 = []
    x = 0
    for np_name in glob.glob(path_yes + '*.npy'):
        np_yesdata[np_name] = np.load(np_name, allow_pickle = True)
        ydata = np_yesdata[np_name]
        ydata200 = ydata[:200, :]
        
        pathy200 = "/Users/banafshehkhazali/Documents/Research/Data/saved_dynamics/Yes200/"
        for filename in os.listdir(pathy200):
            np.save(pathy200 + "200y" + str(x) + ".npy", ydata200)
        x += 1    
        yes_list_200.append(ydata200)
        
    # print(yes_list_200)
    y = 0
    for np_name in glob.glob(path_no + '*.npy'):
        np_nodata[np_name] = np.load(np_name, allow_pickle = True)
        ndata = np_nodata[np_name]
        ndata200 = ndata[:200, :]
        
        pathn200 = '/Users/banafshehkhazali/Documents/Research/Data/saved_dynamics/No200/'
        for filename in os.listdir(pathn200):
            np.save(pathn200 + "200n" + str(y) + ".npy", ndata200)  
        y += 1
        no_list_200.append(ndata200)
        
    # print(no_list_200)

      


# # cut the data until second 400
# # for every arrays in np_nodata save the first 400 elements in a new array
# # Then show them in pandas format

def first_400_rows():
    """
    This function extract the first 400 rows of the yes and no data
    """
    no_list_400 = []
    yes_list_400 = []
    m = 0
    for np_name in glob.glob(path_yes + '*.npy'):
        np_yesdata[np_name] = np.load(np_name)
        ydata = np_yesdata[np_name]
        ydata400 = ydata[:400, :]
        
        pathy400 = "/Users/banafshehkhazali/Documents/Research/Data/saved_dynamics/Yes400/"
        for filename in os.listdir(pathy400):
            np.save(pathy400 + "400y" + str(m) + ".npy", ydata400)
        m += 1    
        yes_list_400.append(ydata400)
        
    # print(yes_list_400)
    n = 0
    for np_name in glob.glob(path_no + '*.npy'):
        np_nodata[np_name] = np.load(np_name)
        ndata = np_nodata[np_name]
        ndata400 = ndata[:400, :]
        
        pathn400 = '/Users/banafshehkhazali/Documents/Research/Data/saved_dynamics/No400/'
        for filename in os.listdir(pathn400):
            np.save(pathn400 + "400n" + str(n) + ".npy", ndata400)  
        n += 1
        no_list_400.append(ndata400)
        
    # print(no_list_400)




  
# cut the data until second 600
# for every arrays in np_nodata save the first 600 elements in a new array
# Then show them in pandas format

def first_600_rows():
    """
    This function extract the first 600 rows of the yes and no data
    """
    no_list_600 = []
    yes_list_600 = []
    i = 0
    for np_name in glob.glob(path_yes + '*.npy'):
        np_yesdata[np_name] = np.load(np_name)
        ydata = np_yesdata[np_name]
        ydata600 = ydata[:600, :]
        
        pathy600 = "/Users/banafshehkhazali/Documents/Research/Data/saved_dynamics/Yes600/"
        for filename in os.listdir(pathy600):
            np.save(pathy600 + "600y" + str(i) + ".npy", ydata600)
        i += 1    
        yes_list_600.append(ydata600)
        
    # print(yes_list_600)
    j = 0
    for np_name in glob.glob(path_no + '*.npy'):
        np_nodata[np_name] = np.load(np_name)
        ndata = np_nodata[np_name]
        ndata600 = ndata[:600, :]
        
        pathn600 = '/Users/banafshehkhazali/Documents/Research/Data/saved_dynamics/No600/'
        for filename in os.listdir(pathn600):
            np.save(pathn600 + "600n" + str(j) + ".npy", ndata600)  
        j += 1
        no_list_600.append(ndata600)
        
    # print(no_list_600)


if __name__ == '__main__':
    main()

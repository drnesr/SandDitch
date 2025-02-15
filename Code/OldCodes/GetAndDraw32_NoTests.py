#!/usr/bin/env python
# coding: utf-8

# <div class="alert alert-block alert-info">
# # pyHydrus3D  
# ## A Python Module to convert  HYDRUS 3D output data to 2D sections
# ### By: Dr. Mohammad Elnesr
# ***

# <div class="alert alert-block alert-success">
# ## **The Basic code**

# In[1]:


get_ipython().run_cell_magic('javascript', '', '// resize ipython notebook output window\nIPython.OutputArea.auto_scroll_threshold = 1000;')


# In[2]:


# Importing important libraries
import numpy as np
import pandas as pd
import linecache #as lc
import math
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import os
# from decimal import Decimal, ROUND_CEILING
import decimal
float_formatter = lambda x: "%.3f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})

from scipy.signal import savgol_filter
from scipy import integrate

from matplotlib.colors import LinearSegmentedColormap as LSCm
import matplotlib.cm as cm
import random
import glob
import shutil
# from matplotlib import rc
# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# ## for Palatino and other serif fonts use:
# #rc('font',**{'family':'serif','serif':['Palatino']})
# rc('text', usetex=True)


# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')


# **Code for importing and converting HYDRUS output to CSV**

# In[4]:


def get_df_from_csv(path, file_name):
    _file = os.path.join(path, file_name)
    if os.path.isfile(_file):
        return pd.read_csv(_file)
    else:
        print ('Warning, the given path does not contain such given file name,         or the path does not exist\n You provided the file name: {}\n ... and the         path as: {}'.format(file_name, path))


# In[5]:


def read_hydrus_data(folder='Current', save_to_csv=True):
    '''
    A function to read both Theat and H files from HYDRUS outputs, 
        then to:
            1- return one dataframe contains both data in a decent format.
            2- save this output to a CSV file (optional, True by default)
    Input:
        The name of the main folder (leave balank for the current folder)
        The option to save_to csv, default =True (Boolean)
    '''
    # Specify the source folder
    if folder=='Current':
        read_dir = os.getcwd()
    else:
        read_dir = folder
        
    # Finding number of nodes in the file
    mesh_file = os.path.join(read_dir, 'MESHTRIA.TXT')
    num_cells = np.array(linecache.getline(mesh_file, 6).split(),int)[0]
    # Define dataframe titles
    titles = ['n', 'x', 'y', 'z'] 
    # Define a list of coordinates
    full_data = [[0,0,0,0]]
    # Set a loop to geather all coordinates from MESHTRIA.TXT file
    for i in range(8, num_cells + 8):
        full_data.append(np.array(linecache.getline(mesh_file, i).split(),float))
    # Convert the list to numpy array then to a dataframe
    coordinates_df = pd.DataFrame(np.array(full_data), columns=titles)
    # Print head and tail of the dataframe to ensure correctness
    # pd.concat([coordinates_df.head(),coordinates_df.tail()])
    
    
    # -----------------------------#
    # To get data from all files   #
    # -----------------------------#
    def get_data_from_file(filename='TH.TXT', caption = 'Theta'):
        '''
        Function to combine all values of a property to a single dataframe 
        inputs:
        filename, the name of the file
        caption, the leading caption of the columns (we will add the portion '_T= xxx')
        where xxx is the timestep
        '''
        # compute number of lines for each timestep
        num_lines = int(math.ceil(num_cells /10.))
        time_steps_remaining = True  # Flag to see if the loop should continue or not.
        times_df = pd.DataFrame([])  # Empty dataframe
        time_loc_start = 2  # The starting cell of the timestep
        while time_steps_remaining:
            line_t = linecache.getline(filename, time_loc_start).split()
            # Check if it is the start of the timestep, otherwise exit
            if line_t[0] == 'Time':
                t = int(line_t[2])
                # Finding the last line of the timestep
                tim_loc_end = num_lines + time_loc_start + 2
                # The starting time is always 0 because steps starts in 1 in HYDRUS
                time_data = [0]  
                # Create the timestep as one long list
                for i in range(time_loc_start + 2, tim_loc_end):
                    time_data.extend(linecache.getline(filename, i).split())
                # Convert the list to DataFrame
                dft=pd.DataFrame(np.array(time_data,float),columns=['{}_T={}'.
                                                                    format(caption,t)])
                if len(times_df) == 0:  # If it is the first timestep
                    times_df = dft
                else:  # Otherwise (for all other timesteps)
                    times_df = pd.concat([times_df, dft], axis=1)
                # Change the start to the probable next timestem (if exist)
                time_loc_start = tim_loc_end + 1
                time_steps_remaining = True if len(linecache.
                                                   getline(filename, 
                                                           time_loc_start)) > 0 else False
                # End IF
        return times_df
    
    # Set the basic dataframe to the coordinates dataframe, to append to it.
    full_df = coordinates_df
    # Looping through the basic output files then to concatenate them all
    for prop in [('TH.TXT','Th'), ('H.TXT','H')]:#, ('V.TXT', 'V')]:
        file_path = os.path.join(read_dir, prop[0])
        # Check if the file exists
        if os.path.isfile(file_path):
            prop_df = get_data_from_file(file_path, prop[1])
            full_df = pd.concat([full_df, prop_df], axis=1)
        else: 
            print ('Warning, the file {} does not exist in the given path'.
                   format(prop[0]))

    # Convert the num column to integer
    full_df[['n']] = full_df[['n']].astype(np.int64)
    # dropping the first row (the zeros row) as it is not necessary
    full_df.drop(0, inplace=True)
    # Saving the resultant dataframe to disk.
    if save_to_csv:
        full_df.to_csv(os.path.join(read_dir, 'nesr_data2.csv'))        
    return full_df


# ## Converting a bunch of HYDRUS files to CSV

# #define source and destination folders
# 
# source = 'C:/Users/DrNesr/Dropbox/@CurrentWork/@Work/NewHydrus/Current'
# 
# output = 'C:/Users/DrNesr/Dropbox/@CurrentWork/@Work/NewHydrus/CSVs'
# #source, output

# In[6]:


def copy_required_files_and_folders(source, destination):
    success, fail =0, 0
    if not os.path.isdir(destination):
        try:
            os.mkdir(destination)
        except:
            print ("Destination directory is not found and cannot be created")
            return None
    items = glob.glob(source + '/*')
    n_folders = list(filter(lambda x: os.path.isdir(x) , items))
    for n_folder in n_folders:
        msg0 = '\nFor the directory {}'.format(n_folder.split('/')[-1])
        n_path = os.path.join(destination, n_folder.split('\\')[-1])
        items2 = glob.glob(n_folder + '/*')
        n_files2 =list(filter(lambda x: not os.path.isdir(x), items2))
        n_files3 = list(filter(lambda x: x[-4:].lower()==".txt", n_files2))
        n_files3_names = list(map(lambda x: x.
                                  split('\\')[-1].lower(), n_files3))
        count_files=len(n_files3_names)
#         print(n_files3_names)
        if count_files==0:
            out_msg= '\tNo (*.TXT) files found!'
        elif count_files < 3:
            out_msg= '\tOnly the files {} exist.'.            format(n_files3_names)
        else:
            out_msg= '\tAll the files copied successfuly.'
        
        if count_files >=2 and 'MESHTRIA.TXT'.                            lower() in n_files3_names:
            # create the folder only if the important files exist
            if not os.path.exists(n_path):
                os.mkdir(n_path)
            # then copy the files
            for file_from in n_files3:
                file_to = file_from.replace(source, destination)
                shutil.copyfile(file_from, file_to)
#                 print ('\t', file_from.split('\\')[-1], ' copied.')
        
        if count_files<3:
            print(msg0)
            print (out_msg)
            fail +=1
        else:
            success +=1
    print ('\nOut of {} scanned folders, we found:'.format(fail+success))
    print ('\t{} folders processed successfully, and'.format(success))        
    print ('\t{} folders contain one or more errors'.format(fail))
    pass

# src='C:\zTest\Current'
# dst= source #'C:\zTest\Current2'
# copy_required_files_and_folders(src, dst) 


# In[7]:


def check_folders_suitability(parent_folder, check_for=['TH.TXT', 
                                                        'H.TXT', 
                                                        'MESHTRIA.TXT']):
    all_folders_are_ok = True
    num_dirs=0
    for _, dirs, _ in os.walk(parent_folder):
        for folder in dirs:
            if 'Genex' not in dirs:
                missing=[]
                num_dirs+=1
                files = os.listdir(os.path.join(parent_folder,folder))
                for file in check_for:
                    if file not in files:
                        missing.append(file)
                        all_folders_are_ok=False
                if len(missing) == 1:
                    print('The file {} is missing from the folder {}'.
                          format(missing[0], folder))
                elif len(missing) > 1:
                    print('The files {} are missing from the folder {}'.
                          format(missing, folder))
                else:
                    pass
                        
                continue
    msg = "Checked {} subfolders, and all of them contain the required files."
    if all_folders_are_ok:
        if num_dirs==0:
            return "No subfolders found in the given directory!"
        return msg.format(num_dirs)
    else:
        return msg.replace('all', 'some').format(num_dirs)
# check_folders_suitability(source)


# In[8]:


def get_data_from_file(filename, num_cells, caption = 'Theta'):
        '''
        Function to combine all values of a property to a single dataframe 
        inputs:
        filename, the name of the file {{with full path}}
        num_cells, the number of nodes in the file.
        caption, the leading caption of the columns (we will add the portion '_T= xxx')
        where xxx is the timestep
        '''
        # compute number of lines for each timestep
        num_lines = int(math.ceil(num_cells /10.))
        time_steps_remaining = True  # Flag to see if the loop should continue or not.
        times_df = pd.DataFrame([])  # Empty dataframe
        time_loc_start = 2  # The starting cell of the timestep
        while time_steps_remaining:
            line_t = linecache.getline(filename, time_loc_start).split()
            # Check if it is the start of the timestep, otherwise exit
            if line_t[0] == 'Time':
                t = int(line_t[2])
                # Finding the last line of the timestep
                tim_loc_end = num_lines + time_loc_start + 2
                # The starting time is always 0 because steps starts in 1 in HYDRUS
                time_data = [0]  
                # Create the timestep as one long list
                for i in range(time_loc_start + 2, tim_loc_end):
                    time_data.extend(linecache.getline(filename, i).split())
                # Convert the list to DataFrame
                dft=pd.DataFrame(np.array(time_data,float),
                                 columns=['{}_T={}'.format(caption,t)])
                if len(times_df) == 0:  # If it is the first timestep
                    times_df = dft
                else:  # Otherwise (for all other timesteps)
                    times_df = pd.concat([times_df, dft], axis=1)
                # Change the start to the probable next timestem (if exist)
                time_loc_start = tim_loc_end + 1
                time_steps_remaining = True if len(linecache.
                                                   getline(filename, 
                                                           time_loc_start)) > 0 else False
                # End IF
        return times_df


# In[9]:


def export_hydrus_data(subfolders):
    for subfolder in subfolders:
        # Finding number of nodes in the file
        mesh_file = os.path.join(source, subfolder, 'MESHTRIA.TXT')
        num_2d = linecache.getline(mesh_file, 1).split()
        num_3d = linecache.getline(mesh_file, 6).split()
        if len(num_2d) == 5: # a 2D simulation
            # reading number of points
            num_cells = int(num_2d[1])
            # Define dataframe titles
            titles = ['n', 'x', 'z']
            # Define a list of coordinates
            full_data = [np.array([0, 0, 0])]
            # define the starting line to read the data
            start_line = 2
        elif len(num_3d) ==2: # a 3D simulation
            # reading number of points
            num_cells = int(num_3d[0])
            # Define dataframe titles
            titles = ['n', 'x', 'y', 'z']
            # Define a list of coordinates
            full_data = [np.array([0, 0, 0, 0])]
            # define the starting line to read the data
            start_line = 8
        else:
            # Not defined
            print ('Error reading number of cells')
            num_cells = 0
        print (subfolder, end="... ")#, num_2d, num_3d, num_cells,type(num_cells))

        # Set a loop to geather all coordinates from MESHTRIA.TXT file
        for i in range(start_line, num_cells + start_line):
            full_data.append(np.array(linecache.getline(mesh_file, i).split(),float)) 
        # print (full_data[:3])
        # Convert the list to numpy array then to a dataframe
        coordinates_df = pd.DataFrame(np.array(full_data), columns=titles)
        # Print head and tail of the dataframe to ensure correctness
        # print(pd.concat([coordinates_df.head(),coordinates_df.tail()]))

        # Reading TH and H files
        
        # 1st: define the path of each
        theta_file = os.path.join(source, subfolder, 'TH.TXT')
        head_file = os.path.join(source, subfolder, 'H.TXT')
        
        # 2nd ensure the file exists
        theta_ok, head_ok = False, False
        if os.path.isfile(theta_file):
            th_df = get_data_from_file(theta_file, num_cells, caption = 'Th')
            theta_ok = True
        else: 
            print ('Warning, the file "TH.TXT" does not exist in the folder: {}'.
                   format(subfolder))
            
        if os.path.isfile(head_file):
            hd_df = get_data_from_file(head_file, num_cells, caption = 'Hd')
            head_ok = True
        else: 
            print ('Warning, the file "H.TXT" does not exist in the folder: {}'.
                   format(subfolder))
    
        # Set the basic dataframe to the coordinates dataframe, to append to it.
        used_dataframes=[coordinates_df]
        # add the existing dataframes only
        if theta_ok: used_dataframes.append(th_df)
        if head_ok: used_dataframes.append(hd_df)
        full_df = pd.concat(used_dataframes, axis=1)
        
        # Convert the num column to integer
        full_df[['n']] = full_df[['n']].astype(np.int64)
        
        # dropping the first row (the zeros row) as it is not necessary
        full_df.drop(0, inplace=True)
        
    #     print(pd.concat([full_df.head(4),full_df.tail(3)]))
        
        # Exporting to CSV
        full_df.to_csv(os.path.join(output, '{}.CSV'.format(subfolders[subfolder])))
        print('saved to {}.CSV'.format(subfolders[subfolder]))


# In[10]:


# Testing the above functions on all the folders in the selected source path
def retrieve_all_csv_files(source_folder, get='all', retrieve_folders_only=False,
                          get_only_new=False, output_folder=None):
    '''
    get= can be 'all', '2d', '3d', or any part of file name
    if retrieve_folders_only=True, the function will retturn a list of the folder
        names, and it will NOT convert any HYDRUS file to CSV.
    If get_only_new=True, the function will look for the files in the output_folder
        if not None, then it will calculate only the files that do not exist.
    '''
    subfolders ={}
    # print('===============Creating Folders list ===============')
    for subdir, dirs, files in os.walk(source_folder):
        for i in dirs:
            if 'Genex' not in dirs:
                if get =='all':
                    subfolders[i]=i[5:]
                elif get.lower() in i[5:].lower():
                    subfolders[i]=i[5:]
                else:
                    continue
    # Creating exceptions list
    
    if get_only_new & (output_folder is not None):
        exceptions = []
        for subdir, dirs, files in os.walk(output_folder):
            for i in files:
                if get =='all':
                    exceptions.append(i[:-4])
                elif get.lower() in i[:-4].lower():
                    exceptions.append(i[:-4])
                else:
                    continue
        
        temp_subfolders={}
        if len(exceptions)>0:
            for k_elem in subfolders.keys():
                exists_flag=False
                for d_pop in exceptions:
                    if subfolders[k_elem] == d_pop:
                        exists_flag=True
                        break
                if not exists_flag:
                    temp_subfolders[k_elem]=subfolders[k_elem]
                        
            subfolders = temp_subfolders         
    if retrieve_folders_only:
        return subfolders
    print('\n'.join('In_Folder: {}, Out_File: {}.CSV'.format(sub,pub) for sub,
                    pub in zip(subfolders, subfolders.values())))
    print('====================================================')
    print('======= Converting HYDRUS files to CSV format ======')
    print('====================================================')
    export_hydrus_data(subfolders)
    print('====================================================')
    print('====  ALL THE FILES WERE CONVERTED SUCCESSFULLY  ===')
    print('====================================================')

# test
# retrieve_all_csv_files(source, get='all', retrieve_folders_only=True,
#                           get_only_new=True, output_folder=output)


# # Reading individual files

# In[11]:


def read_a_level_out(file_path, geom='2D'):
    if geom.lower() =='2d':
        start = 3
    else:
        start = 11
    filename= os.path.join(file_path, 'A_Level.out')
    headers=linecache.getline(filename, start).split()
    #boundary conditions
    bc1=linecache.getline(filename, start+3).split()
    bc2=linecache.getline(filename, start+4).split()
    filename= os.path.join(file_path, 'ATMOSPH.IN')
    headers += linecache.getline(filename, 7).split()
    bc1 += linecache.getline(filename, 8).split()
    bc2 += linecache.getline(filename, 9).split()
    bcs = np.array([bc1, bc2])
    df= pd.DataFrame(data=bcs, columns=headers)
    df=df.apply(pd.to_numeric, errors='ignore')
    return df
    pass
# read_a_level_out(source, '3D').T


# In[12]:


def read_balance_out(file_path):
    filename= os.path.join(file_path, 'Balance.out')
    headers=['Time', 'Volume', 'VolumeW', 'InFlow', 'hMean', 'WatBalT', 'WatBalR']
    reading = True
    start = 5
    balance_info={}
    while reading:
        start +=1
        line_feed = linecache.getline(filename, start).split()
        feed_len = len(line_feed)
        if feed_len < 2: 
            continue
        first_word = line_feed[0].strip()
        if first_word == 'Time':
            # Initiate record
            time = float(line_feed[2])
            balance_info[time]={'Volume':None, 'VolumeW':None, 
                                'InFlow':None, 'hMean':None, 
                                'WatBalT':None, 'WatBalR':None}
        elif first_word in headers:
            balance_info[time][first_word]=float(line_feed[2])
            
#         print(first_word)
        if first_word=='Calculation':# or feed_len>115:
            simulation_time = line_feed[3]
            reading=False
    df = pd.DataFrame.from_dict(data=balance_info).T.reset_index()
    df.columns.values[0] = 'Time'
#     df.rename(columns = {'index':'Time'})
    return simulation_time, df#, columns=headers)
    pass
# results = read_balance_out(source)
# print ('Simulation time = {} seconds.'.format(results[0]))
# results[1] 


# In[13]:


def read_selector_in(file_path, geom='2D'):
    if geom.lower() =='2d':
        is2d= True
        start = 3
    else:
        is2d= False
        start = 11
    filename= os.path.join(file_path, 'SELECTOR.IN')
    headers=['L_Unit', 'T_Unit', 'Category']
    categ={0:'Horizontal plane XY', 
           1:'Axisymmetric Vertical Flow', 
           2:'Vertical Plane XZ',
           3:'3D General Domain'}
    body =[]
    
    def proper_type(x):
        try:
            nf=float(x)
            ni = float(int(nf))
            # print(nf, ni, abs(nf - ni))
            if abs(nf - ni) < 0.0000000000001:
                return int(ni)
            else:
                return nf
        except:
            return x
    
    def replace_text(x):
        if x in ('t', 'f'):
            #return {'t':1, 'f':0}[x]
            return ['f', 't'].index(x)
        elif x in ('mm', 'cm', 'm'):
            return ['mm', 'cm', 'm'].index(x)
        elif x in ('sec', 'min', 'hours', 'days', 'years'):
            return ['sec', 'min', 'hours', 'days', 'years'].index(x)
        elif x in ('s', 'min', 'h', 'd', 'y'):
            return ['s', 'min', 'h', 'd', 'y'].index(x)
        else:
            return x#proper_type(x)
    
    def get_line(pos):
        line_feed=linecache.getline(filename, pos).split()
        return list(map(replace_text,line_feed))
    
    def get_word(pos, loc=0):
        word=get_line(pos)
        if len(word)<1:
            return ''
        else:
            word = word[loc]
        if isinstance(word, str):
            return word.strip()
        else:
            return word
    
    def get_num(p1, p2):
        '''
        p1, the line of 2D file
        p2, the line of 3D file
        '''
        return {True: p1, False:p2}[is2d]
    
    def adjust_body(replaceable):
        for _ in range(len(headers)-len(body)):
            body.append(replaceable)
    

    
    body.append(get_word(6))
    body.append(get_word(7))
    body.append({True: int(get_word(10)), False:3}[is2d])
    headers += get_line(get_num(11,9))[:4]
    body += get_line(get_num(12,10))[:4]

    headers += get_line(get_num(13,11))
    body += get_line(get_num(14,12))
    headers += get_line(get_num(15,13))
    body += get_line(get_num(16,14))

    headers += get_line(get_num(20,18))
    body += get_line(get_num(21,19))
    adjust_body(0)
    
    headers += get_line(get_num(22,20))
    body += get_line(get_num(23,21))
    
    headers += get_line(get_num(24,22))
    body += get_line(get_num(25,23)) 
    
    headers += get_line(27)
    body += get_line(28)    
    headers += get_line(29)
    body += get_line(30)
    
    # Getting data from the DIMENSIO.IN file
    filename= os.path.join(file_path, 'DIMENSIO.IN')
    headers += get_line(2)
    body += get_line(3)
    adjust_body(0)
    
    # Getting data from the Run_Inf.out file
    filename= os.path.join(file_path, 'Run_Inf.out')
    headers += ['TLevel_i', 'Time_i', 'dt_i', 'Iter_i', 'ItCum_i']
    body += get_line(5)
    i=6
    while get_word(i) != 'end':
        i += 1
#         print(i, get_word(i), end='||')
    headers += ['TLevel_e', 'Time_e', 'dt_e', 'Iter_e', 'ItCum_e']
    body += get_line(i-1)
    
    # Getting data from the Balance.out file
    filename= os.path.join(file_path, 'Balance.out')
    headers = ['SimulTime_s'] + headers
    i=10
    while get_word(i) != 'Calculation':
        i += 1
    body = [get_word(i, loc=3)] + body
    
    # finalize
    body = np.array(body)
    headers = np.array(headers)

    df=pd.DataFrame(data=body, index=headers).T
    df=df.apply(pd.to_numeric, errors='ignore')
    return df
    pass
# res=read_selector_in(source, '2d')
# # res.astype(float)
# # res.T.astype(float)
# # res.astype(float).info()
# # res.info()
# # print(res.T)
# res


# In[14]:


def get_one_line_df(folder_path, simulation_name="Nesr simulation", dims='2d'):
    # Get the basic parameters
    df_basic = read_selector_in(folder_path, dims)
    
    # Get the boundary conditions parameters
    df_bcs= read_a_level_out(folder_path, dims).T
    # converting it to one row
    hds0 = ['Time', 'CumQ3', 'hAtm', 'hKode3', 'A-level', 'hCritA', 'rt']
    hdsB = ['Time', 'CumQ3', 'hAtm', 'hKode3', 'A-level', 'hCritA', 'Flux_rt']
    hds = []
    vals = []
    for col in df_bcs.columns:
        hds += list(map(lambda x: x+'BC{}'.format(col), hdsB))
        for idx in hds0:
            if idx =='rt':
                vals.append(df_bcs.loc[idx, col].iloc[0])
            else:
                vals.append(df_bcs.loc[idx, col])
    df_bcs= pd.DataFrame(data=vals, index=hds).T
    
    # Get the mass balance parameters
    df_bal = read_balance_out(folder_path)[1]
    hds0=['InFlow', 'VolumeW', 'WatBalR', 'WatBalT', 'hMean']
    hds=[]
    vals =[]
    for col in hds0:
        for idx in df_bal.index:
            hds.append(col+str(int(df_bal.loc[idx, 'Time'])))
            vals.append(df_bal.loc[idx, col])
    df_bal= pd.DataFrame(data=vals, index=hds).T
    
    # concatenate the 3 dfs
    frames = [df_basic, df_bcs, df_bal]

    df_result = pd.concat(frames,axis=1)
    # df_result.columns
    # df_result.rename({'0':"Custom Name"}, axis='columns')
    df_result = df_result.rename({0:simulation_name}, axis='index')
    return df_result.T


# <div class="alert alert-block alert-success">
# ## **Some Auxillary functions**

# **A function to calculate distance**

# In[15]:


def distance3d(p1, p2=(0, 0, 0)):
    ''' 
    A function to return distance in 3D between two points
    If one point is given, the distance to the origin (0, 0, 0) will be returned
    The function accept only tuples or lists as inputs
    '''
    return math.hypot(math.hypot(p2[0] - p1[0], p2[1] - p1[1]), p2[2] - p1[2])


# **A function to get the output grid of the cross section**

# In[16]:


def get_section_grid(source_df_1, axis_of_section='y', grid_value=1., 
                     default_value=20., output_method='3D', 
                    is_axisymmetric=False):
    ''' 
    if the output_method is 3D, then it will outputs a list of lists, each sublist
    is in the form [x, y, z]
    Otherwise, if the output_method ='2D', then the outputs will be in the form
    [D1, D2], where D1 and D2 are the other axes than that was specified in the 
    axis_of_section, i.e. if axis_of_section='y', then D1 and D2 will be x, z.
    The default_value is the value that will be appended to all list of list in
    the option output_method='3D'
    If the axis_of_section='y' and default_value=20., then the outputs will be
    [[x1, 20., z1], [x2, 20., z2], ...]
    The function returns a tuple of 
        1- a list of sublists in the form [x, y, z] or [D1, D2] as described
            above
        2- a list of two linespace arrays of the two coordinates other than that
            selected
    
    '''
    if is_axisymmetric:
        # if the simulation is Axisymmetric 3D, then take only the 
        # positive quater to compare with the 2D sections
        source_df=source_df_1[(source_df_1.x>=0)&(source_df_1.y>=0)]
    else:
        source_df=source_df_1
        
    # find boundaries of x, y, and z (min then max)
    src_axis = [axs for axs in ['x', 'y','z'] if axs in source_df.columns]
    n_axis = len(src_axis)
    inf = source_df.describe()[src_axis].iloc[[3,7]]
    inf2 = list(inf.values.T)
    mn, mx, iv, lv=[0]*n_axis,[0]*n_axis,[0]*n_axis,[0]*n_axis
    grid = grid_value  # cm
    for i in range(n_axis):
        # minimum and maximum
        mn[i], mx[i] = inf2[i]
        # number of segments
        iv[i] = int(max(abs(mn[i]), abs(mx[i])) // grid) + 1
        # grid of coordinates
        lv[i] = np.linspace(mn[i],mx[i],iv[i])
        pass


     # Now specify the used axes perpendicular to the section
     # I want to define the variable outside the if condition
    used_axes = (0, 2) # if the default axis, y, is used
    if axis_of_section.lower() == 'x':
        used_axes = (1, 2)
    elif axis_of_section.lower() == 'z':
        used_axes = (0, 1)
    else:
        pass  # it is 'y'
    cros_section_grid = []
#     print(output_method)
    if output_method !='3D': # =='2D' for example
        used_axes = (0, 1)
        for outer in lv[used_axes[0]]:
            for inner in lv[used_axes[1]]:
                cros_section_grid.append((outer, inner))
                pass
            pass
    else:  # output_method =='3D'
        for outer in lv[used_axes[0]]:
            for inner in lv[used_axes[1]]:
                if axis_of_section.lower() == 'x':
                    x, y, z = default_value, outer, inner
                elif axis_of_section.lower() == 'z':
                    x, y, z = outer, inner, default_value
                else:  # the default axis_of_section='y'
                    x, y, z = outer, default_value, inner
                cros_section_grid.append((x, y, z))
                pass
            pass
#     return mn, mx, iv, list(lv[0])
    return cros_section_grid, (lv[used_axes[0]], lv[used_axes[1]])


# **A function to get a sliced dataframe for the desired variable at specific axis**

# In[17]:


def get_section_dataframes(source_df, axis_of_section='y', cross_at=20., 
                           tolerance=15., output='before & after'):
    '''
    reads a dataframe, and slices it on x, y, or z axis at a specic location 
    then returns one or two dataframes contain all the points within a specific 
    tolerance around the cross section.
    Inputs:
        1- the main dataframe
        2- the axis of section to slice at (default is y-axis) {'x', 'y', 'z'}
        3- the value at which the cross section occur (default = 20. cm) {float}
        4- the tolerance of the setion (how long the slice will take after and 
            before the cross section)(default = 15. cm) {float}
        5- how to output (what will return?)
            {a- 'before & after', returns two dataframes, one to the left 
                and other to the right of it (default)
             b- 'all', one dataframe contains the two dataframes merged}
        
    '''
    # let us try to get points around the section Y=20
    # it is a section at XZ direction, so we have all values of X and Z
    # but only values of Y=20 plus or minus a tolerance (say 10 cm)
    if axis_of_section.lower() =='x':
        sec_at = source_df.x # the axis of the section
    elif axis_of_section.lower() =='y':
        sec_at = source_df.y # the axis of the section
    else:  # axis_of_section ='z'
        sec_at = source_df.z # the axis of the section
        pass
    
    sec_val = cross_at # the value at which the section occur
    sec_tol = tolerance # the tolerance of the section
    # Find minimum value of the section axis
    sec_min = sec_val - sec_tol
    sec_max = sec_val + sec_tol
    # (to overcome the negative values problem)
    sec_min, sec_max = min(sec_min, sec_max), max(sec_min, sec_max)  
    # theta[(theta.y>=10) & (theta.y<=30)].shape
    if output == 'before & after':
        # Then we find two dataFrames, one after the point, and one before it
        df_after = source_df[(sec_at>=sec_val) & (sec_at<=sec_max)]  #.shape
        df_before = source_df[(sec_at>=sec_min) & (sec_at<=sec_val)]  #.shape
    #     return df_before.shape, df_after.shape
        return df_before, df_after
    else: # outputs one dataframe
        df_full = source_df[(sec_at>=sec_min) & (sec_at<=sec_max)]  #.shape
        return df_full


# In[18]:


def get_grid_values(data_frame, variable=0, time_step = 180, grid = 0.5, 
                    crosses = 35., tol = 10., section ='x', 
                    testing=False, is2d=False, is_axisymmetric=False):
    '''
    
    '''
    
    #Find the variable mask
    v_mask = {0:'Th', 1:'H'}[variable] # , 2:'V'
    if testing: print(v_mask)
        
    # first get the dataframe of the neighbors of the required cross-section
    # (source_df, axis_of_section='y', cross_at=20., tolerance=15., 
    #   output='before & after')
#     print('is2d from get_grid_values: ', is2d)
    if is2d:
        src = data_frame
        scr_cols=[axs for axs in ['x', 'y','z'] if axs in data_frame.columns]     
        points = np.array(src[scr_cols])
    else:
        src= get_section_dataframes(data_frame, axis_of_section=section, 
                                    cross_at=crosses, tolerance=tol, output='full')
        points = np.array(src[['x', 'y','z']])
    z_values = np.array(src[['{}_T={}'.format(v_mask, time_step)]])
    if testing: print ('src shape:{}, points shape:{}, z_values shape::'.
                       format (src.shape, points.shape, z_values.shape))
    if testing: print(src[['{}_T={}'.format(v_mask, time_step)]].head())
    # get the grid info
    # (source_df, axis_of_section='y', grid_value=1., default_value=20., output_method='3D')
    if is2d:
        cs = get_section_grid(data_frame, 
                              axis_of_section='y', grid_value=grid, 
                              default_value=0., output_method='2D')
    else:
        cs = get_section_grid(data_frame, axis_of_section=section, 
                              grid_value=grid, default_value=crosses, 
                              output_method='3D', 
                              is_axisymmetric=is_axisymmetric)
    
    requests = np.array(cs[0])
    # x_vals, z_vals are the two used axes, regardless they are XY, XZ, or YZ
    x_vals, z_vals = cs[1][0], cs[1][1]
    if testing: print ('requests shape:{}, x_vals shape:{}, z_vals shape:{}'.
                       format (requests.shape, x_vals.shape, z_vals.shape))
    
    X, Z = np.meshgrid(x_vals, z_vals)
#     M = griddata(points, z_values, requests).reshape((X.shape[1], X.shape[0])).T
    if testing: print ('points shape:{}, z_values shape:{}, requests shape:{}'.
                       format (points.shape, z_values.shape, requests.shape))

    M = griddata(points, z_values, requests).reshape((X.shape[1], X.shape[0])).T

    return X, Z, M, x_vals, z_vals


# In[19]:


def get_available_timesteps(data_frame):
    '''
    
    '''
    cols = list(data_frame.head())
    mems = list(filter(lambda x: x.find('_T')>0, cols))
    return sorted(list(set(map(lambda x: int(float(x.split('=')[1])),mems))))


# In[20]:


def get_full_dimensions(data_frame):
    xyz = {}
    for dim in ['x', 'y', 'z']:
        if dim in data_frame.columns:
            _t = data_frame[dim]
            _t = _t.min(), _t.max() 
            xyz[dim]=_t
    #     mems = list(filter(lambda x: x.find('h_T')>0, cols))
    return xyz


# In[21]:


def rnd(number, significant_digits=8):
    '''
    
    '''
    return round(number*10**significant_digits)/10.**significant_digits

def round_to_significance(number, significance, direction='up'):
    '''
    
    '''
    if direction == 'up':
        num = math.ceil(number/significance) * significance
    else: 
        num = math.floor(number/significance) * significance
    return rnd (num, 4)


# In[22]:


def smooth_series(series, odd_envelop=51, plynomial_degree=3):
    '''
    A simple function to use savgol_filter to smooth any array-like series
    the odd_envelop must be odd number, if an even is giver, it will be 
    increased by 1, plynomial_degree must be >=2, other wize it will be 
    set to 3.
    one must to add the statement from scipy.signal import savgol_filter 
    at the begining of the code.
    '''
    # correcting the inputs
    if odd_envelop % 2 !=1:
        odd_envelop +=1
    if odd_envelop <3:
        odd_envelop = 3
    if plynomial_degree<2:
        plynomial_degree =3
    odd_envelop = int(odd_envelop)
    plynomial_degree =int(plynomial_degree)
    
#     if isinstance(series, pd.Series):
#     if isinstance(series, np.ndarray):
    if not isinstance(series, pd.Series):
        sss= savgol_filter(series, odd_envelop, plynomial_degree)
        return pd.Series(sss)
    else:
        return pd.Series(savgol_filter(series, odd_envelop, plynomial_degree))


# In[23]:


def get_legend_range(mn, mx):
    '''
    
    '''
    rg = mx - mn
    vnn = '{:.2E}'.format(rg)
    ew=vnn.split('E')
    ws = float(ew[0]), float(ew[1])
    wq = int(float(ew[0])), 10**int(float(ew[1]))/10.
    step = wq[0]*wq[1]
    rn = round_to_significance(mn, step, direction='up')
    rx = round_to_significance(mx, step, direction='dn')
#     return vnn, ew, ws, wq, rr, rnd(rr)
    # return vnn, ws, wq, step, (mn,rn), (mx, rx), np.arange(rn+step, 
    # rx+step, step)
    return np.arange(rn+step, rx+step, step)


# In[24]:


def draw_contour(X, Z, M, levels=None, 
                 plot_title="ElNesr cross sectional contour map",
                x_step=10., z_step=25., mirror_x=False, mirror_z=False,
                return_figure_object=False):
    '''
    
    '''
    fig = plt.figure(num=None, figsize=(18, 7), dpi=80, facecolor='w', edgecolor='k');
    origin = 'lower'
    
    if levels is None:
        try:
            levels = get_legend_range(np.nanmin(M),np.nanmax(M))#np.arange(0.15, 0.42, 0.03)
        except:
            levels = get_legend_range(-.15, 0.15)
        
    CS_lines = plt.contour (X, Z, M, levels, cmap=plt.cm.Accent_r, 
                            linewidths=(0.25,), origin=origin, extend='both')
    
    CS_fill  = plt.contourf(X, Z, M, levels, cmap=plt.cm.YlGn, 
                            origin=origin, extend='both')
    
    CS_fill.cmap.set_under('oldlace')
    CS_fill.cmap.set_over('darkslategrey')
    plt.title(plot_title)
    plt.ylabel("Depth (cm)")
    cols = plt.cm.Accent_r(CS_lines.norm(CS_lines.levels))
    plt.clabel(CS_lines, linewidths=4, fmt='%2.2f', fontsize='x-large', 
               colors=cols, inline=True, inline_spacing=10)
    plt.colorbar(CS_fill)

    def adjust_max_and_min(_min, _max, _step):
        nn, xx, ss = _min, _max, _step
        if xx <= 0.:
            nn, xx = xx, nn
            if ss>0:
                ss = -ss
        return nn,xx, ss
        
    def adjust_axis_labels(_min, _max, _step):
        nn, xx, ss = adjust_max_and_min(_min, _max, _step)

        x_list = np.arange(nn,xx, ss)
        if abs(x_list[-1]-xx)>3: # The last number is far enough from 
                                 # the maximum element
            x_list = np.hstack([x_list, xx])
        else:  # The last number is too close to the maximum element
            x_list = np.hstack([x_list[:-1], xx])
        return x_list

    def adjust_mirrored_labels(_min, _max, _step):
        nn, xx, ss = adjust_max_and_min(_min, _max, _step)
        x_mid=(xx-nn)/2.
        
        if x_mid < 0:
            right_list = adjust_axis_labels(min(x_mid, xx), max(x_mid, xx), ss)
            left_list = right_list - x_mid
            right_list = x_mid - right_list 
            
            left_list.sort()
            label_list = np.hstack([left_list[:-1], right_list])
            real_list =x_mid-label_list 
            label_list=label_list[::-1]
            return real_list, label_list
        elif x_mid > 0:
            right_list = adjust_axis_labels(x_mid, xx, ss)
            left_list = 2 * x_mid - right_list
            left_list.sort()
            real_list = np.hstack([left_list[:-1], right_list])
            label_list =real_list - x_mid
            label_list =tuple(['{:3.1f}'.format(x) for x in label_list])
            return real_list, label_list
        else:
            real_list, label_list = None, None
            return real_list, label_list
        
    if mirror_x:
        if x_step is not None:
            ticks, labels = adjust_mirrored_labels(X.min(),X.max(), x_step)
            plt.xticks(ticks, labels)
    else: # No Mirroring
        if x_step is not None:
            plt.xticks(adjust_axis_labels(X.min(),X.max(), x_step))

    if mirror_z:
        if z_step is not None:
            ticks, labels = adjust_mirrored_labels(Z.min(),Z.max(), z_step)
            plt.yticks(ticks, labels)
    else: # No Mirroring
        if z_step is not None:
            plt.yticks(adjust_axis_labels(Z.min(),Z.max(), z_step))
    ax = plt.gca()
    ax.grid(True, zorder=0)

    if return_figure_object:
        return fig
    else:
        plt.show()


# In[25]:


def draw_full_contour(data_frame,variable=0, time_step=180, grid= 0.5, 
                      crosses=35., tol=10., section= 'x', levels=None,
                      plot_title="ElNesr cross sectional contour map",
                      return_arrays=True, x_step=None, z_step=None, 
                      mirror_x=False, mirror_z=False, is2d=False, 
                      output_the_contour=True, is_axisymmetric=False,
                      return_figure_object=False):
    
    '''
    Either (1) set the return_arrays to True and use on right 
                hand side of equal sign, 
    OR     (2) set the return_arrays to False and use the function as is.
    Examples:
    (1)
       arrays = draw_full_contour(data_frame,variable, time_step, grid, 
                                   crosses, tol, section)
       It will draws the chart AND sets arrays=X, Z, M, levels
    (2)
       draw_full_contour(data_frame,variable, time_step, grid, crosses, 
                           tol, section, return_arrays=False)    
    '''
#     print('is2d=', is2d)
    X, Z, M, x_vals, z_vals = get_grid_values(data_frame, variable, 
                                              time_step, grid, crosses, 
                                              tol, section, is2d=is2d,
                                             is_axisymmetric=is_axisymmetric)
    # print(x_vals.shape, z_vals.shape, X.shape, Z.shape, M.shape)
    if levels is None:
        levels = get_legend_range(np.nanmin(M),np.nanmax(M))#np.arange(0.15, 0.42, 0.03)

    mn, mx = np.nanmin(M),np.nanmax(M)
    # print (mx,mn, mx-mn)
    if mx - mn < 0.000000001:
        print('For the requested contour map of {}'.format(plot_title), end='. ')
        print ("The map has one value only ({}), no contour map will be drawn.".
               format(mn))
        can_draw_figure=False
    else:
        can_draw_figure=True
    
    if not output_the_contour and not return_figure_object:
        fig = None
    else:
        if can_draw_figure:
            fig = draw_contour(X, Z, M, levels, plot_title, x_step, z_step, 
                             mirror_x, mirror_z, return_figure_object);
        else:
            fig = None
        
#     exit()
    if return_arrays:
        if output_the_contour:
            if return_figure_object:
                return X, Z, M, levels, fig
            else: # return_figure_object=False
                display(fig)
                return X, Z, M, levels
        else:  #output_the_contour=False
            if return_figure_object:
                return X, Z, M, levels, fig
            else: # return_figure_object=False
                return X, Z, M, levels
    else:  #return_arrays=False
        if output_the_contour:
            if return_figure_object:
                return fig
            else: # return_figure_object=False
                display(fig)
        else:  #output_the_contour=False
            if return_figure_object:
                return fig


# In[26]:


def reduce_crossed_at_list(cs_list):
        ''' read the reduce_auto_list above'''
        a1, an = cs_list[0], cs_list[-1]
        if len(cs_list) %2 ==1: # Odd list
            cs2= cs_list[1:-1][1::2]
            return np.hstack([a1, cs2, an])
        else: # even list
            cs2= cs_list[1:-1]
            zl = zip (cs2[::2], cs2[1::2])
            return np.hstack([a1, list(map(np.mean, list(zl))), an])


# In[27]:


def draw_cross_sections(input_array, xs_array, zs_array, direction='z', 
                        crossed_at_list=None, number_of_sections=10,
                       measured_value='Th', reduce_auto_list=False):
    '''
    inputs:
        input_array, xs_array, zs_array are two_dim_array is a 2D array, 
        may be numpy array or a list of lists
        xs_array, zs_array might be 1D arrays
        direction:  if = 'z' the curves represent cross sections at z e.g -30, -15
                    if = 'x' the curves represent cross sections at x e.g. 45, 60
                    if = 'y' the curves represent cross sections at y e.g. 45, 60
        crossed_at_list if None, then it requires number_of_sections that 
                                        will be calculated
        measured_value='Th' if the curves represent moisture content (default), or 
                       'H' if the curves represent suction pressure 
        reduce_auto_list: if the crossed_at_list is not provided(None), 
                        then if this is True, 
                   it will take every other line in the generated list. 
                   For example:
                   if the generated list is [0, 10, 20, 30 ,40, 50, 60] 
                   it will display only 
                   [0, *, 20, * ,40, *, 60] only. 
                   while if the provided list is even [0,,,,70]
                   it will display [0, 15, 35, 55, 70]. the default is False
    '''
#     colors=[(1, 0, 0), (0, 1, 0), (0, 0, 1)] # R -> G -> B
#     colors=['indigo', 'darkviolet', 'darkblue', 'blue', 'darkmagenta', 'darkcyan', 
#             'darkgreen', 'darkolivegreen', 'olive', 'darkgoldenrod', 'firebrick', 
#                 'red' ] # R -> G -> B
#     colors=['darkviolet', 'darkblue', 'blue', 'magenta', 'darkcyan', 'darkgreen', 
#             'green', 'y', 'darkorange', 'firebrick', 'red']    
    colors=['darkviolet', 'y','blue', 'gold','darkgreen','darkcyan', 'yellow','red']
#     random.shuffle(colors)
#     print (colors)
    cmap_name = 'Nesr_cmap'
    n_bin=len(colors)*2
    cm = LSCm.from_list(cmap_name, colors, N=n_bin)
    col_map = ['Paired', 'nipy_spectral', 'brg', 'prism', 'tab10', 
               'tab20', 'tab20b'][0]
    col_map = cm
    def reduce_crossed_at_list(cs_list):
            ''' read the reduce_auto_list above'''
            a1, an = cs_list[0], cs_list[-1]
            if len(cs_list) %2 ==1: # Odd list
                cs2= cs_list[1:-1][1::2]
                return np.hstack([a1, cs2, an])
            else: # even list
                cs2= cs_list[1:-1]
                zl = zip (cs2[::2], cs2[1::2])
                return np.hstack([a1, list(map(np.mean, list(zl))), an])            
                
    
    def draw_cs_x(depth_df, cs_range, axis_label, title_part):

        depth_cs_df=[]
        for sec in cs_range:
            current_series= depth_df.loc[sec]
            name = '@{:04.1f} cm'.format(current_series.name)
            idx =  current_series.index
            smoothed = smooth_series(current_series)
            smoothed.index = idx
            smoothed.name = name
            depth_cs_df.append(smoothed)

        depth_cs_df= pd.concat(depth_cs_df,axis=1)
        depth_cs_df.head(10)
        xs =depth_df.columns
        ax = depth_cs_df.plot(figsize=(9,6), grid=True, colormap=col_map, 
                              xlim=(xs.min(), xs.max()) )
        ax.set_ylabel(axis_label, fontsize=12)
        ax.set_xlabel(r'Horizontal distance in {} direction $(cm)$'.
                      format(direction), fontsize=12)
        _ti='The change in {} accross the horizontal distance, at different depths'
        ax.set_title(_ti.format(title_part),
                     fontsize=16, y=1.08)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10),
          fancybox=True, shadow=True, ncol=5)
        pass
    

    def draw_cs_z(x_df, cs_range, axis_label, title_part):
        x_cs_df=[]
        for sec in cs_range:
            ts=x_df.loc[:,sec].reset_index()
            ts.rename(columns={ts.columns[0]: "@{:04.1f} cm".
                               format(ts.columns[1]), 
                               ts.columns[1]: 'Value'}, inplace=True)
            sss=ts['Value'].apply(lambda x: int(x*1000)/1000)
            ts['Value']= smooth_series(sss)
            ts.set_index('Value', inplace=True)
            x_cs_df.append(ts)
        x_cs_df= pd.concat(x_cs_df,axis=0)#, keys='df{}'.format(cs_range))
        # ms =depth_df.values
        zs =depth_df.index
        ax = x_cs_df.plot(figsize=(6,8), grid=True, colormap=col_map,
                          ylim=(zs.min(), zs.max())) #, xlim=(ms.min(), ms.max()) )
        ax.set_xlabel(axis_label, fontsize=12)
        _ti='The change in {} accross depth, at different horizontal distances'
        ax.set_ylabel(r'Depth under soil $(cm)$',fontsize=12)
        ax.set_title(_ti.format(title_part),
                     fontsize=16, y=1.08)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10),
                  fancybox=True, shadow=True, ncol=5)
        pass    
    
    
    # Correct inputs to be numpy arrays
    inputs = [input_array, xs_array, zs_array]
    for i, lst in enumerate(inputs):
        if isinstance(lst, list):
            inputs[i] = np.ndarray(lst)
    input_array, xs_array, zs_array = inputs 
    
    # Check if the xs and zs are passed as 2Dim or 1Dim.
    # If passed as 2D, we take the axis that is corresponding to each.
    if xs_array.ndim == 2:
        _x = xs_array[0,:]
    if zs_array.ndim == 2:
        _z = zs_array[:,0]

    depth_df = pd.DataFrame(input_array, index=_z, columns=_x, dtype=float)
    
    #Create the crossing list if none provided
    if crossed_at_list is None:
        if direction == 'x' or direction == 'y':
            zx, zn = _z.max(), _z.min()
        elif direction == 'z':
            zx, zn = _x.max(), _x.min()
        step = (zx - zn)/number_of_sections
        cs_range = np.arange(zn, zx, step)
        if reduce_auto_list:
            cs_range = reduce_crossed_at_list(cs_range)
    
    # set the axis labels
    if measured_value == 'Th':
        axis_label = r'Moisture content $(cm^{3}/cm^{3})$'
        title_part = 'moisture content'
    elif measured_value == 'H':
        axis_label = r'Pressure head $(cm)$'
        title_part = 'pressure head'
    else:
        axis_label = 'Undefined measure'
        title_part = 'undefined measure'
        
    
    function = {'x': draw_cs_x, 'y': draw_cs_x, 'z': draw_cs_z}
    function[direction](depth_df, cs_range, axis_label, title_part)    


# In[28]:


def integrate_volume(sX, sZ, sM, method='Simp', get_average=False, 
                     separate_negatives=False):
    """
    
    """
    if separate_negatives:
        # in this case, the sM will be divided into two arrays, 
#         one with positives, and one for negatives
        positives_array = np.where(sM>0,sM,0)
        positives_results = integrate_volume(sX, sZ, positives_array, 
                                             method=method, 
                                             get_average=get_average, 
                                             separate_negatives=False)
        negatives_array = np.where(sM<=0,sM,0)
        negatives_results = integrate_volume(sX, sZ, negatives_array, 
                                             method=method, 
                                             get_average=get_average, 
                                             separate_negatives=False)
        if get_average:
            total_results = (positives_results[0]+negatives_results[0], 
                            positives_results[1]+negatives_results[1], 
                            positives_results[2])
        else:
             total_results = positives_results+negatives_results
        return positives_results, negatives_results, total_results
    
    requested_method = method.lower()[:4]
    if requested_method == 'simp':
        vol = integrate.simps(integrate.simps(sM, sX, axis=1),sZ)
    elif requested_method == 'trap':
        vol = np.trapz(np.trapz(sM, sX, axis=1),sZ)
    else: # requested_method == 'mean':
        vol = 0.5 * (integrate.simps(integrate.simps(sM, sX, axis=1),sZ) 
                     + np.trapz(np.trapz(sM, sX, axis=1),sZ))
    if get_average:
        Lx, Lz = (sX.max()-sX.min()), (sZ.max()-sZ.min())
        area= Lx*Lz
        return vol, vol/area, area
    else:
        return vol
    pass


# In[29]:


def draw_difference(arr1, arr2, scale_from=0, custom_levels=None,
                    x_step=10., z_step=25., mirror_x=False, mirror_z=False, 
                    calculate_volume=False, calculate_average=False, 
                    no_contours=False, separate_negatives=False,
                    calculate_volume_percent=False,
                    return_calculations=True, return_figure_object=False):
    """
    A function that calculates the difference between two arrays. 
    Main inputs:
    arr1, arr2: the two arrays that will calculate the difference from each other,
                    Each array shoud be a tuple of (X, Z, M, levels),  
                    all should be nupy arrays of the same size,
                    the difference will be calculated (arr1-arr2)
    scale_from: The levels of the contour will be taken from arr1, unless 
                    this argument is adjusted to be = 2, 
                    then we will take x, y, levels from arr2
                    Default=0
    custom_levels: Only if the scale_from=0, the contour levels will be read from
                    what is provided by custom_levels, they should be a tuple 
                    in the format (min, max, step)
                    Default=None
    x_step: The step of the scale in the horizontal (x) direction, Default=10.
    z_step: The step of the scale in the vertical (z) direction, Default=25.
    mirror_x: If true, the x values will be shown as 0 in the middle, 
                    positive values to its right, and negative values to its left
                    Default=False
    mirror_z: If true, the z values will be shown as 0 in the middle, 
                    positive values upward, and negative values downward
                    Default=False
    calculate_volume: if True, the function will calculate the full volume 
                    of the difference
                    Default= False
    calculate_average: if True, the function will calculate the average 
                    moisture/head of the difference
                    Default= False
    no_contours: if True, the function will not return any drawing, it will 
                    just return volumemand average moisture if they are selected. 
                    Default=False
    separate_negatives: If True, the calculated volumes and heights will 
                    be performed for positives and negatives individually, 
                    then will be added together.
                    
    It has 2 main outputs:
    1- Draws a contour map of the difference if no_contours=False, or not set 
        (the default =False)
    2- Outputs the volume stuff: (IF the calculate_volume_percent=False)
        a- If calculate_average=True, and separate_negatives=False,
                it will output a tuple of:(volume, average, section area)
        b- If calculate_average=False, and separate_negatives=False, 
                it will return the volume only (float)
        c- If calculate_average=True, and separate_negatives=True, 
                it will return a tuple of 3 tuples, each one contains:
                a tuple of  (volume, average, section area). 
                The 3 tuples are for 
                (positives, negatives, and totals) {respectively}
        d- If calculate_average=False, and separate_negatives=True, 
                it will return a tuple of 3 tuples, each one contains:
                a tuple of  (volume). The 3 tuples are for 
                (positives, negatives, and totals) {respectively}
    *- IF the calculate_volume_percent=True:
            The function will return the same outputs from 2 plus
            number of members will be added to the difference tuple:
            a- a tuple of (vol_diff/vol_base, avg_diff/avg_base)
            b- a tuple of (vol_diff/vol_base)
            c- a tuple of tuples ((p), (n), (t)), where each of p, n, t  is
                  of the form(vol_diff/vol_base, avg_diff/avg_base) for 
                  positives, negatives, and totals respectively.
            d- a tuple of (p, n, t), where each of p, n, t  is
                  of the form(vol_diff/vol_base) for 
                  positives, negatives, and totals respectively.
                
    If both calculate_volume=False, and no_contours=False, 
    it will return None with a warning
    """
    # To draw diffecence between two contour maps
    if scale_from ==2:
        _x, _z, _m , _levels= arr2
        difference_matrix = arr2[2]-arr1[2]
    else:
        _x, _z, _m , _levels= arr1
        difference_matrix = arr1[2]-arr2[2]
        pass
    
    if not no_contours:
        if scale_from ==0:
            _ti="Difference between two contours, Specific scale"
            fig = draw_contour(_x, _z, difference_matrix, levels=custom_levels, 
                         plot_title=_ti,
                         x_step=x_step, z_step=z_step, 
                         mirror_x=mirror_x, mirror_z=mirror_z, 
                         return_figure_object=return_figure_object);
        else:
            _ti="Difference between two contours, Normal scale"
            fig = draw_contour(_x, _z, difference_matrix, levels=_levels, 
                         plot_title=_ti,
                         x_step=x_step, z_step=z_step, 
                         mirror_x=mirror_x, mirror_z=mirror_z,
                         return_figure_object=return_figure_object);
    if not return_calculations and return_figure_object:
        return fig
    
    if calculate_volume_percent:
        base_array = _m
        base_volumes = integrate_volume(_x[0], _z[:,0], base_array, 
                                get_average=calculate_average, 
                                separate_negatives=separate_negatives)
        
        diff_volumes = integrate_volume(_x[0], _z[:,0], difference_matrix, 
                                get_average=calculate_average, 
                                separate_negatives=separate_negatives)
#         print (base_volumes, diff_volumes,'\n')
        if isinstance(diff_volumes,tuple):
            prc_vol= list(diff_volumes)
        else: # Numpy array
            prc_vol = diff_volumes.tolist()
        
        if calculate_average & separate_negatives:
            # the resturns will be in the form: ((Vol, Th, A), (Vol, Th, A), 
            #                                   (Vol, Th, A))
            # where the groups are for +ve, -ve, and totals
            for res in range(3):
                prc_vol.append(diff_volumes[res][0] / base_volumes[2][0])
                pass
#             prc_vol[-2] = prc_vol[-1]-prc_vol[-3]
        elif calculate_average & (not separate_negatives):
            # the resturns will be in the form: (Vol, Th, A)
            prc_vol.append(diff_volumes[0] / base_volumes[0])
            pass
        elif (not calculate_average) & separate_negatives:
            # the resturns will be in the form: (Vol, Vol, Vol)
            # where the groups are for +ve, -ve, and totals
            # volume ratio
            for res in range(3):
                prc_vol.append(diff_volumes[res] / base_volumes[res])
                pass
        else: #(not calculate_average) & (not separate_negatives)
            # the resturns will be in the form: Vol
            prc_vol=(prc_vol, diff_volumes / base_volumes)
            pass
        
        if return_figure_object:
            return tuple(prc_vol),fig
        else:
            return tuple(prc_vol)
                           
                           
            
    if calculate_volume:
        Lx, Lz = _x[0], _z[:,0]
        calcs= integrate_volume(Lx, Lz, difference_matrix, 
                                get_average=calculate_average, 
                                separate_negatives=separate_negatives)
        if return_figure_object:
            return calcs, fig
        else:
            return calcs
    
    # If both calculate_volume=False, and no_contours=False, 
    # it will return None with a warning
    if no_contours and not calculate_volume:
        print ("Warning, both calculate_volume and no_contours are set to False")
        print ("         Please set at least one argument to True.")
        return None


# In[30]:


def get_results_for_grid(arr1, arr2, scale_from=0,
                        grid=(7, 5)):
    """
    
    
    """
    # To draw diffecence between two contour maps
    if scale_from ==2:
        _x, _z, _m , _levels= arr2
    else:
        _x, _z, _m , _levels= arr1
        pass
    
    # identify grid limits
    g_x, g_z = grid
    l_x, l_z = _x.shape[0],_x.shape[1]
    # get quotient and remainder of each dimension
    n_x, n_z = l_x // g_x, l_z // g_z
    r_x, r_z = l_x % g_x, l_z % g_z
    #
    lst_x, lst_z= [[0, n_x] for _ in range (g_x)], [[0, n_z] for _ in range (g_z)]
    #determining initiation and ending of the grid
    x_i, x_e, z_i, z_e = r_x // 2, r_x // 2, r_z // 2, r_z // 2
    x_i += r_x % 2
    z_i += r_z % 2
    lst_x[0][1] += x_i + 1
    lst_x[-1][1] = l_x +1
    lst_z[0][1] += z_i + 1
    lst_z[-1][1] = l_z + 1
    for i in range(1, g_x):
        lst_x[i][0]=lst_x[i-1][1]
        lst_x[i][1]=lst_x[i][0] + n_x + 1
    for i in range(1, g_z):
        lst_z[i][0]=lst_z[i-1][1]
        lst_z[i][1]=lst_z[i][0] + n_z + 1

    results=[]
    for j, x_g in enumerate(lst_x):
        x_gi, x_ge = x_g
        for k, z_g in enumerate(lst_z):
            z_gi, z_ge = z_g
            s_arr1=[0,0,0,0]
            s_arr2=[0,0,0,0]
            for i in range(3):
                s_arr1[i] = arr1[i][x_gi:x_ge:,z_gi:z_ge]
                s_arr2[i] = arr2[i][x_gi:x_ge:,z_gi:z_ge]
            s_arr2[3] = arr2[3]
            s_arr1[3] = arr1[3]
            temp_results1 = draw_difference(s_arr1, s_arr2, calculate_volume=True, 
                            scale_from=scale_from,
                            custom_levels=get_legend_range(0, 0.2),
                            calculate_average=True, no_contours=True, 
                            separate_negatives=True, 
                            calculate_volume_percent=True)
            tr1= temp_results1
            temp_results2 = (j, k, s_arr1[0].mean(), s_arr1[1].mean(),
                             len(s_arr1[2]), np.nanmean(s_arr1[2]), 
                             np.nanmin(s_arr1[2]), np.nanmax(s_arr1[2]), 
                             np.nanstd(s_arr1[2]),  
                             tr1[0][2], tr1[0][0], tr1[0][1], 
                             tr1[1][0], tr1[1][1], tr1[2][0], tr1[2][1], 
                             tr1[3]*100., tr1[4]*100., tr1[5]*100.) 
            

            results.append(temp_results2)
    results_head=['x_cord', 'z_cord', 'x_average', 'z_average', 
                  'm_count', 'm_average', 'm_min', 'm_max', 'm_std', 
                  'element_area', 'dif_vol_positive', 'dif_avg_positive', 
                  'dif_vol_negative', 'dif_avg_negative', 
                  'dif_vol_all', 'dif_avg_all', 'pos_vol_ratio%', 'neg_vol_ratio%',
                  'full_vol_ratio%']
    df_vol_results = pd.DataFrame.from_records(results, columns=results_head)
    return df_vol_results


# <div class="alert alert-block alert-success">
# ## **Collective functions for final analysis of all textures**

# In[31]:


# ===========================================================
# ======== First, converting HYDRUS files to CSV ============
# ===========================================================
# ======= 1- Moving TXT files to speciial folder ============
# ===========================================================
def move_hydrus_txt_to_folder():

    ll=list(os.walk(source_folder))
    soils_folders=ll[0][1]
    i = -1
    frames = []
    for soil in soils_folders:
        subfolder= os.path.join(source_folder, soil)
        ll=list(os.walk(subfolder))
        simul_folders =ll[0][1]
        print ('\n',soil, len(simul_folders), end=' Copying... ')

        copy_required_files_and_folders(subfolder, HYDRUS_TXT)

        print ('Done.')
    pass
# move_hydrus_txt_to_folder()   


# In[32]:


# ===========================================================
# ======== First, converting HYDRUS files to CSV ============
# ===========================================================
# ======== 2- Converting TXT files to CSV files =============
# ===========================================================

# To convert HYDRUS files to CSV
def convert_HYDRUS_files_to_CSV():
    retrieve_all_csv_files(HYDRUS_TXT, 
                       get='all', 
                       retrieve_folders_only=False, 
                       get_only_new=True, 
                       output_folder=HYDRUS_CSV)
    pass
# convert_HYDRUS_files_to_CSV()


# In[33]:


# ===========================================================
# ======== First, converting HYDRUS files to CSV ============
# ===========================================================
# =========== 3- Correcting missing CSV files ===============
# ===========================================================

# # 1 file was having errors, we have built its CSV as follows
def add_missing_csv(missing_folder_name):
    retrieve_all_csv_files(HYDRUS_TXT, 
                       get=missing_folder_name, 
                       retrieve_folders_only=False, 
                       get_only_new=False, 
                       output_folder=HYDRUS_CSV)

# missing_folder_name = 'pXZ_3D_srf_Silt_sht_rO1f'
# add_missing_csv(missing_folder_name)


# In[34]:


# ===========================================================
# ======== First, converting HYDRUS files to CSV ============
# ===========================================================
# =============== 4- Checking all files OK ==================
# ===========================================================

def check_file_list():
    # Check everything OK
    FileNames=list(retrieve_all_csv_files(source, get='all', 
                                          retrieve_folders_only=True).values())
    print(FileNames)
    print(list(map(len, FileNames)))
    pass

# check_file_list()


# In[35]:


# ===========================================================
# ===== Second, intiating the dataframes for all files ======
# ===========================================================


def get_dataframes_from_csv(process_from_the_csv_folder = True):
    # If the CSV files already exist, then set this flag to True
    # process_from_the_csv_folder = True

    df2d, df3d=[],[]
    file_names={}
    file_number =-1
    dict_3d={}

    if process_from_the_csv_folder:
        items = glob.glob(output + '/*')
        subfolders=[]
        for n_folder in items:
            subfolders.append(n_folder.split('\\')[-1][:-4])
        pass
    else:
        # define file names
        subfolders= retrieve_all_csv_files(source, get='all', retrieve_folders_only=True)
        subfolders = list(subfolders.values())

    the_2d_files= list(filter(lambda x: '_2D_' in x, subfolders))
    the_3d_files= list(filter(lambda x: '_3D_' in x, subfolders))
    print ('Processing {} files, from which {} 3D files, and {} 2D files.'.
          format(len(subfolders), len(the_3d_files), len(the_2d_files)))    
    # print (the_2d_files, the_3d_files)

    # store dataframes from files
    for file in the_2d_files:
        file_number +=1
        filename = '{}.CSV'.format(file)
        file_names[file_number] = filename
        df2d.append(get_df_from_csv(output, filename))
    for i, file in enumerate(the_3d_files):
        file_number +=1
        filename = '{}.CSV'.format(file)
        file_names[file_number] = filename
        dict_3d[i]= file_number
        df3d.append(get_df_from_csv(output, filename))
    # print info about each dataframe
    # print (file_names)
    dims={}
    time_steps={}
    section_locations = np.array([0.5, 0.6, 0.75, 0.9])
    sections={}
    for i, data_frame in enumerate(df2d + df3d):
        print('{:3d}-For the case study     : {}'.format(i+1, (the_2d_files+the_3d_files)[i]))
        dims[i] = get_full_dimensions(data_frame)
        print('    The full dimensions are: ', dims[i])
        time_steps[i] = get_available_timesteps(data_frame)
        print('    The available timesteps: ', time_steps[i])
        if 'y' in dims[i]:
            sections[i]=dims[i]['y'][0]+section_locations *  (dims[i]['y'][1]-dims[i]['y'][0])
            print('    The sections to study  : ', sections[i])
            
    return df2d, df3d, file_names, dict_3d, dims, time_steps, sections

# df2d, df3d, file_names, dict_3d, dims, time_steps, sections = get_dataframes_from_csv()


# In[1]:


# ===========================================================
# ==== Fourth, performing calculations on the dataframes ====
# ====== Performing studies on full domain and 7*5 grid =====
# ===========================================================


def calculate_gridded_and_full_volumes(variable=0, section='y', tol= 10.,
                                      grid= 0.25, levels= get_legend_range(0.05, 0.45)):
    
    # Define main variables
    # variable   = 0 # Theta
    # section    ='y'
    # tol        = 10.
    # grid       = 0.25 # cm
    # levels     = get_legend_range(0.05, 0.45)
    

    out_frame = []
    # the gridded results dataframe
    grid_res_df = pd.DataFrame()
    # Looping through all 3D files and 2D files
    case_number = -1
    for i, data_frame_3d in enumerate(df3d): #[:1]):######

        name3d= file_names[dict_3d[i]]

        # Finding the characteristics of the 3D file
        # if the file is named 'pXZ_3D_sub_sand_cyl_2f.CSV'
        # it will be ['pXZ', '3D', 'sub', 'sand', 'cyl', '2f']
        name_parts_3d=name3d[:-4].replace("_", " ").split() 

        # Check if the *3D contour* is axisymmetric
        # THis will result in splitting the 3d contour in half to show only 
        # the right quarter
        if name_parts_3d[0]=='Axi':
            axisym=True
        else:
            axisym=False

        # Drawing contours of selected timesteps 
        # for the axisymmetric section, we want only the middle crossection
        # cross_sections = sections[dict_3d[i]][0] if axisym else sections[dict_3d[i]]
        cross_sections = sections[dict_3d[i]]
        for crosses in cross_sections: #[:1]:######
            for time_step in time_steps[dict_3d[i]][1:]: ########## [1:2]
                # Plotting and calculating the 3D file
                plot_title = "Cross sectional contour of {} after time value={}.                 CS@ {} direction, crossing at {}={} cm".                            format({0:'Theta', 1:'Head'}[variable], time_step,
                                  {'x': 'Y-Z', 'y':'X-Z'}[section], section, crosses)
                # cont_y3D will hold a tuple of (X, Y, Z)
                cont_y3D = draw_full_contour(data_frame_3d,variable, time_step, 
                                             grid, crosses, tol, section, 
                                             levels, plot_title, 
                                             x_step=15, z_step=25, 
                                             mirror_x=not axisym, mirror_z=False, 
                                             is2d=False, 
                                             output_the_contour=False,
                                             is_axisymmetric=axisym, 
                                             return_arrays=True,
                                             return_figure_object=False)

                # Plotting and calculating the 2D files
                for j, data_frame_2d in enumerate(df2d): #[:1]):#######
                    name2d=file_names[j]
                    # be sure the 2D file and the 3D file match each other
                    name_parts_2d=name2d[:-4].replace("_", " ").split()
                    fail = False
                    for part in [0, 2, 3]:# axi/pxz, sub/srf, soil_tex
                        if name_parts_3d[part] != name_parts_2d[part]:
                            fail = True
                            pass
                    if fail:
                        # Continue the loop without processing further
                        continue
                    plot_title = "2D cross sectional contour of {} "
                    plot_title += "after time value={}.".                        format({0:'Theta', 1:'Head'}[variable], time_step)
                    # cont will hold a tuple of (X, Y, Z)
                    cont_y2d = draw_full_contour(data_frame_2d,variable, 
                                                 time_step, grid, crosses, 
                                                 tol, section, 
                                                 levels, plot_title, 
                                                 x_step=15, z_step=25, 
                                                 mirror_x=not axisym, mirror_z=False, 
                                                 is2d=True, 
                                                 output_the_contour=False, 
                                                 return_arrays=True,
                                                 return_figure_object=False)

                    Tp, Tn, Tt, Vp, Vn, Vt =  draw_difference(cont_y3D, cont_y2d,
                                                  scale_from=0, 
                                                  custom_levels=get_legend_range(
                                                      -.1, 0.2),
                                                  mirror_x=not axisym, 
                                                  calculate_volume=True, 
                                                  calculate_average=True,
                                                  no_contours=True, 
                                                  separate_negatives=True, 
                                                  calculate_volume_percent= True,
                                                  return_calculations=True,
                                                  return_figure_object=False)

                    part5 = name_parts_2d[5] if len(
                        name_parts_2d[5])<=3 else name_parts_2d[5][1:4]
                    case_number += 1
                    outs = (case_number, name_parts_2d[0], name_parts_2d[2], 
                            name_parts_2d[3], name_parts_2d[4], part5, crosses, 
                            time_step, Tp[0], Tn[0], Tt[0], Tp[1],Tn[1], Tt[1], 
                            Vp*100., Vn*100., Vt*100., name3d[:-4])
                    out_frame.append(outs)

                    # Create the gridded difference dataframe
                    tab_df = get_results_for_grid(cont_y3D, cont_y2d)
                    tab_df['case_number']=case_number
                    if sum(grid_res_df.shape) ==0:
                        grid_res_df=tab_df.copy()
                    else:
                        grid_res_df= pd.concat([grid_res_df, tab_df])


                    print(case_number, name2d, name3d, part5, crosses, time_step)
                    pass
                pass
            pass
        pass
    return out_frame, grid_res_df

# out_data_frame, grid_res_df = calculate_gridded_and_full_volumes()
# title_print='| n |Axi|sub|soil|elm|emtr|CS | TS |PosVol| NegVol \
#     | TotVol | PosAvg | NegAvg  |  TotAvg |PsV_Pr|NgV_Pr|TtV_Pr|  3D_filename  |'

# out_data_frame=pd.DataFrame(out_frame, columns=title_print.replace('|',' ').split())
# out_data_frame.to_csv(os.path.join(output, 'AnalysisResultsGrossFull_New.CSV'))

# grid_res_df.to_csv(os.path.join(output, 'AnalysisResultsGrossGridded_New.CSV'))
# grid_res_df.head(), out_data_frame.head()


# In[37]:


def print_date_and_time():
    import time
    import datetime
    print ('Time now using `datetime.datetime` module')
    print('now()\t\t\t', datetime.datetime.now())
    print('now().time()\t\t', datetime.datetime.now().time())
    print('now().strftime()\t', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print ('\nTime now using `time` module')
    print('strftime(), gmtime()\t', time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
    print('ctime()\t\t\t', time.ctime())
    print ('\nTime now using `pandas` module')
    print ('datetime.now()\t\t', pd.datetime.now())
# print_date_and_time()


# In[38]:


# ===========================================================
# ==== Fifth, exporting PNG cross sections of dataframes ====
# ===========================================================
def export_png_pictures(variable=0, section='y', tol= 10.,
                                      grid= 0.25, levels= get_legend_range(0.05, 0.45)):

    tb="  "
    # Looping through all 3D files and 2D files
    case_number = -1
    for i, data_frame_3d in enumerate(df3d): #[:1]):######

        name3d= file_names[dict_3d[i]]

        # Finding the characteristics of the 3D file
        # if the file is named 'pXZ_3D_sub_sand_cyl_2f.CSV'
        # it will be ['pXZ', '3D', 'sub', 'sand', 'cyl', '2f']
        name_parts_3d=name3d[:-4].replace("_", " ").split() 
        np3d = name_parts_3d
        name3D_mod = '_'.join([np3d[3], np3d[1], np3d[0], np3d[2], np3d[4], np3d[5]])
        # Check if the *3D contour* is axisymmetric
        # THis will result in splitting the 3d contour in half to show only 
        # the right quarter
        if name_parts_3d[0]=='Axi':
            axisym=True
        else:
            axisym=False

        print('{}-For 3D file: {}'.format(i, name3d[:-4]))
        # Drawing contours of selected timesteps 
        # for the axisymmetric section, we want only the middle crossection
        # cross_sections = sections[dict_3d[i]][0] if axisym else sections[dict_3d[i]]
        cross_sections = sections[dict_3d[i]]
        for crosses in cross_sections: #[:1]:######
            for tt, time_step in enumerate(time_steps[dict_3d[i]][1:]): ########## [1:2]
                # Plotting and calculating the 3D file
                fig3D_name= '{}__t{}c{}'.format(name3D_mod,tt,int(crosses))
                print(tb,'Cross section {} and timestep {}'.format(crosses, time_step), end=' ...')
                plot_title = "Cross sectional contour of {} after time value={}.                 CS@ {} direction, crossing at {}={} cm".                            format({0:'Theta', 1:'Head'}[variable], time_step,
                                  {'x': 'Y-Z', 'y':'X-Z'}[section], section, crosses)
                # cont_y3D will hold a tuple of (X, Y, Z)
                if os.path.isfile(os.path.join(figput, '{}.png'.format(fig3D_name))):
                    print('EXISTS, the 3D file {} and its followers will be excluded'.format(fig3D_name))
                    continue
                    # No further commands will be processed for 2D and differences.

                try:
                    cont_y3D_full = draw_full_contour(data_frame_3d,variable, time_step, 
                                     grid, crosses, tol, section,levels, 
                                     plot_title, x_step=15, z_step=25, 
                                     mirror_x=not axisym, mirror_z=False, 
                                     is2d=False, output_the_contour=False,
                                     is_axisymmetric=axisym,
                                     return_arrays=True,
                                     return_figure_object=True)
                except:
                    print('ERROR, while processing the 3D file {}.'.format(fig3D_name))
                    # No further commands will be processed for 2D and differences.
                    continue
                try:
                    #cont_y3D_full is X, Z, M, levels, fig
                    cont_y3D = cont_y3D_full [:-1]
                    fig_3D = cont_y3D_full [ -1]
                    fig_3D.savefig(os.path.join(figput, '{}.png'.format(fig3D_name)))
                    print('Done.')
                except:
                    print('ERROR, while saving the 3D file {}.'.format(fig3D_name))
                    # Only the 3D PNG is not saved, but the 2D calculation will continue.

                # Plotting and calculating the 2D files
                for j, data_frame_2d in enumerate(df2d): #[:1]):#######
                    name2d=file_names[j]
                    # be sure the 2D file and the 3D file match each other
                    name_parts_2d=name2d[:-4].replace("_", " ").split()
                    np2d=name_parts_2d
                    name2D_mod = '_'.join([np2d[3], np2d[1], np2d[0], 
                                           np2d[2], np2d[4], np2d[5], '_t{}'.format(tt)])
                    fig2Dpath = os.path.join(figput, '{}.png'.format(name2D_mod))
                    fail = False
                    for part in [0, 2, 3]:# axi/pxz, sub/srf, soil_tex
                        if name_parts_3d[part] != name_parts_2d[part]:
                            fail = True
                            pass
                    if fail:
                        # Continue the loop without processing further
                        continue
                    plot_title = "2D cross sectional contour of {} "
                    plot_title += "after time value={}.".                        format({0:'Theta', 1:'Head'}[variable], time_step)
                    # cont will hold a tuple of (X, Y, Z)
                    print(tb*2,'{}-For the 2D file: {}'.format(j,name2d[:-4]), end=' ...')
                    try:
                        cont_y2d_full = draw_full_contour(data_frame_2d,variable, 
                                         time_step, grid, crosses, tol, section, 
                                         levels, plot_title, x_step=15, z_step=25, 
                                         mirror_x=not axisym, mirror_z=False, 
                                         is2d=True, output_the_contour=False,
                                         return_arrays=True,
                                         return_figure_object=True)
                    except:
                        print('ERROR, while processing the 2D file {}.'.format(name2D_mod))
                        # No further commands will be processed for differences.
                        continue
                    try:
                        #cont_y2d_full is X, Z, M, levels, fig
                        cont_y2d = cont_y2d_full [:-1]
                        fig_2D   = cont_y2d_full [ -1]
                        if not os.path.isfile(fig2Dpath):          
                            fig_2D.savefig(fig2Dpath)
                            print ('Done')
                        else:
                            print('{} is found'.format(name2D_mod))
                    except:
                        print('ERROR, while saving the 2D file {}.'.format(name2D_mod))
                        # Only the 2D PNG is not saved, but the difference calculation will continue.
                        pass

                    # custom_levels=get_legend_range(-.1, 0.2),
                    # custom_levels=None,

                    print(tb*3,'The difference figure ', end=' ...')
                    fig_dif_name = 'dif_{}_{}'.format(fig3D_name, name2d[16:-4])

                    fig_dif = None
                    try:
                        fig_dif =  draw_difference(cont_y3D, cont_y2d, scale_from=0, 
                                      custom_levels=get_legend_range(-.3, 0.3),
                                      mirror_x=not axisym, calculate_volume=False, 
                                      calculate_average=False, no_contours=False, 
                                      separate_negatives=False, 
                                      calculate_volume_percent= False,
                                      return_calculations=False,
                                      return_figure_object=True)
                    except:
                        print('ERROR, while processing the DIF file {}.'.format(fig_dif_name))
                        continue
                    try:    
                        if not fig_dif is None:
                            fig_dif_path = os.path.join(figput, '{}.png'.format(fig_dif_name))
                            if not os.path.isfile(fig_dif_path):          
                                fig_dif.savefig(fig_dif_path)
                                print ('Done')
                            else:
                                print('{} is found'.format(fig_dif_name))
                        else: 
                            print('Fail.')
                    except:
                        print('ERROR, while saving the DIF file {}.'.format(fig_dif_name))
                    pass
                pass
            pass
        pass
    print('==============================================')    
    print('All the files were processed as printed above.')
    print('==============================================')
    pass

# export_png_pictures()


# In[38]:


# ===========================================================
# ==== Fifth, exporting PNG cross sections of dataframes ====
# ===========================================================
# ============ More efficient handling to the files =========
# ===========================================================

def export_png_pictures_more_efficient(variable=0, section='y', tol= 10.,
                                      grid= 0.25, levels= get_legend_range(0.05, 0.45)):

    tb="  "
    # Looping through all 3D files and 2D files
    case_number = -1
    for i, data_frame_3d in enumerate(df3d): #[:1]):######

        name3d= file_names[dict_3d[i]]

        # Finding the characteristics of the 3D file
        # if the file is named 'pXZ_3D_sub_sand_cyl_2f.CSV'
        # it will be ['pXZ', '3D', 'sub', 'sand', 'cyl', '2f']
        name_parts_3d=name3d[:-4].replace("_", " ").split() 
        np3d = name_parts_3d
        name3D_mod = '_'.join([np3d[3], np3d[1], np3d[0], np3d[2], np3d[4], np3d[5]])
        # Check if the *3D contour* is axisymmetric
        # THis will result in splitting the 3d contour in half to show only 
        # the right quarter
        if name_parts_3d[0]=='Axi':
            axisym=True
        else:
            axisym=False

        print('{}-For 3D file: {}'.format(i, name3d[:-4]))
        # Drawing contours of selected timesteps 
        # for the axisymmetric section, we want only the middle crossection
        # cross_sections = sections[dict_3d[i]][0] if axisym else sections[dict_3d[i]]
        cross_sections = sections[dict_3d[i]]
        dict_2d_contours={}
        for crosses in cross_sections: #[:1]:######
            for tt, time_step in enumerate(time_steps[dict_3d[i]][1:]): ########## [1:2]
                # Plotting and calculating the 3D file
                fig3D_name= '{}__t{}c{}'.format(name3D_mod,tt,int(crosses))
                print(tb,'Cross section {} and timestep {}'.format(crosses, time_step), end=' ...')
                plot_title = "Cross sectional contour of {} after time value={}.                 CS@ {} direction, crossing at {}={} cm".                            format({0:'Theta', 1:'Head'}[variable], time_step,
                                  {'x': 'Y-Z', 'y':'X-Z'}[section], section, crosses)
                # cont_y3D will hold a tuple of (X, Y, Z)
                if os.path.isfile(os.path.join(figput, '{}.png'.format(fig3D_name))):
                    print('EXISTS, the 3D file {} and its followers will be excluded'.format(fig3D_name))
                    continue
                    # No further commands will be processed for 2D and differences.

                try:
                    cont_y3D_full = draw_full_contour(data_frame_3d,variable, time_step, 
                                     grid, crosses, tol, section,levels, 
                                     plot_title, x_step=15, z_step=25, 
                                     mirror_x=not axisym, mirror_z=False, 
                                     is2d=False, output_the_contour=False,
                                     is_axisymmetric=axisym,
                                     return_arrays=True,
                                     return_figure_object=True)
                except:
                    print('ERROR, while processing the 3D file {}.'.format(fig3D_name))
                    # No further commands will be processed for 2D and differences.
                    continue
                    
                try:
                    #cont_y3D_full is X, Z, M, levels, fig
                    cont_y3D = cont_y3D_full [:-1]
                    fig_3D = cont_y3D_full [ -1]
                    fig_3D.savefig(os.path.join(figput, '{}.png'.format(fig3D_name)))
                    print('Done.')
                except:
                    print('ERROR, while saving the 3D file {}.'.format(fig3D_name))
                    # Only the 3D PNG is not saved, but the 2D calculation will continue.

                # Plotting and calculating the 2D files
                
                for j, data_frame_2d in enumerate(df2d): #[:1]):#######
                    name2d=file_names[j]
                    # be sure the 2D file and the 3D file match each other
                    name_parts_2d=name2d[:-4].replace("_", " ").split()
                    np2d=name_parts_2d
                    name2D_mod = '_'.join([np2d[3], np2d[1], np2d[0], 
                                           np2d[2], np2d[4], np2d[5], '_t{}'.format(tt)])
                    if name2D_mod not in dict_2d_contours.keys():
                        fig2Dpath = os.path.join(figput, '{}.png'.format(name2D_mod))
                        fail = False
                        for part in [0, 2, 3]:# axi/pxz, sub/srf, soil_tex
                            if name_parts_3d[part] != name_parts_2d[part]:
                                fail = True
                                pass
                        if fail:
                            # Continue the loop without processing further
                            continue
                        plot_title = "2D cross sectional contour of {} "
                        plot_title += "after time value={}.".                            format({0:'Theta', 1:'Head'}[variable], time_step)
                        # cont will hold a tuple of (X, Y, Z)
                        print(tb*2,'{}-For the 2D file: {}'.format(j,name2d[:-4]), end=' ...')
                        try:
                            cont_y2d_full = draw_full_contour(data_frame_2d,variable, 
                                             time_step, grid, crosses, tol, section, 
                                             levels, plot_title, x_step=15, z_step=25, 
                                             mirror_x=not axisym, mirror_z=False, 
                                             is2d=True, output_the_contour=False,
                                             return_arrays=True,
                                             return_figure_object=True)
                        except:
                            print('ERROR, while processing the 2D file {}.'.format(name2D_mod))
                            # No further commands will be processed for differences.
                            continue
                        try:
                            #cont_y2d_full is X, Z, M, levels, fig
                            cont_y2d = cont_y2d_full [:-1]
                            fig_2D   = cont_y2d_full [ -1]
                            if not os.path.isfile(fig2Dpath):          
                                fig_2D.savefig(fig2Dpath)
                                print ('Done')
                            else:
                                print('{} is found'.format(name2D_mod))
                        except:
                            print('ERROR, while saving the 2D file {}.'.format(name2D_mod))
                            # Only the 2D PNG is not saved, but the difference calculation will continue.
                            pass
                        # saving the results to the dictionary
                        dict_2d_contours[name2D_mod] = name2d, cont_y2d
                    else:
                        name2d, cont_y2d = dict_2d_contours[name2D_mod]

                    # custom_levels=get_legend_range(-.1, 0.2),
                    # custom_levels=None,

                    print(tb*3,'The difference figure ', end=' ...')
                    fig_dif_name = 'dif_{}_{}'.format(fig3D_name, name2d[16:-4])

                    fig_dif = None
                    try:
                        fig_dif =  draw_difference(cont_y3D, cont_y2d, scale_from=0, 
                                      custom_levels=get_legend_range(-.3, 0.3),
                                      mirror_x=not axisym, calculate_volume=False, 
                                      calculate_average=False, no_contours=False, 
                                      separate_negatives=False, 
                                      calculate_volume_percent= False,
                                      return_calculations=False,
                                      return_figure_object=True)
                    except:
                        print('ERROR, while processing the DIF file {}.'.format(fig_dif_name))
                        continue
                    try:    
                        if not fig_dif is None:
                            fig_dif_path = os.path.join(figput, '{}.png'.format(fig_dif_name))
                            if not os.path.isfile(fig_dif_path):          
                                fig_dif.savefig(fig_dif_path)
                                print ('Done')
                            else:
                                print('{} is found'.format(fig_dif_name))
                        else: 
                            print('Fail.')
                    except:
                        print('ERROR, while saving the DIF file {}.'.format(fig_dif_name))
                    pass
                pass
            pass
        pass
    print('==============================================')    
    print('All the files were processed as printed above.')
    print('==============================================')
    pass

# export_png_pictures_more_efficient()


# <div class="alert alert-block alert-warning">
# # Testing the previous functions to confirm they work fine.

# #### *REMOVED, can be found at HydrusGetAndDraw31 and earlier*

# <div class="alert alert-block alert-warning">
# # Applying the previous works to draw the desired contour

# **Check the dataframe headers**

# # The application of all functions 
# ## (on the testing dataset)

# #### *REMOVED, can be found at HydrusGetAndDraw31 and earlier*

# ## Applications on real datasets

# #### *REMOVED, can be found at HydrusGetAndDraw31 and earlier*

# # The gross analysis of sand files

# #### *REMOVED, can be found at HydrusGetAndDraw31 and earlier*

# # The gross analysis of ALL TEXTURES

# In[39]:


import warnings
warnings.filterwarnings('ignore')

source_folder= 'C:/Users/DrNesr/Documents/Current2'

HYDRUS_TXT= 'C:/Users/DrNesr/Documents/HYDRUS_TXT/Em2Lph'
HYDRUS_CSV= 'C:/Users/DrNesr/Documents/HYDRUS_CSV/Em2Lph'
HYDRUS_PNG= 'C:/Users/DrNesr/Documents/HYDRUS_PNG/Em2Lph'

source = HYDRUS_TXT
output = HYDRUS_CSV
figput = HYDRUS_PNG

# Stage 1.1 (uncomment if not done yet)
# ---------------------------------------
# move_hydrus_txt_to_folder()

# Stage 1.2 (uncomment if not done yet)
# ---------------------------------------
# convert_HYDRUS_files_to_CSV()

# Stage 1.3 (uncomment if not done yet)
# ---------------------------------------
# missing_folder_name = 'pXZ_3D_srf_Silt_sht_rO1f'
# add_missing_csv(missing_folder_name)

# Stage 1.4 (uncomment if not done yet)
# ---------------------------------------
# check_file_list()

# Stage 2.0 (A MUST for next stages)
# ---------------------------------------
df2d, df3d, file_names, dict_3d, dims, time_steps, sections = get_dataframes_from_csv()

# Stage 4.0 (uncomment if not done yet)
# ---------------------------------------
# out_data_frame, grid_res_df = calculate_gridded_and_full_volumes()
# title_print='| n |Axi|sub|soil|elm|emtr|CS | TS |PosVol| NegVol \
#     | TotVol | PosAvg | NegAvg  |  TotAvg |PsV_Pr|NgV_Pr|TtV_Pr|  3D_filename  |'

# out_data_frame=pd.DataFrame(out_frame, columns=title_print.replace('|',' ').split())
# out_data_frame.to_csv(os.path.join(output, 'AnalysisResultsGrossFull_New.CSV'))

# grid_res_df.to_csv(os.path.join(output, 'AnalysisResultsGrossGridded_New.CSV'))
# grid_res_df.head(), out_data_frame.head()

# print_date_and_time()

# Stage 5.0 (uncomment if not done yet)
# ---------------------------------------
# export_png_pictures()
# or
# export_png_pictures_more_efficient()


# In[ ]:


export_png_pictures()


# In[ ]:


export_png_pictures_more_efficient()


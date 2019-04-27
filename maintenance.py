from nptdms import TdmsFile
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats

df = pd.DataFrame()
results_dir = 'RESULTS'

for root,dirs,files in os.walk(results_dir):
    if root[len(results_dir)+1:].count(os.sep)<1:
        if len(dirs)>0:
            directory = dirs[0]
        print('DIR: ',directory)
        for f in files:
            extension = os.path.splitext(f)[1]
            if extension == '.tdms':
                print('FILE: ', os.path.join(root,f))
                oTDMS_file = TdmsFile(os.path.join(root,f))
                # create day dataframe
                file_df = pd.DataFrame()
                ###### Get Data From the Channel
                for column in oTDMS_file.group_channels('MEASUREMENTS'):
                    file_df[column.channel] =  oTDMS_file.channel_data('MEASUREMENTS',column.channel)
                for index, row in file_df.iterrows():
                    abs_file_wave = os.path.join(root,directory, row['Waves file'])
                    wave_filename = row['Waves file']
                    if wave_filename in os.listdir(os.path.join(root,directory)):
                        oTDMS_file_wave = TdmsFile(abs_file_wave)
                        valve_channel_4 = oTDMS_file_wave.channel_data('Untitled', 'Canale 4')
                        file_df.loc[index, 'valve_min'] = np.min(valve_channel_4)
                        file_df.loc[index, 'valve_max'] = np.max(valve_channel_4)
                        file_df.loc[index, 'valve_mean'] = np.mean(valve_channel_4)
                        file_df.loc[index, 'valve_std'] = np.std(valve_channel_4)
                df = df.append(file_df, ignore_index=True)
      
for root,dirs,files in os.walk(results_dir):
    if root[len(results_dir)+1:].count(os.sep)==1:
        if len(dirs)>0:
            directory = dirs[0]
            print('DIR: ',directory)
        for f in files:
            # check blacklist
            if not (any(x in f for x in ('GIORNO23-I4,0','GIORNO25-I4,0','Test-02'))):
                # get file extension
                extension = os.path.splitext(f)[1]
                # check extension == .tdms
                if extension == '.tdms':
                    print('FILE: ',os.path.join(root,f))
                    oTDMS_file = TdmsFile(os.path.join(root,f))
                    # create day dataframe
                    file_df = pd.DataFrame()
                    ###### Get Data From the Channel
                    for column in oTDMS_file.group_channels('MEASUREMENTS'):
                        file_df[column.channel] =  oTDMS_file.channel_data('MEASUREMENTS',column.channel)
                    for index, row in file_df.iterrows():
                        abs_file_wave = os.path.join(root,directory, row['Waves file'])
                        wave_filename = row['Waves file']
                        if wave_filename in os.listdir(os.path.join(root,directory)):
                            oTDMS_file_wave = TdmsFile(abs_file_wave)
                            valve_channel_4 = oTDMS_file_wave.channel_data('Untitled', 'Canale 4')
                            file_df.loc[index, 'valve_min'] = np.min(valve_channel_4)
                            file_df.loc[index, 'valve_max'] = np.max(valve_channel_4)
                            file_df.loc[index, 'valve_mean'] = np.mean(valve_channel_4)
                            file_df.loc[index, 'valve_std'] = np.std(valve_channel_4)
                    df = df.append(file_df, ignore_index=True)

df['Datetime'] = pd.to_datetime(df['DATE [dd/mm/yy]'] + ' ' + df['TIME [hh/mm/ss]'], format='%d/%m/%Y %H:%M:%S')

df = df.drop(['Pressure 2'
              ,'Temp. 1','Temp. 2'
              ,'TFree'
              ,'BFree 1','BFree 2','BFree 3','BFree 4'
              ,'CFree 1','CFree 2','CFree 3'
              ,'STATUS'
              ,'Efficiency time'
              ,'Waves file'
              ,'DATE [dd/mm/yy]'
              ,'TIME [hh/mm/ss]'
              #,'BT11 Temp.', 'BT22 Temp.', 'BT23 Temp.'
              , 'BT21 Temp.'                          # inconsistency on data
              ,'Wearing time'
              #,'Wearing cycles'
              ,'valve_min', 'valve_max','valve_std'
              ], axis=1)

df = df[~np.isnan(df['valve_mean'])]
df = df.sort_values(by=['Datetime'])
df = df.reset_index()
df.to_csv('maintenance_measurments.csv')

df = df.drop('index', axis=1)

######################## FINDING OUTLIERS USING Z-SCORE ##########################################
#%%#

# Pressure 1
outliers_pressure = df[np.abs(stats.zscore(df['Pressure 1'])) > 0.7]['Pressure 1']
# valve_mean
outliers_valveMean = df[np.abs(stats.zscore(df['valve_mean'])) > 1]['valve_mean']
# BT11 Temp.
outliers_BT11 = df[np.abs(stats.zscore(df['BT11 Temp.'])) > 5]['BT11 Temp.']

# LVDT
LVDT_mean = np.mean(df['LVDT'])
df['LVDT_mean_diff'] = np.abs(df['LVDT'] - LVDT_mean)
outliers_LVDT_diff = df[np.abs(stats.zscore(df['LVDT_mean_diff'])) > 1.67]['LVDT_mean_diff']

###################### FINDING ALARMING VALUES USING MEDIAN ########################################
# SL22
SL22_median = np.median(df['SL22 Level'])
alarming_SL22 = df[df['SL22 Level'] > SL22_median]['SL22 Level']

# SL23
SL23_median = np.median(df['SL23 Level'])
alarming_SL23 = df[df['SL23 Level'] > SL23_median]['SL23 Level']

##################### FINDING ALARMING VALUES USING MEAN ##########################################
# BT22 Temp.
BT22_threshold = 30.5
alarming_BT22 =  df[df['BT22 Temp.'] > BT22_threshold]['BT22 Temp.']
temp_mean = np.mean(df['BT22 Temp.'])
###################################################################################################
#%%#
            
'''
df['Dist_from_mean'] = np.abs(df['BT22 Temp.'] - temp_mean)
df['HourOfDay'] = df['Datetime'].dt.hour
kmeans = KMeans(n_clusters=8).fit(df[['Dist_from_mean','HourOfDay']])
df['TempCluster'] = kmeans.predict(df[['Dist_from_mean','HourOfDay']])
'''

df['Date'] = df['Datetime'].dt.date
df['index'] = df.index
dates_df = df.groupby('Date').first()['index']
resets = []

sept_5 = datetime(2017, 9, 5)
sept_8 = datetime(2017, 9, 8)
oct_8 = datetime(2017, 10, 8)
oct_11 = datetime(2017, 10, 11)
oct_22 = datetime(2017, 10, 22)         # No data before
oct_25 = datetime(2017, 10, 25)


df_broke = df[df['Datetime'] >= sept_5]
df_broke = df_broke[df_broke['Datetime'] <= sept_8]
df_broke = df_broke.reset_index()

df_9_10_oct = df[df['Datetime'] >= oct_8]
df_9_10_oct = df_9_10_oct[df_9_10_oct['Datetime'] <= oct_11]
df_9_10_oct = df_9_10_oct.reset_index()

df_23_oct = df[df['Datetime'] >= oct_22]
df_23_oct = df_23_oct[df_23_oct['Datetime'] <= oct_25]
df_23_oct = df_23_oct.reset_index()

with PdfPages('measurments.pdf') as pdf:
    for column in df.columns:
        if not column == 'Datetime' and (not column == 'Date') and (not column == 'index') and (not column == 'Datetime'):
            fig = plt.figure(figsize =(30 , 10))
            # set the "xticks"
            ticks_to_use = df.index[::100]
            labels = [ i.strftime("%d") for i in df.loc[ticks_to_use,'Datetime']]
            ax = plt.gca()
            ax.set_xticks(ticks_to_use)
            ax.set_xticklabels(labels)
            # create plot
            for date in dates_df:
                plt.axvline(x=date, c='r')
            if column == 'BT22 Temp.':
                plt.plot(df.index, df['BT22 Temp.'])
               # plt.scatter(df.loc[df['TempCluster']==2].index, df.loc[df['TempCluster']==2][column], c='r')
                plt.axhline(y=30.5, c='orange')
                for index, row in df.iterrows():
                    if index > 0:
                        difference = row['Wearing cycles'] - df.loc[index-1,'Wearing cycles']
                        if (row['Wearing cycles'] < df.loc[index-1,'Wearing cycles']):
                            resets.append(index)
                for index in resets:
                    plt.axvline(x=index, c='cyan')
            else:
                plt.plot(df.index, df[column])
                    
            plt.title(column)
            plt.xlabel('Day')
            plt.ylabel(column)
            pdf.savefig()
            plt.close()
#%%#
'''
# create plots
with PdfPages('measurments_boxplots.pdf') as pdf:
    for column in df.columns:
        if not column == 'Datetime':
            plt.boxplot(df[column])
            plt.title(column)
            plt.xlabel('Day')
            plt.ylabel(column)
            pdf.savefig()
            plt.close()
'''
with PdfPages('Output/Measurments_broke.pdf') as pdf:
    for column in df.columns:
        if not column == 'Datetime' and (not column == 'Date') and (not column == 'index') and (not column == 'Datetime'):
            fig = plt.figure(figsize =(30 , 10))
            # set the "xticks"
            ticks_to_use = df_broke.index[::10]
            labels = [ i.strftime("%d : %H") for i in df_broke.loc[ticks_to_use,'Datetime']]
            ax = plt.gca()
            ax.set_xticks(ticks_to_use)
            ax.set_xticklabels(labels)
            # create plot
            if column == 'BT22 Temp.':
                plt.plot(df_broke.index, df_broke[column])
            else:
                plt.plot(df_broke.index, df_broke[column])
            plt.title(column)
            plt.xlabel('Day')
            plt.ylabel(column)
            pdf.savefig()
            plt.close()
            
with PdfPages('Output/Measurments_09_10_oct.pdf') as pdf:
    for column in df.columns:
        if not column == 'Datetime' and (not column == 'Date') and (not column == 'index') and (not column == 'Datetime'):
            fig = plt.figure(figsize =(30 , 10))
            # set the "xticks"
            ticks_to_use = df_9_10_oct.index[::10]
            labels = [ i.strftime("%d : %H") for i in df_9_10_oct.loc[ticks_to_use,'Datetime']]
            ax = plt.gca()
            ax.set_xticks(ticks_to_use)
            ax.set_xticklabels(labels)
            # create plot
            if column == 'BT22 Temp.':
                plt.plot(df_9_10_oct.index, df_9_10_oct[column])
            else:
                plt.plot(df_9_10_oct.index, df_9_10_oct[column])
            plt.title(column)
            plt.xlabel('Day')
            plt.ylabel(column)
            pdf.savefig()
            plt.close()
            
with PdfPages('Output/Measurments_23_oct.pdf') as pdf:
    for column in df.columns:
        if not column == 'Datetime' and (not column == 'Date') and (not column == 'index') and (not column == 'Datetime'):
            fig = plt.figure(figsize =(30 , 10))
            # set the "xticks"
            ticks_to_use = df_23_oct.index[::10]
            labels = [ i.strftime("%d : %H") for i in df_23_oct.loc[ticks_to_use,'Datetime']]
            ax = plt.gca()
            ax.set_xticks(ticks_to_use)
            ax.set_xticklabels(labels)
            # create plot
            if column == 'BT22 Temp.':
                plt.plot(df_23_oct.index, df_23_oct[column])
            else:
                plt.plot(df_23_oct.index, df_23_oct[column])
            plt.title(column)
            plt.xlabel('Day')
            plt.ylabel(column)
            pdf.savefig()
            plt.close()
            

#%%
df.to_csv('Output/Maintenance_df.csv')
df_valve = pd.DataFrame()
df_valve = df['valve_mean']
df_valve.to_csv('Output/Maintenance_valve_df.csv')

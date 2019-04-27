#!/usr/bin/python

# imports for observers
import sys
import os
import time  
from watchdog.observers import Observer  
from watchdog.events import PatternMatchingEventHandler

# imports for calculations
import pandas as pd
from nptdms import TdmsFile
import numpy as np
import csv

global main_folder_observer
global channels_observer

"""
event.event_type 
    'modified' | 'created' | 'moved' | 'deleted'
event.is_directory
    True | False
event.src_path
    path/to/observed/file
"""
        
class Main_Folder_Handler(PatternMatchingEventHandler):
    #patterns = ["*.tdms"]

    def process(self, event):

        # the file will be processed there
        if  event.is_directory and (event.event_type == 'created'):
            print('MAIN')
            measurements_observer.unschedule_all()
            print(event.src_path, event.event_type)  # print now only for debug    
            measurements_observer.schedule(Measurements_Handler(), path=event.src_path)
    
    def on_modified(self, event):
        self.process(event)

    def on_created(self, event):
        self.process(event)
        
class Measurements_Handler(PatternMatchingEventHandler):
    #patterns = ["*.tdms"]

    def process(self, event):
        # the file will be processed there
        if  event.is_directory and (event.event_type == 'created'):
            print('MEASUREMENTS')
            channels_observer.unschedule_all()
            print(event.src_path, event.event_type)  # print now only for debug    
            channels_observer.schedule(Channels_Handler(), path=event.src_path)
        
        # the file will be processed there
        '''
        filename, file_extension = os.path.splitext(event.src_path)
        if file_extension == '.tdms':
            print(event.src_path, event.event_type)  # print now only for degug
            if event.event_type == 'modified':
                oTDMS_file = TdmsFile(event.src_path)
                # create day dataframe
                file_df = pd.DataFrame()
                ###### Get Data From the Channel
                for column in oTDMS_file.group_channels('MEASUREMENTS'):
                    file_df[column.channel] =  oTDMS_file.channel_data('MEASUREMENTS',column.channel)
                if not file_df.tail(1) == df.tail(1):
                    df.append(file_df)
                    df.to_csv('live_measurements.csv', delimiter=",")
               ''' 
    def on_modified(self, event):
        self.process(event)

    def on_created(self, event):
        self.process(event)  
        
class Channels_Handler(PatternMatchingEventHandler):
    patterns = ["*.tdms"]

    def process(self, event):
        # the file will be processed there
        print('CHANNELS')
        print(event.src_path, event.event_type)  # print now only for degug
        '''
        if event.event_type == 'modified':
            channels_file = TdmsFile(event.src_path)
            valve_channel_4 = channels_file.channel_data('Untitled', 'Canale 4')
            
            channel_df = pd.DataFrame(columns=['valve_min','valve_max','valve_mean','valve_std'])
            valve_min = np.min(valve_channel_4)
            valve_max = np.max(valve_channel_4)
            valve_std = np.std(valve_channel_4)
            values = [valve_min, valve_max, valve_mean, valve_std]
            channel_df.loc[len(df)] = values
            
            valve_mean = np.mean(valve_channel_4)
            if not valve_mean == valve_array[len(valve_array)-1]:
                valve_array.append(valve_mean)
                np.savetxt("live_channels.csv", delimiter=",")
            '''
    def on_modified(self, event):
        self.process(event)
        
    def on_created(self, event):
        self.process(event)    
#%%
if __name__ == '__main__':
    args = sys.argv[1:]
    exec(open("./maintenance.py").read())
    #%%#
    df = pd.read_csv('Output/Maintenance_df.csv', delimiter=",")
    valve_array = []
    data = csv.reader(open('Output/Maintenance_valve_df.csv'), delimiter=",")
    for row in data:
        valve_array.append(row[1])
    measurements_df = pd.DataFrame()
    ######################################### RUN OBSERVERS ###############################
    # main folder observer
    main_folder_observer = Observer()
    main_folder_observer.schedule(Main_Folder_Handler(), path=args[0] if args else '.')
    main_folder_observer.start()
    # measurements observer
    measurements_observer = Observer()
    measurements_observer.start()
    # channels observer
    channels_observer = Observer()
    channels_observer.start()
    print('\nObservers initiated on folder: ', args[0])
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        main_folder_observer.stop()
        channels_observer.stop()

    main_folder_observer.join()
    
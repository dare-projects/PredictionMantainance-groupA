
RESULTS = Input Data

maintenace.py = Script for data elaboration, data analysis 
                and data visualization.
                
                Input -> Input data's folder relative path.
                         'RESULTS'
                Output -> 1. .csv files containing elaborated data
                              'Output/Maintenance_df.csv'
                              'Output/Maintenance_valve_df.csv'
                          2. Three pdf files with plots regarding the data
                             (one for each 'anomaly' detected)
                              'Output/Measurments_broke.pdf'
                              'Output/Measurments_09_10_oct.pdf'
                              'Output/Measurments_23_oct.pdf'
                              
watch_for_changes.py = Script that runs maintenance.py and 
                       initiates target folder observers
                       based on the input data structure.
                          
                         STRUCTURE: 
                         
                                Main Folder 
                                  -> Day1 folder
                                      -> Measurements file
                                      -> Channels folder
                                         -> Channels files
                                   -> Day2 folder
                                       -> Measurements file
                                       -> Channels folder
                                          -> Channels files
                                   ...
                                
                        Input -> Folder path for observers to watch
                        
                        Output -> Whenever a new folder gets created or
                                  an old one gets modified, prints out:
                                  1. Observer label
                                  2. Path of the event that triggered the observer
                                  3. Type of the event that triggered the observer
                                      (Created or Modified)
                                      

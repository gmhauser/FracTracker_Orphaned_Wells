#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 09:16:15 2025

@author: gracehauser
"""

#%%

# =============================================================================
# Set-up 
# =============================================================================

# Title: "Orphaned_Wells_06_2025.py"
# Author: Grace Hauser
# Affiliations: YSPH Department of EHS & FracTracker Alliance Western Division
# Date of last update: 06/20/2025
# Script aim: answer the following questions
#             (1) How many orphaned wells are there as of mid-2025?
#             (2) How many wells have become orphaned since the USGS report in ____? 
#             (3) How many wells have been plugged since the USGS report in ____?

# USGS publication: https://pubs.usgs.gov/publication/dr1167/full
# USGS dataset: https://www.sciencebase.gov/catalog/item/62ebd67bd34eacf539724c56
# USGS methodology: https://www.sciencebase.gov/catalog/file/get/62ebd67bd34eacf539724c56?f=__disk__fc%2F1e%2Fc2%2Ffc1ec2c6bd83535801cbaea9e17cf4dbf091a946&transform=1&allowOpen=true
# Fractracker dataset: available upon request at https://www.fractracker.org/data/

# Set working directory
import os
os.chdir('/Users/gracehauser/Desktop/Publication/Data/Wells')

# Load packages
import numpy as np
import pandas as pd
import warnings
import geopandas as gpd
import matplotlib.pyplot as plt
from pyproj import Transformer

#%%

# =============================================================================
# =============================================================================
# =============================================================================
# AIM 1 : HOW MANY ORPHANED WELLS ARE THERE AS OF JUNE 2025?
# =============================================================================
# =============================================================================
# =============================================================================

#%%
# =============================================================================
# Set-up: FracTracker Dataset
# =============================================================================

# Import dataset
warnings.filterwarnings("ignore")
ft_prelim = pd.read_csv("FRACTRACKER/full_dataset.csv")

# Add in Tennessee, which was accidentally not included in FT dataset
tn_ft = pd.read_csv("FRACTRACKER/tennessee_wells_071624.csv")
ft = pd.concat([ft_prelim, tn_ft], ignore_index= True)

# Clean FracTracker API number attribute
ft = ft.dropna(subset=['api_num'])
ft = ft[ft['api_num'] != 0000000000]
ft = ft[ft['api_num'] != '0000000000']
ft['api_num'] = ft.api_num.str.replace('-','')
ft['api_num'] = ft['api_num'].replace('-', '', regex=True).astype("string")
ft['api_num'] = ft['api_num'].replace(',', '', regex=True).astype("string")
ft['api_num'] = ft['api_num'].apply(lambda x: x.strip())
ft['api_num'] = ft['api_num'].astype(str)
ft = ft[ft['api_num'].str.len() >= 10]

# Create orphaned and plugged dictionaries
# Orphaned dictionary 
state_status_dict = { 
    'Alabama' : ['Abandoned'],
    # Alaska : no category
    'Arkansas' : ['Abandoned Orphaned Well'],
    # California : no category
    # Colorado : no category
    # Florida : no category
    'Indiana' : ['Orphaned'],
    'Kansas': ['D&A'],
    'Kentucky' : ['AB',
                  'D&A',
                  'ABD'],
    'Louisiana' : ['23',
                   '26'],
    'Michigan' : ['Orphan'],
    'Mississippi' : ['PO - Potential Orphan Well',
                     'O - Orphaned Well'],
    'Missouri' : ['Abandoned',
                  'Abandoned, Unknown Location',
                  'Abandoned, Known Location and Verified',
                  'Abandoned, No evidence of existence/ Unable to find',
                  'Orphaned'],
    # Montana : no category
    'Nebraska' : ['AB', 'SI'],
    'Nevada' : ['AB'],
    'New Mexico' : ['Reclamation Fund Approved'],
    'New York' : ['UN',
                  'UL',
                  'UM'],
    'North Dakota' : ['AB'],
    'Ohio' : ['OR',
              'OP'],
    'Oklahoma' : ['OR'],
    'Pennsylvania' : ['DEP Orphan List'],
    'South Dakota' : ['Abandoned-Not Regulated']
    # Tennessee : no category
    # Texas : no category
    # Utah : no category
    # West Virginia : no category
    # Wyoming : no category
}

# Plugged dictionary
plugged_dict = { 
    'Alabama' : ['Plugged and Abandoned',  
                'Plugged Back'],
    'Alaska' : ['Plugged & Abandoned',
              'Surface Plug'],
    'Arkansas' : ['Plugged and Abandoned'],
    'California' : ['Plugged',
                  'PluggedOnly'],
    'Colorado' : ['PA',
                'pa'],
    'Florida' : ['P&A',
                 'DRY HOLE/P&A'],
    'Indiana' : ['Prsmd Plggd(I)',
                 'Plugd & Abandnd',
                 'Prsmd Plggd',
                 'Prsmd Plggd(I)',
                 'Inadqtly Plggd'],
    'Kansas' : ['OIL-P&A',
                'GAS-P&A',
                'EOR-P&A',
                'SWD-P&A',
                'OTHER-P&A(INJ or EOR)',
                'O&G-P&A',
                'OTHER-P&A(LH)',
                'CBM-P&A',
                'OTHER-P&A(STRAT)',
                'OTHER-P&A(CATH)',
                'OTHER-P&A()',
                'INJ-P&A',
                'OTHER-P&A(GAS-INJ)',
                'OTHER-P&A(OBS)',
                'OTHER-P&A(TA)',
                'OTHER-P&A(OIL&GAS-INJ)',
                'OTHER-P&A(GAS-STG)',
                'OTHER-P&A(GSW)',
                'OTHER-P&A(Plugged)',
                'OTHER(Plugged)',
                'OTHER-P&A(SWD-P&A)',
                'OTHER-P&A(Inj)',
                'OTHER-P&A(CLASS ONE (OLD))',
                'OTHER-P&A(2 OIL)'],
    # Kentucky : no category
    'Louisiana' : ['29', '30', '35', '90',
                  29, 30, 35, 90],
    'Michigan' : ['Plugging Approved',
                  'Plugging Completed'],
    'Missouri' : ['Plugged - Approved',
                  'Plugged - Not Approved'],
    'Montana' : ['P&A - Approved'],
    'Nebraska' : ['PA'],
    'Nevada' : ['P & A',
                'P&A',
                'P & A 7/27/95',
                'P & A (?)',
                'P & A 7/17/95'],
    'New Mexico' : ['Plugged (site released)',
                    'Plugged (not released)',
                    'Zone plugged (permanent)',
                    'Zone plugged (temporary)'],
    'New York' : ['PA',
                  'PB'],
    'North Dakota' : ['PA'],
    'Ohio' : ['PA'],
    'Oklahoma' : ['PA'],
    'Pennsylvania' : ['Plugged OG Well',
                      'DEP Plugged',
                      'Plugged Unverified',
                      'Plugged Mined Through'],
    'South Dakota' : ['Abandoned-Not Regulated',
                      'Plugged and Abandoned'],
    # Tennessee : no category
    'Texas' : [7,
               8,
               10,
               116,
               117,
               118,
               119,
               136,
               137,
               138,
               139,
               152,
               153,
               154,
               155,
               '7',
               '8',
               '10',
               '116',
               '117',
               '118',
               '119',
               '136',
               '137',
               '138',
               '139',
               '152',
               '153',
               '154',
               '155'],
    'Utah' : ['PA'],
    'West Virginia' : ['Plugged'],
    'Wyoming' : ['PA']
}

# Combine both dictionaries into one function for standardizing
def standardize_well_status(row):
    state = row['stusps']  
    status = row['well_status']
    
    # Check if the status belongs to the orphaned or plugged categories
    if state in state_status_dict and status in state_status_dict[state]:
        return 'ORPHANED'
    elif state in plugged_dict and status in plugged_dict[state]:
        return 'PLUGGED'
    return status  # If status doesn't match, keep the original

# Apply the mapping function to the 'well_status' column
ft['well_status'] = ft.apply(standardize_well_status, axis=1)

#%%
# =============================================================================
# 3. Work with FracTracker duplicate APIs
# =============================================================================

# Select only states of interest to make this more efficient
non_states = ['Arizona', 'Idaho', 'Illinois', 'Maryland', 'Oregon', 'Virginia',
              'Washington', 'Arizona', 'Illinois']
ft = ft[~ft['stusps'].isin(non_states)]
print("Starting length:", len(ft))

# Methodology:
# [STEP 1]: If they have the same api, well status, lat, and lon keep the last entry
# [STEP 2]: If they have the same api but different lat & lon, delete both
# [STEP 3]: If they have the same api, lat, and lon, but different well statuses:
#      [STEP 3a]: Keep the one listed as plugged
#      [STEP 3b]: If no well status is plugged, keep the one listed as orphaned
#      [STEP 3c]: If no well status is plugged or orphaned, keep the last entry

# [STEP 1]: Drop exact duplicates, keeping the last entry
ft = ft.drop_duplicates(subset=['api_num', 'well_status', 'latitude', 'longitude'], keep='last')
print("Step 1 Complete: Keep one version of exact duplicates")
print("Length after Step 1:", len(ft))
print('')

# [STEP 2]: Identify and delete APIs with multiple lat/lon entries
duplicate_api_mask = ft.duplicated(subset=['api_num'], keep=False)
api_groups = ft[duplicate_api_mask]
ft = ft[~duplicate_api_mask]
print("Step 2 Complete: Removed entries with the same API but different lat/lon.")
print("Length after Step 2:", len(ft))
print('')

# [STEP 3]: Handle same API, but diff status, prioritizing plugged or orphaned
# Define a function that prioritizes rows based on well status
def prioritize_status(group):
    # Check if "PLUGGED" status exists in the group
    if "PLUGGED" in group['well_status'].values:
        return group[group['well_status'] == "PLUGGED"].iloc[-1]
    # If no "PLUGGED" status, check for "ORPHANED"
    elif "ORPHANED" in group['well_status'].values:
        return group[group['well_status'] == "ORPHANED"].iloc[-1]
    # If neither "PLUGGED" nor "ORPHANED", keep the last entry
    else:
        return group.iloc[-1]

# Apply the prioritize_status function to each group of api nums
# Since we've deleted all api duplicates now, these are the ones that remain
# Therefore we only have to match on api
ft = ft.groupby(['api_num']).apply(prioritize_status)

# Reset the index after grouping to clean up the df structure
ft = ft.reset_index(drop=True)

# Summary after final filtering step
print("Step 3 Complete: Prioritized removal by well status")
print("Length after Step 3: ", len(ft))
print('')

#%%

# =============================================================================
# 1. Import and standardize well status column in USGS dataset
# =============================================================================

# Ignore storage space warnings
warnings.filterwarnings("ignore")

# Import USGS dataset
usgs = pd.read_csv("USGS/US_orphaned_wells.csv")

# Clean USGS API number attribute
usgs['Well identifier'] = usgs['Well identifier'].str[4:-4]
usgs['Well identifier'] = usgs['Well identifier'].replace('-', '', regex=True).astype("string")
usgs['Well identifier'] = usgs['Well identifier'].replace(',', '', regex=True).astype("string")
usgs['Well identifier'] = usgs['Well identifier'].apply(lambda x: x.strip())

# Standardize well status attribute
usgs['Status'] = "ORPHANED"

#%%

# =============================================================================
# 2. Import state data
# =============================================================================

# Alabama
alabama = pd.read_csv("Alabama/Alabama_May30_25.csv")
alabama.loc[alabama['StatusDesc'] == "Abandoned"]

# Alaska
alaska = pd.read_excel("Alaska/Official AOGCC Alaska Orphan Well List.xlsx")
alaska = alaska[alaska['General Location'] != 'Iniskin Peninsula, AK']
alaska = alaska[alaska['Surface Location Coordinates (NAD 83)'] != 'unknown']
alaska[['County', 'State']] = alaska['General Location'].str.split(',', n=1, expand=True)
alaska[['Lat', 'Lon']] = alaska['Surface Location Coordinates (NAD 83)'].str.split(',', n=1, expand=True)

# Arkansas
arkansas_shp = gpd.read_file('Arkansas/OIL_AND_GAS_WELLS_AOGC.shp')
arkansas = arkansas_shp.to_crs('EPSG:26915')
arkansas.to_csv('Arkansas/arkansas.csv', index=False)
arkansas = arkansas.loc[arkansas['wl_status'] == "AOW"]

# California
california = pd.read_csv("California/Well Prioritization.csv")

# Colorado
colorado_shp = gpd.read_file('Colorado/OWP_Shapefile.shp')
colorado = colorado_shp.to_crs('EPSG:26913')
colorado.to_csv('Colorado/colorado.csv', index=False)
colorado["well_name"] = colorado["Project"] + ' ' + colorado["LocationID"].astype(str)

# Florida
florida = pd.read_excel("Florida/OrphanWell_List_Florida_CurrentlyWorking_8_09_2024.xlsx")

# Indiana
indiana = pd.read_csv("Indiana/OilAndGasWells_-7355386120110653967.csv")
indiana["WellName"] = indiana["Lease_Name"] + ' ' + indiana["Well_Number"]
transformer_16N = Transformer.from_crs("EPSG:32616", "EPSG:4326")  # UTM Zone 16N to WGS84
transformer_17N = Transformer.from_crs("EPSG:32617", "EPSG:4326")  # UTM Zone 17N to WGS84
# Function to convert UTM to Latitude/Longitude with default Zone 16N
def utm_to_latlon_with_zone(easting, northing):
    # First transform using Zone 16
    lon_16, lat_16 = transformer_16N.transform(easting, northing)  # Correct order: (easting, northing)
    # Determine if the point might belong to Zone 17
    if -84 <= lon_16 < -78:  # Check if longitude suggests Zone 17
        lon_17, lat_17 = transformer_17N.transform(easting, northing)  # Correct order: (easting, northing)
        return lat_17, lon_17, 17
    return lat_16, lon_16, 16
# Apply the conversion and directly unpack the results into new columns
indiana[['Latitude', 'Longitude', 'UTM_Zone']] = indiana.apply(
    lambda row: pd.Series(utm_to_latlon_with_zone(row['Utmx'], row['Utmy'])), axis=1)

# Kansas
kansas = pd.read_csv("Kansas/Oil_and_Gas_Wells_Download_-5818358308320799179.csv")
kansas = kansas[kansas['Status'].isin(["KCC Fee Fund Plugging",
                                       "Federal Plugging Project"])]

# Kentucky
kentucky = pd.read_csv("Kentucky/Kentucky.csv")

# Louisiana
louisiana = pd.read_csv("Louisiana/Results.csv")

# Michigan
michigan = pd.read_csv("Michigan/Michigan_Orphan_Wells.csv")

# Mississippi
mississippi_1 = pd.read_csv("Mississippi/Well Search_O.csv")
mississippi_2 = pd.read_csv("Mississippi/Well Search_PO.csv")
mississippi = pd.concat([mississippi_1, mississippi_2])

# Missouri 
missouri = pd.read_excel("Missouri/Oil and Gas Well List Updated August 30,2024.xlsx")
missouri = missouri[missouri['Well Status'].isin(['Abandoned, Unknown Location                                                     ',
                                                  'Abandoned                                                                       ',
                                                  'Orphaned',
                                                  'Abandoned, No evidence of existence/ Unable to find                             ',
                                                  'Abandoned, Known Location and Verified                                          '])]
missouri["WellName"] = missouri["Lease Name"] + ' ' + missouri["Well Name"]

# Montana
montana = pd.read_csv("Montana/download.csv")

# Nebraska
nebraska_shp = gpd.read_file('Nebraska/NE_WELLS/NE_WELLS.shp')
nebraska = nebraska_shp.to_crs('EPSG:4269')
nebraska.to_csv('Nebraska/nebraska.csv', index=False)
nebraska = nebraska[nebraska['Well_Statu'].isin(["AB", "SI"])]

# Nevada
nevada = pd.read_excel("Nevada/oilgas_well_index_20200106.xlsx")
nevada = nevada[nevada['status'].isin(["Abandoned", "D & A"])]

# New Mexico
newmexico = pd.read_csv("New Mexico/New_Mexico_OCD_Oil_and_Gas_Wells (1).csv")

# New York
newyork1 = pd.read_csv("New York/Unknown_Located.csv")
newyork2 = pd.read_csv("New York/Unknown.csv")
newyork3 = pd.read_csv("New York/Unknown_Not_Found.csv")
newyork = pd.concat([newyork1, newyork2, newyork3])

# North Dakota
northdakota_shp = gpd.read_file('North Dakota/OGD_Wells/OGD_Wells.shp')
northdakota = northdakota_shp.to_crs('EPSG:4269')
northdakota.to_csv('North Dakota/northdakota.csv', index=False)
northdakota = northdakota.loc[northdakota['status'] == "AB"]

# Ohio
ohio = pd.read_excel("Ohio/Orphan Wells Ohio.xlsx")

# Oklahoma
oklahoma = pd.read_excel("Oklahoma/orphan_well_list.xlsx")

# Pennsylvania
pennsylvania = pd.read_csv("Pennsylvania/Abandoned_Orphan_Web.csv")

# South Dakota
southdakota = pd.read_excel("South Dakota/SDOILexport/Wells.xlsx")
southdakota = southdakota.loc[southdakota['Administrative Status'] == "Abandoned-Not Regulated"]

# Tennessee
tennessee = pd.read_excel("Tennessee/Forfeited Operator Wells 02_05_2025.xlsx")

# Texas
texas = pd.read_excel("Texas/Public Orphan Well List March.xlsx")
texas["well_name"] = texas["LEASE_NAME"] + ' ' + texas["WELL_NO"]

# Utah
utah = pd.read_excel("Utah/WellInformation Lat Long.xlsx")


# West Virginia
westvirginia = pd.read_excel("West Virginia/2025-07-30 Orphaned Well Counts.xlsx")
transformer = Transformer.from_crs("epsg:26917", "epsg:4326", always_xy=True)
westvirginia[['Longitude', 'Latitude']] = westvirginia.apply(
    lambda row: pd.Series(transformer.transform(row['UTM_E'], row['UTM_N'])),
    axis=1
)

# Wyoming
wyoming = pd.read_excel("Wyoming/OrphanWellsxls.xlsx")
wyoming = wyoming[~wyoming['F2Status'].isin(["SR", "PA"])]

# Define the column mapping for each state
state_fields_dict = { 
                    'Alabama' : 
                     {'api_10' : 'API',
                      'lat' : 'Latitude',
                      'lon' : 'Longitude',
                      #'state',
                      'county' : 'County',
                      'well_name' : 'WellName',
                      'operator' : 'Operator',
                      'well_status' : 'StatusDesc',
                      'status_date' : 'StatusDate',
                      'spud_date' : 'SpudDate'
                      }, 
                     
                     'Alaska' :
                     {'api_10' : 'API#',
                      'lat' : 'Lat',
                      'lon' : 'Lon',
                      'state' : 'State',
                      'county' : 'County',
                      'well_name' : 'Well Designation',
                      'operator' : 'Original Operator'
                      #'well_status',
                      #'status_date',
                      #'spud_date'
                      },
                     
                     'Arkansas' :
                     {'api_10' : 'api_wellno',
                      'lat' : 'latitude',
                      'lon' : 'longitude',
                      #'state'
                      'county' : 'county',
                      'well_name' : 'well_nm',
                      'operator' : 'coname',
                      'well_status' : 'wl_status',
                      'status_date' : 'dt_status',
                      #'spud_date'
                      },
                     
                     'California' :
                     {'api_10' : 'Well API',
                      'lat' : 'Latitude',
                      'lon' : 'Longitude',
                      #'state'
                      'county' : 'County',
                      'well_name' : 'Well Designation',
                      'operator' : 'Operator Name',
                      #'well_status',
                      #'status_date',
                      #'spud_date'
                      },
                     
                     'Colorado' :
                     {'api_10' : 'API',
                      'lat' : 'Latitude',
                      'lon' : 'Longitude',
                      #'state'
                      #'county",
                      'well_name' : 'well_name',
                      #'operator'
                      'well_status' : 'Status'
                      #'status_date',
                      #'spud_date'
                      },
                     
                     'Florida' : 
                      {'api_10' : 'API',
                       'lat' : 'Latitude',
                       'lon' : 'Longitude',
                       #'state',
                       'county' : 'COUNTY',
                       'well_name' : 'WELL_NAME',
                       'operator' : 'COMPANY',
                       'well_status' : 'Current Plugging Stage',
                       #'status_date'
                       #'spud_date' 
                       }, 
                      
                      'Indiana' : 
                       {'api_10' : 'Permit_Number', #Indiana doesnt use API
                        'lat' : 'Latitude',
                        'lon' : 'Longitude',
                        #'state',
                        'county' : 'County',
                        'well_name' : 'WellName',
                        'operator' : 'Operator_Name',
                        'well_status' : 'Status'
                        #'status_date'
                        #'spud_date'
                        },
                       
                       'Kansas' : 
                        {'api_10' : 'API_NUMBER',
                         'lat' : 'Latitude (NAD27)',
                         'lon' : 'Longitude (NAD27)',
                         #'state',
                         'county' : 'County',
                         'well_name' : 'WELL_LABEL',
                         'operator' : 'Original Operator',
                         'well_status' : 'Status',
                         #'status_date' : 'StatusDate',
                         'spud_date' : 'Spud Date'
                         }, 
                      
                        'Kentucky' : 
                         {'api_10' : ' API No ',
                          'lat' : 'LAT',
                          'lon' : 'LONG',
                          #'state',
                          'county' : 'County',
                          'well_name' : 'Well Name',
                          #'operator' : 'Original Operator',
                          'well_status' : 'Well Type',
                          #'status_date' : 'StatusDate',
                          #'spud_date' : 'Spud Date'
                          },
                         
                         'Louisiana' : 
                         {'api_10' : 'API Num',
                          'lat' : 'Latitude',
                          'lon' : 'Longitude',
                          #'state',
                          'county' : 'Parish Name',
                          'well_name' : 'Well Name',
                          'operator' : 'Operator Name',
                          'well_status' : 'Well Status Code Description',
                          'status_date' : 'Well Status Date',
                          'spud_date' : 'Spud Date'
                          },
                          
                          'Michigan' : 
                           {'api_10' : 'US_Well_ID_API',
                            'lat' : 'Latitude',
                            'lon' : 'Longitude',
                            'state' : 'State',
                            'county' : 'CountyName',
                            'well_name' : 'FacilityName',
                            'operator' : 'Company',
                            'well_status' : 'Data_Element',
                            'status_date' : 'last_edited_date',
                            #'spud_date' : 'Spud Date'
                            },
                           
                           'Mississippi' : 
                           {'api_10' : 'API',
                            'lat' : 'Lat(NAD83)',
                            'lon' : 'Long(NAD83)',
                            #'state',
                            'county' : 'County',
                            'well_name' : 'Name',
                            'operator' : 'Operator',
                            'well_status' : 'Well Status',
                            #'status_date' : 'StatusDate',
                            #'spud_date' : 'Spud Date'
                            }, 
                           
                           'Missouri' : 
                           {'api_10' : 'API Number',
                            'lat' : 'Well Latitude Decimal',
                            'lon' : 'Well Longitude Decimal',
                            #'state',
                            'county' : 'County',
                            'well_name' : 'WellName',
                            'operator' : 'Operator',
                            'well_status' : 'Well Status',
                            'status_date' : 'Well Status Date',
                            'spud_date' : 'Spud Date'
                            },
                           
                           
                           
                           'Nebraska' : 
                            {'api_10' : 'API_WellNo',
                             'lat' : 'Lat',
                             'lon' : 'Long',
                             #'state',
                             'county' : 'County',
                             'well_name' : 'Well_Name',
                             'operator' : 'Co_Name',
                             'well_status' : 'Well_Statu',
                             #'status_date' : 'StatusDate',
                             #'spud_date' : 'SpudDate'
                             }, 
                            
                            'Nevada' : 
                             {'api_10' : 'apino',
                              'lat' : 'latdegree',
                              'lon' : 'longdegree',
                              'state' : 'state_',
                              'county' : 'county',
                              'well_name' : 'wellname',
                              'operator' : 'operator_',
                              'well_status' : 'status',
                              'status_date' : 'statusdatetime',
                              #'spud_date' : 'SpudDate'
                              }, 
                             
                             'New Mexico' : 
                              {'api_10' : 'id',
                               'lat' : 'latitude',
                               'lon' : 'longitude',
                               #'state' : '',
                               'county' : 'county',
                               'well_name' : 'name',
                               #'operator' : '',
                               'well_status' : 'status',
                               'status_date' : 'statusdatetime',
                               'spud_date' : 'year_spudded'
                               },
                              
                    'New York' : 
                     {'api_10' : 'API Well Number',
                      'lat' : 'Surface Latitude',
                      'lon' : 'Surface Longitude',
                      #'state',
                      'county' : 'County',
                      'well_name' : 'Well Name',
                      'operator' : 'Company Name',
                      'well_status' : 'Well Status',
                      'status_date' : 'Status Date',
                      'spud_date' : 'Spud/Start Drilling Date'
                      },
                     
                    'North Dakota' : 
                     {'api_10' : 'api',
                      'lat' : 'latitude',
                      'lon' : 'longitude',
                      #'state',
                      'county' : 'County',
                      'well_name' : 'well_name',
                      'operator' : 'operator',
                      'well_status' : 'status',
                      #'status_date',
                      'spud_date' : 'spud_date'
                      }, 
                     
                     'Ohio' : 
                      {'api_10' : 'API_WELLNO',
                       'lat' : 'WHLat',
                       'lon' : 'WHLong',
                       #'state',
                       'county' : 'County',
                       'well_name' : 'WellName',
                       #'operator' : 'operator',
                       'well_status' : 'WL_STATUS',
                       'status_date' : 'DT_STATUS'
                       #'spud_date' : 'spud_date'
                       }, 
                      
                     'Oklahoma' : 
                      {'api_10' : 'API',
                       'lat' : 'Y',
                       'lon' : 'X',
                       #'state',
                       'county' : 'CountyName',
                       'well_name' : 'WellName',
                       'operator' : 'OperatorName',
                       'well_status' : 'WellStatus',
                       'status_date' : 'OrphanDate'
                       #'spud_date' : 'spud_date'
                       }, 
                      
                     'Pennsylvania' : 
                      {'api_10' : 'API',
                       'lat' : 'LATITUDE_DECIMAL',
                       'lon' : 'LONGITUDE_DECIMAL',
                       #'state',
                       'county' : 'COUNTY',
                       'well_name' : 'FARM_NAME',
                       'operator' : 'OPERATOR',
                       'well_status' : 'WELL_STATUS',
                       'status_date' : 'STATUS_DATE'
                       #'spud_date' : 'spud_date'
                       }, 
                      
                     'South Dakota' : 
                      {'api_10' : 'API Number',
                       'lat' : 'Latitude (GCS83)',
                       'lon' : 'Longitude (GCS83)',
                       #'state',
                       'county' : 'County',
                       'well_name' : 'Well Name',
                       'operator' : 'Operator',
                       'well_status' : 'Administrative Status',
                       #'status_date' : 'STATUS_DATE',
                       'spud_date' : 'Spud Date'
                       }, 
                      
                     'Tennessee' : 
                      {'api_10' : 'API',
                       'lat' : 'LAT',
                       'lon' : 'LONG',
                       #'state',
                       'county' : 'COUNTYNAME',
                       'well_name' : 'WELLNAME',
                       'operator' : 'OPNAME',
                       #'well_status' : 'Administrative Status',
                       #'status_date' : 'STATUS_DATE',
                       #'spud_date' : 'Spud Date'
                       }, 
                      
                     'Texas' : 
                      {'api_10' : 'API',
                       'lat' : 'latitude',
                       'lon' : 'longitude',
                       #'state',
                       'county' : 'COUNTY_NAME',
                       'well_name' : 'well_name',
                       'operator' : 'OPERATOR_NAME',
                       #'well_status' : 'Administrative Status',
                       #'status_date' : 'STATUS_DATE',
                       #'spud_date' : 'Spud Date'
                       }, 
                      
                     'Utah' : 
                      {'api_10' : 'API',
                       'lat' : 'Latitude',
                       'lon' : 'Longitude',
                       #'state',
                       'county' : 'County',
                       'well_name' : 'Well Name',
                       #'operator' : 
                       'well_status' : 'Operator'
                       #'status_date' : 'STATUS_DATE',
                       #'spud_date' : 'Spud Date'
                       }, 
                      
                     'West Virginia' :
                         {'api_10' : 'wellID',
                          'lat' : 'Latitude',
                          'lon' : 'Longitude',
                          #'state',
                          'county' : 'countyname',
                          'well_name' : 'entityname',
                          #'operator' : 
                          #'well_status' : 'Operator'
                          #'status_date' : 'STATUS_DATE',
                          #'spud_date' : 'Spud Date'
                          }, 
    
                      
                     'Wyoming' : 
                      {'api_10' : 'Apino',
                       'lat' : 'Lat',
                       'lon' : 'Lon',
                       #'state',
                       #'county' : 'COUNTY_NAME',
                       'well_name' : 'Wellname',
                       'operator' : 'Company',
                       #'well_status' : 'Administrative Status',
                       #'status_date' : 'STATUS_DATE',
                       #'spud_date' : 'Spud Date'
                       }, 
                      }

    # Define the required fields for Hauser df
required_fields = ['api_10', 'lat', 'lon', 'state', 'county', 'well_name', 'operator', 'well_status', 'spud_date']

#%%

# =============================================================================
# 7. Define function to clean & process states 
# =============================================================================

# Dictionary of state dfs that I need
states_data = {
    'Alabama' : pd.DataFrame(alabama),
    'Alaska' : pd.DataFrame(alaska),
    'Arkansas' : pd.DataFrame(arkansas),
    'California' : pd.DataFrame(california),
    'Colorado' : pd.DataFrame(colorado),
    'Florida' : pd.DataFrame(florida),
    'Indiana' : pd.DataFrame(indiana),
    'Kansas' : pd.DataFrame(kansas),
    'Kentucky' : pd.DataFrame(kentucky),
    'Louisiana' : pd.DataFrame(louisiana),
    'Michigan' : pd.DataFrame(michigan),
    'Mississippi' : pd.DataFrame(mississippi),
    'Missouri' : pd.DataFrame(missouri),
    'Nebraska' : pd.DataFrame(nebraska),
    'Nevada' : pd.DataFrame(nevada),
    'New Mexico' : pd.DataFrame(newmexico),
    'New York' : pd.DataFrame(newyork),
    'North Dakota' : pd.DataFrame(northdakota),
    'Ohio' : pd.DataFrame(ohio),
    'Oklahoma' : pd.DataFrame(oklahoma),
    'Pennsylvania' : pd.DataFrame(pennsylvania),
    'South Dakota' : pd.DataFrame(southdakota),
    'Tennessee' : pd.DataFrame(tennessee),
    'Texas' : pd.DataFrame(texas),
    'Utah' : pd.DataFrame(utah),
    'West Virginia' : pd.DataFrame(westvirginia),
    'Wyoming' : pd.DataFrame(wyoming)
    }

# Define a function to standardize and combine datasets into one
def standardize_and_combine(states_data, state_fields_dict, required_fields):
    # Initialize an empty df for the final result
    combined_df = pd.DataFrame()  
    
    for state_name, state_df in states_data.items():
        print('---------------------')
        print('Cleaning ' + state_name)
        # Create an empty df for the current state's cleaned data
        clean_df = pd.DataFrame()
        
        # Loop through the required fields and map them to the state-specific columns
        for std_col in required_fields:
            # Use .get() to avoid KeyErrors when a field is missing
            state_col = state_fields_dict[state_name].get(std_col, None)
            if state_col in state_df.columns:
                clean_df[std_col] = state_df[state_col]
            else:
                # Fill missing columns with NaN
                clean_df[std_col] = np.nan  
        
        # Add the state name as a new column
        clean_df['state'] = state_name
        
        print(state_name + ": " + str(len(clean_df)) + " orphaned wells before cleaning duplicates")
        print('')
        
        # Append the cleaned data to the combined DataFrame
        combined_df = pd.concat([combined_df, clean_df], ignore_index=True)
    
    return combined_df


#%%
# =============================================================================
# 9. Call standardize and combine function
# =============================================================================

# Call the function
hauser_2025 = standardize_and_combine(states_data, state_fields_dict, required_fields)
print('-----------------------------------------')


#%%
# =============================================================================
# 10. Clean Hauser df
# =============================================================================

# Delete those with missing lat/lons
hauser_2025 = hauser_2025.dropna(subset=['lat'])
hauser_2025 = hauser_2025.dropna(subset=['lon'])
hauser_2025 = hauser_2025[hauser_2025['lat'] != 0]
hauser_2025 = hauser_2025[hauser_2025['lon'] != 0]
hauser_2025 = hauser_2025[hauser_2025['lat'] != 'nan']
hauser_2025 = hauser_2025[hauser_2025['lon'] != 'nan']

# Convert lat & lon to numeric dtypes
hauser_2025['lat'] = pd.to_numeric(hauser_2025['lat'], errors='coerce')
hauser_2025['lon'] = pd.to_numeric(hauser_2025['lon'], errors='coerce')

# Make sure all lons are negative
# (lats are within bounds)
hauser_2025['lon'] = hauser_2025['lon'].abs()
hauser_2025['lon'] = hauser_2025['lon']*-1 

# Drop NA API #s
hauser_2025 = hauser_2025.dropna(subset=['api_10'])
# Make API formatting consistent
hauser_2025['api_10'] = hauser_2025['api_10'].astype("string")
hauser_2025['api_10'] = hauser_2025['api_10'].replace('-', '', regex=True)
hauser_2025['api_10'] = hauser_2025['api_10'].replace(',', '', regex=True)
hauser_2025['api_10'] = hauser_2025['api_10'].replace(' ', '', regex=True)
hauser_2025['api_10'] = hauser_2025['api_10'].apply(lambda x: x.strip())
hauser_2025['api_10'] = hauser_2025['api_10'].apply(lambda x: x[:10] if pd.notna(x) and len(x) > 10 else x)

# Apply special formatting for relevant states
# Add leading zeros for CA and FL
hauser_2025.loc[hauser_2025['state'] == 'California', 'api_10'] = hauser_2025.loc[hauser_2025['state'] == 'California', 'api_10'].str.zfill(10)
hauser_2025.loc[hauser_2025['state'] == 'Florida', 'api_10'] = hauser_2025.loc[hauser_2025['state'] == 'Florida', 'api_10'].str.zfill(10)
# Add state digits to start of PA, TN, TX, and WY
hauser_2025.loc[hauser_2025['state'] == 'Pennsylvania', 'api_10'] = '37' + hauser_2025.loc[hauser_2025['state'] == 'Pennsylvania', 'api_10']    
hauser_2025.loc[hauser_2025['state'] == 'Tennessee', 'api_10'] = '41' + hauser_2025.loc[hauser_2025['state'] == 'Tennessee', 'api_10']
hauser_2025.loc[hauser_2025['state'] == 'Texas', 'api_10'] = '42' + hauser_2025.loc[hauser_2025['state'] == 'Texas', 'api_10']  
hauser_2025.loc[hauser_2025['state'] == 'Wyoming', 'api_10'] = '490' + hauser_2025.loc[hauser_2025['state'] == 'Wyoming', 'api_10']  

# Delete duplicate APIs from each state
hauser_2025 = hauser_2025.drop_duplicates(subset='api_10', keep=False)

# Add state abbreviation column
#List of states
state2abbrev = {'Alaska': 'AK',
                'Alabama': 'AL',
                'Arkansas': 'AR',
                'California': 'CA',
                'Florida': 'FL',
                'Indiana': 'IN',
                'Kansas': 'KS',
                'Kentucky': 'KY',
                'Louisiana': 'LA',
                'Michigan': 'MI',
                'Missouri': 'MO',
                'Mississippi': 'MS',
                'Montana': 'MT',
                'North Dakota': 'ND',
                'Nebraska': 'NE',
                'New Mexico': 'NM',
                'Nevada': 'NV',
                'New York': 'NY',
                'Ohio': 'OH',
                'Oklahoma': 'OK',
                'Pennsylvania': 'PA',
                'South Dakota': 'SD',
                'Tennessee': 'TN',
                'Texas': 'TX',
                'Utah': 'UT',
                'Virginia': 'VA',
                'West Virginia': 'WV',
                'Wyoming': 'WY'}
hauser_2025['st_abbrev'] = hauser_2025['state'].map(state2abbrev)

# Head count
hauser_2025.groupby('state').agg(
    count = ("api_10", "count"))


#%%
# =============================================================================
# 10. Delete wells that are listed as plugged in FracTracker 
# =============================================================================

# Using API: if a well is listed as plugged in FracTracker, remove it from Hauser_2025
# Filter FT dataset to only include plugged wells
plugged_wells_ft = ft[ft['well_status'] == 'PLUGGED']
plugged_wells_ft = plugged_wells_ft[['stusps', 'api_num', 'operator', 'well_name']]

# Make sure both are the same datatype
plugged_wells_ft['api_num'] = plugged_wells_ft['api_num'].astype("string")
hauser_2025['api_10'] = hauser_2025['api_10'].astype("string")

# Split the data into Indiana and other datasets for different merge conditions
indiana_wells = hauser_2025[hauser_2025['state'] == 'Indiana']
other_wells = hauser_2025[hauser_2025['state'] != 'Indiana']


# Merge Indiana wells on operator name and lease name
indiana_merged = pd.merge(indiana_wells, plugged_wells_ft,
                          left_on=['operator', 'well_name'], 
                          right_on=['operator', 'well_name'],
                          how='left', indicator=True)

# Merge other wells on API numbers
other_merged = pd.merge(other_wells, plugged_wells_ft,
                        left_on='api_10', right_on='api_num', how='left',
                        indicator=True)

# Drop the temporary merge columns & rename duplicates
indiana_merged = indiana_merged.drop(['api_num', 'stusps', 'latitude', 'longitude'], axis=1, errors='ignore')
other_merged = other_merged.drop(['stusps', 'api_num', 'operator_y', 'well_name_y',], axis=1, errors='ignore')
other_merged = other_merged.rename(columns={'well_name_x': 'well_name', 'operator_x': 'operator'})

# Combine both merged datasets
hauser_2025f = pd.concat([indiana_merged, other_merged])

# Create DataFrame of wells actually plugged in FracTracker
actually_plugged = hauser_2025f[hauser_2025f['_merge'] == 'both']
print(actually_plugged.groupby('state').size().reset_index(name='Actually_plugged'))

# Remove plugged wells from Hauser_2024 based on merge results
hauser_2025f = hauser_2025f[hauser_2025f['_merge'] == 'left_only']


# Drop the temporary merge columns
hauser_2025f = hauser_2025f.drop(['_merge'], axis=1, errors='ignore')

# Display the final grouped count by state
print('-----------------------------------------')
print(hauser_2025f.groupby('state').size().reset_index(name='Hauser_well_count'))
print(hauser_2025f.columns)
hauser_2025_grouped = hauser_2025f.groupby('state').size().reset_index(name='Hauser_well_count')


#%%

# =============================================================================
# =============================================================================
# =============================================================================
# AIM 2 : HOW MANY WELLS HAVE BECOME ORPHANED SINCE THE USGS REPORT?
# =============================================================================
# =============================================================================
# =============================================================================

# =============================================================================
# 1. Compare APIs in Hauser_2024 to USGS
# =============================================================================

# Separate Indiana, Kansas and other wells
indiana_wells = hauser_2025f[hauser_2025f['state'] == 'Indiana']
other_wells = hauser_2025f[hauser_2025f['state'] != 'Indiana']

# Ensure columns to compare have the same data type
usgs[['County', 'Well name', 'Well number']] = usgs[['County', 'Well name', 'Well number']].astype("string")
indiana_wells[['well_name', 'spud_date']] = indiana_wells[['well_name', 'spud_date']].astype("string")

# Find Indiana wells that are not in USGS based on well name and number
indiana_newly_orphaned = indiana_wells[
    ~indiana_wells[['well_name', 'spud_date']].apply(tuple, axis=1).isin(
        usgs[['Well name', 'Well number']].apply(tuple, axis=1))]

# Find non-Indiana wells that are not in USGS based on API
other_newly_orphaned = other_wells[~other_wells['api_10'].isin(usgs['Well identifier'])]

# Concatenate the results
newly_orphaned = pd.concat([indiana_newly_orphaned, other_newly_orphaned])

# Get a count of newly orphaned wells by state
print('-----------------------------------------')
print(newly_orphaned.groupby('state').size().reset_index(name='new_orphaned_well_count'))
newly_orphaned_grouped = newly_orphaned.groupby('state').size().reset_index(name='new_orphaned_well_count')

#%%

# =============================================================================
# 2. Make hauser_status column
# =============================================================================
# Add default "Orphaned since USGS" status to all wells in hauser_2024f
hauser_2025f['hauser_status'] = 'Orphaned since USGS'

# Update status to "Newly orphaned" for Indiana wells in newly_orphaned
hauser_2025f.loc[
    hauser_2025f[['state', 'well_name', 'spud_date']].apply(tuple, axis=1).isin(
        indiana_newly_orphaned[['state', 'well_name', 'spud_date']].apply(tuple, axis=1)
    ),
    'hauser_status'
] = 'Newly orphaned'

# Update status to "Newly orphaned" for other states based on API match
hauser_2025f.loc[
    hauser_2025f['api_10'].isin(other_newly_orphaned['api_10']),
    'hauser_status'] = 'Newly orphaned'

# Check the final DataFrame
print(hauser_2025f[['state', 'hauser_status']].value_counts())


#%%


# =============================================================================
# =============================================================================
# =============================================================================
# AIM 3 : HOW MANY WELLS HAVE BEEN PLUGGED SINCE THE USGS REPORT?
# =============================================================================
# =============================================================================
# =============================================================================

# =============================================================================
# 1. Compare APIs in USGS to FT Plugged
# =============================================================================
 
# THIS WON'T WORK FOR INDIANA

# Find APIs in USGS but not in Hauser 2024
newly_plugged = usgs[~usgs['Well identifier'].isin(hauser_2025['api_10'])]

# From this, drop APIs that have a status other than "PLUGGED" in ft
newly_plugged = newly_plugged[newly_plugged['Well identifier'].isin(plugged_wells_ft['api_num'])]

# From this, drop APIs that aren't in hauser_2024 bc they're actually plugged while currently listed as orphaned
newly_plugged = newly_plugged[~newly_plugged['Well identifier'].isin(actually_plugged['api_10'])]

# From this, drop APIs that are fake (USGS assigned value)
newly_plugged = newly_plugged[~newly_plugged['Well identifier'].astype(str).str.startswith('ID')]
newly_plugged = newly_plugged[~newly_plugged['Well identifier'].astype(str).str.startswith('D')]

# View
newly_plugged_grouped = newly_plugged.groupby('State').size().reset_index(name='since_plugged_well_count')
print('-----------------------------------------')
print(newly_plugged_grouped) 

#%%

# =============================================================================
# =============================================================================
# =============================================================================
# VISUALIZE & EXPORT
# =============================================================================
# =============================================================================
# =============================================================================

# =============================================================================
# 1. Validate that wells are within their specified states
# ============================================================================= 

# Fix Indiana attributes
hauser_2025f.loc[hauser_2025f['state'] == 'Indiana', 'spud_date'] = pd.NA

# Drop spud date column
hauser_2025f['spud_date'] = hauser_2025f['spud_date'].astype(str)
#hauser_2025f.drop('spud_date', axis=1, inplace=True)
#hauser_2025f.drop('plug_date', axis=1, inplace=True)

# Convert to gdf
hauser_2025_gdf = gpd.GeoDataFrame(hauser_2025f,
                       geometry=gpd.points_from_xy(hauser_2025f.lon, hauser_2025f.lat),
                       crs="EPSG:4326")

# Make sure there are no hidden datetimes
hauser_2025_gdf['api_10'] = hauser_2025_gdf['api_10'].astype(str)
hauser_2025_gdf['state'] = hauser_2025_gdf['state'].astype(str)
hauser_2025_gdf['county'] = hauser_2025_gdf['county'].astype(str)
hauser_2025_gdf['well_name'] = hauser_2025_gdf['well_name'].astype(str)
hauser_2025_gdf['operator'] = hauser_2025_gdf['operator'].astype(str)
hauser_2025_gdf['well_status'] = hauser_2025_gdf['well_status'].astype(str)
hauser_2025_gdf['st_abbrev'] = hauser_2025_gdf['st_abbrev'].astype(str)
hauser_2025_gdf['hauser_status'] = hauser_2025_gdf['hauser_status'].astype(str)


from pygris import states
import us  # us library provides mappings for state names and abbreviations

# Retrieve all state boundaries
state_boundaries = states()

# Create a dictionary to map state names to abbreviations
state_name_to_abbr = {state.name.upper(): state.abbr for state in us.states.STATES}

def validate_points_in_state(gdf):
    # Standardize and map full state names to abbreviations
    gdf['state_abbr'] = gdf['state'].str.upper().map(state_name_to_abbr)
    
    # Initialize an empty list to store validation results
    validation_results = []
    
    for idx, row in gdf.iterrows():
        # Filter state boundaries for the claimed state
        state_geom = state_boundaries[state_boundaries['STUSPS'] == row['state_abbr']].geometry
        
        # Check if point is within the claimed state boundary
        if not state_geom.empty and row['geometry'].within(state_geom.iloc[0]):
            validation_results.append(True)
        else:
            validation_results.append(False)
    
    # Add validation results to the original geodataframe
    gdf['is_within_claimed_state'] = validation_results
    # Drop the temporary abbreviation column
    gdf.drop(columns=['state_abbr'], inplace=True)
    return gdf

hauser_2025_gdf = validate_points_in_state(hauser_2025_gdf) 
hauser_2025_gdf = hauser_2025_gdf[hauser_2025_gdf.is_within_claimed_state != False]

#%%

# =============================================================================
# 2. Map all pts
# ============================================================================= 

from pygris import states
from pygris.utils import shift_geometry

us = states(cb = True, resolution = "20m")
us_rescaled = shift_geometry(us)

orphans_rescaled = shift_geometry(hauser_2025_gdf)
fig, ax = plt.subplots()

us_rescaled.plot(ax = ax, color = "grey") 
orphans_rescaled.plot(ax = ax, color = "black", marker='o', markersize=2)

# Set axis limits for the contiguous US
ax.set_xlim(us_rescaled.total_bounds[0], us_rescaled.total_bounds[2])
ax.set_ylim(us_rescaled.total_bounds[1], us_rescaled.total_bounds[3])

# Add a title for context
ax.set_title("Hauser_2025 Wells (Newly Orphaned, Newly Plugged, etc.")

# Show the plot
plt.show()

#%%

# =============================================================================
# 3. Export 
# ============================================================================= 

# Change directory
os.chdir('/Users/gracehauser/Desktop/Publication/Results') 

# Hauser final ds
hauser_2025_gdf.to_file('hauser_2025.shp', driver='ESRI Shapefile')

# Newly plugged ds
newly_plugged_gdf = gpd.GeoDataFrame(newly_plugged,
                        geometry=gpd.points_from_xy(newly_plugged.Longitude, newly_plugged.Latitude),
                        crs="EPSG:4326")
newly_plugged_gdf.to_file('newly_plugged.shp', driver='ESRI Shapefile')

# Newly orphaned ds
newly_orphaned['api_10'] = newly_orphaned['api_10'].astype(str)
newly_orphaned['state'] = newly_orphaned['state'].astype(str)
newly_orphaned['county'] = newly_orphaned['county'].astype(str)
newly_orphaned['well_name'] = newly_orphaned['well_name'].astype(str)
newly_orphaned['operator'] = newly_orphaned['operator'].astype(str)
newly_orphaned['well_status'] = newly_orphaned['well_status'].astype(str)
newly_orphaned['spud_date'] = newly_orphaned['spud_date'].astype(str)
newly_orphaned['st_abbrev'] = newly_orphaned['st_abbrev'].astype(str)
newly_orphaned_gdf = gpd.GeoDataFrame(newly_orphaned,
                        geometry=gpd.points_from_xy(newly_orphaned.lon, newly_orphaned.lat),
                        crs="EPSG:4326")
newly_orphaned_gdf.to_file('newly_orphaned.shp', driver='ESRI Shapefile') 


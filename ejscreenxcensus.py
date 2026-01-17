#!/usr/bin/env python
# coding: utf-8

#%%


# title: "ejscreenxcensus"
# author: Grace Hauser
# affiliations: YSPH Department of EHS & FracTracker Alliance
# date of last update: 07/26/2025
# script aim: combine EJScreen and ACS Census data into one dataframe


#%%


# set working directory
import os
os.chdir('/Users/gracehauser/Desktop/Publication/Data/Census')

# load packages
import numpy as np
import pandas as pd
import geopandas as gpd
from functools import reduce
import requests, zipfile, io
import math


#%%


### 1. import census data

# ignore storage space warnings
import warnings
warnings.filterwarnings("ignore")

# education
acs_b15003 = pd.read_csv('EDUCATION/B15003_EDUCATIONAL_ATTAINMENT/ACSDT5Y2021.B15003-Data.csv')

# poverty
acs_c17002 = pd.read_csv('EMPLOYMENT_INCOME/C17002_RATIO_INCOMExPOVERTY/ACSDT5Y2021.C17002-Data.csv')

# government programs
acs_b19058 = pd.read_csv('GOVERNMENT_PROGRAMS/B19058_PUBLIC_ASSISTANCE_SNAP/ACSDT5Y2021.B19058-Data.csv')
acs_b27010 = pd.read_csv('GOVERNMENT_PROGRAMS/B27010_HEALTH_INSURANCExAGE/ACSDT5Y2021.B27010-Data.csv')

# housing
acs_b11012 = pd.read_csv('HOUSING/B11012_HOUSEHOLDSxTYPE/ACSDT5Y2021.B11012-Data.csv')
acs_b25009 = pd.read_csv('HOUSING/B25009_TENURExHOUSEHOLD_SIZE/ACSDT5Y2021.B25009-Data.csv')
acs_b25024 = pd.read_csv('HOUSING/B25024_UNITS_IN_STRUCTURE/ACSDT5Y2021.B25024-Data.csv')
acs_b25047 = pd.read_csv('HOUSING/B25047_PLUMBING_FACILITIES/ACSDT5Y2021.B25047-Data.csv')
acs_b25070 = pd.read_csv('HOUSING/B25070_GROSS_RENT_AS_PCT_HOUSEHOLD_INCOME/ACSDT5Y2021.B25070-Data.csv')

# technology
acs_b28001 = pd.read_csv('TECHNOLOGY/B28001_COMPUTERS/ACSDT5Y2021.B28001-Data.csv')
acs_b28002 = pd.read_csv('TECHNOLOGY/B28002_INTERNET/ACSDT5Y2021.B28002-Data.csv')


#%%


### 2. select relevant columns & combine all census data

# education
acs_b15003 = acs_b15003[['GEO_ID', 'NAME', 'B15003_001E', 'B15003_001M', 'B15003_002E',
                     'B15003_002M', 'B15003_003E', 'B15003_003M', 'B15003_004E',
                     'B15003_004M', 'B15003_005E', 'B15003_005M', 'B15003_006E',
                     'B15003_006M', 'B15003_007E', 'B15003_007M', 'B15003_008E',
                     'B15003_008M', 'B15003_009E', 'B15003_009M', 'B15003_010E',
                     'B15003_010M', 'B15003_011E', 'B15003_011M', 'B15003_012E',
                     'B15003_012M', 'B15003_013E', 'B15003_013M', 'B15003_014E',
                     'B15003_014M', 'B15003_015E', 'B15003_015M', 'B15003_016E',
                     'B15003_016M', 'B15003_017E', 'B15003_017M', 'B15003_018E',
                     'B15003_018M', 'B15003_019E', 'B15003_019M', 'B15003_020E',
                     'B15003_020M', 'B15003_021E', 'B15003_021M', 'B15003_022E',
                     'B15003_022M', 'B15003_023E', 'B15003_023M', 'B15003_024E',
                     'B15003_024M', 'B15003_025E', 'B15003_025M']]

# poverty
acs_c17002 = acs_c17002[['GEO_ID', 'NAME','C17002_001E','C17002_001M','C17002_002E','C17002_002M',
                         'C17002_003E', 'C17002_003M', 'C17002_004E', 'C17002_004M', 
                         'C17002_005E', 'C17002_005M', 'C17002_006E', 'C17002_006M', 
                         'C17002_007E', 'C17002_007M', 'C17002_008E', 'C17002_008M']]

# government programs
acs_b19058 = acs_b19058[['GEO_ID', 'NAME', 'B19058_001E', 'B19058_001M', 'B19058_002E', 'B19058_002M']]


acs_b27010 = acs_b27010[['GEO_ID', 'NAME', 'B27010_002E', 'B27010_006E', 'B27010_007E', 'B27010_013E',
                         'B27010_017E', 'B27010_002M', 'B27010_006M', 'B27010_007M',
                         'B27010_013M', 'B27010_017M', 'B27010_001E', 'B27010_055E',
                         'B27010_062E', 'B27010_066E', 'B27010_001M', 'B27010_055M', 
                         'B27010_062M', 'B27010_066M', 'B27010_051E','B27010_033E',
                         'B27010_050E', 'B27010_051M', 'B27010_033M', 'B27010_050M']]

# housing
acs_b11012 = acs_b11012[['GEO_ID', 'NAME','B11012_001E', 'B11012_008E', 'B11012_013E', 'B11012_001M',
                         'B11012_008M', 'B11012_013M']]
acs_b25009 = acs_b25009[['GEO_ID', 'NAME','B25009_001E', 'B25009_010E', 'B25009_001M', 'B25009_010M']]
acs_b25024 = acs_b25024[['GEO_ID', 'NAME','B25024_010E', 'B25024_001E', 'B25024_010M', 'B25024_001M']]
acs_b25047 = acs_b25047[['GEO_ID', 'NAME','B25047_001E', 'B25047_003E', 'B25047_001M', 'B25047_003M']]
acs_b25070 = acs_b25070[['GEO_ID', 'NAME','B25070_001E', 'B25070_007E', 'B25070_008E', 'B25070_009E',
                         'B25070_010E', 'B25070_001M', 'B25070_007M', 'B25070_008M',
                         'B25070_009M', 'B25070_010M']]

# technology
acs_b28001 = acs_b28001[['GEO_ID', 'NAME','B28001_001E', 'B28001_011E', 'B28001_001M', 'B28001_011M']]
acs_b28002 = acs_b28002[['GEO_ID', 'NAME','B28002_001E', 'B28002_013E', 'B28002_001M', 'B28002_013M']]


#%%


### 2. combine all census data

# make list of all census dataframes
dfs = [acs_b15003,
       acs_c17002,
       acs_b19058, acs_b27010,
       acs_b11012, acs_b25009, acs_b25024, acs_b25047, acs_b25070,
       acs_b28001, acs_b28002]

# merge
acs = reduce(lambda left,right: pd.merge(left,right,on=['GEO_ID', 'NAME'], how='outer'), dfs)


#%%


### 3. clean census data

# delete first row
acs = acs.iloc[1:]

# split name column into many columns
acs[['Block_Group', 'Census_Tract', 'County', 'State']] = acs['NAME'].str.split(', ', expand=True)

# clean block group and census tract columns
acs['Block_Group'] = acs.Block_Group.str[12:]
acs['Census_Tract'] = acs.Census_Tract.str[13:]

# delete hawaii and puerto rico
acs = acs[acs.State != 'Hawaii']
acs = acs[acs.State != 'Puerto Rico']

# make ID column since acs's geoid column is the full reference and we only want the 9-digit reference to merge on
acs['ID'] = acs.GEO_ID.str[9:]

# change type to int to get rid of leading 0s
acs['ID'] = acs['ID'].astype(int)


#%%


### 4. import ejscreen data
ejscreen = pd.read_csv('EJSCREEN_2023_BG_with_AS_CNMI_GU_VI.csv', encoding='utf-8', encoding_errors='ignore')


#%%


### 5. clean ejscreen data

# delete hawaii, northern mariana island, guam, puerto rico, US virgin islands, and american samoa
ejscreen = ejscreen[ejscreen.STATE_NAME != 'Hawaii']
ejscreen = ejscreen[ejscreen.STATE_NAME != 'Northern Mariana Is']
ejscreen = ejscreen[ejscreen.STATE_NAME != 'Guam']
ejscreen = ejscreen[ejscreen.STATE_NAME != 'Puerto Rico']
ejscreen = ejscreen[ejscreen.STATE_NAME != 'Virgin Islands']
ejscreen = ejscreen[ejscreen.STATE_NAME != 'American Samoa']

# select columns of interest
ejscreen = ejscreen[['ID', 'ACSTOTPOP',
                     'PEOPCOLOR', 'PEOPCOLORPCT',
                     'LINGISO', 'LINGISOPCT',
                     'UNDER5', 'UNDER5PCT','OVER64', 'OVER64PCT', 
                     'PM25', 'DSLPM', 'OZONE', 'CANCER', 'RESP',
                     'RSEI_AIR', 'NPL_CNT', 'PNPL', 'TSDF_CNT', 'PTSDF',
                     'PWDIS', 'UST', 'PRE1960', 'PRE1960PCT', 'PRMP',
                     'AREALAND', 'AREAWATER', 'Shape_Length', 'Shape_Area']]


#%%

### 6. merge census and ejscreen data

# merge acs data to ejscreen data, keeping all ejscreen data
acs_ej = ejscreen.merge(acs, on = "ID", how = "left", indicator=True)

# there are 2 block groups that are in the census TIGERLINE file & ejscreen file but not in the ACS files...
# thus, there are stored as NaNs and throw errors as pandas reads the 2 NaNs as duplicates later on
# let's delete these for now
acs_ej.drop_duplicates(subset=['GEO_ID'], inplace = True)


#%%


### 7. clean merged dataset

# bring identifiers to the front of the dataframe
st = acs_ej['State']
acs_ej.drop(labels=['State'], axis=1,inplace = True)
acs_ej.insert(1, 'State', st)

cty = acs_ej['County']
acs_ej.drop(labels=['County'], axis=1,inplace = True)
acs_ej.insert(2, 'County', cty)

ct = acs_ej['Census_Tract']
acs_ej.drop(labels=['Census_Tract'], axis=1,inplace = True)
acs_ej.insert(3, 'Census_Tract', ct)

bg = acs_ej['Block_Group']
acs_ej.drop(labels=['Block_Group'], axis=1,inplace = True)
acs_ej.insert(4, 'Block_Group', bg)

# delete unneeded columns
acs_ej.drop(labels=['NAME'], axis=1,inplace = True)
acs_ej.drop(labels=['_merge'], axis=1,inplace = True)

# clean acs data with jam values
# source: https://www.census.gov/programs-surveys/acs/technical-documentation/code-lists.html "jam values"

# - : margin of error for median > median
acs_ej = acs_ej.replace({"-": np.nan}) 

# N : data can't be displayed because there were an insufficient number of samples
acs_ej = acs_ej.replace({"N": np.nan})

# (X) : data isn't applicable or isn't available
acs_ej = acs_ej.replace({"(X)": np.nan})

# ** : the margin of error could not be computed because there weren't enough samples
acs_ej = acs_ej.replace({"**": np.nan})

# ***** : margin of error isn't appropriate because the measure corresponds to a single measure
# effectively, the margin of error should be treated as 0
acs_ej = acs_ej.replace({"*****": 0})


#%%


### 8. create EJ metrics of my own - percentages with no aggregation required

# list of acs data I want to work with
acs_list = ['B25009', # for %rented
            'B25024', # for %mobile home
            'B28002', # for %no internet access
            'B28001', # for %no computer at home
            'B25047', # for %no plumbing
            'B19058', # for %receiving SNAP/public assistance
            'C17002_und0.5', # for %families whose income is ½x the poverty threshold for their family size
            'B25070_50pls'] # for %extremely cost-burdened ppl (spend over 50% of income on rent)

# initialize empty list to append my finished dfs to later
list_pct_ej_metrics = []

for file in acs_list:
    print("Working on " + file + "...")
    
    # select columns relevant for %rented calculation
    if file == 'B25009':
        df = acs_ej[['GEO_ID','B25009_001E','B25009_001M','B25009_010E','B25009_010M']]
    
    # select columns relevant for %mobile home calculation
    if file == 'B25024':
        df = acs_ej[['GEO_ID','B25024_001E','B25024_001M','B25024_010E','B25024_010M']]
    
    # select columns relevant for %no internet access calculation
    if file == 'B28002':
        df = acs_ej[['GEO_ID','B28002_001E','B28002_001M','B28002_013E','B28002_013M']]
    
    # select columns relevant for %no computer at home calculation
    if file == 'B28001':
        df = acs_ej[['GEO_ID','B28001_001E','B28001_001M','B28001_011E','B28001_011M']]
    
    # select columns relevant for %no plumbing calculation
    if file == 'B25047':
        df = acs_ej[['GEO_ID','B25047_001E','B25047_001M','B25047_003E','B25047_003M']]
    
    # select columns relevant for %receiving SNAP/public assistance calculation
    if file == 'B19058':
        df = acs_ej[['GEO_ID','B19058_001E','B19058_001M','B19058_002E','B19058_002M']]
    
    # select columns relevant for %extreme poverty calculation
    if file == 'C17002_und0.5':
        df = acs_ej[['GEO_ID','C17002_001E','C17002_001M','C17002_002E','C17002_002M']]
    
    # select columns relevant for %extremely cost-burdened ppl (spend over 50% of income on rent) calculation
    if file == 'B25070_50pls':
        df = acs_ej[['GEO_ID','B25070_001E','B25070_001M','B25070_010E','B25070_010M']]  
    
    # rename columns to streamline process
    df.rename(columns={df.columns[1]: 'TOT_EST',
                       df.columns[2]: 'MOE_TOT_EST',
                       df.columns[3]: 'NUM',
                       df.columns[4]: 'MOE_NUM'}, inplace= True)
    
    # convert to floats to do calculations
    df[['TOT_EST', 'MOE_TOT_EST', 'NUM', 'MOE_NUM']] = df[['TOT_EST', 'MOE_TOT_EST', 'NUM', 'MOE_NUM']].astype(float)
    # delete first row (it contains column descriptions, no data)
    df = df[1:]
    
    # calculate proportion and percent of interest
    df['PROP'] = df['NUM'] / df['TOT_EST']
    df['PCT'] = df['PROP']
    
    # calculate margin of error corresponding to the pct of interest
    # formulas on pg 63-64: https://www.census.gov/content/dam/Census/library/publications/2020/acs/acs_general_handbook_2020_ch08.pdf
    df['MOE_PCT'] = (1/df['TOT_EST']) * np.sqrt(pow(df['MOE_NUM'],2) - (pow(df['PROP'],2) * pow(df['MOE_TOT_EST'],2)))
    
    # keep only columns of interest
    df = df[['GEO_ID','PCT', 'MOE_PCT']]
    # rename columns
    df.rename(columns={df.columns[1]: str(file) + '_PCT',
                       df.columns[2]: str(file) + '_PCT_MOE'}, inplace= True)
    # set index to geoid to keep geoid unique
    df.set_index('GEO_ID', inplace = True)
    
    # append this df to list
    list_pct_ej_metrics.append(df)
    
    print("Complete!")


#%%


### 10. create EJ metrics of my own - percentages with aggregation required

# list of acs data I want to work with
metrics_to_agg = ['B11012', # for %single parent household
                  'B15003_nohsgrad', # for %no diploma or GED
                  'B27010_18und', # for %ppl aged 18 and under w/gov-provided or no healthcare
                  'B27010_65pls', # for %ppl aged 65 and up w/gov-provided or no healthcare
                  'B27010_uninsured', # for %uninsured ppl 
                  'C17002_und1', # for %families whose income is equal to the poverty threshold for their family size
                  'C17002_und1.5', # for %families whose income is 3/2x the poverty threshold for their family size
                  'C17002_und2', # for %families whose income is 2x the poverty threshold for their family size
                  'B25070_30pls'] # for %cost-burdened ppl (spend over 30% of income on rent)

# initialize empty list to append my finished dfs to later                  
agg_ej_metric_list = []

for file in metrics_to_agg:
    print("Working on " + file + "...")
    
    # select columns relevant for %single parent household calculation
    if file == 'B11012':
        df = acs_ej[['GEO_ID', 'B11012_001E', 'B11012_001M', 'B11012_008E',
                     'B11012_008M', 'B11012_013E', 'B11012_013M']]
        
    # select columns relevant for %no diploma or GED calculation
    if file == 'B15003_nohsgrad':
        df = acs_ej[['GEO_ID', 'B15003_001E', 'B15003_001M', 'B15003_002E',
                     'B15003_002M', 'B15003_003E', 'B15003_003M', 'B15003_004E',
                     'B15003_004M', 'B15003_005E', 'B15003_005M', 'B15003_006E',
                     'B15003_006M', 'B15003_007E', 'B15003_007M', 'B15003_008E',
                     'B15003_008M', 'B15003_009E', 'B15003_009M', 'B15003_010E',
                     'B15003_010M', 'B15003_011E', 'B15003_011M', 'B15003_012E',
                     'B15003_012M', 'B15003_013E', 'B15003_013M', 'B15003_014E',
                     'B15003_014M', 'B15003_015E', 'B15003_015M', 'B15003_016E',
                     'B15003_016M']]
        
    # select columns relevant for %ppl aged 18 and under w/medicaid or no healthcare calculation    
    if file == 'B27010_18und':
        df = acs_ej[['GEO_ID', 'B27010_002E', 'B27010_002M',
                     'B27010_007E', 'B27010_007M',
                     'B27010_013E', 'B27010_013M',
                     'B27010_017E', 'B27010_017M']]
    
    # select columns relevant for %ppl aged 65+ w/medicaid or no healthcare calculation
    if file == 'B27010_65pls':
        df = acs_ej[['GEO_ID', 'B27010_051E', 'B27010_051M',
                     'B27010_062E','B27010_062M',
                     'B27010_066E','B27010_066M']] 
        
    # select columns relevant for %uninsured ppl
    if file == 'B27010_uninsured':
        df = acs_ej[['GEO_ID', 'B27010_001E', 'B27010_001M',
                     'B27010_017E','B27010_017M',
                     'B27010_033E', 'B27010_033M',
                     'B27010_050E','B27010_050M',
                     'B27010_066E', 'B27010_066M']]
    
    # for %families whose income is equal to the poverty threshold for their family size
    if file == 'C17002_und1':
        df = acs_ej[['GEO_ID', 'C17002_001E', 'C17002_001M', 'C17002_002E', 'C17002_002M', 'C17002_003E', 'C17002_003M']]

    # for %families whose income is 3/2x the poverty threshold for their family size
    if file == 'C17002_und1.5': 
        df = acs_ej[['GEO_ID', 'C17002_001E', 'C17002_001M', 'C17002_002E', 'C17002_002M', 'C17002_003E', 'C17002_003M',
                     'C17002_004E', 'C17002_004M', 'C17002_005E', 'C17002_005M']]
        
    # for %families whose income is 2x the poverty threshold for their family size
    if file == 'C17002_und2': 
        df = acs_ej[['GEO_ID', 'C17002_001E', 'C17002_001M', 'C17002_002E', 'C17002_002M', 'C17002_003E', 'C17002_003M',
                     'C17002_004E', 'C17002_004M', 'C17002_005E', 'C17002_005M',
                     'C17002_006E', 'C17002_006M', 'C17002_007E', 'C17002_007M', 'C17002_008E', 'C17002_008M']]
    
    # select columns relevant for %cost-burdened ppl (spend over 30% of income on rent)
    if file == 'B25070_30pls':
        df = acs_ej[['GEO_ID', 'B25070_001E', 'B25070_001M', 'B25070_007E',
                     'B25070_007M', 'B25070_008E', 'B25070_008M', 'B25070_009E', 'B25070_009M', 'B25070_010E', 'B25070_010M']]      
    
    # change data to float type for calculations later on
    df.iloc[:, 1:] = df.iloc[:, 1:].astype(float)
    # replace 0s with NaNs
    df.replace(0, np.nan, inplace=True)
    # drop first row (it contains column descriptions, no data)
    df = df[1:]
    # rename columns to streamline things
    df.rename(columns={df.columns[1]: 'TOT_EST',
                       df.columns[2]: 'MOE_TOT_EST'}, inplace= True)
    
    # filter for estimate columns (this doesn't include the total estimate column since we renamed it)
    ests = [col for col in df.columns if col.endswith('E')]
    # sum estimates
    df['AGG_EST'] = df[ests].sum(axis=1)
    
    # calculate proportion and pct of interest
    df['PROP'] = df['AGG_EST'] / df['TOT_EST']
    df['PCT'] = df['PROP']
    
    # filter for moe columns (this doesn't include the total estimate moe column since we renamed it)
    moes = df[[col for col in df.columns if col.endswith('M')]]
    # create function to calculate margins of error corresponding to the aggregated estimates
    # formulas on pg 61-63: https://www.census.gov/content/dam/Census/library/publications/2020/acs/acs_general_handbook_2020_ch08.pdf
    def agg_moe_calc(x):
        moe_sq = x * x
        add = np.sum(moe_sq)
        sqrtd = np.sqrt(add)
        return(sqrtd)
    # run function on aggregated estimates, row by row
    moes_final = moes.apply(agg_moe_calc, axis=1)
    df = pd.concat([df, moes_final], axis=1)
    # rename column
    df.rename(columns={df.columns[-1]: 'MOE'}, inplace = True)

    # calculate pct moe
    # formulas on pg 63-64: https://www.census.gov/content/dam/Census/library/publications/2020/acs/acs_general_handbook_2020_ch08.pdf
    df['MOE_PCT'] = (1/df['TOT_EST']) * np.sqrt(pow(df['MOE'], 2) - (pow(df['PROP'], 2) * pow(df['MOE_TOT_EST'], 2)))

    # keep only relevant columns
    df = df[['GEO_ID','PCT', 'MOE_PCT']]
    # rename columns
    df.rename(columns={df.columns[1]: str(file) + '_PCT',
                       df.columns[2]: str(file) + '_PCT_MOE'}, inplace= True)
    # set index to geoid to keep geoid unique
    df.set_index('GEO_ID', inplace = True)
    
    # append this df to list
    agg_ej_metric_list.append(df)
    
    print("Complete!")


#%%


### 12. create EJ metrics of my own - educational attainment score

# creating a complex variable here, and my previous cells aren't equipped to include this

# select columns relevant for educational attainment score
# not including "no school completed", "nursery school", or "kindergarden" towards educational attainment score
df = acs_ej[['GEO_ID', 'B15003_001E', 'B15003_001M',
             'B15003_005E', 'B15003_005M', 'B15003_006E',
             'B15003_006M', 'B15003_007E', 'B15003_007M', 'B15003_008E',
             'B15003_008M', 'B15003_009E', 'B15003_009M', 'B15003_010E',
             'B15003_010M', 'B15003_011E', 'B15003_011M', 'B15003_012E',
             'B15003_012M', 'B15003_013E', 'B15003_013M', 'B15003_014E',
             'B15003_014M', 'B15003_015E', 'B15003_015M', 'B15003_016E',
             'B15003_016M', 'B15003_017E', 'B15003_017M', 'B15003_018E',
             'B15003_018M', 'B15003_019E', 'B15003_019M', 'B15003_020E',
             'B15003_020M', 'B15003_021E', 'B15003_021M', 'B15003_022E',
             'B15003_022M', 'B15003_023E', 'B15003_023M', 'B15003_024E',
             'B15003_024M', 'B15003_025E', 'B15003_025M']]

# recategorize data as floats for calculations later
df.iloc[:, 1:] = df.iloc[:, 1:].astype(float)
# rename columns for ease (not necessary, but allows for copy-paste of formula above with minimal changes)
df.rename(columns={df.columns[1]: 'TOT_EST',
                   df.columns[2]: 'MOE_TOT_EST'}, inplace= True)

# assign each grade-level an ascending point value (ex: 1st grade = 1 pt, 12th grade = 12 pts)
# this effectively weights the population by their educational attainment
# sum these weighted values to get the total educational attainment score for each block group
df['sum_educ'] = ((df['B15003_005E'] * 1) + # 1st grade
                  (df['B15003_006E'] * 2) + # 2nd grade
                  (df['B15003_007E'] * 3) + # 3rd grade
                  (df['B15003_008E'] * 4) + # 4th grade
                  (df['B15003_009E'] * 5) + # 5th grade
                  (df['B15003_010E'] * 6) + # 6th grade
                  (df['B15003_011E'] * 7) + # 7th grade
                  (df['B15003_012E'] * 8) + # 8th grade
                  (df['B15003_013E'] * 9) + # 9th grade
                  (df['B15003_014E'] * 10) + # 10th grade
                  (df['B15003_015E'] * 11) + # 11th grade 
                  (df['B15003_016E'] * 12) + # 12th grade, no diploma
                  (df['B15003_017E'] * 13) + # 12th grade, HS diploma
                  (df['B15003_018E'] * 13) + # 12th grade, GED or alternative credential
                  (df['B15003_019E'] * 14) + # some college, less than 1 yr
                  (df['B15003_020E'] * 14) + # some college, 1+ yrs, no degree
                  (df['B15003_021E'] * 15) + # associates degree
                  (df['B15003_022E'] * 16) + # bachelors degree
                  (df['B15003_023E'] * 17) + # masters degree
                  (df['B15003_024E'] * 18) + # professional school degree
                  (df['B15003_025E'] * 19))  # doctorate degree

# replace any block groups with 0 people with educational attainment data so we don't divide by 0 and throw an error
df['TOT_EST'] = df['TOT_EST'].replace(0, np.NaN)
# calculate per-capita educational attainment score
df['PROP'] = df['sum_educ'] / df['TOT_EST']

# filter for moe columns
moes = df[[col for col in df.columns if col.endswith('M')]]
# aggregated moe function
# formulas on pg 61-63: https://www.census.gov/content/dam/Census/library/publications/2020/acs/acs_general_handbook_2020_ch08.pdf
def agg_moe_calc(x):
    moe_sq = x * x
    add = np.sum(moe_sq)
    sqrtd = np.sqrt(add)
    return(sqrtd)
moes_final = moes.apply(agg_moe_calc, axis = 1)
df = pd.concat([df, moes_final], axis=1)
df.rename(columns={df.columns[-1]: 'MOE'}, inplace = True)

# calculate ratio moe column
# formulas on pg 65: https://www.census.gov/content/dam/Census/library/publications/2020/acs/acs_general_handbook_2020_ch08.pdf
# I had to split this up because I was struggling with the NaNs, but it's the same calculation shown in the documentation
df['step1'] = pow(df['MOE'], 2) + (pow(df['PROP'], 2) * pow(df['MOE_TOT_EST'], 2))
# deal with nans
df['step1'] = df['step1'].replace([np.inf, -np.inf], np.nan).fillna(0)
df['step2'] = np.sqrt(df['step1'])
df['step3'] = (1/df['TOT_EST']) * df['step2']

df = df[['GEO_ID','PROP', 'step3']]  
df.rename(columns={df.columns[1]: 'B15003_educscore',
                   df.columns[2]: 'B15003_educscore_MOE'}, inplace= True)
df.set_index('GEO_ID', inplace = True)

agg_ej_metric_list.append(df)
 

#%%


### 9. merge all new fields into one df and attach it to the acs_ej df
pct_ej_metrics = pd.concat(list_pct_ej_metrics[:8], axis=1, sort=False).reset_index() 

agg_ej_metrics = pd.concat(agg_ej_metric_list[:10],axis=1, sort=False).reset_index()

my_metrics = pd.merge(pct_ej_metrics,agg_ej_metrics, on = "GEO_ID", how = "inner")

# delete acs data used to make my own metrics

acs_ej = acs_ej.iloc[:, :34]

# merge
acs_ej_final = pd.merge(acs_ej, my_metrics, on = "GEO_ID", how = "inner")

# make geoid_12 and state abbreviation columns
acs_ej_final['GEOID_12'] = acs_ej_final.GEO_ID.str[9:]
acs_ej_final['GEOID_12'] = acs_ej_final['GEOID_12'].astype('|S')

state2abbrev = {'Alaska': 'AK',
                'Alabama': 'AL',
                'Arkansas': 'AR',
                'California': 'CA',
                'Florida': 'FL',
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
acs_ej_final['ST_ABBREV'] = acs_ej_final['State'].map(state2abbrev)

# rename cols for ease!!!!!!
acs_ej_final.columns = ['FIPS', 'STATE', 'COUNTY', 'TRACT', 'CBG',
                        'POP', 'NUM_POC', 'PCT_POC',
                        'NUM_LINGISO', 'PCT_LINGISO',
                        'NUM_UND5', 'PCT_UND5',
                        'NUM_OV64', 'PCT_OV64',
                        'PM25', 'PMDIESL', 'O3',
                        'AIRTOXCANCER', 'AIRTOXRESPHI', 'AIRTOX',
                        'SUPERFUND', 'SUPERFUNDSCORE',
                        'HAZWST', 'HAZWSTSCORE',
                        'WWDISCHRG', 'UNDGTANKS',
                        'LEAD', 'PCT_LEAD', 'RMPSCORE',
                        'AREALAND', 'AREAWATER',
                        'Shape_Length', 'Shape_Area', 'GEOID_21',
                        'PCT_RENT', 'MOE_RENT',
                        'PCT_MOBILE', 'MOE_MOBILE',
                        'PCT_NOINT', 'MOE_NOINT',
                        'PCT_NOCOMP', 'MOE_NOCOMP',
                        'PCT_INCPLUMB', 'MOE_INCPLUMB',
                        'PCT_PUBASSIST', 'MOE_PUBASSIST',
                        'PCT_05POV', 'MOE_05POV',
                        'PCT_EXTRENTBURD', 'MOE_EXTRENTBURD',
                        'PCT_SINGPARENT', 'MOE_SINGPARENT',
                        'PCT_NONHSGRAD', 'MOE_NONHSGRAD',
                        'PCT_UND18INSUR', 'MOE_UND18INSUR',
                        'PCT_OV64INSUR', 'MOE_OV64INSUR',
                        'PCT_UNINSUR', 'MOE_UNINSUR',
                        'PCT_POV', 'MOE_POV',
                        'PCT_15POV', 'MOE_15POV',
                        'PCT_2POV', 'MOE_2POV',
                        'PCT_RENTBURD', 'MOE_RENTBURD',
                        'EDUCSCORE', 'MOE_EDUCSCORE',
                        'GEOID_12', 'ST_ABBREV'] 

# reorder
acs_ej_final = acs_ej_final[['FIPS', 'GEOID_21', 'GEOID_12',
                             'ST_ABBREV', 'STATE', 'COUNTY', 'TRACT', 'CBG',
                        'POP', 'NUM_UND5', 'PCT_UND5',
                        'NUM_OV64', 'PCT_OV64', 
                        'NUM_POC', 'PCT_POC',
                        'NUM_LINGISO', 'PCT_LINGISO',
                        'PCT_RENT', 'MOE_RENT',
                        'PCT_MOBILE', 'MOE_MOBILE',
                        'PCT_NOINT', 'MOE_NOINT',
                        'PCT_NOCOMP', 'MOE_NOCOMP',
                        'PCT_INCPLUMB', 'MOE_INCPLUMB',
                        'PCT_PUBASSIST', 'MOE_PUBASSIST',
                        'PCT_05POV', 'MOE_05POV',
                        'PCT_POV', 'MOE_POV',
                        'PCT_15POV', 'MOE_15POV',
                        'PCT_2POV', 'MOE_2POV',
                        'PCT_RENTBURD', 'MOE_RENTBURD',
                        'PCT_EXTRENTBURD', 'MOE_EXTRENTBURD',
                        'PCT_SINGPARENT', 'MOE_SINGPARENT',
                        'PCT_NONHSGRAD', 'MOE_NONHSGRAD',
                        'EDUCSCORE', 'MOE_EDUCSCORE',
                        'PCT_UND18INSUR', 'MOE_UND18INSUR',
                        'PCT_OV64INSUR', 'MOE_OV64INSUR',
                        'PCT_UNINSUR', 'MOE_UNINSUR',
                        'PM25', 'PMDIESL', 'O3',
                        'AIRTOXCANCER', 'AIRTOXRESPHI', 'AIRTOX',
                        'SUPERFUND', 'SUPERFUNDSCORE',
                        'HAZWST', 'HAZWSTSCORE',
                        'WWDISCHRG', 'UNDGTANKS',
                        'LEAD', 'PCT_LEAD', 'RMPSCORE',
                        'AREALAND', 'AREAWATER',
                        'Shape_Length', 'Shape_Area']] 

#%%

### Export dataset 
os.chdir('/Users/gracehauser/Desktop/Publication/Results')
acs_ej_final.to_csv('acs_ej_final.csv', sep=',', index=False, encoding='utf-8')


#%%
# # Extra code

# ### 10. create EJ metrics of my own - download VRE tables for percentages with aggregation required

# # make list of tables to download
# vre_list = [#'B25014', # tenure x occupants per room (for use in renter:total households ratio)
#             #'B25024', # units in structure (for mobile home categorization)
#             'B27010', # types of health insurance x age
#             #'B28002', # internet subscription
#             #'B28001' # computer access
#            ]
# # initialize empty list for ______
# vre_df_list = []
# # for B27010 only: list of which rows we want to import (file is really big)
# hc = ['Under 19 years:', 'With Medicare coverage only', 'With Medicaid/means-tested public coverage only', 'No health insurance coverage',
#       '65 years and over:', 'With Medicare coverage only', 'With Medicaid/means-tested public coverage only', 'No health insurance coverage']

# for file in vre_list:
#     print("Working on " + file + "...")
#     appended_data = []
    
#     for state_num in range(1,3):
#         if state_num <= 9:
#             filename = file + "_0" + str(state_num) + ".csv" 
#         else: 
#             filename = file + "_" + str(state_num) + ".csv"
#         url = 'https://www2.census.gov/programs-surveys/acs/replicate_estimates/2021/data/5-year/150/' + filename + '.zip'
#         r = requests.get(url)
#         try:
#             z = zipfile.ZipFile(io.BytesIO(r.content))
#             z.extractall()

#             df = pd.read_csv(filename, sep=',', encoding='latin-1' if state_num == 35 else None)
#             if file == 'B27010':
#                 df = df[df['TITLE'].isin(hc)]
#                 appended_data.append(df)
#             else:
#                 df = df.iloc[2:]
#                 appended_data.append(df)
            
#         except zipfile.BadZipFile as err:
#             continue
#         print("Successfully loaded state FIPS code " + str(state_num))
            
              
#     vre_concat = pd.concat(appended_data) 
#     vre_df_list.append(vre_concat)



# ### 11. create EJ metrics of my own - percentages with aggregation required (VRE table method)
# hlth = vre_df_list[0]

# # total pop under 19 y/o
# undr19yo = hlth[(hlth['ORDER'] == 2.0)] # all ppl under 19 y/os
# undr19yo_bgs = undr19yo[['GEOID','ESTIMATE']]

# # under 19 y/o: healthcare groups of interest
# undr19yohc = hlth[(hlth['ORDER'] == 6.0) | # 19 y/os: medicare coverage only
#                 (hlth['ORDER'] == 7.0) | # 19 y/os: medicaid/means-tested public coverage only
#                 (hlth['ORDER'] == 17.0)] # 19 y/os: uninsured

# # sum estimates by GEOID and merge to new df
# hc_df = undr19yo_bgs.merge(undr19yohc, on = 'GEOID', how = 'right')
# hc_df.head(n=5)
# # sum MOEs
# #undr19yo
# #for blockgroup in geoid:
# #    estimate = renter_ratio['ESTIMATE']
# #    for i in range(1,81):
# #        col_name = "Var_Rep" + str(i)
# #        var_rep = renter_ratio.groupby('GEOID')[col_name].sum()['Renter occupied:']
# #        sq_diff = pow((var_rep - estimate), 2)
# #        st_err = math.sqrt(sum(sq_diff) * (4/80))
# #        moe = st_err * 1.645
# #        print([blockgroup, col_name, moe])
        



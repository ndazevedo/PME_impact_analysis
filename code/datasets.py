# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 12:15:44 2023

@author: nicol
"""

import pandas as pd
import os
import numpy as np

path = r'C:\Users\nicol\OneDrive - The University of Chicago\Desktop\Thesis'
fnameMUNIC = 'DB2018MUNIC.xlsx'
# fnameCENSUS2018 = 'DB2018CENSUS.xlsx'
# fname3 = 'DBIDEB.xlsx'
# fname4 = 'DBIDEB2013ai.xlsx'
# fname5 = 'DBIDEB2013af.xlsx'
fnameCENSUS = 'DBCENSUS.xlsx'
fnameGDP = 'municipio.csv'
fnameSES = 'DBSES2013.xlsx'
fnameIDEB1 = 'divulgacao_anos_iniciais_municipios_2019.xlsx'
fnameIDEB2 = 'divulgacao_anos_finais_municipios_2019.xlsx'

df_gdp = pd.read_csv(os.path.join(path, fnameGDP))
df_munic2018 = pd.read_excel(os.path.join(path, fnameMUNIC)) ## PME DATA ##
# df_census2018 = pd.read_excel(os.path.join(path, fnameCENSUS2018))
# df_ideb = pd.read_excel(os.path.join(path, fname3) ### useless
# df_ideb_ai = pd.read_excel(os.path.join(path, fname4)) ## not using anymore, didnt have all ideb data
# df_ideb_af = pd.read_excel(os.path.join(path, fname5)) ## not using anymore, didnt have all ideb data
df_census = pd.read_excel(os.path.join(path, fnameCENSUS))
df_ses = pd.read_excel(os.path.join(path, fnameSES))
df_ideb1 = pd.read_excel(os.path.join(path, fnameIDEB1)).tail(-8) # dropping first rows (no data)
df_ideb2 = pd.read_excel(os.path.join(path, fnameIDEB2)).tail(-8) # dropping first rows (no data)

############ IDEB. IDEB1 = k1-5 dataset; IDEB2 = k6-9 DATASET
#### Cleaning Datasets
# Making first rows into columns
df_ideb1.columns = df_ideb1.iloc[0]
df_ideb1 = df_ideb1[1:].reset_index()
del df_ideb1[df_ideb1.columns[0]]



# print(df_ideb1.columns)
# print(df_ideb2.columns)

new_col_list1 = ['CO_MUNICIPIO',
                  'REDE',
                  'VL_APROVACAO_2013_SI_4', 
                  'VL_APROVACAO_2013_SI', 
                  'VL_APROVACAO_2013_1',
                  'VL_APROVACAO_2013_2', 
                  'VL_APROVACAO_2013_3', 
                  'VL_APROVACAO_2013_4',
                  'VL_INDICADOR_REND_2013', 
                  'VL_APROVACAO_2015_SI_4',
                  'VL_APROVACAO_2015_SI', 
                  'VL_APROVACAO_2015_1', 
                  'VL_APROVACAO_2015_2',
                  'VL_APROVACAO_2015_3', 
                  'VL_APROVACAO_2015_4', 
                  'VL_INDICADOR_REND_2015',
                  'VL_APROVACAO_2017_SI_4', 
                  'VL_APROVACAO_2017_SI', 
                  'VL_APROVACAO_2017_1',
                  'VL_APROVACAO_2017_2', 
                  'VL_APROVACAO_2017_3',
                  'VL_APROVACAO_2017_4',
                  'VL_INDICADOR_REND_2017', 
                  'VL_APROVACAO_2019_SI_4',
                  'VL_APROVACAO_2019_SI', 
                  'VL_APROVACAO_2019_1', 
                  'VL_APROVACAO_2019_2',
                  'VL_APROVACAO_2019_3', 
                  'VL_APROVACAO_2019_4', 
                  'VL_INDICADOR_REND_2019',
                  'VL_NOTA_MATEMATICA_2013', 
                  'VL_NOTA_PORTUGUES_2013',
                  'VL_NOTA_MEDIA_2013', 
                  'VL_NOTA_MATEMATICA_2015',
                  'VL_NOTA_PORTUGUES_2015', 
                  'VL_NOTA_MEDIA_2015',
                  'VL_NOTA_MATEMATICA_2017', 
                  'VL_NOTA_PORTUGUES_2017',
                  'VL_NOTA_MEDIA_2017', 
                  'VL_NOTA_MATEMATICA_2019',
                  'VL_NOTA_PORTUGUES_2019', 
                  'VL_NOTA_MEDIA_2019',
                  'VL_OBSERVADO_2013', 
                  'VL_OBSERVADO_2015', 
                  'VL_OBSERVADO_2017',
                  'VL_OBSERVADO_2019']

new_col_list2 = ['CO_MUNICIPIO',
                  'REDE',
                  'VL_APROVACAO_2013_SI_4', 'VL_APROVACAO_2013_1', 'VL_APROVACAO_2013_2',
                  'VL_APROVACAO_2013_3', 'VL_APROVACAO_2013_4', 'VL_INDICADOR_REND_2013',
                  'VL_APROVACAO_2015_SI_4', 'VL_APROVACAO_2015_1', 'VL_APROVACAO_2015_2',
                  'VL_APROVACAO_2015_3', 'VL_APROVACAO_2015_4', 'VL_INDICADOR_REND_2015',
                  'VL_APROVACAO_2017_SI_4', 'VL_APROVACAO_2017_1', 'VL_APROVACAO_2017_2',
                  'VL_APROVACAO_2017_3', 'VL_APROVACAO_2017_4', 'VL_INDICADOR_REND_2017',
                  'VL_APROVACAO_2019_SI_4', 'VL_APROVACAO_2019_1', 'VL_APROVACAO_2019_2',
                  'VL_APROVACAO_2019_3', 'VL_APROVACAO_2019_4', 'VL_INDICADOR_REND_2019',
                  'VL_NOTA_MATEMATICA_2013', 'VL_NOTA_PORTUGUES_2013',
                  'VL_NOTA_MEDIA_2013', 'VL_NOTA_MATEMATICA_2015',
                  'VL_NOTA_PORTUGUES_2015', 'VL_NOTA_MEDIA_2015',
                  'VL_NOTA_MATEMATICA_2017', 'VL_NOTA_PORTUGUES_2017',
                  'VL_NOTA_MEDIA_2017', 'VL_NOTA_MATEMATICA_2019',
                  'VL_NOTA_PORTUGUES_2019', 'VL_NOTA_MEDIA_2019',
                  'VL_OBSERVADO_2013', 'VL_OBSERVADO_2015', 'VL_OBSERVADO_2017',
                  'VL_OBSERVADO_2019']

df_ideb2.REDE.unique()

## Filtering for Municipal schools
df_ideb1 = df_ideb1[df_ideb1['REDE'] == 'Municipal'].dropna()
df_ideb2 = df_ideb2[df_ideb2['REDE'] == 'Municipal'].dropna()

### Creating Math dataset
math_cols = ['CO_MUNICIPIO',
             'VL_NOTA_MATEMATICA_2005',
             'VL_NOTA_MATEMATICA_2007',
             'VL_NOTA_MATEMATICA_2009',
             'VL_NOTA_MATEMATICA_2011',
             'VL_NOTA_MATEMATICA_2013',
             'VL_NOTA_MATEMATICA_2015',
             'VL_NOTA_MATEMATICA_2017',
             'VL_NOTA_MATEMATICA_2019']

df_ideb1_math = df_ideb1[math_cols]
df_ideb1_math = df_ideb1_math.dropna(subset=math_cols[1:], thresh=len(math_cols[1:]))

df_ideb1_math_melt = df_ideb1_math.melt(id_vars=['CO_MUNICIPIO'], value_vars=['VL_NOTA_MATEMATICA_2005',
                                                                              'VL_NOTA_MATEMATICA_2007',
                                                                              'VL_NOTA_MATEMATICA_2009',
                                                                              'VL_NOTA_MATEMATICA_2011', 
                                                                              'VL_NOTA_MATEMATICA_2013',
                                                                              'VL_NOTA_MATEMATICA_2015',
                                                                              'VL_NOTA_MATEMATICA_2017',
                                                                              'VL_NOTA_MATEMATICA_2019'],
                                        var_name='year', value_name='math_score')

df_ideb1_math_melt['year'] = df_ideb1_math_melt['year'].str[-4:].astype(int)
df_ideb1_math_melt = df_ideb1_math_melt.sort_values(by=['CO_MUNICIPIO', 'year'])
df_ideb1_math_melt = df_ideb1_math_melt.reset_index(drop=True)

df_ideb1_math_melt.columns = ['CODMUN', 'year', 'math_score']

df_ideb1_math_melt['math_score'] = df_ideb1_math_melt['math_score'].replace('-', np.nan).apply(lambda x: str(x).replace(',', '.'))
df_ideb1_math_melt['math_score'] = df_ideb1_math_melt['math_score'].str.replace('\*\*', '').str.replace('\*', '').replace('ND', np.nan)
df_ideb1_math_melt['math_score'] = df_ideb1_math_melt['math_score'].astype(float)

### Creating reading dataset
read_cols = [col.replace("MATEMATICA", "PORTUGUES") if "MATEMATICA" in col else col for col in math_cols]

df_ideb1_read = df_ideb1[read_cols]

df_ideb1_read = df_ideb1_read.dropna(subset=read_cols[1:], thresh=len(read_cols[1:]))

# for i in df_ideb1_read.columns:
#     print(df_ideb1_read[i].isnull().describe())

df_ideb1_read_melt = df_ideb1_read.melt(id_vars=['CO_MUNICIPIO'], value_vars=read_cols, var_name='year', value_name='reading_score')
df_ideb1_read_melt['year'] = df_ideb1_read_melt['year'].str[-4:].astype(int)
df_ideb1_read_melt = df_ideb1_read_melt.sort_values(by=['CO_MUNICIPIO', 'year'])
df_ideb1_read_melt = df_ideb1_read_melt.reset_index(drop=True)

df_ideb1_read_melt.columns = ['CODMUN', 'year', 'reading_score']

df_ideb1_read_melt['reading_score'] = df_ideb1_read_melt['reading_score'].replace('-', np.nan).apply(lambda x: str(x).replace(',', '.'))
df_ideb1_read_melt['reading_score'] = df_ideb1_read_melt['reading_score'].str.replace('\*\*', '').str.replace('\*', '').replace('ND', np.nan)
df_ideb1_read_melt['reading_score'] = df_ideb1_read_melt['reading_score'].astype(float)

## Creating approval rates dataset
approval_cols = [col.replace("MATEMATICA", "APROVACAO").replace("_NOTA", "") + '_4' if "MATEMATICA" in col else col for col in math_cols]

df_ideb1_appr = df_ideb1[approval_cols]

df_ideb1_appr = df_ideb1_appr.dropna(subset=approval_cols[1:], thresh=len(approval_cols[1:]))

df_ideb1_appr_melt = df_ideb1_appr.melt(id_vars=['CO_MUNICIPIO'], value_vars=approval_cols, var_name='year', value_name='approval_rates')

df_ideb1_appr_melt['year'] = df_ideb1_appr_melt['year'].str[-4:].astype(int)
df_ideb1_appr_melt = df_ideb1_appr_melt.sort_values(by=['CO_MUNICIPIO', 'year'])
df_ideb1_appr_melt = df_ideb1_appr_melt.reset_index(drop=True)

df_ideb1_appr_melt.columns = ['CODMUN', 'year', 'approval_rates']

df_ideb1_appr_melt['approval_rates'] = df_ideb1_appr_melt['approval_rates'].replace('-', np.nan).apply(lambda x: str(x).replace(',', '.'))
df_ideb1_appr_melt['approval_rates'] = df_ideb1_appr_melt['approval_rates'].str.replace('\*\*', '').str.replace('\*', '').replace('ND', np.nan)
df_ideb1_appr_melt['approval_rates'] = df_ideb1_appr_melt['approval_rates'].astype(float)
## Merging datasets into Y variable datasets

#### USE THIS df = df.dropna(subset=['year_2011', 'year_2013', 'year_2015', 'year_2017', 'year_2019'], thresh=5)

dfy1 = pd.merge(df_ideb1_math_melt, df_ideb1_read_melt, on=['CODMUN', 'year'], how='left')
dfy1 = pd.merge(dfy1, df_ideb1_appr_melt, on=['CODMUN', 'year'], how='left')
dfy1 = dfy1[dfy1['year'] > 2010]

### MUNIC: PME and district data
## Selecting relevant variables and renaming
munic2018 = df_munic2018[['Cod Municipio', 'MEDU01' , 'MEDU03','MEDU04', 'MEDU05', 'MEDU06', 'MEDU14', 'MEDU141B']]
munic2018.columns = ['CODMUN', 'office', 'sex', 'age', 'race',  'schooling', 'pme', 'pme_year']

### Renaming MUNIC variables
## 1 if there is a specific district office (secretaria de educação)
munic2018['office'] = munic2018['office'].apply(lambda x: 1 if x == 'Secretaria exclusiva' else 0)
## 1 if sex = female
munic2018['sex'] = munic2018['sex'].apply(lambda x: 1 if x == 'Feminino' else 'Masculino')
## 1 if race = white
munic2018['race'] = munic2018['race'].apply(lambda x: 1 if x == 'Branca' else 0)
## 1 if has pme
munic2018['pme'] = munic2018['pme'].apply(lambda x: 1 if x == 'Sim' else 0)
## years as integers
munic2018['pme_year'] = pd.to_numeric(munic2018['pme_year'], errors='coerce')
## adding treatment variable (i.e. had PME only after 2013)
munic2018['treatment'] = munic2018['pme_year'].apply(lambda x: 1 if x > 2013 else 0)

munic2018_pre = munic2018[munic2018['pme_year']>2013]
munic2018_post = munic2018[munic2018['pme_year']<=2013]

## Education Census from 2011 to 2020, filtering to years with math/reading scores
## and merging with PME and scores data
df_census = df_census[df_census['year'].isin(dfy1.year.unique())]
df = pd.merge(df_census, dfy1, on=['CODMUN', 'year'], how='inner')

## GDP DATA
df_gdp.columns = ['CODMUN', 'year', 'gdp', 'taxes', 'va1', 'va2', 'va3', 'va4', 'va5']
df_gdp = df_gdp[df_gdp['year'].isin(dfy1.year.unique())]
df_gdp = df_gdp[['CODMUN', 'year', 'gdp']]
df = pd.merge(df, df_gdp, on=['CODMUN', 'year'], how='inner')

## Merging PME data with df
df = pd.merge(df, munic2018[['CODMUN', 'office', 'pme_year', 'treatment']], on='CODMUN', how='left')

##### Removing columns from final DF
df = df[['CODMUN',
         'CO_UF',
         'year', 
         'schools',
         'teachers',
         'students',
         'office',
         'SexoFem',
         'Branco',
         'water', 
         'energy',
         'ESCompleto', 
         'Concursado',
         'Urbano',
         'math_score', 
         'reading_score', 
         'approval_rates', 
         'gdp',
         'pme_year',
         'treatment']]

df.to_csv('final_df.csv', index=False)

### Source: https://basedosdados.org/dataset/br-ibge-pib?bdm_table=municipio

## SES DATA based on 2011 and 2013 surveys, school-level

## Limiting data to district schools
df_ses = df_ses[df_ses['REDE']=='Municipal']
# Renaming
df_ses = df_ses[['COD_ESCOLA', 'COD_MUNICIPIO', 'QTD_ALUNOS_INSE', 'INSE - VALOR ABSOLUTO']]
df_ses.columns = ['school', 'CODMUN', 'students', 'ses']

# Weighting the average per municipality
df_ses['w_ses'] = df_ses['ses']*df_ses['students']
gp_df_ses = df_ses.groupby('CODMUN').agg({'students':'sum', 'w_ses':'sum'}).reset_index()
gp_df_ses['ses'] = gp_df_ses['w_ses'] / gp_df_ses['students']

df_ses_final = gp_df_ses[['CODMUN', 'ses']]

df = pd.merge(df, df_ses_final, on='CODMUN', how='inner')




########### FUND 2

df_ideb2.columns = df_ideb2.iloc[0]
df_ideb2 = df_ideb2[1:].reset_index()
del df_ideb2[df_ideb2.columns[0]]


## Filtering for Municipal schools
df_ideb2 = df_ideb2[df_ideb2['REDE'] == 'Municipal'].dropna()

### Creating Math dataset
math_cols = ['CO_MUNICIPIO',
             'VL_NOTA_MATEMATICA_2005',
             'VL_NOTA_MATEMATICA_2007',
             'VL_NOTA_MATEMATICA_2009',
             'VL_NOTA_MATEMATICA_2011',
             'VL_NOTA_MATEMATICA_2013',
             'VL_NOTA_MATEMATICA_2015',
             'VL_NOTA_MATEMATICA_2017',
             'VL_NOTA_MATEMATICA_2019']

df_ideb2_math = df_ideb2[math_cols]
df_ideb2_math = df_ideb2_math.dropna(subset=math_cols[1:], thresh=len(math_cols[1:]))

df_ideb2_math_melt = df_ideb2_math.melt(id_vars=['CO_MUNICIPIO'], value_vars=['VL_NOTA_MATEMATICA_2005',
                                                                              'VL_NOTA_MATEMATICA_2007',
                                                                              'VL_NOTA_MATEMATICA_2009',
                                                                              'VL_NOTA_MATEMATICA_2011', 
                                                                              'VL_NOTA_MATEMATICA_2013',
                                                                              'VL_NOTA_MATEMATICA_2015',
                                                                              'VL_NOTA_MATEMATICA_2017',
                                                                              'VL_NOTA_MATEMATICA_2019'],
                                        var_name='year', value_name='math_score')

df_ideb2_math_melt['year'] = df_ideb2_math_melt['year'].str[-4:].astype(int)
df_ideb2_math_melt = df_ideb2_math_melt.sort_values(by=['CO_MUNICIPIO', 'year'])
df_ideb2_math_melt = df_ideb2_math_melt.reset_index(drop=True)

df_ideb2_math_melt.columns = ['CODMUN', 'year', 'math_score']

df_ideb2_math_melt['math_score'] = df_ideb2_math_melt['math_score'].replace('-', np.nan).apply(lambda x: str(x).replace(',', '.'))
df_ideb2_math_melt['math_score'] = df_ideb2_math_melt['math_score'].str.replace('\*\*', '').str.replace('\*', '').replace('ND', np.nan)
df_ideb2_math_melt['math_score'] = df_ideb2_math_melt['math_score'].astype(float)

### Creating reading dataset
read_cols = [col.replace("MATEMATICA", "PORTUGUES") if "MATEMATICA" in col else col for col in math_cols]
df_ideb2_read = df_ideb2[read_cols]
df_ideb2_read = df_ideb2_read.dropna(subset=read_cols[1:], thresh=len(read_cols[1:]))

# for i in df_ideb2_read.columns:
#     print(df_ideb2_read[i].isnull().describe())

df_ideb2_read_melt = df_ideb2_read.melt(id_vars=['CO_MUNICIPIO'], value_vars=read_cols, var_name='year', value_name='reading_score')
df_ideb2_read_melt['year'] = df_ideb2_read_melt['year'].str[-4:].astype(int)
df_ideb2_read_melt = df_ideb2_read_melt.sort_values(by=['CO_MUNICIPIO', 'year'])
df_ideb2_read_melt = df_ideb2_read_melt.reset_index(drop=True)

df_ideb2_read_melt.columns = ['CODMUN', 'year', 'reading_score']

df_ideb2_read_melt['reading_score'] = df_ideb2_read_melt['reading_score'].replace('-', np.nan).apply(lambda x: str(x).replace(',', '.'))
df_ideb2_read_melt['reading_score'] = df_ideb2_read_melt['reading_score'].str.replace('\*\*', '').str.replace('\*', '').replace('ND', np.nan)
df_ideb2_read_melt['reading_score'] = df_ideb2_read_melt['reading_score'].astype(float)




### Creating reading dataset
read_cols = [col.replace("MATEMATICA", "PORTUGUES") if "MATEMATICA" in col else col for col in math_cols]

df_ideb2_read = df_ideb2[read_cols]

df_ideb2_read = df_ideb2_read.dropna(subset=read_cols[1:], thresh=len(read_cols[1:]))

# for i in df_ideb2_read.columns:
#     print(df_ideb2_read[i].isnull().describe())

df_ideb2_read_melt = df_ideb2_read.melt(id_vars=['CO_MUNICIPIO'], value_vars=read_cols, var_name='year', value_name='reading_score')
df_ideb2_read_melt['year'] = df_ideb2_read_melt['year'].str[-4:].astype(int)
df_ideb2_read_melt = df_ideb2_read_melt.sort_values(by=['CO_MUNICIPIO', 'year'])
df_ideb2_read_melt = df_ideb2_read_melt.reset_index(drop=True)

df_ideb2_read_melt.columns = ['CODMUN', 'year', 'reading_score']

df_ideb2_read_melt['reading_score'] = df_ideb2_read_melt['reading_score'].replace('-', np.nan).apply(lambda x: str(x).replace(',', '.'))
df_ideb2_read_melt['reading_score'] = df_ideb2_read_melt['reading_score'].str.replace('\*\*', '').str.replace('\*', '').replace('ND', np.nan)
df_ideb2_read_melt['reading_score'] = df_ideb2_read_melt['reading_score'].astype(float)

## Creating approval rates dataset
approval_cols = [col.replace("MATEMATICA", "APROVACAO").replace("_NOTA", "") + '_4' if "MATEMATICA" in col else col for col in math_cols]

df_ideb2_appr = df_ideb2[approval_cols]

df_ideb2_appr = df_ideb2_appr.dropna(subset=approval_cols[1:], thresh=len(approval_cols[1:]))

df_ideb2_appr_melt = df_ideb2_appr.melt(id_vars=['CO_MUNICIPIO'], value_vars=approval_cols, var_name='year', value_name='approval_rates')

df_ideb2_appr_melt['year'] = df_ideb2_appr_melt['year'].str[-4:].astype(int)
df_ideb2_appr_melt = df_ideb2_appr_melt.sort_values(by=['CO_MUNICIPIO', 'year'])
df_ideb2_appr_melt = df_ideb2_appr_melt.reset_index(drop=True)

df_ideb2_appr_melt.columns = ['CODMUN', 'year', 'approval_rates']

df_ideb2_appr_melt['approval_rates'] = df_ideb2_appr_melt['approval_rates'].replace('-', np.nan).apply(lambda x: str(x).replace(',', '.'))
df_ideb2_appr_melt['approval_rates'] = df_ideb2_appr_melt['approval_rates'].str.replace('\*\*', '').str.replace('\*', '').replace('ND', np.nan)
df_ideb2_appr_melt['approval_rates'] = df_ideb2_appr_melt['approval_rates'].astype(float)
## Merging datasets into Y variable datasets

dfy2 = pd.merge(df_ideb2_math_melt, df_ideb2_read_melt, on=['CODMUN', 'year'], how='left')
dfy2 = pd.merge(dfy2, df_ideb2_appr_melt, on=['CODMUN', 'year'], how='left')
dfy2 = dfy2[dfy2['year'] > 2010]


## Education Census from 2011 to 2020, filtering to years with math/reading scores
## and merging with PME and scores data
df_census = df_census[df_census['year'].isin(dfy2.year.unique())]
df = pd.merge(df_census, dfy2, on=['CODMUN', 'year'], how='inner')

## GDP DATA
df_gdp.columns = ['CODMUN', 'year', 'gdp', 'taxes', 'va1', 'va2', 'va3', 'va4', 'va5']
df_gdp = df_gdp[df_gdp['year'].isin(dfy2.year.unique())]
df_gdp = df_gdp[['CODMUN', 'year', 'gdp']]
df = pd.merge(df, df_gdp, on=['CODMUN', 'year'], how='inner')

## Merging PME data with df
df = pd.merge(df, munic2018[['CODMUN', 'office', 'pme_year', 'treatment']], on='CODMUN', how='left')

##### Removing columns from final DF
df = df[['CODMUN',
         'CO_UF',
         'year', 
         'schools',
         'teachers',
         'students',
         'office',
         'SexoFem',
         'Branco',
         'water', 
         'energy',
         'ESCompleto', 
         'Concursado',
         'Urbano',
         'math_score', 
         'reading_score', 
         'approval_rates', 
         'gdp',
         'pme_year',
         'treatment']]

df.to_csv('final_df_2.csv', index=False)














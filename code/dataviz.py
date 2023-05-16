# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 20:55:10 2023

@author: nicol
"""


import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

path = r'C:\Users\nicol\OneDrive - The University of Chicago\Desktop\Thesis'
df_file = 'final_df.csv'
df_file_2 = 'final_df_2.csv'


df = pd.read_csv(os.path.join(path, df_file))
df2 = pd.read_csv(os.path.join(path, df_file_2))
# Defining cities with PME prior to 2013. 2013 was the last year with tests
# before the law in 2014.

#### CONTROL AND TREATMENT DF
comp_cols = ['CODMUN', 
             'CO_UF', 
             'year', 
             'schools',
             'teachers', 
             'students',
             'office',
             'sex',
             'race', 
             'water', 
             'energy', 
             'teacher_col', 
             'teacher_contr', 
             'urban',
             'math_score', 
             'reading_score', 
             'approval_rates', 
             'gdp', 
             'pme_year',
             'treatment']

df.columns = comp_cols

df['gdp'] = np.log(df['gdp'])
df = df.drop('treatment', axis=1)
df['pme_year'].fillna(2025, inplace=True)
df = df[df['pme_year']>=2014]

df['treatment'] = df.apply(lambda x: 1 if x['pme_year'] < 2025 else 0, axis=1)

control_group = df[df['treatment'] == 0]['CODMUN']
treatment_group = df[df['treatment'] == 1]['CODMUN']

df_control = df[df['CODMUN'].isin(control_group)].copy()
df_treatment = df[df['CODMUN'].isin(treatment_group)].copy()


df2.columns = comp_cols

df2['gdp'] = np.log(df2['gdp'])
df2 = df2.drop('treatment', axis=1)
df2['pme_year'].fillna(2025, inplace=True)
df2 = df2[df2['pme_year']>=2014]

df2['treatment'] = df2.apply(lambda x: 1 if x['pme_year'] < 2025 else 0, axis=1)

control_group = df2[df2['treatment'] == 0]['CODMUN']
treatment_group = df2[df2['treatment'] == 1]['CODMUN']

df2_control = df2[df2['CODMUN'].isin(control_group)].copy()
df2_treatment = df2[df2['CODMUN'].isin(treatment_group)].copy()

# df_control = df_control[df_control['pme_year']>=2014]
# df_treatment = df_treatment[df_treatment['pme_year']>=2014]


sns.set_style("whitegrid")

#### PME implementation years
ax = sns.histplot(df['pme_year'], color='blue')
plt.title('PME implementation years')
plt.xlim(2008, 2020)
plt.xticks(range(2011, 2019))
plt.show
########### DEMOGRAPHICS

ax = sns.histplot(df[df['year']==2019].students, color='blue')
plt.title('Brazilian School Districts Number of Students')
plt.xlabel('Students')
plt.xlim(0, 50000)
plt.ylim(0, 1000)
plt.show

ax = sns.histplot(df[df['year']==2019].gdp, color='green')
plt.title('Brazilian Municipalities GDP')
plt.xlim(1000, 10000000000)
plt.ylim(0, 400)
plt.show

df.gdp.describe()
df.race.describe()
# plt.xticks(range(2011, 2019))

############################################################ 5TH GRADE
# Math
sns.lineplot(x="year", y="math_score", data=df_control, label='Control Math Scores')
sns.lineplot(x="year", y="math_score", data=df_treatment, label="Treatment Math Scores")
plt.legend()
plt.title('Treatment and Control 5th-grade Math scores over time')
plt.xlabel("Year")
plt.ylabel("Math Scores")
plt.xlim(2011, 2019)
plt.xticks(range(2011, 2021, 2))
plt.axvline(2014, color='red', linestyle='--')
plt.show()

# Reading
sns.lineplot(x="year", y="reading_score", data=df_control, label="Control Reading Scores")
sns.lineplot(x="year", y="reading_score", data=df_treatment, label="Treatment Reading Scores")
plt.legend()
plt.title('Treatment and Control 5th-grade Reading scores over time')
plt.xlabel("Year")
plt.ylabel("Reading Scores")
plt.xlim(2011, 2019)
plt.xticks(range(2011, 2021, 2))
plt.axvline(2014, color='red', linestyle='--')
plt.show()

# Teacher contracts
sns.lineplot(x="year", y="teacher_contr", data=df_control, label="Control Reading Scores")
sns.lineplot(x="year", y="teacher_contr", data=df_treatment, label="Treatment Reading Scores")
plt.legend()
plt.title('Treatment and Control teacher_contr over time')
plt.xlabel("Year")
plt.ylabel("teacher_contr")
plt.xlim(2011, 2019)
plt.xticks(range(2011, 2021, 2))
plt.axvline(2014, color='red', linestyle='--')
plt.show()

df.columns

######################## HETEROGENEITY TEST
### SEC EXCLUSIVA
## 5th grade
# Math w/office
sns.lineplot(x="year", y="math_score", data=df_control[df_control['office']==1], label='Control Math Scores')
sns.lineplot(x="year", y="math_score", data=df_treatment[df_treatment['office']==1], label="Treatment Math Scores")
plt.legend()
plt.title('Treatment and Control 5th-grade Math Scores - Exclusive offices')
plt.xlabel("Year")
plt.ylabel("Math Scores")
plt.xlim(2011, 2019)
plt.xticks(range(2011, 2021, 2))
plt.axvline(2014, color='red', linestyle='--')
plt.show()

# Reading w/office
sns.lineplot(x="year", y="reading_score", data=df_control[df_control['office']==1], label="Control Reading Scores")
sns.lineplot(x="year", y="reading_score", data=df_treatment[df_treatment['office']==1], label="Treatment Reading Scores")
plt.legend()
plt.title('Treatment and Control 5th-grade Reading Scores - Exclusive offices')
plt.xlabel("Year")
plt.ylabel("Reading Scores")
plt.xlim(2011, 2019)
plt.xticks(range(2011, 2021, 2))
plt.axvline(2014, color='red', linestyle='--')
plt.show()

# Math without office
sns.lineplot(x="year", y="math_score", data=df_control[df_control['office']==1], label='Control Math Scores')
sns.lineplot(x="year", y="math_score", data=df_treatment[df_treatment['office']==1], label="Treatment Math Scores")
plt.legend()
plt.title('Treatment and Control 5th-grade Math Scores - Non-Exclusive offices')
plt.xlabel("Year")
plt.ylabel("Math Scores")
plt.xlim(2011, 2019)
plt.xticks(range(2011, 2021, 2))
plt.axvline(2014, color='red', linestyle='--')
plt.show()

# Reading without office
sns.lineplot(x="year", y="reading_score", data=df_control[df_control['office']==1], label="Control Reading Scores")
sns.lineplot(x="year", y="reading_score", data=df_treatment[df_treatment['office']==1], label="Treatment Reading Scores")
plt.legend()
plt.title('Treatment and Control 5th-grade Reading Scores - Non-Exclusive offices')
plt.xlabel("Year")
plt.ylabel("Reading Scores")
plt.xlim(2011, 2019)
plt.xticks(range(2011, 2021, 2))
plt.axvline(2014, color='red', linestyle='--')
plt.show()



############################################################ 9TH GRADE
# Math w/office
sns.lineplot(x="year", y="math_score", data=df2_control[df2_control['office']==1], label='Control Math Scores')
sns.lineplot(x="year", y="math_score", data=df2_treatment[df2_treatment['office']==1], label="Treatment Math Scores")
plt.legend()
plt.title('Treatment and Control 9th-grade Math Scores - Exclusive offices')
plt.xlabel("Year")
plt.ylabel("Math Scores")
plt.xlim(2011, 2019)
plt.xticks(range(2011, 2021, 2))
plt.axvline(2014, color='red', linestyle='--')
plt.show()

# Reading w/office
sns.lineplot(x="year", y="reading_score", data=df2_control[df2_control['office']==1], label="Control Reading Scores")
sns.lineplot(x="year", y="reading_score", data=df2_treatment[df2_treatment['office']==1], label="Treatment Reading Scores")
plt.legend()
plt.title('Treatment and Control 9th-grade Reading Scores - Exclusive offices')
plt.xlabel("Year")
plt.ylabel("Reading Scores")
plt.xlim(2011, 2019)
plt.xticks(range(2011, 2021, 2))
plt.axvline(2014, color='red', linestyle='--')
plt.show()

# Math without office
sns.lineplot(x="year", y="math_score", data=df2_control[df2_control['office']==1], label='Control Math Scores')
sns.lineplot(x="year", y="math_score", data=df2_treatment[df2_treatment['office']==1], label="Treatment Math Scores")
plt.legend()
plt.title('Treatment and Control 9th-grade Math Scores - Non-Exclusive offices')
plt.xlabel("Year")
plt.ylabel("Math Scores")
plt.xlim(2011, 2019)
plt.xticks(range(2011, 2021, 2))
plt.axvline(2014, color='red', linestyle='--')
plt.show()

# Reading without office
sns.lineplot(x="year", y="reading_score", data=df2_control[df2_control['office']==1], label="Control Reading Scores")
sns.lineplot(x="year", y="reading_score", data=df2_treatment[df2_treatment['office']==1], label="Treatment Reading Scores")
plt.legend()
plt.title('Treatment and Control 9th-grade Reading Scores - Non-Exclusive offices')
plt.xlabel("Year")
plt.ylabel("Reading Scores")
plt.xlim(2011, 2019)
plt.xticks(range(2011, 2021, 2))
plt.axvline(2014, color='red', linestyle='--')
plt.show()

####### LOW ACHEVING
#### 5TH GRADE

perc_25_math = df[df['year']==2013]['math_score'].quantile(0.25)
perc_25_read = df[df['year']==2013]['reading_score'].quantile(0.25)
m_districts = df.loc[(df['year']==2013) &
                     (df['math_score']<=perc_25_math)]['CODMUN'].tolist()

r_districts = df.loc[(df['year']==2013) &
                     (df['reading_score']<=perc_25_read)]['CODMUN'].tolist()

df_low_m_control = df_control[df_control['CODMUN'].isin(m_districts)].copy()
df_low_m_treatment = df_treatment[df_treatment['CODMUN'].isin(m_districts)].copy()

df_low_r_control = df_control[df_control['CODMUN'].isin(r_districts)].copy()
df_low_r_treatment = df_treatment[df_treatment['CODMUN'].isin(r_districts)].copy()

# Math Low
sns.lineplot(x="year", y="math_score", data=df_low_m_control, label='Control Math Scores')
sns.lineplot(x="year", y="math_score", data=df_low_m_treatment, label="Treatment Math Scores")
plt.legend()
plt.title('Treatment and Control 5th-grade Math scores over time - Low Achieving Districts')
plt.xlabel("Year")
plt.ylabel("Math Scores")
plt.xlim(2011, 2019)
plt.xticks(range(2011, 2021, 2))
plt.axvline(2014, color='red', linestyle='--')
plt.show()

# Reading Low
sns.lineplot(x="year", y="reading_score", data=df_low_r_control, label="Control Reading Scores")
sns.lineplot(x="year", y="reading_score", data=df_low_r_treatment, label="Treatment Reading Scores")
plt.legend()
plt.title('Treatment and Control 5th-grade Reading scores over time - Low Achieving Districts')
plt.xlabel("Year")
plt.ylabel("Reading Scores")
plt.xlim(2011, 2019)
plt.xticks(range(2011, 2021, 2))
plt.axvline(2014, color='red', linestyle='--')
plt.show()

#### 9th grade
perc_25_math = df2[df2['year']==2013]['math_score'].quantile(0.25)
perc_25_read = df2[df2['year']==2013]['reading_score'].quantile(0.25)
m_districts = df2.loc[(df2['year']==2013) &
                     (df2['math_score']<=perc_25_math)]['CODMUN'].tolist()

r_districts = df2.loc[(df2['year']==2013) &
                     (df2['reading_score']<=perc_25_read)]['CODMUN'].tolist()

df2_low_m_control = df2_control[df2_control['CODMUN'].isin(m_districts)].copy()
df2_low_m_treatment = df2_treatment[df2_treatment['CODMUN'].isin(m_districts)].copy()

df2_low_r_control = df2_control[df2_control['CODMUN'].isin(r_districts)].copy()
df2_low_r_treatment = df2_treatment[df2_treatment['CODMUN'].isin(r_districts)].copy()

# Math Low
sns.lineplot(x="year", y="math_score", data=df2_low_m_control, label='Control Math Scores')
sns.lineplot(x="year", y="math_score", data=df2_low_m_treatment, label="Treatment Math Scores")
plt.legend()
plt.title('Treatment and Control 9th-grade Math scores over time - Low Achieving Districts')
plt.xlabel("Year")
plt.ylabel("Math Scores")
plt.xlim(2011, 2019)
plt.xticks(range(2011, 2021, 2))
plt.axvline(2014, color='red', linestyle='--')
plt.show()

# Reading Low
sns.lineplot(x="year", y="reading_score", data=df2_low_r_control, label="Control Reading Scores")
sns.lineplot(x="year", y="reading_score", data=df2_low_r_treatment, label="Treatment Reading Scores")
plt.legend()
plt.title('Treatment and Control 9th-grade Reading scores over time - Low Achieving Districts')
plt.xlabel("Year")
plt.ylabel("Reading Scores")
plt.xlim(2011, 2019)
plt.xticks(range(2011, 2021, 2))
plt.axvline(2014, color='red', linestyle='--')
plt.show()



####### HIGH ACHEVING
#### 5TH GRADE
perc_75_math = df[df['year']==2013]['math_score'].quantile(0.75)
perc_75_read = df[df['year']==2013]['reading_score'].quantile(0.75)
m_districts = df.loc[(df['year']==2013) &
                     (df['math_score']<=perc_75_math)]['CODMUN'].tolist()

r_districts = df.loc[(df['year']==2013) &
                     (df['reading_score']<=perc_75_read)]['CODMUN'].tolist()

df_high_m_control = df_control[df_control['CODMUN'].isin(m_districts)].copy()
df_high_m_treatment = df_treatment[df_treatment['CODMUN'].isin(m_districts)].copy()

df_high_r_control = df_control[df_control['CODMUN'].isin(r_districts)].copy()
df_high_r_treatment = df_treatment[df_treatment['CODMUN'].isin(r_districts)].copy()

# Math High
sns.lineplot(x="year", y="math_score", data=df_high_m_control, label='Control Math Scores')
sns.lineplot(x="year", y="math_score", data=df_high_m_treatment, label="Treatment Math Scores")
plt.legend()
plt.title('Treatment and Control 5th-grade Math scores over time - High Achieving Districts')
plt.xlabel("Year")
plt.ylabel("Math Scores")
plt.xlim(2011, 2019)
plt.xticks(range(2011, 2021, 2))
plt.axvline(2014, color='red', linestyle='--')
plt.show()

# Reading High
sns.lineplot(x="year", y="reading_score", data=df_high_r_control, label="Control Reading Scores")
sns.lineplot(x="year", y="reading_score", data=df_high_r_treatment, label="Treatment Reading Scores")
plt.legend()
plt.title('Treatment and Control 5th-grade Reading scores over time - High Achieving Districts')
plt.xlabel("Year")
plt.ylabel("Reading Scores")
plt.xlim(2011, 2019)
plt.xticks(range(2011, 2021, 2))
plt.axvline(2014, color='red', linestyle='--')
plt.show()

#### 9th grade
perc_75_math = df2[df2['year']==2013]['math_score'].quantile(0.75)
perc_75_read = df2[df2['year']==2013]['reading_score'].quantile(0.75)
m_districts = df2.loc[(df2['year']==2013) &
                     (df2['math_score']<=perc_75_math)]['CODMUN'].tolist()

r_districts = df2.loc[(df2['year']==2013) &
                     (df2['reading_score']<=perc_75_read)]['CODMUN'].tolist()

df2_high_m_control = df2_control[df2_control['CODMUN'].isin(m_districts)].copy()
df2_high_m_treatment = df2_treatment[df2_treatment['CODMUN'].isin(m_districts)].copy()

df2_high_r_control = df2_control[df2_control['CODMUN'].isin(r_districts)].copy()
df2_high_r_treatment = df2_treatment[df2_treatment['CODMUN'].isin(r_districts)].copy()

# Math High
sns.lineplot(x="year", y="math_score", data=df2_high_m_control, label='Control Math Scores')
sns.lineplot(x="year", y="math_score", data=df2_high_m_treatment, label="Treatment Math Scores")
plt.legend()
plt.title('Treatment and Control 9th-grade Math scores over time - High Achieving Districts')
plt.xlabel("Year")
plt.ylabel("Math Scores")
plt.xlim(2011, 2019)
plt.xticks(range(2011, 2021, 2))
plt.axvline(2014, color='red', linestyle='--')
plt.show()

# Reading High
sns.lineplot(x="year", y="reading_score", data=df2_high_r_control, label="Control Reading Scores")
sns.lineplot(x="year", y="reading_score", data=df2_high_r_treatment, label="Treatment Reading Scores")
plt.legend()
plt.title('Treatment and Control 9th-grade Reading scores over time - High Achieving Districts')
plt.xlabel("Year")
plt.ylabel("Reading Scores")
plt.xlim(2011, 2019)
plt.xticks(range(2011, 2021, 2))
plt.axvline(2014, color='red', linestyle='--')
plt.show()




#### BALANCE TABLE

# df_control['2013achievement'] = df_control['2013achievement'].astype(float)
# df_treatment['2013achievement'] = df_treatment['2013achievement'].astype(float)

# df_control1 = df_control[df_control['office']==1]
# df_treatment1 = df_treatment[df_treatment['office']==1]

# Calculate the t-statistic and p-value for each variable

from tabulate import tabulate
from scipy import stats


df_table = df[df.year==2013].copy()
table_cols =  comp_cols[5:-2]
table_cols.remove('sex')
table_cols.remove('approval_rates')

control_group_table = df_table[df_table['CODMUN'].isin(control_group)].copy()
treatment_group_table = df_table[df_table['CODMUN'].isin(treatment_group)].copy()

balance_table = []
for col in table_cols:
    mean_c = control_group_table[col].mean()
    mean_t = treatment_group_table[col].mean()
    dif = mean_t - mean_c
    t, p = stats.ttest_ind(control_group_table[col], treatment_group_table[col])
    balance_table.append([col, mean_c, mean_t, dif, p])

col_names = ['Variable', 'Control Average', 'Treatment Average',
             'T - C', 'p-value']

print(tabulate(balance_table, headers=col_names, tablefmt="fancy_grid"))

# , caption='Table 1 - Effect of a School Latrine on Math and Reading scores')      

### only with municipalities that have education offices
balance_table1 = []

for col in comp_cols:
    mean_c = df_control[col].mean()
    mean_t = df_treatment[col].mean()
    dif = mean_t - mean_c
    t, p = stats.ttest_ind(df_control[col], df_treatment[col])
    balance_table1.append([col, mean_c, mean_t, dif, t, p])

print(tabulate(balance_table1, headers=col_names, tablefmt="fancy_grid"))


###################################################

## Control vs Treatment over time

# df_census_control = df_census[df_census['CODMUN'].isin(control_group)]
# df_census_treatment = df_census[~df_census['CODMUN'].isin(control_group)]


# sns.lineplot(x="year", y="schools", data=df_census, label="Column1")
# sns.lineplot(x="year", y="water", data=df, label="Column2")

# sns.lineplot(x="year", y="students", data=df_census, label="Students")
# sns.lineplot(x="year", y="ESCompleto", data=df, label="Column3")
# sns.lineplot(x="year", y="students", data=df_census, label="Students")
# sns.lineplot(x="year", y="ESCompleto", data=df, label="Column3")


sns.lineplot(x="year", y="teachers", data=df_control, label="Teachers Control")
sns.lineplot(x="year", y="teachers", data=df_treatment, label="Teachers Treatment")
plt.legend()
plt.xlabel("Year")
plt.ylabel("Number of teachers")
plt.xlim(2011, 2020)
plt.xticks(range(2011, 2021))
plt.axvline(2014, color='red', linestyle='--')
plt.show()

sns.lineplot(x="year", y="students", data=df_control, label="Students Control")
sns.lineplot(x="year", y="students", data=df_treatment, label="Students Treatment")
plt.legend()
plt.xlabel("Year")
plt.ylabel("Number of students")
plt.xlim(2011, 2020)
plt.xticks(range(2011, 2021))
plt.axvline(2014, color='red', linestyle='--')
plt.show()


sns.lineplot(x="year", y="2013achievement", data=df_control, label="Control Achievement")
sns.lineplot(x="year", y="2013achievement", data=df_treatment, label="Treatment Achievement")
plt.legend()
plt.xlabel("Year")
plt.ylabel("2013achievement")
plt.xlim(2011, 2020)
plt.xticks(range(2011, 2021))
plt.axvline(2014, color='red', linestyle='--')
plt.show()

## COMPARING Y VARIABLES OVER TIME

# Math
sns.lineplot(x="year", y="math_score", data=df_control, label="Control Math Scores")
sns.lineplot(x="year", y="math_score", data=df_treatment, label="Treatment Math Scores")
plt.legend()
plt.title('Treatment and Control Math scores over time')
plt.xlabel("Year")
plt.ylabel("Math Scores")
plt.xlim(2011, 2019)
plt.xticks(range(2011, 2021, 2))
plt.axvline(2014, color='red', linestyle='--')
plt.show()

# Reading
sns.lineplot(x="year", y="reading_score", data=df_control, label="Control Reading Scores")
sns.lineplot(x="year", y="reading_score", data=df_treatment, label="Treatment Reading Scores")
plt.legend()
plt.title('Treatment and Control Reading scores over time')
plt.xlabel("Year")
plt.ylabel("Reading Scores")
plt.xlim(2011, 2019)
plt.xticks(range(2011, 2021, 2))
plt.axvline(2014, color='red', linestyle='--')
plt.show()


# Approval
sns.lineplot(x="year", y="approval_rates", data=df_control, label="Control Approval Rates")
sns.lineplot(x="year", y="approval_rates", data=df_treatment, label="Treatment Approval Rates")
plt.legend()
plt.xlabel("Year")
plt.ylabel("Approval Rates")
plt.xlim(2011, 2019)
plt.xticks(range(2011, 2021, 2))
plt.axvline(2014, color='red', linestyle='--')
plt.show()


## seeing district size distribution
df_test = df[df.year == 2019]

df_test['students'].describe()

ax = sns.histplot(df_test['students'], color='blue')
plt.title('District size')
plt.xlim(0, 50000)
# plt.xticks(range(2011, 2019))
plt.show

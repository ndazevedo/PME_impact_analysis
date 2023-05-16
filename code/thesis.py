# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 12:28:36 2023

@author: nicol
"""

import pandas as pd
import os
import statsmodels.api as sm
import numpy as np

path = r'C:\Users\nicol\OneDrive - The University of Chicago\Desktop\Thesis'
df_file = 'final_df.csv' #5th grade data
df_file_2 = 'final_df_2.csv' #9th grade data

df = pd.read_csv(os.path.join(path, df_file))
df_2 = pd.read_csv(os.path.join(path, df_file_2))

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
df = df.drop(columns=['approval_rates']).dropna()
df['treatment'] = df.apply(lambda x: 1 if x['pme_year'] < 2025 else 0, axis=1)

#### CONTROL AND TREATMENT DF

df_2.columns = comp_cols

df_2['gdp'] = np.log(df_2['gdp'])
df_2 = df_2.drop('treatment', axis=1)
df_2['pme_year'].fillna(2025, inplace=True)
df_2 = df_2[df_2['pme_year']>=2014]
df_2 = df_2.drop(columns=['approval_rates']).dropna()
df_2['treatment'] = df_2.apply(lambda x: 1 if x['pme_year'] < 2025 else 0, axis=1)


################################# DID ANALYSIS

##### 2013 and 2015 only
analysis_years_2019 = [2011, 2013, 2015, 2017, 2019]
analysis_years_2017 = [2011, 2013, 2015, 2017]
analysis_years_2015 = [2011, 2013, 2015]


def did_func(df, analysis_years, office=False, no_office=False, low_achiev=False, high_achiev=False, small=False, big=False):
    df_fix = df[df['year'].isin(analysis_years)].copy()
    df_fix['const'] = 1
    df_fix['post'] = df_fix['year'].apply(lambda x: 1 if x > 2014 else 0)
    df_fix['treat_post'] = df_fix['treatment'] * df_fix['post']
    df_fix = df_fix.set_index(['CODMUN', 'CO_UF','year'])
    exog_vars = df.columns[8:14].tolist()
    exog_vars.extend(['students' ,'gdp', 'const', 'post', 'treatment', 'treat_post'])
    
    if (office == False) & (no_office == False) & (low_achiev == False) & (high_achiev == False) & (small == False) & (big == False):
        math_model = sm.OLS(df_fix['math_score'], df_fix[exog_vars], entity_effects=True, time_effects=True)
        math_results = math_model.fit()
        reading_model = sm.OLS(df_fix['reading_score'], df_fix[exog_vars], entity_effects=True,  time_effects=True)
        reading_results = reading_model.fit()
        
        print(math_results.summary())
        print(reading_results.summary())

        # treat_post_coef_math = math_results.params['treat_post']
        # treat_post_pvalue_math = math_results.pvalues['treat_post']
        # print(f"{treat_post_coef_math:.2f}")
        # print(f"{treat_post_pvalue_math:.3f}")
        # print(f"{math_results.rsquared:.3f}")


        # treat_post_coef_read = reading_results.params['treat_post']
        # treat_post_pvalue_read = reading_results.pvalues['treat_post']
        # print(f"{treat_post_coef_read:.2f}")
        # print(f"{treat_post_pvalue_read:.3f}")
        # print(f"{reading_results.rsquared:.3f}")

        
    elif office == True:
        math_model_office = sm.OLS(df_fix[df_fix['office']==1]['math_score'], df_fix[df_fix['office']==1][exog_vars], entity_effects=True, time_effects=True)
        reading_model_office = sm.OLS(df_fix[df_fix['office']==1]['reading_score'], df_fix[df_fix['office']==1][exog_vars], entity_effects=True, time_effects=True)

        math_results_office = math_model_office.fit()
        reading_results_office = reading_model_office.fit()

        treat_post_coef_math = math_results_office.params['treat_post']
        treat_post_pvalue_math = math_results_office.pvalues['treat_post']
        print(f"{treat_post_coef_math:.2f}")
        print(f"{treat_post_pvalue_math:.3f}")
        print(f"{math_results_office.rsquared:.3f}")
        
        treat_post_coef_read = reading_results_office.params['treat_post']
        treat_post_pvalue_read = reading_results_office.pvalues['treat_post']
        print(f"{treat_post_coef_read:.2f}")
        print(f"{treat_post_pvalue_read:.3f}")
        print(f"{reading_results_office.rsquared:.3f}")
        
    elif no_office == True:
        math_model_office = sm.OLS(df_fix[df_fix['office']==0]['math_score'], df_fix[df_fix['office']==0][exog_vars], entity_effects=True, time_effects=True)
        reading_model_office = sm.OLS(df_fix[df_fix['office']==0]['reading_score'], df_fix[df_fix['office']==0][exog_vars], entity_effects=True, time_effects=True)

        math_results_office = math_model_office.fit()
        reading_results_office = reading_model_office.fit()

        treat_post_coef_math = math_results_office.params['treat_post']
        treat_post_pvalue_math = math_results_office.pvalues['treat_post']
        print(f"{treat_post_coef_math:.2f}")
        print(f"({treat_post_pvalue_math:.3f})")
        print(f"{math_results_office.rsquared:.3f}")
        
        treat_post_coef_read = reading_results_office.params['treat_post']
        treat_post_pvalue_read = reading_results_office.pvalues['treat_post']
        print(f"{treat_post_coef_read:.2f}")
        print(f"({treat_post_pvalue_read:.3f})")
        print(f"{reading_results_office.rsquared:.3f}")
        
    elif low_achiev == True:
        perc_25_math = df[df['year']==2013]['math_score'].quantile(0.25)
        perc_25_read = df[df['year']==2013]['reading_score'].quantile(0.25)
        m_districts = df.loc[(df['year']==2013) &
                             (df['math_score']<=perc_25_math)]['CODMUN'].tolist()

        r_districts = df.loc[(df['year']==2013) &
                             (df['reading_score']<=perc_25_read)]['CODMUN'].tolist()

        df_fix = df[df['year'].isin(analysis_years)].copy()
        df_fix = df_fix.drop(columns=['approval_rates']).dropna()
        df_fix['const'] = 1
        df_fix['post'] = df_fix['year'].apply(lambda x: 1 if x > 2014 else 0)
        df_fix['treat_post'] = df_fix['treatment'] * df_fix['post']
        df_low_m = df_fix[df_fix['CODMUN'].isin(m_districts)].copy()
        df_low_r = df_fix[df_fix['CODMUN'].isin(r_districts)].copy()

        df_low_m = df_low_m.set_index(['CODMUN', 'CO_UF','year'])
        df_low_r = df_low_r.set_index(['CODMUN', 'CO_UF','year'])

        math_model_low_achiev = sm.OLS(df_low_m['math_score'], df_low_m[exog_vars], entity_effects=True, time_effects=True)
        reading_model_low_achiev = sm.OLS(df_low_r['reading_score'], df_low_r[exog_vars], entity_effects=True, time_effects=True)

        math_results_low_achiev = math_model_low_achiev.fit()
        reading_results_low_achiev = reading_model_low_achiev.fit()
               
        treat_post_coef_math = math_results_low_achiev.params['treat_post']
        treat_post_pvalue_math = math_results_low_achiev.pvalues['treat_post']
        print(f"{treat_post_coef_math:.2f}")
        print(f"{treat_post_pvalue_math:.3f}")
        print(f"{math_results_low_achiev.rsquared:.3f}")
        
        treat_post_coef_read = reading_results_low_achiev.params['treat_post']
        treat_post_pvalue_read = reading_results_low_achiev.pvalues['treat_post']
        print(f"{treat_post_coef_read:.2f}")
        print(f"{treat_post_pvalue_read:.3f}")
        print(f"{reading_results_low_achiev.rsquared:.3f}")
        
    elif high_achiev == True:
        perc_75_math = df[df['year']==2013]['math_score'].quantile(0.75)
        perc_75_read = df[df['year']==2013]['reading_score'].quantile(0.75)
        m_districts_75 = df.loc[(df['year']==2013) &
                                (df['math_score']>=perc_75_math)]['CODMUN'].tolist()
         
        r_districts_75 = df.loc[(df['year']==2013) &
                                (df['math_score']>=perc_75_read)]['CODMUN'].tolist()
        
        df_fix = df[df['year'].isin(analysis_years)].copy()
        df_fix = df_fix.drop(columns=['approval_rates']).dropna()
        df_fix['const'] = 1
        df_fix['post'] = df_fix['year'].apply(lambda x: 1 if x > 2014 else 0)
        df_fix['treat_post'] = df_fix['treatment'] * df_fix['post']
        df_high_m = df_fix[df_fix['CODMUN'].isin(m_districts_75)].copy()
        df_high_r = df_fix[df_fix['CODMUN'].isin(r_districts_75)].copy()
        
        math_model_high_achiev = sm.OLS(df_high_m['math_score'], df_high_m[exog_vars], entity_effects=True, time_effects=True)
        reading_model_high_achiev = sm.OLS(df_high_r['reading_score'], df_high_r[exog_vars], entity_effects=True, time_effects=True)
        
        math_results_high_achiev = math_model_high_achiev.fit()
        reading_results_high_achiev = reading_model_high_achiev.fit()
        
        treat_post_coef_math = math_results_high_achiev.params['treat_post']
        treat_post_pvalue_math = math_results_high_achiev.pvalues['treat_post']
        print(f"Math treat_post coefficient: {treat_post_coef_math:.2f}")
        print(f"Math treat_post p-value: {treat_post_pvalue_math:.3f}")
        print(f"Math R-square statistic: {math_results_high_achiev.rsquared:.3f}")

        treat_post_coef_read = reading_results_high_achiev.params['treat_post']
        treat_post_pvalue_read = reading_results_high_achiev.pvalues['treat_post']
        print(f"{treat_post_coef_read:.2f}")
        print(f"{treat_post_pvalue_read:.3f}")
        print(f"{reading_results_high_achiev.rsquared:.3f}")
        
    elif small == True:
        perc_25_size = df[df['year']==2013]['students'].quantile(0.25)
        districts_size_25 = df.loc[(df['year']==2013) &
                                   (df['students']<=perc_25_size)]['CODMUN'].tolist()

        df_fix = df[df['year'].isin(analysis_years)].copy()
        df_fix = df_fix.drop(columns=['approval_rates']).dropna()
        df_fix['const'] = 1
        df_fix['post'] = df_fix['year'].apply(lambda x: 1 if x > 2014 else 0)
        df_fix['treat_post'] = df_fix['treatment'] * df_fix['post']
        df_small = df_fix[df_fix['CODMUN'].isin(districts_size_25)].copy()
        df_small = df_small.set_index(['CODMUN', 'CO_UF','year'])

        math_model_small = sm.OLS(df_small['math_score'], df_small[exog_vars], entity_effects=True, time_effects=True)
        reading_model_small = sm.OLS(df_small['reading_score'], df_small[exog_vars], entity_effects=True, time_effects=True)

        math_results_small = math_model_small.fit()
        reading_results_small = reading_model_small.fit()
               
        treat_post_coef_math = math_results_small.params['treat_post']
        treat_post_pvalue_math = math_results_small.pvalues['treat_post']
        print("Small Municipalities:")
        print(f"{treat_post_coef_math:.2f}")
        print(f"{treat_post_pvalue_math:.3f}")
        print(f"{math_results_small.rsquared:.3f}")
        
        treat_post_coef_read = reading_results_small.params['treat_post']
        treat_post_pvalue_read = reading_results_small.pvalues['treat_post']
        print(f"{treat_post_coef_read:.2f}")
        print(f"{treat_post_pvalue_read:.3f}")
        print(f"{reading_results_small.rsquared:.3f}")
        
    elif big == True:
        perc_75_size = df[df['year']==2013]['students'].quantile(0.75)
        districts_size_75 = df.loc[(df['year']==2013) &
                                   (df['students']>=perc_75_size)]['CODMUN'].tolist()
         
        df_fix = df[df['year'].isin(analysis_years)].copy()
        df_fix = df_fix.drop(columns=['approval_rates']).dropna()
        df_fix['const'] = 1
        df_fix['post'] = df_fix['year'].apply(lambda x: 1 if x > 2014 else 0)
        df_fix['treat_post'] = df_fix['treatment'] * df_fix['post']
        df_big = df_fix[df_fix['CODMUN'].isin(districts_size_75)].copy()
        df_big = df_big.set_index(['CODMUN', 'CO_UF','year'])

        math_model_big = sm.OLS(df_big['math_score'], df_big[exog_vars], entity_effects=True, time_effects=True)
        reading_model_big = sm.OLS(df_big['reading_score'], df_big[exog_vars], entity_effects=True, time_effects=True)

        math_results_big = math_model_big.fit()
        reading_results_big = reading_model_big.fit()
               
        treat_post_coef_math = math_results_big.params['treat_post']
        treat_post_pvalue_math = math_results_big.pvalues['treat_post']
        print("Big Municipalities:")
        print(f"{treat_post_coef_math:.2f}")
        print(f"{treat_post_pvalue_math:.3f}")
        print(f"{math_results_big.rsquared:.3f}")
        
        treat_post_coef_read = reading_results_big.params['treat_post']
        treat_post_pvalue_read = reading_results_big.pvalues['treat_post']
        print(f"{treat_post_coef_read:.2f}")
        print(f"{treat_post_pvalue_read:.3f}")
        print(f"{reading_results_big.rsquared:.3f}")
        
#### Generating regressions

#### 2015, 5th grade
did_func(df, analysis_years_2015)
did_func(df, analysis_years_2015, office=True)
did_func(df, analysis_years_2015, no_office=True)
did_func(df, analysis_years_2015, low_achiev=True)
did_func(df, analysis_years_2015, high_achiev=True)
did_func(df, analysis_years_2015, small=True)
did_func(df, analysis_years_2015, big=True)

#### 2015, 9th grade
did_func(df_2, analysis_years_2015)
did_func(df_2, analysis_years_2015, office=True)
did_func(df_2, analysis_years_2015, no_office=True)
did_func(df_2, analysis_years_2015, low_achiev=True)
did_func(df_2, analysis_years_2015, high_achiev=True)
did_func(df_2, analysis_years_2015, small=True)
did_func(df_2, analysis_years_2015, big=True)

#### 2017, 5th grade
did_func(df, analysis_years_2017)
did_func(df, analysis_years_2017, office=True)
did_func(df, analysis_years_2017, no_office=True)
did_func(df, analysis_years_2017, low_achiev=True)
did_func(df, analysis_years_2017, high_achiev=True)
did_func(df, analysis_years_2017, small=True)
did_func(df, analysis_years_2017, big=True)

#### 2017, 9th grade
did_func(df_2, analysis_years_2017)
did_func(df_2, analysis_years_2017, office=True)
did_func(df_2, analysis_years_2017, no_office=True)
did_func(df_2, analysis_years_2017, low_achiev=True)
did_func(df_2, analysis_years_2017, high_achiev=True)
did_func(df_2, analysis_years_2017, small=True)
did_func(df_2, analysis_years_2017, big=True)

#### 2019, 5th grade
did_func(df, analysis_years_2019)
did_func(df, analysis_years_2019, office=True)
did_func(df, analysis_years_2019, no_office=True)
did_func(df, analysis_years_2019, low_achiev=True)
did_func(df, analysis_years_2019, high_achiev=True)
did_func(df, analysis_years_2019, small=True)
did_func(df, analysis_years_2019, big=True)

#### 2019, 9th grade
did_func(df_2, analysis_years_2019)
did_func(df_2, analysis_years_2019, office=True)
did_func(df_2, analysis_years_2019, no_office=True)
did_func(df_2, analysis_years_2019, low_achiev=True)
did_func(df_2, analysis_years_2019, high_achiev=True)
did_func(df_2, analysis_years_2019, small=True)
did_func(df_2, analysis_years_2019, big=True)


##### ROBUSTNESS CHECK (DUFLO)
## 5th-Grade
df_r1 = df[df['year']<=2013].copy()
df_r1 = df_r1.groupby('CODMUN').mean().reset_index()

df_r2 = df[df['year']>2013].copy()
df_r2 = df_r2.groupby('CODMUN').mean().reset_index()

df_r = pd.concat([df_r1, df_r2], axis=0)
df_r = df_r.reset_index(drop=True)

df_r['const'] = 1
df_r['post'] = df_r['year'].apply(lambda x: 1 if x > 2014 else 0)
df_r['treat_post'] = df_r['treatment'] * df_r['post']
df_r = df_r.set_index(['CODMUN', 'CO_UF','year'])
exog_vars = df.columns[8:14].tolist()
exog_vars.extend(['students' ,'gdp', 'const', 'post', 'treatment', 'treat_post'])

math_model = sm.OLS(df_r['math_score'], df_r[exog_vars], entity_effects=True, time_effects=True)
math_results = math_model.fit()
reading_model = sm.OLS(df_r['reading_score'], df_r[exog_vars], entity_effects=True,  time_effects=True)
reading_results = reading_model.fit()

print(math_results.summary())
print(reading_results.summary())


## 9th-Grade
df_2_r1 = df_2[df_2['year']<=2013].copy()
df_2_r1 = df_2_r1.groupby('CODMUN').mean().reset_index()

df_2_r2 = df_2[df_2['year']>2013].copy()
df_2_r2 = df_2_r2.groupby('CODMUN').mean().reset_index()

df_2_r = pd.concat([df_2_r1, df_2_r2], axis=0)
df_2_r = df_2_r.reset_index(drop=True)

df_2_r['const'] = 1
df_2_r['post'] = df_2_r['year'].apply(lambda x: 1 if x > 2014 else 0)
df_2_r['treat_post'] = df_2_r['treatment'] * df_2_r['post']
df_2_r = df_2_r.set_index(['CODMUN', 'CO_UF','year'])
exog_vars = df_2.columns[8:14].tolist()
exog_vars.extend(['students' ,'gdp', 'const', 'post', 'treatment', 'treat_post'])

math_model2 = sm.OLS(df_2_r['math_score'], df_2_r[exog_vars], entity_effects=True, time_effects=True)
math_results2 = math_model2.fit()
reading_model2 = sm.OLS(df_2_r['reading_score'], df_2_r[exog_vars], entity_effects=True,  time_effects=True)
reading_results2 = reading_model2.fit()

print(math_results2.summary())
print(reading_results2.summary())

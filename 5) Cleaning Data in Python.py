#####CLEANING DATA IN PYTHON#####

'''Inspecting data'''
    df.head() #first 5 rows
    df.tail() #last 5 rows
    df.columns #index of column names
    df.shape #row, columns
    df.info() #returns additional info

'''Exploratory data analysis'''
    #count unique values in data
        df.column.value_counts(dropna=False)
        df['column'].value_counts(dropna=False)
    #summary statistics
        df.describe()
        #count, mean, std, min, 25%, 50% (median), 75%, max
        #Only columns with numeric dtype returned

'''Visual exploratory data analysis'''
    #Histograms
        df.column.plot('hist')
        
        import matplotlib.pyplot as plt
        df['col'].plot(kind='hist', rot=70, logx=True, logy=True) #rotate labels 70deg, log scale
        plt.show()

    #Box plots
        df.boxplot(column='population', by='continent')

    #Scatter plots
        df.plot(kind='scatter', x='initial_cost', y='total_est_fee', rot=70)

'''Tidy Data'''
    #Tidy Data - Paper by Hadley Wickham, PhD
        #https://vita.had.co.nz/papers/tidy-data.html

        #Principles of tidy data
            #Columns represent seperate variables
            #Rows represent individual observations
            #Observational units form tables

    #Melting: seperate columns -> unique rows
        pd.melt(frame=df, id_vars='name',
            value_vars=['treatment a', 'treatment b'],
            var_name = 'treatment', value_name = 'result')

    #Pivoting data: unique values -> seperate columns
        #Pivot: unique indices
        weather_tidy = weather.pivot(index='date',
             columns='element', values='value')

        #Pivot_table: aggregates multiple entries for index
        weather_tidy = weather.Pivot_table(values='value',
            index='date', columns='element', aggfunc=np.mean)

        #Get back to original df from pivoted df
        df.reset_index()

    #Splitting columns
       df['new_col'] = df.col.str[0] #Slicing string
       df['new_col'] = df.col.str.split('_') #Split by delimiter
       df['new_col'] = df.col.str.get(0) #get by index

'''Concatenating Data'''
    #Combining dataframes
        pd.concat([df1,df2]) #list of dfs to combine
        #row index labels retained
        #loc will return multiple rows
        
        pd.concat([df1,df2], ignore_index=True)
        #resets index to number sequentially

        pd.concat([df1,df2],axis=1)
        #column-wise concatenation
        #stitching together from sides

    #Glob function
        #find files based on a pattern
        #wildcards
            # * - any string e.g. *.csv
            #  ? - any character e.g. file_?.csv
        
        import glob

        csv_files = glob.glob('*.csv')

        list_data=[]
        for filename in csv_files:
                data = pd.read_csv(filename)
                list_data.append(data)
        
        pd.concat(list_data)

'''Merge Data'''
    #Combining datasets based on a common set of columns
    #Similar to joining tables in SQL

    pd.merge(left=df1, right=df2, on=None
                left_on='L_col', right_on='R_col')

    #Type of merges
        #One-to-one
        #Many-to-one/One-to-Many:
            #Duplicates values for df from duplicate keys
        #Many to Many:
            #Every pairwise combination created

'''Data Types'''
    df.dtypes #lists type of data stored in columns

    #Converting datatypes
        df['col'] = df['col'].astype(str)

        #Convert to categorical variable
            df['col'] = df['col'].astype('category')
            #Can make smaller in memory
            #Can make them be utilised by other libraries for analysis

        #Cleaning bad data
            df['col'] = pd.to_numeric(df['col'], errors'coerce')
            #converts all to numeric, invalid values to NaN

'''String Manipulation'''
    #'re' library for regular expressionsa 
        #E.g.   17      \d* decimal
        #       $17     \$\d* decimal with dollar sign
        #       $17.00  \$\d*\.\d*
        #       $17.89  \$\d*\.\d{2}
        #       $17.895 ^\$\d*\.\d{2}$

    #Testing string with regular expression
        import re

        pattern = re.compile('\$\d*\.\d{2}')
        result = pattern.match('17.89')
        bool(result)

    #Extracting numerical values from strings
        matches = re.findall('\d+', 'the recipe calls for 10 strawberries and 1 banana')
        # '+' after \d so that the previous element
        # is matched one or more times. This ensures
        # that 10 is viewed as one number and not as
        # 1 and 0.

        #Pattern Examples
        # Phone number
        pattern1 = bool(re.match(pattern='\d{3}-\d{3}-\d{4}', string='123-456-7890'))

        # Currency
        pattern2 = bool(re.match(pattern='\$\d*.\d{2}', string='$123.45'))
        #\d* : arbitrary number of digits

        # Capitalised
        pattern3 = bool(re.match(pattern='[A-Z]\w*', string='Australia'))
        #[A-Z] : capital letter
        #\w* : arbitrary number of alphanumeric characters

'''Using functions to clean data'''

    def diff_money(row, pattern):

        icost = row['Initial Cost']
        tef = row['Total est. Fee']

        if bool(pattern.match(icost)) and bool(pattern.match(tef)):
            icost = icost.replace("$","")
            tef = tef.replace("$","")

            icost = float(icost)
            tef = float(tef)

            return icost - tef
        else:
            return(NaN)

    df_subset['diff'] = df_subset.apply(diff_money,
                                        axis = 1,
                                        pattern = pattern)
    #axis=1 => row-wise

    #Using lambda functions
    tips['total_dollar_replace'] = tips.total_dollar.apply(lambda x: x.replace('$', ''))
    tips['total_dollar_re'] = tips.total_dollar.apply(lambda x: re.findall('\d+\.\d+', x)[0])

'''Duplicate and missing data'''

    #Removing duplicate values
        df.drop_duplicates()

    #Dealing with missing values

        #Inspect missing values by column
        df.info()

        #Removing rows with missing values
        df.dropna()

        #Fill missing values
        df.fillna(0)
        df.fillna('missing')
        df['column'].fillna(df['column'].mean())

'''Testing with asserts'''
    
    #Evaluates condition and raises error is not held
    assert 1==1 #returns nothing
    assert 1==2 #raises error

    #Checking for missing values
    #column
    assert df.column.notnull().all()
    #whole df
    assert pd.notnull(df)

    assert pd.notnull().all().all()
    #first returns True/False per column
    #second True/False overall

'''Cleaning Data in Python Case Study'''
    #Checking data
        def check_null_or_valid(row_data):
            """Function that takes a row of data,
            drops all missing values,
            and checks if all remaining values are greater than or equal to 0
            """
            no_na = row_data.dropna()
            numeric = pd.to_numeric(no_na)
            ge0 = numeric >= 0
            return ge0

        # Check whether the first column is 'Life expectancy'
        assert g1800s.columns[0] == 'Life expectancy'

        # Check whether the values in the row are valid
        assert g1800s.iloc[:, 1:].apply(check_null_or_valid, axis=1).all().all()

        # Check that there is only one instance of each country
        assert g1800s['Life expectancy'].value_counts()[0] == 1

    #Assembling data
        # Concatenate the DataFrames row-wise
        gapminder = pd.concat([g1800s,g1900s,g2000s])

        import pandas as pd

        # Melt gapminder: gapminder_melt
        gapminder_melt = pd.melt(gapminder,id_vars='Life expectancy')

        # Rename the columns
        gapminder_melt.columns = ['country', 'year', 'life_expectancy']

        #Save clean dataset to csv
        gapminder_melt.to_csv('gapminder.csv')

    #Checking data types
        #Load cleaned dataset
        gapminder = pd.read_csv('gapminder.csv')

        # Convert the year column to numeric
        gapminder.year = pd.to_numeric(gapminder.year)

        # Test if country is of type object
        assert gapminder.country.dtypes == np.object

        # Test if year is of type int64
        assert gapminder.year.dtypes == np.int64

        # Test if life_expectancy is of type float64
        assert gapminder.life_expectancy.dtypes == np.float64

    #Checking spellings
        # Create the series of countries: countries
        countries = gapminder.country

        # Drop all the duplicates from countries
        countries = countries.drop_duplicates()

        # Write the regular expression: pattern
        pattern = '^[A-Za-z\.\s]*$'

        # Create the Boolean vector: mask
        mask = countries.str.contains(pattern)

        # Invert the mask: mask_inverse
        mask_inverse = ~mask

        # Subset countries using mask_inverse: invalid_countries
        invalid_countries = countries.loc[mask_inverse]

        # Print invalid_countries
        print(invalid_countries)

    #Cleaning data
        # Assert that country does not contain any missing values
        assert pd.notnull(gapminder.country).all()

        # Assert that year does not contain any missing values
        assert pd.notnull(gapminder.year).all()

        # Drop the missing values
        gapminder = gapminder.dropna()

    #Visualising data
        # Print the shape of gapminder
        print(gapminder.shape)

        # Add first subplot
        plt.subplot(2, 1, 1) 

        # Create a histogram of life_expectancy
        gapminder.life_expectancy.plot(kind='hist')

        # Group gapminder: gapminder_agg
        gapminder_agg = gapminder.groupby('year')['life_expectancy'].mean()

        # Print the head of gapminder_agg
        print(gapminder_agg.head())

        # Print the tail of gapminder_agg
        print(gapminder_agg.tail())

        # Add second subplot
        plt.subplot(2, 1, 2)

        # Create a line plot of life expectancy per year
        gapminder_agg.plot()

        # Add title and specify axis labels
        plt.title('Life expectancy over the years')
        plt.ylabel('Life expectancy')
        plt.xlabel('Year')

        # Display the plots
        plt.tight_layout()
        plt.show()

        # Save both DataFrames to csv files
        gapminder.to_csv('gapminder.csv')
        gapminder_agg.to_csv('gapminder_agg.csv')
    
 
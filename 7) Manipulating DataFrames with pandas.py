#####Manipulation DataFrames with pandas#####

'''Indexing DataFrames'''

    #Indexing using square brackets
        df['col']['row']
    
    #Using column attribute and row label
        df.eggs['Mar']

    #Using accesors
        df.loc['row_name', 'col_name']
        df.iloc[row_num, col_num]

    #Selecting only some columns
        df[['salt', 'eggs']]

'''Slicing DataFrames'''
    #Selecting a column as Series
        df['col']
    
    #Slicing and indexing a Series
        df['col'][1:4]
        df['col'][4]

    #Using loc/iloc
        df.loc[:, 'eggs':'salt'] #All rows, col slice
        #Label slicing is inclusive of right endpoint

        df.loc['Jan':'Apr', :] #Row slice, All cols

        df.loc['Mar':'May', 'salt': 'spam']
        df.iloc[2:5, 1:]

        #Using lists rather than slices
        df.loc['Jan':'May', ['eggs','spam']]
        df.iloc[[0, 4, 5], 0:2]

        #Slice in reverse order
        election_df.loc['Potter' : 'Perry': -1] #Step -1

'''Filtering DataFrames'''

    #Filtering with a Boolean Series
        bool_series = df.col > 50
        df[bool_series]
    
    #Combing filters
        df[(df.col > 50) & (df.col < 200)] #And operator
        df[(df.col == 10) | (df.col == 20)] #Or operator

    #DataFrames with zeroes and NaNs
        
        #Select columns with all non-zeroes
        df.loc[:, df.all()]

        #Select columns with any non-zeroes
        df.loc[:, df.any()]

        #Select columns with any NaNs
        df.loc[:, df.isnull().any()]

        #Select columns without NaNs
        df.loc[: df.notnull().all()]

        #Drop NA values
        df.dropna(how='any') #Drop rows with any na
        df.dropna(how='all') #Drop rows with al na values
        titanic.dropna(thresh=1000, axis='columns') #Threshold for no. na to drop

    #Filtering a column based on another
    df.eggs[df.salt > 55]

    #Modifying a column based on another
    df.eggs[df.salt > 55] += 5

'''Transforming DataFrames'''

    #DataFrame vectorised methods
    df.floordiv(12) # Convert to dozens unit
    np.floor_divide(df, 12)

    #Plain Python functions
    df.apply(dozens) # Apply pre-defined dozens func
    df.apply(lambda n: n//12) # Define function on the fly

    #Working with string values
    df.index = df.index.str.upper()
    df.index.map(str.lower)










#####pandas Foundations#####

'''Building DataFrames from scratch'''

    #From dictionary
        data = {'weekday' : ['Sun', 'Sun', 'Mon', 'Mon'],
                'city' : ['Austin', 'Dallas', 'Austin', 'Dallas'],
                'visitors' : [139, 237, 326, 456],
                'signups' : [7, 12, 3, 5]}

        users = pd.DataFrame(data)

    #From lists
        cities = ['Austin', 'Dallas', 'Austin', 'Dallas']
        signups = [7, 12, 3, 5]
        visitors = [139, 237, 326, 456]
        weekdays = ['Sun', 'Sun', 'Mon', 'Mon']
        list_labels = ['city', 'singups', 'visitors', 'weekday']
        list_cols = [cities, signups, visitors, weekdays]
        
        zipped = list(zip(list_labels, list_cols))
        data = dict(zipped)
        users = pd.DataFrame(data)

    #Broadcasting
        #Allows column creation on the fly

        #Single scalar value
        users['fees'] = 0 #Broadcasts to entire column
        
        #Broadcasting with a dict
        heights = [59.0, 65.2, 62.9, 65.4]
        data = {'height' : heights, 'sex' : 'M'}
        results = pd.DataFrame(data)
        results.columns = ['height (in)', 'sex']
        results.index = ['A', 'B', 'C', 'D']
    
    #Importing and Exporting data
        col_names = ['year', 'month', 'day', 'dec_date',
                    'sunspots', 'definite']

        sunspots = pd.read_csv(filepath, header=None, names=col_names,
                                 na_values={' sunspots' : [' -1']},
                                 parse_dates[[0,1,2]])
        
        #na_values param defines values to be read as NaN
        #parse_dates param defines date values to amalgamate
        
        sunspots.index = sunspots['year_month_day']
        sunspots.index.name = 'date'
        sunsports.info()

        sunspots.to_csv('sunspots.csv')
        sunspots.to_csv('sunspots.tsv', sep='\t')
        sunspots.to_excel('sunspots.xlsx')

        #Importing clean df
        df2 = pd.read_csv(file_messy, delimiter=' ',
                             header=3, comment='#')
        #Skips first three rows
        #Defines comment values
        #Sets tab delimited

    #Plotting with pandas
        import pandas as pd
        import matplotlib.pyplot as plt
        aapl = pd.read_csv('aapl.csv', index_col='date',
                            parse_dates=True)
        
        #Method 1 - Array
            close_arr = aapl['close'].values #array of close column vals
            plt.plot(close_arr)
            plt.show()

        #Method 2 - Series with matplotlib
            close_series = aapl['close'] #pandas series from close column
            plt.plot(close_series)
            plt.show()
            #Datetimes autoformatted

        #Method 3 - Series with pandas
            close_series.plot()
            pt.show()
            #Datetimes autoformatted and axis labels

        #Method 4 - DataFrame with pandas
            aapl.plot()
            plt.show()

        #Method 5 - DataFrame matplotlib
            plt.plot(aapl)
            plt.show()

        #Fixing scales
            aapl.plot()
            plt.yscale('log')
            plt.show()

        #Customising plots
            aapl['open'].plot(color='b', style='.-',
                                legend=True)

            aapl['close'].plot(color='r', style='.',
                                legend=True)

            plt.axis(('2001', '2002', 0, 100))
            plt.title('Apple Stock')
            plt.xlabel('x_label')
            plt.ylabel('y_label')
            plt.show()

        #Plot df as subplots
            df.plot(subplots=True)
            plt.show()

        #Saving plots
            plt.savefig('aapl.png')
            plt.savefig('aapl.jpg')
            plt.savefig('aapl.pdf')

'''Visual Exploratory Data Analysis'''
    
    #Types of plot
        import pandas as pd
        import matplotlib.pyplot as plt

        iris = pd.read_csv('iris.csv', index_col=0)

        #Scatter plot
            iris.plot(x='sepal length', y='sepal width', kind='scatter')

        #Box plot
            iris.plot(x='sepal length', y='sepal width', kind='box')
            #Min, Max, IQR, Median

        #Histogram
            iris.plot(x='sepal length', y='sepal width', kind='hist')
            #bins(int) : number of bins
            #range(tuple) : extrema of bins (min,max)
            #normed(bool) : whether to normalise to 1 (e.g. plotting PDF)
            #cumulative(bool) : compute CDF (P(X<=x))

            iris.plot(x='sepal length', y='sepal width', kind='hist',
                        bins=30, range=(4,8), normed=True)
        
        #Alternative syntaxes
            iris.plot(kind='hist')
            iris.plt.hist()
            iris.hist()

'''Statistical exploratory data analysis'''
    
    #Summary statistics
        df.describe()

        #count
        df.count()
        #mean
        df.mean()
        #std
        df.std()
        #min
        df.min()
        #q1
        df.quantile(0.25)
        #median
        df.median()
        df.quantile(0.5)
        #q3
        df.quantile(0.75)
        #max
        df.max()
        #IQR - interquartile range
        q = [q1,q2]
        df.quantile(q)





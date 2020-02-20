#####IMPORTING DATA IN PYTHON#####
'''Flat Files'''

    '''Importing Plain Text'''

        #Flat file: text files containing records (table data), e.g. txt, csv...

        file = open(filename, mode='r') #open read only
        text = file.read()
        file.close()

        #Context manager method
        with open(filename, mode='r') as file: #no need to close manually
            print(file.read())

    '''Importing flat files using NumPy'''

        data = np.loadtxt(filename, delimiter=',', skiprows=,
            dtype=str, usecols=[0,2]) #handles data of dtype

        data = np.genfromtxt('titanic.csv', delimiter=',',
            names=True, dtype=None) #figures out which dtype each col should be

        data = np.recfromcsv(file) #default: delimiter=',' names=True dtype=None

    '''Importing flat files using Pandas'''

        data = pd.read_csv(file, sep='\t', na_values = 'Nothing',
        nrows=5, header=False)

        #Contains comments after the '#' character
        #Tab delimited
        #Recognises 'Nothing' string as NA value

'''Other file types'''

    '''Pickled files'''

        #Native to Python, serialized (convert object to bytestream)

        import pickle

        with open('pickled_fruit.pkl', 'rb') as file:
            data = pickle.load(file)
        print(data)

        'rb' - Read-only, binary

    '''Excel Spreadsheets'''

        import pandas as pd

        file = 'urbanpop.xlsx'

        data = pd.ExcelFile(file)
        print(data.sheet_names)
        df1 = data.parse('Sheet1')

    '''Importing SAS/Stata files'''

        import pandas as pd
        from sas7bdat import SAS7BDAT
        with SAS7BDAT('urbanpop.sas7bdat') as file:
            df_sas = file.to_data_frame() 
            
        import pandas as pd
        data = pd.read_stata('urbanpop.dta')

    '''Importing HDF5 files'''
        import h5py

        data = h5py.File(filename, 'r') #r read 
        
        Importing MATLAB files
        import scipy.io
        scipy.io.loadmat(filename)#read .mat files
        scipy.io.savemat(filename) #write .mat files

'''Relational Databases'''

    #Each column represents an attribute of each instance, analogous to a dataframe.
    #Each row has unique identifier- primary key
    #Tables are linked 

    #Common relational database management systems:
    #    PostgreSQL
    #    MySQL
    #    SQlite

    '''Creating a database engine'''

        from sqlalchemy import create_engine
        engine = create_engine('sqlite:///Northwind.sqlite') #fires up SQL engine to communicate with database
                                                            #single string arg defining type of database, name
        
        table_names = engine.table_names() #returns names of tables contained withing database

        print(table_names)

    '''Querying relational databases in Python'''

        #Workflow
            #Import packages and functions
                from sqlalchemy import create_engine
                import pandas as pd
            #Create database engine
                engine = create_engine('sqlite:///Northwind.sqlite')
            #Connect to the engine
                con = engine.connect()
            #Query the database
                rs = con.execute("SELECT * FROM Orders") #create results object
            #Save query results to a dataframe
                df = pd.DataFrame(rs.fetchall()) #fetchall gets all rows from reults object
            #Close connection
                df.columns = rs.keys() #sets headers
                con.close()

            #Context manager method
                from sqlalchemy import create_engine
                import pandas as pd

                engine = create_engine('sqlite:///Northwind.sqlite')

                with engine.connect() as con:
                        rs = con.execute("SELECT OrderID, OrderDate, ShipName FROM Orders") #Selecting specific column names
                        df = pd.DataFrame(rs.fetchmany(size = 5)) #importing 5 rows
                        df.columns  = rs.keys)()

        #SQL Querying

            #Filtering
            SELECT * FROM Customer WHERE Country = 'Canada'
            SELECT * FROM Employee WHERE EmployeeId >= 6

            #Ordering
                SELECT * FROM Customer ORDER BY SupportRepId

    '''Querying relational databases directly with pandas'''
        
        pd.read_sql_query("SELECT * FROM Orders", engine)
        #condenses query execution and df creation to one line

    '''Advanced Querying: exploiting table relationships'''

        SELECT OrderId, CompanyName FROM Orders INNER JOIN Customers on Orders.CustomerID = Customers.CustomerID
        #Selects records with matching values in both tables

        df = pd.read_sql_query('''SELECT * FROM PlaylistTrack INNER JOIN
            Track ON PlaylistTrack.TrackId = Track.TrackId
            WHERE Milliseconds < 250000''', engine)

            # Print head of DataFrame
            print(df.head())

'''Importing data from the internet'''

    '''Importing flat files from the web'''  
            
            from urllib.request import urlretrieve

            url = '''http://archive.ics.uci.edu/ml/machine-learning-
                databases/wine-quality/winequality-white.csv'''
            
            urlretrieve(url, 'winequality-white.csv') 
            #downloads file to local storage

            #Reading directly to dataframe
            df = pd.read_csv(url, sep=';')


            import pandas as pd

            url = '''http://s3.amazonaws.com/assets.datacamp.com/
                course/importing_data_into_r/latitude.xls'''

            xl = pd.read_excel(url, sheetname=None) #read all sheets of Excel file

            print(xl.keys()) #print sheetnames

            print(xl['1700'].head()) #print head of first sheet

    '''HTTP requests to import files from the web'''
        
        #Using urllib
            from urllib.request import urlopen, Request

            url = "http://www.datacamp.com/teach/documentation"
            
            request = Request(url) #packages request
            
            response = urlopen(request) #sends request, catches response
            
            html = response.read() #extracts responses as string
            
            print(html)
            
            response.close() #closes response
        
        #Using requests
            import requests

            url = "https://www.wikipedia.org/"

            r = requests.get(url) 
            #packages, sends and catches response

            text = r.text #returns html as string

    '''Scraping the web in Python'''
        from bs4 import BeautifulSoup

        import requests

        url = '''https://www.crummy.com/software/
            BeautifulSoup/'''

        r = requests.get(url)
        html_doc = r.text
        soup = BeautifulSoup(html_doc)
        soup.prettify()
        soup.title
        soup.get_text()
        soup.find_all()

        a_tags = soup.find_all('a') #find all hyperlinks (HTML tag: <a>)
        for link in a_tags:
            print(link.get('href')) #prints hyperlink url

'''Interacting with APIs to import data from the web'''

    '''Introduction to APIs and JSONs'''
        #API: Application Programming Interface
            #Set of protocols and routines for building and
            #interacting with software applications

        #JSON: JavaScript Object Notation
            #Real-time server-to-browser communication

    '''Loading JSONs'''
        import json
        
        with open('snakes.json', 'r') as json_file:
            json_data = json.load(json_file)
        #Loads JSON file as dict

        for k in json_data.keys():
            print(k + ': ', json_data[k])

    '''Connecting to an API in Python'''
        import requests

        url = 'http//www.omdbapi.com/?t=hackers'
        r = requests.get(url)
        json_data = r.json()

        for key, value in json_data.items():
            print(key + ':', value)

        #Breaking down URL:
            #http : making an HTTP request
            #www.omdbapi.com : querying OMDB API
            #?t=hackers : query string
            #             returning data with title (t) 'hackers'

    '''Twitter API'''
        import tweepy, json

        access_token = "..."
        access_token_secret = "..."
        consumer_key = "..."
        consumer_secret = "..."

        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)

#####INTERMEDIATE PYTHON#####

'''Matplotlib'''
    Import matplotlib.pyplot as plt

    plt.plot(x_list, y_list) #Line plot

    plt.scatter(x_list, y_list, s=size, c=color, #Scatter plot
        alpha=opacity) 

    plt.hist(values, bins = int) #histogram

    plt.xlabel('X_label')
    plt.title('My Title')
    plt.xscale('log')
    plt.yticks([tick value list], [tick names list])
    plt.grid(True)
    plt.text(x_pos, y_pos, text) 
    plt.cla() #Clear axes
    plt.clf() #Clear current figure
    plt.close() #Close Window
    plt.show()

'''Dictionaries'''

    my_dict = {"key1":"value1", "key2":"value2"} #defining dict

    #accessing value
	My_dict["key2"] #one value
	My_dict.keys() #all keys
	My_dict #prints all
	"key1" in my_dict #Returns True/False if exists/doesn't
	Del(my_dict["key1"])  #delete key-value pair
	My_dict['key3'] = 'value3' #Adding key-value pair
	
	#Dictionary of a dictionary
	europe = { 'spain': { 'capital':'madrid', 'population':46.77 },
	           'france': { 'capital':'paris', 'population':66.03 },
	           'germany': { 'capital':'berlin', 'population':80.62 },
	           'norway': { 'capital':'oslo', 'population':5.084 } }

    Europe['france']['capital'] #access values

'''Pandas'''

    #Importing
        Import pandas as pd

    #Reading data
        My_dataframe = pd.DataFrame(dict) #DataFrame from dictionary
        My_dataframe = pd.read_csv("path", index_col = 0) #DataFrame from csv

    #Index and Select Data

        ##Assigning indices

            #Will automatically assign indices 0,1,2…
            my_dataframe.index = index_list #manually specify index

        ##Square Bracket Select

            #Panda Series: 1D labelled array (can be pasted together to form DF)

            My_dataframe["column_label"] #Prints entire column as Panda Series
            
            My_dataframe[["column_label1", "column_label2"]]  #Prints selected columns as DataFrame

            My_dataframe[1:4] #Selecting rows as slice of DF

        ##loc function
            My_dataframe.loc["row_index"] #Selecting row as Panda Series
            My_dataframe.loc[["row_index1", "row_index2"]] #Selecting rows as DF
            My_dataframe.loc[["row1","row2"],["col1","col2"]] #Selecting rows/cols as DF
            My_dataframe.loc[:,["col1","col2"]] #All rows, selected cols

        ##iloc function
            #Same syntax as loc, but with row and columns referenced by index rather than label.
    
    #Adding column to DataFrame
        dataframe["new_column"] = dataframe["column"].apply(len)

'''Logic, Control Flow and Filtering'''

    #Boolean Operators in Numpy
        np.logical_and(array > 5, array < 10)
        np.logical_or(…)
        np.logical_not(…)

    #Filtering Pandas DataFrame
        condition = my_dataframe["column_label"] > 0
        my_dataframe[condition]
        #Pandas built on numpy, so numpy logical operators can be applied to DataFrames too.

    #Logical statements
        if condition :
            execute
        elif condition :
            execute
        else :
            execute
            
        while condition:
            execute
            
        for var in range(n):
            execute
            
    #Looping data structures
        ##Dictionary
            for key, value in dict.items():
                print(key + value)
            
        ##Array
            for val in np.nditer(my_array)
                print(val) 
            
        ##DataFrame
            for lab, row in dataframe.iterrows():
                print (lab + row["column_name")
            
'''Random Numbers'''
    np.random.seed(seed) #setting seed
    np.random.rand() #returns value

    #Generates pseudo-random number from seed.
    #Same seed will give same rand number, ensuring reproducibility for multiple trials.

    np.random.randint(0,2) #random integer
	





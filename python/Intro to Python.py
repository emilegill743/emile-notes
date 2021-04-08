##########INTRODUCTION TO PYTHON##########

'''Lists'''

    #Named collection of values
    #Can contain any data type
    #Can contain different data types   

    my_list = [1,2,3] #defining list

    #Subsetting a list
    my_list[start:end] (end exclusive)
    my_list[3]
    my_list[-2]

    #Manipulating a list
    my_list.append(value) #Add element
	my_list[2] = 3.14 #Assign value
	my_list[:3] = [0,1,2] #Assign mutiple values
    del(my_list[4]) #delete element

    #Copying lists

    ##Lists store references to memory, not values themselves 
    x = [0,1,2,3]
	y=x
	y[1] = 2
	print(x)
    Returns: [0,2,2,3] (overwriting list values in memory)

    ##To copy values to independent list
    y = list(x) or y= x[:]

'''Functions and Methods'''

    function(arg) #Calling a function
    help(function) #Function documentation

    #Methods: functions that belong to specific objects
    obj.method(arg) #Calling a method

'''Packages'''
    
    #Package: Directory of Python scripts, containing modules
    # which may be imported

    import numpy as np
	np.array([1,2,3])
	
	from numpy import array
	array([1,2,3])
	
	from numpy import array as ar
	ar([1,2,3])
	
	from numpy import *
    array([1,2,3])

'''NumPy'''
    import numpy as np
    np.array(list)

    #Numpy arrays contain only one data type
    #If different types are input they will be converted

'''Miscellaneous'''
    type() #returns data type of variable
    
    #A semi-colon may be used to place commands on a single line
        # Same line
        command1; command2
        # Separate lines
        command1
        command2








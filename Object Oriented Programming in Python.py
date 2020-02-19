#####OBJECT ORIENTED PROGRAMMING IN PYTHON#####

'''Introduction to OOP'''

    # What is OOP?
        # A way to build flexible, reproducible code
        # Developing building blocks to developing more advanced modules and libraries

    class PrintList:

        def __init__(self, numberlist):
            self.numberlist = numberlist

        def print_list(self):
            for item in self.numberlist:
                print(f"Item {item}")

    A = PrintList([1,2,3])
    A.print_list()

    '''Introduction to Objects and Classes'''

        # OOP Vocabulary
            # Imperative --> OOP
            # Variable   --> Attribute/Field
            # Function   --> Method

        # A class is a template for an object

        # Declaring a Class

            class Dinosaur:
                pass

            Tyrannosaurus = Dinosaur()

'''Deep-Dive on Classes'''

    '''Intro to Classes'''

        # Constructor - initialises class
                def __init__(self, filename):
                    self.filename = filename

        # Method
                def create_datashell(self):
                    self.array = np.genfromtxt(self.filename, delimiter=',', dtype=None)
                    return self.array

        # Attribute
            self.filename
        
        #Example Class
            class DataShell:
                
                def __init__(self, filename):
                    self.filename = filename
                
                def create_datashell(self):
                    self.array = np.genfromtxt(self.filename, delimiter=',', dtype=None)
                    return self.array

                def rename_column(self, old_colname, new_colname):
                    for index, value in enumerate(self.array[0]):
                        if value == old_colname.encode('UTF-8'):
                            self.array[0][index] = new_colname
                    return self.array

                def show_shell(self):
                    print(self.array)

                def five_figure_summary(self, col_pos):
                    statistics = stats.describe(self.array[1:, col_pos].astype(np.float))
                    return f"Five-figure stats of column {col_pos}: {statistics}" 

        # Instantiating class
            our_data_shell = DataShell('mtcars.csv')

    '''Initialising a Class and Self'''

        # Constructor/__init__ method :
            # Sets up initial object, before passing in information
            # Called automatically, when object created
            # Defines memory allocation of object

        # Self
            # Represents instance of class
            # Self not a keyword, but common practice

    '''Class and Instance Variables'''

        class Dinosaur:

            # Class variable - fixed for all class instances
            eyes = 2

            # Instance variable - defined on instantiation of object
            def __init__(self, teeth):
                self.teeth = teeth        

    '''Methods in Classes'''

        def rename_column(self, old_colname, new_colname):
            
            for index, value in enuerate(self.array[0]):
                if value == old_colname.encode('UTF-8'):
                    self.array[0][index] = new_colname

            return self.array

        myDataShell.rename_column('cyl', 'cylinders')

'''OOP Best Practices'''
    
    # PEP-8 Style guid for Python Code
        'https://www.python.org/dev/peps/pep-0008/#maximum-line-length'
    
    # Class names should be in CamelCase

    # Max 79 char per line of code

    # Docstring to explain purpose of class
        # E.g.
            class DataShell:
                """
                A simple class that brings a csv object in-memory as a 
                numpy matrix so you can perform operations on it.
                """
                def __init__(self, filename):
                    self.filename = filename

'''Inheritance'''

    # We can define a new class, inheriting properties from a base Class

    # E.g. New class, inheriting from DataShell class
        class DataStDev(DataShell):
            
            def __init__(self, filename):
                DataShell.filename = filename

            def get_stdev(self, col_position):
                column = self.array[1:, col_position].astype(np.float)
                stdev = np.ndarray.std(column, axis=0)
                return f"Standard Deviation of column {col_position} : {stdev}"

'''Composition'''

    # Inheritance - using the structure of a base class to build a new class
    # Composition - taking elements of several different classes to build a new class

    # E.g. DataShell class composed of methods from pandas Class
        class DataShellComposed:

            def __init__(self, filename):
                self.filename = filename
            
            def create_datashell(self):
                self.df = pandas.read_csv()
                return self.df

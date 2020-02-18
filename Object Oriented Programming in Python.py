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

    '''Introduction to NumPy Internals'''

        # NumPy - package for scientific computing in python
        # Uses matrices and vectors as data structures

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

'''Fancy classes, fancy objects'''
    




#####WRITING EFFICIENT PYTHON CODE#####

'''Defining Efficient'''
    # Minimal completion time (fast runtime)
    # Minimal resource consumtption (small memory footprint)

'''Defining Pythonic'''
    # Readability

    # E.g.
        # Non Pythonic
        doubled_numbers = []

        for i in range((len(numbers))):
            doubled_numbers.append(numbers[i]*2)

        # Pythonic
        doubled_numbers = [x * 2 for x in numbers]

'''Building with built-ins'''

    # Python Standard Library

        # Built in types:
            # list, tuple, set, dict

        # Built in functions:
            # print(), len(), range(), round(), enumerate(), map(), zip()

            # range()
                nums = range(start, stop) # Stop exclusive
                nums = range(stop) # Assumes 0 start
                nums = range(start, stop, step)

                # Unpack range to list
                nums_list = [*range(1, 12, 2)]

            # enumerate()
                # Creates indexed list of objects
                letters = ['a', 'b', 'c', 'd']
                indexed_letters = enumerate(letters)
                indexed_letters_list = list(indexed_letters)
                print(indexed_letters_list)

                Output: [(0, 'a'), (1, 'b'), (2, 'c'), (3, 'd')]

                # Can specify start index
                indexed_letters = enumerate(letters, start=5)
                indexed_letters_list = list(indexed_letters)
                print(indexed_letters_list)

                Output: [(5, 'a'), (6, 'b'), (7, 'c'), (8, 'd')]

                # Efficiency Example
                    # Enumerate Loop
                        indexed_names = []
                        for i,name in enumerate(names):
                            index_name = (i,name)
                            indexed_names.append(index_name) 

                    # List Comprehension
                        indexed_names_comp = [(i,name) for i,name in enumerate(names)]

                    # Unpack enumerate object
                        indexed_names_unpack = [*enumerate(names, start=1)]

            # map()
                # Applies a function over an object
                nums = [1.5, 2.3, 3.4, 4.6, 5.0]
                rnd_nums = map(round, nums)
                print(list(rnd_nums))

                Output: [2, 2, 3, 5, 5]

                # One step map and unpack
                names_uppercase  = [*map(str.upper, names)]

                # Can be used with lambda (anonymous) function
                nums = [1, 2, 3, 4, 5]
                sqrd_nums = map(lambda x : x ** 2, nums)
                print(list(sqrd_nums))

                Output: [1, 4, 9, 16, 25]             


        # Built in modules:
            # os, sys, itertools, collections, math
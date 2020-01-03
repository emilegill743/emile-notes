#####WRITING EFFICIENT PYTHON CODE#####

'''Foundations for Efficiencies'''

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

    '''NumPy Arrays'''

        # Fast and memory efficient alternative to lists
        
        import numpy as np

        nums_np = np.array(range(5))

        # Homogeneous - all elements of same type

        nums_np.dtype
        Output: dtype('int64')

        # NumPy array broadcasting - vectorised operations are performed on all elements at once

        nums_np * 2
        Output: array([0, 2, 4, 6, 8])

        # Indexing
        
        nums2 = [ [1, 2, 3],
                [4, 5, 6] ]
        nums2_np = np.array(nums2) # 2D array

        nums2_np[0,1] # Access 2nd value of 1st row
        nums2_np[:,0] # Access first column

        # Boolean Indexing

        nums = [-2, -1, 0, 1, 2]
        nums_np = np.array(nums)

        nums_np > 0
        Output: array([False, False, False, True, True])

        nums_np[nums_np > 0]
        Output: array([1, 2])

    '''Bringing it all together'''
        # Welcoming guests to party due to arrive at 10min increments, noting lateness

        # Create a list of arrival times
        arrival_times = [*range(10,60,10)]

        # Convert arrival_times to an array and update the times
        arrival_times_np = np.array(arrival_times)
        new_times = arrival_times_np - 3

        # Use list comprehension and enumerate to pair guests to new times
        guest_arrivals = [(names[i],time) for i,time in enumerate(new_times)]

        # Map the welcome_guest function to each (guest,time) pair
        welcome_map = map(welcome_guest, guest_arrivals)

        guest_welcomes = [*welcome_map]
        print(*guest_welcomes, sep='\n')

'''Timing and profiling code'''

    '''Timing Code'''

        # IPython Magic Commands
            # Enhancements on top of normal Python syntax
            
            # See all magic commands
                %lsmagic

        # Calculate runtime
            %timeit # Single line
            %%timeit # Multiple lines
            -r # Set number of runs
            -n # Set number of loops

            # E.g.
                %timeit -r2 -n10 rand_nums = np.random.rand(1000)

                %%timeit
                nums = []
                for x in range(10)
                    nums.append(x)

            #Saving output
            -o # Saving output to a variable
            times = %timeit -o rand_nums = np.random.rand(1000)
            
            times.timings # All runs
            times.best # Best run
            times.worst # Worst run

    '''Code profiling for runtime'''

        # Code profiling - Detailed stats on frequency and duration of function calls
        # Allows line-by-line analysis

        # Install
        pip install line_profiler

        # Load into Session
        %load_ext line_profiler

        # Magic command for line-by-line times
        %lprun -f func_name func_name(arg1, arg2)
        -f # Profile function

    '''Code profiling for memory usage'''

        # Basic approach
            
            sys.getsizeof() # Returns size of object in bytes

            # E.g.
                import sys
                nums_list = [*range(1000)]
                sys.getsizeof(nums_list)

        # Inspecting line-by-line memory footprint

            # Install
            pip install memory_profiler

            # Load into Session
            %load_ext memory_profiler

            # Magic command
            %mprun -f func_name func_name(arg1, arg2)

            # Function must be defined in physical file and imported

            # Memory useage defined in Mebibytes

'''Gaining Efficiencies'''

    # Efficiently combining, counting, iterating

        # Combining objects

            # Combining with loop
            names = ['Bulbasaur', 'Charmander', 'Squirtle']
            hps = [45, 39, 44]

            combined = []

            for i, pokemon in enumerate(names):
                combined.append((pokemon, hps[i]))

            print(combined)

            # Combining objects with zip
            combined_zip = zip(names, hps)

        # Counting Objects
            # Counting with loop
            poke_types = ['Grass', 'Dark', 'Fire', 'Fire', ... ]

            type_counts = []

            for poke_type in poke_types:
                if poke_type not in type_counts:
                    type_counts[poke_type] = 1
                else:
                    type_counts[poke_type] += 1

            # collections.Counter()
            from collections import Counter
            type_counts = Counter(poke_types)

        # Iterating
            # Combinations with loop
            poke_types = ['Bug', 'Fire', 'Ghost', 'Grass', 'Water']
            combos = []

            for x in poke_types:
                for y in poke_types:
                    if x==y:
                        continue
                    if ((x,y) not in combos) & ((y,x) not in combos):
                        combos.append((x,y))

            # itertools
            from itertools import combinations
            combos_obj = combinations(poke_types, 2)
            combos = [*combos_obj]

        # Set theory  

            # Branch of Mathematics applied to collections of objects
            # set - collection of distinct elements

            # Python has built in set datatype with methods:
                intersection() # all elements in both sets
                difference() # all elements in one set but not other
                symmetric_difference() # all elements in excactly one set
                union() # all elements in either set

            # Fast membership tetsing
            element in set

            # Comparing objects in lists
            list_a = ['Bulbasaur', 'Charmander', 'Squirtle']
            list_b = ['Caterpie', 'Pidgey', 'Squirtle']

            set_a = set(list_a)
            set_b = set(list_b)

            set_a.intersection(set_b)
            set_a.difference(set_b)
            set_a.symmetric_difference(set_b)
            set_a.union(set_b)

            # Uniques with set
            # Since set comprises distinct elements it can be used to find unique values in list
            primary_types = ['Grass', 'Psychic', 'Dark', 'Bug', ...]
            unique_types_set = set(primary_types)
        
        # Eliminating Loops

            # Eliminate loops with built ins
                poke_stats = [[90, 92, 75, 60],
                            [25, 20, 15, 90],
                            [65, 130, 60, 75]]

                # For loop approach
                totals = []
                for row in poke_stats:
                    totals.append(sum(row))
                
                # List comprehension
                totals_comp = [sum(row) for row in poke_stats]

                # Map function
                totals_map = [*map(sum, poke_stats)]

            # Eliminate loops with NumPy
                poke_stats = np.array([ [90, 92, 75, 60],
                                        [25, 20, 15, 90],
                                        [65, 130, 60, 75]])

                # For loop approach
                avgs = []
                for row in poke_stats:
                    avg = np.mean(row)
                    avgs.append(avg)

                # NumPy approach
                avgs_np = poke_stats.mean(axis=1)
            
        # Writing better loops
            
            # In the case that loops are unavoidable we can make them more efficient by considering:
                # Understand what is being done with each iteration
                # Move one-time calcs outside (above) the loop
                # Move datatype conversions outside (below) the loop
                    
                    #Example

                    # Converting tuple to list in loop
                    names = ['Pikachu', 'Squirtle', 'Articuno']
                    legend_status = [False, False, True]
                    generations = [1, 1, 1]

                    poke_data = []
                    for poke_tuple in zip(names, legend_status, generations):
                        poke_list = list(poke_tuple)
                        poke_data.append(poke_list)
                    
                    # Use map to convert after loop
                    poke_data_tuples
                    for poke_tuple in zip(names, legend_status, generations):
                        poke_data_tuples.append(poke_tuple)
                    
                    poke_data = [*map(list, poke_data_tuple)]

'''Basic pandas Optimisations'''

    '''pandas Iteration'''

        # Iterating with iloc
            win_perc_list = []

            for i in range(len(baseball_df))
                row = baseball_df.iloc[i]

                wins = row['W']
                games_played = row['G']
                win_perc = calc_win_perc(wins, games_played)
                win_perc_list.append(win_perc)

            baseball_df['WP'] = win_perc_list

        # Iterating with iterrows
            win_perc_list = []

            for row in baseball_df.iterrows():
                wins = row['W']
                games_played = row['G']
                win_perc = calc_win_perc(wins, games_played)
                win_perc_list.append(win_perc)

            baseball_df['WP'] = win_perc_list

        # Iterating with itertuples
            for row_namedtuple in team_wins_df.itertuples():
                print(row_namedtuple)
                
                # Column values and index accessible as attributes of named tuple
                print(row_namedtuple.Index)
                print(row_namedtuple.Team)

        # pandas alternative to looping

            # apply() method
                0 # column-wise
                1 # row-wise

                baseball_df.apply(
                    lambda row: calc_run_diff(row['RS'], row['RA']),
                    axis = 1
                )

        # Optimal pandas iterating

            # Built on NumPy
            # Can take advantage of vectorised methods

            # Get column of dataframe as NumPy array
            wins_np = baseball_df['W'].values

            # Utilise broadcasting ability of np array
            run_diffs_np = baseball_df['RS'].values - baseball_df['RA'].values
            baseball_df['RD'] = run_diffs_np















#####DATA SCIENCE TOOLBOX#####

'''User-Defined Functions'''

    #Defining function
        def square(value):
            """Return the square of a value.""" #docstring
            new_value = value ** 2
            return new_value

    #Returning multiple values

        #Tuple: Like a list but immutable (cannot modify values). Defined using parenthesis ().
            my_tuple = (2,4,6)

        #Unpacking tuple to variables
            var1, var2, var3 = my_tuple

        #Can return multiple values from function as tuple
            def raise_both(value1, value2):
                """Raise value1 to power of value 2 and vice versa"""
                
                new_value1 = value1 ** value2
                new_value2 = value2 ** value1
                new_tuple = (new_value1, new_value2)
                
                return new_tuple

            var1, var2 = new_tuple
            print(var1)
            print(var2)

    #Scope and user-defined functions

        Global scope: defined in main body of script
        Local scope: defined inside function
        Built-in-scope: names in pre-defined built-in modules

        #LEGB rule:
            local > enclosing functions > global > built-in
            
            When variable name called local scope is searched first then,
            if it cannot be found, the local scope of enclosing functions,
            the global scope and finally the built-in scope.

        Calling global() function on variable allows global value to be accessed and altered within the local scope.
        global var1

        Similarly nonlocal() function allows value from enclosing scope to be accessed and edited within nested function.
        nonlocal var1

    #Nested function

        def raise_val(n)
            """Return the inner function"""
            
            def inner(x):
            """Raise x to the power of n."""
                raised = x ** n
                return raised
                
            return inner
            
        In [1]: square = raise_val(2)
        In[2]: cube = raise_val(3)
        In[3]: print(square(2), cube(4))
        Out: 4 64

        (Closure: square function will remember n=2, despite enclosing scope having finished execution)

    #Lambda Functions

        In[1]: raise_to_power = lambda x, y: x ** y
        In[2]: raise_to_power(2,3)
        Out[2]: 8 
        Quick way to define functions as and when needed.

        #Anonymous Functions
        Lambda functions can be passed to functions without being named, 
        in this case they are referred to as anonymous functions.

        nums = [48, 6, 9, 21]
        square_all = map(lambda num: num ** 2, nums)
        print(list(square_all))

'''Error Handling'''
    raise ValueError('Error message')
    #https://docs.python.org/2/tutorial/errors.html 

'''Iterators'''

    #Examples
        In [1]: word = 'Data'
        In [2]: it = iter(word)
        In [3]: next(it)
        Out [3]: 'D'
        In [4]: next(it)
        Out[4]: 'a'

        In [1]: word = 'Data'
        In [2]: it = iter(word)
        In [3]: print(*it)
        Out [3]: D a t a

        In [1]: file = open('file.txt')
        In [2]: it = iter(file)
        In [3]: print(next(it))
        This is the first line.
        In [4]: print(next(it)
        This is the second line.

    #Functions
        enumerate()
        #function that takes iterator and returns list of tuples
        #with original iterable and their index.

            for index, value in  enumerate(iterable):
                print(index, value)
            
        zip(iterable1, iterable2, …)
        #function that takes an arbitrary number of iterators and
        #combines elements at each index to form tuples.

    #Using iterators to load large files into memory

        #We can load data in chunks if it is too large to fit in the memory
        #all at once.

        total = 0
        for chunk in pd.read_csv('data.csv', chunksize=1000):
            total += sum(chunk['x'])
        print(total) 

'''List Comprehensions'''

    #List comprehensions offer a method to condense a for loop
    #used to build a list into a single line of code.

    nums = [2,4,5,18]
    new_nums = [num + 1 for num in nums]

    #Can be applied over any iterable
        result = [num for num in range(11)]

    #Can also be used to replace nested for loops
        pairs = [(num1, num2) for num1 in range(0, 2) for num2 in range(6,8)]

    #Conditionals in comprehensions

        [output expression for iterator variable in iterable if predicate expression]

        #If condition
            fellowship = ['frodo', 'samwise', 'merry', 'aragorn', 'legolas', 'boromir', 'gimli']
            
            new_fellowship = [member for member in fellowship if len(member)>=7]
            
            print(new_fellowship)

        #If-else condition
            fellowship = ['frodo', 'samwise', 'merry', 'aragorn', 'legolas', 'boromir', 'gimli']

            new_fellowship = [member if len(member)>=7 else '' for member in fellowship]

            print(new_fellowship)

        #Dictionary comprehension

            dict_variable = {key:value for (key,value) in dictonary.items()}

            new_fellowship = {member:len(member) for member in fellowship}

'''Generators'''

    #Like a list comprehension but does not store list in memory,
    #instead it is an object which produces the list elements as required.
    
    #Syntax
        #Uses () instead of [].
        result = (num for num in range(6))

    #Print elements
        for num in result:
            print(num)

    #Generate list
        list(result)
        print(list)

    #Iterate through elements
        print(next(results))
        #Useful when we want to iterate through a large sequence
        #without having to store the whole thing in our memory

    #Generator functions

        def num_sequence(n)
            """Generate values from 0 to n."""
            i=0
            while i<n:
                yield i
                i += 1

'''Miscellaneous'''

    filter() function
    #Filters elements that don't satisfy certain criteria

    fellowship = ['frodo', 'samwise', 'merry', 'pippin', 'aragorn', 'boromir', 'legolas', 'gimli', 'gandalf']

    result = filter(lambda member: len(member) > 6, fellowship)

    result_list = list(result)



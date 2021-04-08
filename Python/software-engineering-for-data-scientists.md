# Software Engineering for Data Scientists

- [Software Engineering for Data Scientists](#software-engineering-for-data-scientists)
  - [Introduction to Software Engineering and Data Science](#introduction-to-software-engineering-and-data-science)
    - [Software Engineering Concepts](#software-engineering-concepts)
    - [Introduction to Packages and Documentation](#introduction-to-packages-and-documentation)
    - [Conventions and PEP 8](#conventions-and-pep-8)
  - [Writing a Python Module](#writing-a-python-module)
    - [Writing your first package](#writing-your-first-package)
    - [Importing a local package](#importing-a-local-package)
    - [Adding functionality to packages](#adding-functionality-to-packages)
    - [Importing functionality with `__init__.py`](#importing-functionality-with-__init__py)
    - [Extending package structure](#extending-package-structure)
    - [Making your package portable](#making-your-package-portable)
  - [Utilising Classes](#utilising-classes)
    - [Adding Classes to a Package](#adding-classes-to-a-package)
    - [Adding Functionality to Classes](#adding-functionality-to-classes)
    - [Classes and the DRY principle](#classes-and-the-dry-principle)
    - [Multilevel inheritance](#multilevel-inheritance)
  - [Maintainability](#maintainability)
    - [Documentation](#documentation)
    - [Readability](#readability)
    - [Unit Testing](#unit-testing)
      - [**doctest**](#doctest)
      - [**pytest**](#pytest)
    - [Documentation & testing in practice](#documentation--testing-in-practice)
      - [**Documenting projects with Sphinx**](#documenting-projects-with-sphinx)
      - [**Continuous integration testing**](#continuous-integration-testing)
      - [Other tools](#other-tools)

## Introduction to Software Engineering and Data Science

### Software Engineering Concepts

- Modularity
  
  > Code divided into short functional units: packages, classes and methods
  - Improves readability
  - Improves maintainability
  - Resusable
  - 
- Documentation
  > Comments, docstrings, self-documenting code.
  - Show users how to use your project
  - Prevent confusion from collaborators
  - Prevent frustration from future self
  - 
- Testing
  - Saves time over manual testing
  - Find and fix more bugs
  - Run tests anytime/anywhere

### Introduction to Packages and Documentation

**Installing packages**:

We can install packages from **Python Package Index** (PyPi) using **pip**.

`pip install numpy`

pip will install the required package, as well as all of its dependecies (as long as they are available in PyPi).

**Accessing documentation**:

```python
help(numpy.busday_count)
```

### Conventions and PEP 8

> PEP 8 - De facto style guide for Python code

https://www.python.org/dev/peps/pep-0008/

There are several ways we can ensure that PEP 8 rules are being enforced. One of these methods is to us the `pycodestyle` package. Another is to use an IDE which included code linting.

```bash
# Installing pycodestyle
pip install pycodestyle

# pycodestyle CLI
pycodestyle dict_to_array.py
```

`pycodestyle` will output a file, line number, column number, error code and error description for each error detected.

As well as its command line interface, we can also make use of the pycodestyle Python package within a Python script:

```python
# Import needed package
import pycodestyle

# Create a StyleGuide instance
style_checker = pycodestyle.StyleGuide()

# Run PEP 8 check on multiple files
result = style_checker.check_files(['nay_pep8.py', 'yay_pep8.py'])

# Print result of PEP 8 style check
print(result.messages)
```

## Writing a Python Module

### Writing your first package

Minimum package structure:
```
package_name
|-- __init_.py
```
`__init__.py` lets Python know that a directory should be treated as a package.

### Importing a local package

```
work_dir
|-- my_script.py   
|-- package_name
    |-- __init__.py
```

Importing local package from `my_script.py`
```python
import my_package
help(my_package)
```

### Adding functionality to packages

```
work_dir
|-- my_script.py
|-- my_package
    |-- __init__.py
    |-- utils.py
```

```python
#### work_dir/my_package/utils.py ###

def we_need_to_talk(break_up=False):
    """Helper for communicating with significant other"""
    if break_up:
        print("It's not you it's me...")
    else:
        print("I <3 U!")
```

```python
### work_dir/my_script.py ###

import my_package.utils

my_package.utils.we_need_to_talk(break_up=True)
```

### Importing functionality with `__init__.py`

We can make the functions in `utils.py` more easily accessible to the user by importing them in the `__init__.py` file.

```python
### work_dir/my_package/__init__.py ###

from .utils import we_need_to_talk
```

```python
### work_dir/my_script.py ###
import my_package

my_package.we_need_to_talk(break_up=False)
```

### Extending package structure

Package structure can be extended indefinitely. However for larger packages we must be mindful of organisation.

As a general rule, only the key functionality of a module should be imported in `__init__.py` to make it directly and easily accessible. Less central functionality should be accessed through the longer `module.sub_module` structure.

In addition to adding submodules within a package, we can also add sub-packages to a package, by including subdirectories which follow the same package conventions (i.e. containing a `__init__.py` file).

```
work_dir
|-- my_script.py
|-- my_package
    |-- __init__.py
    |-- utils.py
    |-- sub_package
        |-- __init__.py
        |-- sub_utils.py
```

### Making your package portable

Including a `setup.py` and `requirements.txt` provides the information required to install your package and recreate its required environment.

```
work_dir
|-- setup.py
|-- requirements.txt
|-- my_package
    |-- __init__.py
    |-- utils.py
```

The `requirements.txt` describes the dependencies of our package.

`work_dir/requirements.txt`
```
# Needed packages/versions
matplotlib
numpy==1.15.4
pycodestyle>=2.4.0
```

- Exact version of package specified by `==`
- Minimum version specified by `>=`

With a `requirements.txt` file in place we can install all the required dependencies for a package using:

```bash
pip install -r requirements.txt
```

The `setup.py` file describes how to install our package.

A simple and common method for defining the `setup.py` file is to use the `setuptools` package:
```python
from setuptools import setup

setup(name='my_package',
      version='0.0.1',
      description='An example package',
      author='Emile Gill',
      author_email='emilegill743@hotmail.com',
      packages=['my_package'],
      install_requires=['matplotlib',
                        'numpy==1.15.4',
                        'pycodestyle>=2.4.0'])
```

Once we have defined a `setup.py` we can install our packed from `pip` from inside the directory of our package using:

```bash
pip install .
```

This will install our package at an environment level, so that we can import it into any python script using the same environment.


## Utilising Classes

### Adding Classes to a Package

Anatomy of a class:

- Class name should be written in CamelCase
- Docstring explaining utilisation of class
- `__init__` method to instantiate class

```python
### work_dir/my_package/my_class.py ###

# Define a minimal class with an attribute
class MyClass:
  """A minimal example class

  :param value: value to set as the ``attribute`` attribute
  :ivar attribute: contains the contents of ``value`` passed in init
  """

  # Method to create a new instance of MyClass
  def __init__(self, value):
    # Define attribute with the contents of the value param
    self.attribute = value
```

Using a class in a package:

- Adding class to `__init__`, for easy access.
```python
### work_dir/my_package/__init__.py ###

from .my_class import MyClass
```

- Creating instance of class
```python
### work_dir/my_script.py ###

import my_package

# Create instance of MyClass
my_instance = my_package.MyClass(value='class attribute value')

# Print out class attribute value
print(my_instance.attribute)
```

### Adding Functionality to Classes

We can call a method on instantiation of the object by including it in the `__init__` function. Here we tokenize the text in our Document object as soon as the object is created.

```python
from .token_utils import tokenize

class Document:
    def __init__(self, text, token_regex=r'[a-zA-z]+'):
        self.text = text
        self.tokens = self._tokenize()
      
    def _tokenize(self):
        return tokenize(self.text)
```

Since there is no need for us to use the `_tokenize` function we have defined other than on instantiation, we define it as a **non-public method**. The leading underscore before the function name is a PEP-8 convention which indicates to the end user that the function is not intended for public usage; although the user still may call this function at their own risk.

### Classes and the DRY principle

DRY principle: **Don't Repeat Yourself**

To avoid rewriting code or copy-pasting code from another script, we can make use of **inheritance** in Python to extend a parent class with additional attributes and methods.

```python
# Import ParentClass object
from .parent_class import ParentClass

# Create a child class with inheritance
class ChildClass(ParentClass):
    def __init__(self):
        # Call parent's __init__ method
        ParentClass.__init__(self)

        # Add attribute unique to child class
        self.child_attribute = "I'm a child class attribute!"

  # Create a ChildClass instance
  child_class = ChildClass()
  print(child_class.child_attribute)
  print(child_class.parent_attribute)
  ```

  ### Multilevel inheritance

  A grandchild class may inherit from a child class, which itself inherits from a parent class. There is no limitation on the levels of inheritance and, in fact, many child classes may inherit from a single parent. It is also possible for a child class to inherit from two parents (multiinheritance), although this is not covered here.

  For multiple levels of class inheritance, to save us from having to call the `__init__` method of each level explicity, we may use the `super()` method. This implicitly deals with calling `__init__()` on all levels of inheritance.

  ```python
  class Parent:
      def __init__(self):
          print("I'm a parent!")
  
  class SuperChild(Parent):
      def __init__(self):
          super().__init__()
          print("I'm a super child!")

  class GrandChild(SuperChild):
      def __init__(self):
          super().__init__()
          print("I'm a grandchild!")
```

To keep track of inherited attributes we can use `help(obj)` or `dir(obj)`.

## Maintainability

### Documentation

**Comments**

- Comments are used to document what a particular line of code is doing and why.

- Comments are not visible to end users unless they look directly at the source code.

```python
# This is a valid comment
x = 2

y = 3 # This is also a valid comment 
```

Effective commenting should not repeat what is clearly implied by the code and should focus on the 'why' rather than the 'what' of the code.

e.g.
```python
# Define people as 5
people = 5
```
vs
```python
# There will be 5 people attending the party
people = 5
```

**Docstrings**

- Documentation for end users, accessible using `help()`



```python
def function(x):
    """High level description of function

    Additional details of function

    :param x: description of parameter x
    :return: description of return value

    >>> # Example function usage
    Expected output of example function usage
    """

    # function code
```

e.g.

```python

def square(x):
    """Square the number x

    :param x: number to square
    :return: x squared

    >>> square(2)
    4
    """

    # `x * x` is faster than x**2
    # reference: https://stackoverflow.com/a/29055266/5731525
    return x * x
    """
```

### Readability

**The Zen of Python**

```python
import this
```
> **The Zen of Python, by Tim Peters (abridged)**
> 
> Beautiful is better than ugly.
> 
> Explicit is better than implicit.
> 
> Simple is better than complex.
> 
> The complex is better than complicated.
> 
> Readability counts.
> 
> If the implementation is hard to explain, it's a bad idea.
> 
> If the implementation is easy to explain, it may be a good idea.



**Descriptive Naming**

Code which is self-descriptive (self-documenting code) is always preferable to the same code defined in a vague manner.

```python
# Poor naming
def check(x, y=100):
    return x >= y

# Descriptive naming
def is_boiling(temp, boiling_point=100):
    return temp >= boiling_point
```

**Simplicity**

> **The Zen of Python, by Tim Peters (abridged)**
>
> Simple is better than complex.
> 
> Complex is better than complicated.

Functions should aim to have do only one thing and if comments are required to break up sections, then it is probable that the code should be refactored in order to simplify it.


e.g.
```python
# Complex function
def make_pizza(ingredients):
    # Make dough
    dough = mix(ingredients['yeast'],
                ingredients['flour'],
                ingredients['water'],
                ingredients['salt'],
                ingredients['shortening'])

    kneaded_dough = knead(dough)
    risen_dough = prove(kneaded_dough)

    # Make sauce
    sauce_base = sautee(ingredients['onion'],
                                ingredients['garlic'],
                                ingredients['olive oil'])

    sauce_mixture = combine(sauce_base,
                            ingredients['tomato_paste'],
                            ingredients['water'],
                            ingredients['spices'])

    sauce = simmer(sauce_mixture)
    ...
```

```python
# Refactored function
def make_pizza(ingredients):

    dough = make_dough(ingredients)
    sauce = make_sauce(ingredients)
    assembled_pizza = assemble_pizza(dough, sauce, ingredients)

    return bake(assembled_pizza)
```

- Code is shorter and now fits on one screen, making it easy to follow the high level processes taking place.

- More modular code means that we could use our defined functions in other 'recipes'.

### Unit Testing

- Confirm code is working as intended.
- Ensure changes in one function don't break another
- Protect against changes in dependency

#### **doctest**

Tests example code in a module.

```python
def square(x):
    """Square the number x

    :param x: number to square
    :return: x squared

    >>> square(3)
    9
    """
    return x ** x

import doctest
doctest.testmod()
```

```
Failed example:
    square(3)
Expected:
    9
Got:
    3
```

#### **pytest**

For more extensive testing, beyond that which can be defined in a docstring, we can use `pytest`

*pytest structure*:

```
work_dir
|-- setup.py
|-- requirements.txt
|-- my_package
    |-- __init__.py
    |-- utils.py
|-- tests
    |-- test_unit.py
    |-- test_this.py
    |-- test_that.py
```

```
test
|-- test_unit.py
|-- test_this.py
|-- test_that.py
|-- subpackage_tests
    |-- test_x.py
    |-- test_y.py
|-- subpackage2_tests
    |-- test_i.py
    |-- test_j.py
```

*Writing unit tests*:

```python
### workdir/test/test_document.py ###

from text_analyzer import Document

# Test tokens attribute on Document object
def test_document_tokens():
    doc = Document('a e i o u')

    assert doc.tokens == ['a', 'e', 'i', 'o', 'u']

# Test edge case of blank document
def test_document_empty():
    doc = Document('')

    assert doc.tokens == []
    assert doc.word_counts == Counter()
```

Note:

- It is not wise to attempt to compare two objects with `==`.
- Instead we should compare the attributes of these objects.

```python
# Create 2 identical Document objects
doc_a = Document('a e i o u')
doc_b = Document('a e i o u')

# Check if objects are ==
print(doc_a == doc_b)

# Check if attributes are ==
print(doc_a.tokens == doc_b.tokens)
print(doc_a.word_counts == doc_b.word_counts)
```
```python
False
True
True
```

*Running pytest*:

Working in the terminal.

- Run all tests.
```bash
~/workdir $ pytest
```

- Run specific test
```
~/work_dir $ pytest tests/test_document.py
```

### Documentation & testing in practice

#### **Documenting projects with Sphinx**

Sphynx automatically transforms docstrings into documentation pages.

https://www.sphinx-doc.org/en/master/#

Documenting classes:

```python

class Document:
    """Analyze text data

    :param text: text to analyse

    :ivar text: text originally passed to the instance on creation
    :ivar tokens: Parsed list of words from text
    :ivar word_counts: Counter containing counts of hashtags used in text
    """

    def __init__(self, text):
        ...
```

- Document parameters for `__init__` method in class docstring, so that users may find out how to instantiate a class by calling `help`.
- Document attributes using the `ivar` (instance variable) keyword.

#### **Continuous integration testing**

To save us from having to continually run test from the command line we can set up **continuous integration testing**, using a tool like Travis CI, to test automatically when new code is added.

https://travis-ci.org/

We can also schedule builds such that tests are run even if we are not adding new code- useful for picking up on bugs introduced by updates to dependencies.

#### Other tools

Codecov -  Discover where to improve your project tests. Keeping test coverage high will ensure out code is less prone to bugs.

Code Climate - Analyze your code for improvements in readability.

(both may be integrated with Travis CI to run automatically when new code added)












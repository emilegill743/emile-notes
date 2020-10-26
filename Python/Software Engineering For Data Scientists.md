# Software Engineering for Data Scientists

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
      author='Emile Gill,
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




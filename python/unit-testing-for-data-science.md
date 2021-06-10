# Unit Testing for Data Science in Python

- [Unit Testing for Data Science in Python](#unit-testing-for-data-science-in-python)
  - [Unit Testing Basics](#unit-testing-basics)
    - [Write a Simple unit test using pytest](#write-a-simple-unit-test-using-pytest)
    - [Understanding the test result report](#understanding-the-test-result-report)
    - [More benefits and test types](#more-benefits-and-test-types)
  - [Intermediate Unit Testing](#intermediate-unit-testing)
    - [Assert statements](#assert-statements)
    - [Testing for exceptions instead of return values](#testing-for-exceptions-instead-of-return-values)
    - [How many tests should one write for a function?](#how-many-tests-should-one-write-for-a-function)
    - [Test Driven Development (TDD)](#test-driven-development-tdd)
  - [Test Organisation and Execution](#test-organisation-and-execution)
    - [Organising tests](#organising-tests)
    - [Mastering Test Execution](#mastering-test-execution)
    - [Expected Failures and Conditional Skipping](#expected-failures-and-conditional-skipping)
    - [Continuous integration and code coverage](#continuous-integration-and-code-coverage)
  - [Testing Models, Plots and Much More](#testing-models-plots-and-much-more)

## Unit Testing Basics

### Write a Simple unit test using pytest

The `test_` prefix naming convention indicates the presence of unit tests. Units tests may also be reffered to as test modules.

Unit tests are python functions whose names start with `test_`.

```python
### test_row_to_list.py ###

import pytest
import row_to_list

def test_for_clean_row():
     assert row_to_list("2,081\t314,942\n") == \
         ["2,081", "314,942"]

def test_for_missing_area():
    assert row_to_list("\t293,410\n") is None

def test_for_missing_tab():
    assert row_to_list("1,463238,765\n") is None
```

Test requires `assert` statement, to evaluate success/failure of test.

Test can be run from the command line:

```bash
pytest test_row_to_list.py
```

### Understanding the test result report

**Section 1** - General information about OS, Python version, pytest package versions, working directory and plugins.

```
================== test session start ==================
platform linux -- Python 3.6.7, pytest-4.0.1, py-1.8.0, pluggy-0.9.0
rootdir: /tmp/tmpvdblq9g7, inifile:
plugins: mock-1.10.0
```

**Section 2** - Test results

```
collecting...
collected 3 items

test_ro_to_list.py .F.                  [100%]
```

- 3 tests found
- test results indicated by characters `.` indicating a pass and `F` indicating that the second test failed.

**Section 3** - Information on failed tests

```
================== FAILURES ==================
--------------test_for_missing_area-----------

    def test_for_missing_area():
>       assert row_to_list("\t293,410\n") is None
E       AssertionError: assert ['', '293,410'] is None
E        + where ['','293,410'] = row_to_list('\t293,410\n')

test_row_to_list.py:7 AssertionError
```

**Section 4** - Test Summary

```
======== 1 failed, 2 passed in 0.03 seconds ========
```

- Result summary from all units tests that ran: 1 failed, 2 passed.
- Total time to run 0.03 seconds

### More benefits and test types

- Unit tests serve as documentation of how code should run.
- Greater trust in a package, since users can run tests and verify that the package works.
- Incorporating tests in a CI/CD pipeline ensures bad code is never pushed to a production system.

Unit test
  - A unit is a small, independent piece of code such as a Python function or class.

Integration test
- Check if multiple units work correctly togther.

## Intermediate Unit Testing

### Assert statements

```python
assert boolean_expression, message
```

Message is only printed if boolean expression doesn't pass and should give information about why the assertion error was raised.

```python
### test_row_to_list.py ###

import pytest
...
def test_for_missing_area():
    actual = row_to_list("\t293,410\n")
    expected = None
    message = ("row_to_list)'\t293,410\n') "
               "returned {0} instead "
               "of {1}".format(actual, expected)
               
    assert actual is expected, message
```

**Assertions with Floats:**

Because of the way Python evaluates floats we may get unexpected results when comparing float values, e.g. `0.1 + 0.1 + 0.1 == 0.3` => `False`.

Instead when comparing float values we should wrap the expected return value using `pytest.approx()`.

```python
assert 0.1 + 0.1 + 0.1 == pytest.approx(0.3)
```

This method also works for NumPy arrays:

```python
assert np.array([0.1 + 0.1, 0.1 + 0.1 + 0.1]) == pytest.approx(np.array([0.2, 0.3]))
```

**Multiple assertions in one unit test:**

A unit test may contain mutliple assert statements to test the output of a function.

```python
### test_convert_to_int.py ###

import pytest
...
def test_on_string_with_one_comma():
    return_value = convert_to_int("2,081")
    assert isinstance(return_value, int)
    assert return_value == 2081
```

### Testing for exceptions instead of return values

Sometimes, rather than testing the return value of a function, we want to check that a function correctly raises an Exception.

```python
with pytest.raises(ValueError):
    # <--- Does nothing on entering the context
    print("This is part of the context")
    # <-- If context raise ValueError, silence it.
    # <-- If the context did not raise ValueError, raise an exception
```

```python
def test_valueerror_on_one_dimensional_arg():
    example_argument = np.array([2081, 314942, 1059, 186606, 1148, 206186])
    with pytest.raises(ValueError):
        split_into_training_and_testing_sets(example_argument)
```

We can also check that the error message on exception is correct by storing the exception_info from the context.

```python
def test_valueerror_on_one_dimensional_arg():
    example_argument = np.array([2081, 314942, 1059, 186606, 1148, 206186])
    with pytest.raises(ValueError) as exception_info:
        split_into_training_and_testing_sets(example_argument)

    assert exception_info.match(
                "Argument data array must be 2 dimensional. "
                "Got 1 dimentional array instead"
                )
```

### How many tests should one write for a function?

**Test argument types:**

- Bad Arguments
  - Arguments for which a function raises and exception instead of returning a value.
- Special arguments
  - Boundary values
  - Special logic
- Normal arguments
  - Recommened to test at least 2/3 normal arguments

Note: not all functions will have special or bad arguments.

### Test Driven Development (TDD)

It is good practice to always write unit tests prior to implementation of any code.
- Unit tests cannot be deprioritised.
- Time for writing unit tests must be factored into implementation time.
- Requirements are clearer and implementation is easier.

## Test Organisation and Execution

### Organising tests

**Project Structure:**

```bash
src/                                    # All application code lives here
|-- data/                               # Package for data preprocessing
      |-- __init__.py
      |-- preprocessing_helpers.py      # Contains row_to_list(), convert_to_int()
|-- features                            # Package for feature generation from preprocessed data
      |-- __init__.py
      |-- as_numpy.py                   # Contains get_data_as_numpy_array()
|-- models                              # Package for training/testing linear regression model
      |-- __init__.py
      |-- train.py                      # Contains split_into_training_and_testing_sets()
tests/                                  # Test suite: all tests live here
|-- data/
      |-- __init__.py
      |-- test_preprocessing_helpers.py # Corresponds to module src/data/preprocessing_helpers.py
|-- features
      |-- __init__.py
      |-- test_as_numpy.py              # Contains TestGetDataAsNumpyArray
|-- models
      |-- __init__.py
      |-- test_train.py                 # Contains TestSplitIntoTrainingAndTestSets
```

For each module `my_module.py` there should exist an equivalent `test_my_module.py`. This mirroring of the internal structure of our project ensures that if we know the location of a module in our project, we can infer the location of its test set.

**Test module structure:**

```python
import pytest
from data.preprocessing_helpers import row_to_list, convert_to_int

class TestRowToList(object):      # Use CamelCase
    
    def test_on_no_tab_no_missing_value(self):
        ...
    
    def test_on_two_tabs_no_missing_value(self):
        ...

class TestConvertToInt(object):

    def test_with_no_comma(self):
        ...
    
    def test_with_one_comma(self):
        ...
```

### Mastering Test Execution

- **Running all tests**

  ```bash
  cd tests
  pytest
  ```

- **Stop after first failure**

  ```bash
  pytest -x
  ```

- **Running a subset of tests**

  ```bash
  pytest data/test_preprocessing_helpers.py
  ```

  pytest assigns a unique **Node ID** to each test class and unit test it encounters:

  - **Node ID of a test class**:

    `<path to test module>::<test class name>`

    ```bash
    pytest data/test_preprocessing_helpers.py::TestRowToList
    ```

  - **Node ID of a unit test**:
  
    `<path to test module>::<test class name>::<unit test name>`

    ```bash
    pytest data/test_preprocessing_helpers.py::TestRowToList::test_on_one_tab_with_missing_value
    ```

- **Running tests using keyword expressions**

  ```bash
  pytest -k "pattern"
  ```

  e.g.
  ```bash
  pytest -k "TestSplitIntoTrainingAndTestingSets"
  ```
  ```bash
  pytest -k "TestSplit"
  ```

  ```bash
  pytest -k "TestSplit and not test_on_one_row"
  ```

### Expected Failures and Conditional Skipping

If we want to mark a test that is expected to fail, e.g. the function is not yet implemented, we may do so with the `xfail` decorator. This will result in the tets being marked as xfail in the test results, but not cause a failure of the test suite.

```python
import pytest

class TestTrainModel(object):
    @pytest.mark.xfail(reason="Using TDD, train_model() is not implemented")
    def test_on_linear_data(self):
         ...
```

If we wish to skip a test baseed on a given condition, e.g. a test which only works on a particular Python version, we may use the `skipif` decorator. If the boolean expression passed to the decorator is `True` then the test will be skipped. We must also pass the `reason` argument to describe the reason for this test being skipped.

```python
import pytest
import sys

class TestConvertToInt(object):
     @pytest.mark.skipif(sys.version_info > (2,7, reason="requires Python 2.7")
     def test_with_no_comma(self):
          """Only runs on Python 2.7 or lower"""
          test_argument = "756"
          expected = 756
          actual = convert_to_int(test_argument)
          message = unicode("Expected: 2081, Actual: {0}".format(actual))
          assert actual == expected, message
```

**Showing reason for skipping/xfail:**

```bash
pytest -r[set of characters]

pytest -rs # Show reason for skipping

pytest -rx # Show reason for xfail

pytest -rsx # Show both skipping and xfail reasons
```

Skipping/xfailing may also be applied to entire test classes.

```python
@pytest.mark.xfail(reason="Using TDD, train_model() is not implemented")
class TestTrainModel(object):
     ...
```

### Continuous integration and code coverage

Developers $\rightarrow$ GitHub $\rightarrow$ CI Server

**Build status badge:**

  - Build passing $\implies$ Stable Project

  - Build Failing $\implies$ Unstable Project

**Setting up a CI Server with Travis:**

1) Create a configuration file (`.travis.yml`)

```yaml
language: python

python:
  - "3.6"

install:
  - pip install -e .

script:
  - pytest tests
```

2) Push the file to GitHub

```bash
git add .travis.yml
git push origin master
```

3) Install the Travis CI app

  - The Travis CI app can be installed through the GitHub Marketplace.
  - Travis CI is free for public respositories.
  - Every commit pushed to our GitHub repo will now trigger a build in Travis.

4) Show the build status badge

  - Click on the badge show in Travis
  - Select "Markdown".
  - Paste markdown code in GitHub `README.md`

**Adding Code Coverage Badge to GitHub:**

$\text{code coverage} = \frac{\text{num lines of application code that ran during testing}}{\text{total num lines of application code}}$

High percentages (75% and above) indicate a well tested code base.

1) Modify .travis.yml

```yaml
language: python
python:
  - "3.6"
install:
  - pip install -e .
  - pip install pytest-cov codecov # Install packages for code coverage report
script:
  - pytest --cov=src tests        # Point to the source directory
after_success:
  - codecov                       # Uploads report to codecov.io
```

2) Install Codecov app from Github Marketplace

3) Paste markdown link for badge in `README.md`






  




## Testing Models, Plots and Much More

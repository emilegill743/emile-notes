# Conda Essentials

## Contents

  - [Installing Packages](#installing-packages)
    - [Installing packages](#installing-packages-1)
    - [Listing installed packages](#listing-installed-packages)
    - [Update a conda package](#update-a-conda-package)
    - [Remove a conda package](#remove-a-conda-package)
    - [Search for available package versions](#search-for-available-package-versions)
    - [Find dependencies of package version](#find-dependencies-of-package-version)
  - [Utilising Channels](#utilising-channels)
    - [Searching with channels](#searching-with-channels)
    - [Searching across channels](#searching-across-channels)
    - [Default, non-default and special channels](#default-non-default-and-special-channels)
    - [Installing from a channel](#installing-from-a-channel)
  - [Working with Environments](#working-with-environments)
    - [What is a Conda Environment?](#what-is-a-conda-environment)
    - [Why do we need Environments?](#why-do-we-need-environments)
    - [Find current environment](#find-current-environment)
    - [What packages are installed in an environment](#what-packages-are-installed-in-an-environment)
    - [Switching between environments](#switching-between-environments)
    - [Remove an environment](#remove-an-environment)
    - [Create a new environment](#create-a-new-environment)
    - [Export an environment](#export-an-environment)
    - [Create and environment from a shared specification](#create-and-environment-from-a-shared-specification)

## Installing Packages

Conda packages are files containing a bundle of resources: usually libraries and executables, but not always. In principle, Conda packages can contain data, images, notebooks, or other assets. The `conda` command-line tool is used to install, remove and examine packages.

### Installing packages

```bash
# installing a package
conda install package

# installing a specific version of a package

conda install package=13 # select at MAJOR version level
conda install package=12.3 # select at MINOR version level
conda install package=14.3.2 # select exact PATCH level

conda install 'bar-lib=1.0|1.4*' 
# prefer 1.4 but if not install 1.0 (e.g. if known bug introduced in 1.0 and solved in 1.4)

conda install 'bar-lib>1.3.4,<1.1'
# prefer greater than 1.3.4 but if not, before 1.1 will do

```

### Listing installed packages
```bash
# list packages in current env
conda list

#list packages in named env
conda list -n myenv

# save packages for future use
conda list --export > package-list.txt

# re-install packages from and export file
conda create -n myenv --file package-list.txt
```

### Update a conda package
```bash
conda update package
```

### Remove a conda package
```bash
conda remove package
```

### Search for available package versions
```bash
# print available versions
conda search package
```

### Find dependencies of package version
```bash
# print details of package version/s
conda search package --info

# print details of specific version/build
conda search package=0.8.2=build --info

# The * wildcard can be used to match a pattern in the build, since recent build will include a hash at the end of the python version string
conda search numpy=1.13.1=py36*
```

## Utilising Channels

A conda channel is an identifier of a path from which Conda packages may be obtained. Using the public cloud , installing without a specified channel points to the main channel at https://repo.anaconda.com/pkgs/main, where hundreds of packages are available.

Although covering a wide swath, the main channel only contains packages which are curated by Anaconda. Anyone may register with Anaconda cloud, thereby creating their own personal Conda channel.

### Searching with channels
```bash
# search for package in a specific channel
conda search --channel davidmertz --override-channels --platform linux-64 package

# --override-channels : prevents searching on default channel
# --platform : selects a platform that differs from the current computer's 
```

### Searching across channels

The package `anaconda-client` provides the command `anaconda` that searches in a different manner that is often more useful. You can search across all channels and platforms (not having to specify these) using:

```bash
anaconda search package
```

### Default, non-default and special channels

The default channel on Anaconda cloud is curated by Anaconda Inc., but another channel called `conda-forge` also has special status. It does not operate any differently from other channels but it acts as a kind of "community curation" of relatively well vetted packages.

### Installing from a channel
```bash
conda install --channel my-organisation the-package
```

## Working with Environments

### What is a Conda Environment?

Conda environments allow multiple incompatible versions of the same package to coexist on your system. An **environment** is simply a file path containing a collection of mutually compatable packages. By isolating distict versions (and their dependencies) in distinct environments, those versions are all available to work on particular projects or tasks.

### Why do we need Environments?

There are a large number of reasons why it is best practice to use environments, whether as a data scientist, software developer, or domain specialist. Without the concept of environments, users essentially rely on and are restricted to whichever particular package versions are installed globally (or in their own user accounts) on a particular machine. Even when one user moves scripts between machines (or shares them with a colleague), the configuration is often inconsistent in ways that interfere with seamless functionality. Conda environments solve both these problems. You can easily maintain and switch between as many environments as you like, and each one has exactly the collection of packages that you want.

### Find current environment
```bash
# list conda environments
conda env list
```

### What packages are installed in an environment
```bash
# list installed packages
conda list

# list specific packages
conda list 'numpy|pandas'

#list packages in environment other than the active one
conda list --name ENVNAME
```

### Switching between environments
```bash
# activate an environment
conda activate ENVNAME

# deactivate an environment
conda deactivate
```

### Remove an environment
```bash
conda env remove --name ENVNAME
```

### Create a new environment

```bash
# create a new environment
conda create --name ENVNAME

# create a new environment with specified packages
conda create --name recent-pd python=3.6 pandas=0.22 scipy statsmodels
```

### Export an environment
```bash
conda env export --name ENVNAME --file environment.yml
```

### Create and environment from a shared specification
```bash
conda env create --file file-name.yml
```
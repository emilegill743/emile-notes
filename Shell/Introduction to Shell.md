# Introduction to Shell

## Contents
 - [Manipulating Files and Directories](#Manipulating-Files-and-Directories')
 - [Manipulating Data](#Manipulating-Data)
 - [Combining Tools](#Combining-Tools)
 - [Batch Processing](#Batch-Processing)
 - [Creating New Tools](#Creating-New-Tools)

## Manipulating Files and Directories

### Print working directory
```bash
pwd
```

### List contents of directory
```bash
ls /home/repl/seasonal
```

### Change directory
```bash
cd /home/repl/seasonal
cd .. # move up to parent
cd 

```

### Special paths

```bash
. # Current Directory
.. # Parent Directory
~ # Home Directory
```

### Copy files
```bash
cp original.txt duplicate.txt # Copy file
cp seasonal/autumn.csv seasonal/winter.csv backup # Copy files into a directory
```

### Move files
```bash
mv autumn.csv winter.csv .. # Move files to parent dir
mv course.txt old-course.txt # Rename file
```

### Delete files
```bash
rm thesis.txt backup/thesis-2017-08.txt # Remove files
```

### Create and delete directories
```bash
mkdir directory_name
rmdir directory_name
```

## Manipulating Data

### Viewing a file's contents
```bash
cat agarwal.txt
```

### View a files contents piece by piece
```bash
less course.txt
# display one page at a time, spacebar scrolls down, q to quit

less seasonal/spring.csv seasonal/summer.csv
# view files in order, spacebar scrolls down, :n to go to next file, :p to go to previos, :q to quit
```

### Preview the first few lines of a file
```bash
head seasonal/summer.csv
```

### Tab completion

The tab key allows us to autocomplete the command. If the command is ambiguous it will display a list of possibilities.

### Command flags

Command flags allow us to change the behaviour of a command, e.g. printing the first n lines of a file:

```bash
head -n 3 seasonal/summer.csv
```

### List everything below a directory

```bash
ls -R # recursive list
ls -F # prints / after the name of each dir, * after the name of a runable program
```

### Get help for a command

```bash
man head # open manual for command
```

### Select columns from a file
```bash
cut -f 2-5,8 -d , values.csv
# select columns 2 through 5 and columns 8, using a comma as the seperator (-f stands for fields and -d delimiter)
```

### Repeat commands

Pressing up and down cycles through previous commands.

```bash
history # print list of recent commands
!55 # run 55th command
!head # re-run the most recent use of the command
```
### Select lines containing specific values

```grep``` selects lines according to what they contain.

```bash
grep bicuspid seasonal/winter.csv # selects lines containing 'bicuspid'

# common flags
    -c # print count of matching lines rather than lines themselves
    -h # do not print names of files when searching multiple files
    -i # ignore case
    -l # print names of files containing matches, not matches
    -n # print line numbers for matching lines
    -v # invert the match (only show lines that don't match)
```

## Combining Tools

### Store command's output in file
```bash
head -n 5 seasonal/summer.csv > top.csv
```

### Use a command's output as an input
```bash
# via an intermediate file
head -n 5 seasonal/winter.csv > top.csv
tail -n 3 top.csv

# using a pipe 
# (uses output of command on left as input of command on right)
head -n 5 seasonal/summer.csv | tail -n 3
```

### Combining many commands
```bash
cut -d , -f 1 seasonal/spring.csv | grep -v Date | head -n 10
```

### Count the number of records in a file
```bash
wc file # print number of characters, words and lines in a file
    -c # characters
    -w # words
    -l # lines
```

### Specify many files at once
```bash
* # zero or more characters
? # single character
[...] # any one of the characters inside the square brackets
{...} # any of the comma seperated patterns inside the curly brackets
```

### Sort lines of text
```bash
sort file # sort data in alphabetical order
    -n # sort numerically
    -r # reverse order
    -b # ignore leading blanks
    -f # fold case (ignore case)
```

### Remove duplicate lines
```bash
uniq file # removes adjacent duplicate lines

# Get second column of csv, remove 'Tooth' from text, sort and display count of each occurance
cut -d , -f 2 seasonal/winter.csv | grep -v Tooth | sort | uniq -c
```

### Save the output of a pipe
```bash
cut -d , -f 2 seasonal/*.csv | grep -v Tooth > teeth-only.txt
```

## Batch Processing

### Environment Variables

The shell stores information in variables. Some of these, called **environment variables**, are available all the time.

```bash
# Common environment variables
HOME # user's home directory
PWD # present working directory
SHELL # which shell program is being used
USER # user's id
```

```bash
set # list all environment variables
set | grep PATH # find specific environment variable
```

### Print variable's value

```bash
echo $USER # prints value of USER to terminal
```

### Shell variables

The other kind of variable is known as a **shell variable**, which is like a local variable in a programming language.

```bash
training=seasonal/summer.csv
```

### Repeating commands in loops

```bash
for filetype in gif jpg png; do echo $filetype; done

for filename in seasonal/*.csv; do echo $filename; done

for file in seasonal/*.csv; do head -n 2 $file | tail -n 1; done

# multiple commands in a loop, seperate by a semi colon
for f in seasonal/*.csv; do echo $f; head -n 2 $f | tail -n 1; done
```

## Creating New Tools

### Edit a file

Unix has many text editors, one of which is Nano.

```bash
# open file for editing
nano filename

Ctrl + K # delete a line
Ctrl + U # un-delete a line
Ctrl + O # save the file
Ctrl + X # exit the editor
```

### Recording actions
```bash
history | tail -n 10 > figure-5.history
```

### Save commands to run later
```bash
# save commands to shell script, headers.sh
nano headers.sh
'head -n 1 seasonal/*.csv'

# run commands from saved script
bash headers.sh

# save output
bash headers.sh > headers.out
```

### Passing parameters to scripts

`$@` represents "all of the command line parameters passed to a script"

```bash
# save commands to script, including $@
nano uniq-lines.sh
'sort $@ | uniq'

# run script passing parameters
bash unique-lines.sh seasonal/summer.csv
bash unique-lines.sh seasonal/*.csv
```

As well as `$@` we can also use $1, $2, etc. to refer to specific parameters that have been passed.

```bash
# save commands to script, including $1, $2
nano column.sh
'cut -d , -f $2 $1'

# run script passing parameters
bash column.sh seasonal/autumn.csv 1
```

### Writing loops in shell scripts

Shell scripts can contain also loops, these can be written using semi-colons or split across lines without semi-colons for better readability.

```bash
# Print thr first and last data records of each file
for filename in $@
do
    head -n 2 $filename | tail -n 1
    tail -n 1 $filename
done
```




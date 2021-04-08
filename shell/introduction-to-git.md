# Introduction To Git

- [Introduction To Git](#introduction-to-git)
  - [Basic Workflow](#basic-workflow)
    - [What is version control?](#what-is-version-control)
    - [Checking state of repository:](#checking-state-of-repository)
    - [Checking differences to last saved version:](#checking-differences-to-last-saved-version)
    - [Commiting changes:](#commiting-changes)
    - [View repository history:](#view-repository-history)
  - [Repositories](#repositories)
    - [Commit structure:](#commit-structure)
    - [View a specific commit:](#view-a-specific-commit)
    - [View who made changes to file and when:](#view-who-made-changes-to-file-and-when)
    - [Show changes between two commits:](#show-changes-between-two-commits)
    - [Tell git to ignore certain files:](#tell-git-to-ignore-certain-files)
    - [Remove unwanted files:](#remove-unwanted-files)
    - [See how Git is configured:](#see-how-git-is-configured)
    - [Change Git configuration:](#change-git-configuration)
  - [Undo](#undo)
    - [Commit changes selectively:](#commit-changes-selectively)
    - [Reverse accidental staging:](#reverse-accidental-staging)
    - [Undo changes to unstaged files:](#undo-changes-to-unstaged-files)
    - [Undo changes to staged files:](#undo-changes-to-staged-files)
    - [Restore old version of a file:](#restore-old-version-of-a-file)
    - [Display contents of file:](#display-contents-of-file)
    - [Undo all changes made:](#undo-all-changes-made)
  - [Branches](#branches)
    - [What is a branch?](#what-is-a-branch)
    - [List branches of repository:](#list-branches-of-repository)
    - [Difference between branches:](#difference-between-branches)
    - [Switch between branches:](#switch-between-branches)
    - [Delete file:](#delete-file)
    - [Create branch:](#create-branch)
    - [Merging branches:](#merging-branches)
    - [Merging branches with conflicts:](#merging-branches-with-conflicts)
  - [Collaborating](#collaborating)
    - [Create new repository:](#create-new-repository)
    - [Turn existing project into Git repository:](#turn-existing-project-into-git-repository)
    - [Create copy of existing repository:](#create-copy-of-existing-repository)
    - [Find out where a cloned repo originated:](#find-out-where-a-cloned-repo-originated)
    - [Define remotes:](#define-remotes)
    - [Pull in changes from remote repo:](#pull-in-changes-from-remote-repo)
    - [Push changes to remote repo:](#push-changes-to-remote-repo)
    - [What if push conflicts with someone else's work?:](#what-if-push-conflicts-with-someone-elses-work)

## Basic Workflow

### What is version control?

> **Version control system**- manages changes made to files and directories in a project.

- Saves changes made to projects
- Automatically notifies when conflicts arise
- Synchronises work across different users and machines

> **Repository**- the files and directories in a git project, in addition to infromation recorded about the project's history.

---

### Checking state of repository:

`Working directory --> Staging Area --> .git Directory`

```bash
git status # shows files in the staging Area
```

- Displays list of files which have been modified since the last changes were saved

---

### Checking differences to last saved version:

```bash
git diff filename #for file
git diff directory #for directory
git diff #shows all changes
```

e.g.

```
diff --git a/report.txt b/report.txt

index e713b17..4c0742a 100644
--- a/report.txt
+++ b/report.txt
@@ -1,4 +1,5 @@
-# Seasonal Dental Surgeries 2017-18
+# Seasonal Dental Surgeries (2017) 2017-18
+# TODO: write new summary

```

- `a`, `b` are placeholders representing 'first version', 'second version'
- `-` represents lines removed
- `+` represents lines added
- `@@` tells us where changes are being made (start line, end line)
- 
---

### Commiting changes:

```bash
git commit -m "log message"

# Amend mistyped comment
git commit --amend -m "new message"
```
Make longer comment:

`git commit` with `-m` absent launches text editor.

**`Ctrl-O`** + **`Enter`** to save.

**`Ctrl-X`** to exit editor.

---

### View repository history:
```bash
git log

git log [path] # specific file's history

git log -n # See last n commits
```
--- 

## Repositories

### Commit structure:

- `commit` - metadata e.g. author, commit message, time
- `tree` - names and locations in repository when commit happened.
- `blob` - (binary large object) compressed snapshot of contents of file when commit happened
- `hash` - pseudo-random number generated for each unique commit

---

### View a specific commit:
```bash
git show

# Specific commit
git show [hash- first 6 char]

git show -r HEAD
# HEAD refers to most recent commit
# HEAD~1 is the commit before
# HEAD~2 is the commit before that etc.
```

---

### View who made changes to file and when:

```bash
git annotate [filename]
```

**Returns 5 elements**:
1. first 8 digits of hash
2. author
3. time of commit
4. line number
5. contents of line

---

### Show changes between two commits:

```bash
git show [ID] #show changes made in particular commit
git diff ID1..ID2 #show difference between two commits
```

e.g.
```bash
git diff abc123..def456
git diff HEAD~1..HEAD~3
```

---

### Tell git to ignore certain files:

- Create a file in root directory called `.gitignore`

- Store list of wildcard patterns for files Git should ignore

e.g.
```bash
build #ignore directory called build
*.mpl #ignore files ending in .mpl
```

---

### Remove unwanted files:

```bash
git clean -n #shows list of files in repository but who's history is not being tracked
git clean -f #will delete these files
```

---

### See how Git is configured:

```bash
git config --list --system #settings for all users on computer
git config --list --global #settings for all projects
git config --list --local #settings for one project
```

---

### Change Git configuration:

```bash
git config --global [setting] [value]
```
e.g.

```
git config --global user.email rep.loop@datacamp.com
```

---

## Undo

### Commit changes selectively:

```bash
git add path/to/file # to stage single file
```

---

### Reverse accidental staging:

```bash
git reset HEAD
```

---

### Undo changes to unstaged files:

```bash
git checkout -- filename # discard changes that have not been staged
```

---

### Undo changes to staged files:

```bash
git reset HEAD path/to/file # unstages file
git checkout -- path/to/file #undo changes since last commit
```

---

### Restore old version of a file:

```bash
git log #shows old versions
git checkout [hash_6_char] [filename] #replaces curret version with previous version
```
Restoring file doesn't delete history of repository, just saves the restore as another commit

---

### Display contents of file:

```bash
cat [filename]
```

---

### Undo all changes made:

```bash
git reset HEAD [directory] #unstage all files from directory
```

Head default command so 'git reset' had same effect

```bash
git checkout --dir #restore files to previous state
git checkout -- . #restore files in current dir
```

---

## Branches

### What is a branch?

Branches are multiple versions of the same work, to which changes may be made independently, until merged back together.

Three part data structure:

- blob - files
- trees - saved states
- commits - record changes

Branches are the reason trees and commits are needed.
A commit will have two parents when branches are being merged.

---

### List branches of repository:

```bash
git branch
```

Current branch shown with *

---

### Difference between branches:

Branches and revisions are closely connected, commands that work on the later normally work on the former.

```bash
git diff branch-1..branch-2y
```

---

### Switch between branches:

```bash
git checkout [branch]
```

Git will only allow you to do this if you have committed all changes to the branch.

---

### Delete file:

```bash
git rm
```

---

### Create branch:

Can use `git branch`. But more commonly want to create branch and switch to it.

`-b` flag using `checkout` creates and switches in one step.
    
```bash
git checkout -b branch-name
```

---

### Merging branches:

If changes to two branches aren't conflicting we can incorporate the changes made to the source branch into the destination branch.

```bash
git merge source destination
```

---

### Merging branches with conflicts:

`git status` after merge will remind you of conflicts that you need to resolve. Git will leave markers inside the file to show where these conflicts occur.
    
e.g.    
```bash
<<<<<<< destination-branch-name
...changes from the destination branch...
=======
...changes from the source branch...
>>>>>>> source-branch-name
```

Can open the file using `nano [file]` and resolve marker conflicts.

Add the file to the staging area `git add [file]`
Commit changes `git commit -m [message]`

---

## Collaborating

### Create new repository:

```bash
git init [project-name]
```

Avoid creating repositories inside other repositories.

---

### Turn existing project into Git repository:

```bash
git init #when in projects working directory
git init [path to project]
```

---

### Create copy of existing repository:
    
```bash
git clone url
git clone [path to project] [new project name]
```

---

### Find out where a cloned repo originated:

```bash
git remote #lists names of remotes
git remote -v #lists remote URLs
```

---

### Define remotes:

Git automatically assigns *'origin'*: the original repo

Add additional remotes:

```bash
git remote add [remote-name] URL
git remote rm remote-name #remove remote name
```

---

### Pull in changes from remote repo:

Git keeps track of remote repositories so you can pull and puch changes to them.

A typical workflow is that you pull or push work to and from an online hosting service like GitHub.

```bash
git pull [remote-name] [branch-name]
```
e.g. `git pull thunk latest-analysis`

If you have unsaved changes Git will stop you from pulling before you have either commited your changes locally or reverted them.

---

### Push changes to remote repo:

Push changes from local to remote repo:

```bash
git push [remote-name] [branch-name]
```

---

### What if push conflicts with someone else's work?:

To prevent overwriting someones work Git requires you to merge the contents of a remote repo into your own, before pushing

```bash
git pull [remote-name] [branch-name] #first merges remote with local
git push [remote-name] [branch-name] #then pushes merged dir to remote
```



    


    











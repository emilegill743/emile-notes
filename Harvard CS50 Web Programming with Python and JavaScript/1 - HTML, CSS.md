# Lecture 1: HTML, CSS

## Resources

- [Lecture Recording](https://video.cs50.net/web/2018/spring/lectures/1)
- [Lecture Notes](https://cs50.harvard.edu/web/notes/1/)
- [Lecture Slides](http://cdn.cs50.net/web/2018/spring/lectures/1/lecture1.pdf)
- [Source Code](http://cdn.cs50.net/web/2018/spring/lectures/1/src1.zip)
- [Course Page](https://courses.edx.org/courses/course-v1:HarvardX+CS50W+Web/course/)

---

## Contents

* [More on Git](#More-on-Git)

---

## More on Git

### Branching

>'**Branching**' is a feature of Git that allows a project to move in multiple different directions simultaneously. The `master` branch is always useable, but any number of new branches can be created to develop new features. These can then be merged back into the master branch, once ready.

In a Git repository `HEAD` refers to the current branch being worked on. When a different branch is 'checked out', the `HEAD` changes to indicate the new working branch.

When merging a branch back into master, there is the possibility for merge conflicts to arise, which can be resolved as discussed in Lecture 0.

#### Git commands related to branching:

`git branch` : lists all branches in repository.

`git branch <name>` : create a new branch called `name`.

`git checkout <name>` : switch current working branch to `name`.

`git merge <name>` : merge branch `name` into current working branch.

### Remote repositories

>Any version of a repository that is not stored locally is called a '**remote**'. We use '**origin**' to refer to the remote from which the local repository was originally downloaded.

#### Git commands related to remote repositories:

`git fetch` : download all of the latest commits from a remote to a local device.

`git merge origin/master` : merge `origin/master`, the remote version of our repository downloaded with `git fetch`, into the local branch.

`git pull` : equivalent to running `git fetch` and then `git merge origin/master`

### Forks

>A '**fork**' of a repository is an entirely seperate repository which is a copy of the original: it may be managed and modified without affecting the original copy.

Open source projects are often developed using forks. There will be one central version of the software which is forked, improved upon and finally merged to the central repository via a '**pull request**'.

A **pull request** can be made to merge a branch of a repository with another branch of the same repository or a different repository. They provide a good way to get feedback from collaborators on the same project.

## More on HTML

### Linking content

We can link to a URL, or some other content using the `href` tag, pasing a URL or `id`.

```html
<a href="path/to/hello.html">Click here!</a>
```

```html
<a href="#Section2">Go to Section2</a>
```

### Form input

There are various types of `input` tags that we can use to gather form data from a webpage:

**Text input**

<input name="name" type="text" placeholder="Name">

```html
<input name="name" type="text" placeholder="Name">
```

**Password input**

<input name="password" type="password" placeholder="Password">

```html
<input name="password" type="password" placeholder="Password">
```

**Radio-button option**

<div>
    Favorite color?
    <input name="color" type="radio" value="red"> Red
    <input name="color" type="radio" value="green"> Green
    <input name="color" type="radio" value="blue"> Blue
    <input name="color" type="radio" value="other"> Other
</div>

```html
<div>
    Favorite color?
    <input name="color" type="radio" value="red"> Red
    <input name="color" type="radio" value="green"> Green
    <input name="color" type="radio" value="blue"> Blue
    <input name="color" type="radio" value="other"> Other
</div>
```

**Datalist input**

<input name="country" list="countries" placeholder="Country">
<datalist id="countries">
    <option value="country1">
    <option value="country2">
    <option value="country3">
</datalist>

```html
<input name="country" list="countries" placeholder="Country">
<datalist id="countries">
    <option value="country1">
    <option value="country2">
    <option value="country3">
</datalist>
```

## CSS

>**CSS Selectors** are patterns used to select different parts of a website to style.

#### Common CSS selectors:

- Select `h1` and `h2`

```css
h1, h2 {
    color: red;
}
```

- Select all `li` (list items) that are descendents of `ol` (not neccessarily immediate descendents)

```css
ol li {
    color: red;
}
```

- Select all `li` that are immediate children of `ol`

```css
ol > li {
    color: red;
}
```

- Select all `input` fields with the attribute `type=text`

```css
input[type=text] {
    background-color: red;
}
```

- Select all `button`s with the pseudoclass (special state of HTML element, e.g whether the cursor is hovering over it) `hover`

```css
button:hover {
    background-color: orange;
}
```

- Select all `before` pseudoelements (a way to affect certain parts of a HTML element, e.g. applying content before content of `a` elements) of the element `a`

```css
a::before {
    content: "\21d2 Click here";
    font-weight: bold;
}
```

- Select all `selection` pseudoelements of the element `p`

```css
p::selection {
    color: red;
    background-color: yellow;
}
```


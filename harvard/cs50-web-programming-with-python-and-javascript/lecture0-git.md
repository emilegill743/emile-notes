# Lecture 0: Git, HTML, CSS

## Resources

- [Lecture Recording](https://video.cs50.net/web/2018/spring/lectures/0)
- [Lecture Notes](https://cs50.harvard.edu/web/notes/0/)
- [Lecture Slides](http://cdn.cs50.net/web/2018/spring/lectures/0/lecture0.pdf)
- [Source Code](http://cdn.cs50.net/web/2018/spring/lectures/0/src0.zip)
- [Course Page](https://courses.edx.org/courses/course-v1:HarvardX+CS50W+Web/course/)

---

## Contents

* [Git](#Git)

    * [Git Commands](#Git-Commands)
    * [Merge Conflicts](#Merge-Conflicts)

* [HTML](#HTML)

    * [Common HTML Tags](#Common-HTML-tags)
    * [Document Object Model](#Document-Object-Model)

* [CSS](#CSS)

    * [Common CSS Properties](#Common-CSS-Properties)

* [Sectioning with HTML and CSS](#Sectioning-with-HTML-and-CSS)

* [GitHub Pages](#GitHub-Pages)

---

## Git

> Git is a version control system which can be used to keep track of changes to code, synchronise code between different people, test changes to code without losing the original, and revert back to old versions of code.

### Git Commands

 ```git clone <url>``` : Takes a repository stored on a server (such as GitHub) and downloads it.

 ```git add <filename(s)>``` : Add files to staging area to be included in next commit.

 ```git commit -m "message"``` : Take a snapshot of the repository and save it with a message about the changes.

 ```git commit -am <filename(s)> "message"``` : Add files to staging area and commit all in one.

 ```git status``` : Prints current status of repository.

 ```git push``` : Push any local changes (commits) to remote server.

```git pull``` : Pull any remote changes from remote server to local computer.

```git log``` : Print history of all commits made.

```git reflog``` : Print list of all the different references to commits.

```git reset --hard <commit>``` : reset repository to a given commit.

```git reset --hard origin/master``` : reset repository to its original state (version on GitHub).

### Merge Conflicts

When combining different versions of code, Git will attempt to automatically merge the two versions. However, if, for example, two users attempt to edit the same line of code, Git's automatic merging will fail. This is known as a **merge conflict**.

The file containing the merge conflict will now show the conflicting lines as:

```
<<<<<<< HEAD
your changes
=======
conflicting changes
>>>>>>> conflicting commit

```

To resolve the merge conflict, we simply remove all the lines we do not want and then stage and commit edited file.

---

## HTML

> HTML (HyperText Markup Language) is a language used to lay out the structure of a webpage.


`<!DOCTYPE html>` is placed at the start of an HTML file to indicate to the browser that HTML5 is being used.

HTML is made up of tags, generally used in pairs with data in between.

```html
 <title>My Web Page!</title>
 ```

Tags are indented to help visualise their heirarchy, but this is purely stylistic.

```html
<head>
    <title>My Web Page!</title>
<head>
```

Tags can also have attributes, which are data fields, sometimes required and sometimes optional, that provide additional information to the browser about how to render the data.

```html
<img src="path/to/img.jpg" height="200", width="300>
```

### Common HTML tags

```html
<html></html> : contents of website
```

```html
<head></head> : metadata about the page that's useful when displaying the page
```

```html
<title></title> : title of the page
```

```html
<body></body> : body of the page
```

```html
<h1></h1> : header (h1 largest --> h6 smallest)
```


```html
<ul></ul> : unordered list

<ol></ol> : ordered list

<li></li> : list item - must be inside <ul></ul> or <ol></ol>
```

```html
<img src="path/to/img.jpg" height="200", width="300"> : image

src - provides path, url to image

height, width - optional (browser will auto-size if omitted)
              - can also be specified as a % to scale with page
```

```html
<table></table> : table

<th></th> : table header

<tr></tr> : table row

<td></td> : table data (cell)
```

```html
<form></form> : form that can be filled out and submitted by the user

<input type="text" placeholder="Full Name" name="name"> : input field

type - type of data

placeholder - greyed-out text shown before field is filled

name - identifier for input field

<button></button> : button to submit form
```

### Document Object Model

The **Document Object Model** is a way to conceptualise webpages by representing them as an interconnected hierarchy of nodes.

In HTML, the nodes of the DOM would be the different tags and their contained data, with the `<html></html>` tag being at the very top of the tree.

---

## CSS

> CSS (Cascading Style Sheets) is a language used to interact with and style HTML, changing the way it looks according to a series of rules.

CSS can be applied to HTML in a variety of different ways:

1. Defining the `style` attribute for a particular tag.

    ```html
    <h5 style="color:blue;text-align:center;"></h5>
    ```
2. Using `<style></style>` tags within the documents `<head></head>` tags.

    ```html
    <head>
        <title>My Web Page!</title>
        <style>
            h1, h2 {
                color: blue;
                text-align: center;
            }
        </style>
    </head>

3. A seperate `.css` file, add link to document head.

    ```html
    <head>
        <title>My Web Page!</title>
        <link rel="stylesheet" href="path/to/styles.css">  
    </head>
    ```

    ```css
    /* styles.css*/
    h1, h2 {color: blue;
        text-align: center;
    }
    ```

### Common CSS Properties

[**Extensive documentation of properties**](https://developer.mozilla.org/en-US/docs/Web/CSS)

```css
color: blue
```
 Define color as one of the ~140 named colors, or hexadecimal RGB value (e.g. #0c8e05)

 ```css
 text-align: left
 ```

 Define text alignment; possible arguments are `center`, `right`, `left`, `justified`.

 ```css
 background color: teal
 ```

 Set background color, same format as `color` property.

 ```css
 height: 150px
 width: 150px
 ```

 Set height, width of area. Pixel arguments may often be specified as a `%` or, simply, `auto`.

 ```css
 margin: 30px
 ```

 Set margin around all four sides of an area. May also be broken up into `margin-left`, `margin-right`, `margin-top`, `margin-bottom`.

 ```css
 padding: 20px
 ```

 Set padding around text inside an area. Can be broken up in the same way as with `margin`.

 ```css
 font-family: Arial, sans-serif
 ```

 Set font family yo be used. A comma seperated list provides alternatives in case a browser doesn't support a particular font. Generic families such as `sans-serif` will use browser defaults.

 ```css
font-size: 28px
 ```

Set font size.

```css
font-weight: bold
```

Set font weight to a relative measure, e.g. `lighter` or a number.

```css
border: 3px solid blue
```

Set border around an area.

## Sectioning with HTML and CSS

Two special tags allow us to break up our webpage into sections.

```html
<div></div> : vertical division of a webpage
```

```html
<span></span> : section of a webpage inside, for example text
```

Different sections of a webpage can be referenced with the `id` and `class` attributes. `id`'s uniquely identify elements, but there can be an arbitrary number of elements with a given `class`.

```html
<body>
    <div id="top">
        This is the <span class="name">top</span> of my webpage.
    </div>

    <div id="middle">
        This is the <span class="name">middle</span> of my webpage.
    </div>

    <div id="bottom">
        This is the <span class="name">bottom</span> of my webpage.
    </div>
</body>
```


`id`'s can be referenced in CSS with `#id` and `class`es with `.class`. This allows us to style the same types of areas in different ways.

```css
#top {
    font-size: 36px;
}

#middle {
    font-size: 24px;
}

#bottom {
    font-size: 12px
}

.name {
    font-weight: bold
}
```

## GitHub Pages

GitHub Pages is a feature of GitHub which allows for a repository to be deployed to the internet.

Simply scroll to GitHub Pages under Settings, select the master branch, and click save.

By default, the repository will be deployed to username.github.io/repository.

GitHub Pages is automatically updated when the repository is updated.
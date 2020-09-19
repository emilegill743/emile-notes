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

|  Type   | CSS Selector               |
|  ----   | ------------               |
| `a, b`  | Multiple Element Selector  |
| `a b`   | Descendant Selector        |
| `a > b` | Child Selector             |
| `a + b` | Adjacent Siblings Selector |
| `[a=b]` | Attribute Selector         |
| `a:b`   | Pseudoclass Selector       |
| `a::b`  | Pseudoelement Selector     |

## Responsive Design

> **Responsive Design** is the idea that a website should look good regardless of the platform it is viewed from.

### Media Query

A **media query** is a method which allows us to apply CSS depending on the context of how we are viewing the site.

For example:

#### Adding content to a webpage which will appear on-screen, but not when printed

```css
<style>
    @media print {
        .screen-only {
            display: none;
        }
    }
</style>
<body>
    <p class="screen-only">This will not appear when printed</p>
</body>
```

`.screen-only` is a class selector which identifies what content we want to be print only.

#### Varying styling with screen size

```css
@media (min-width: 500px) {
    body {
        background-color: red;
    }
}

@media (max-width:499px) {
    body {
        background-color: yellow;
    }
}
```
[print.html](Source%20Code\src1\print.html)

When the width of the screen is at least 500px, the background color of `body` will be red, while if it is less than 499px, the background color of `body` will be yellow.

In order to interact with the screen size, we must include the following in `head':

```html
<meta name="viewport" content="width=device-width, initial scale=1.0">
```
`viewport` is the visible area on which the screen is being displayed, `content` refers to the entire webpage, the `wifth` of which is being set to `device-width`.

Varying heading with screen size:

```html
<meta name="viewport" content="width=device-width, initial scale=1.0">
<style>
    @media (min-width: 500px) {
        h1::before {
            content: "Welcome to My Web Page!";
        }
    }

    @media (max-width: 499px) {
        h1::before{
            content: "Welcome!";
        }
    }
</style>
```

### Flexbox

> **Flexbox** allows for the reorganisation of content based on the size of the viewport

```css
.container {
    display: flex;
    flex-wrap: wrap;
}
```
[flexbox.html](Source%20Code\src1\flexbox.html)

By setting `display: flex` and `flex-wrap: wrap`, content will wrap vertically if neccessary, so no content is lost when the width of the screen is shrunk.

### Grid

A grid of content may be achieved in a similar fashion.

```css
.grid {
    display: grid;
    grid-column-gap: 20px;
    grid-row-gap: 10px;
    grid-template-columns: 200px 200px auto;
}
```
[grid.html](Source%20Code\src1\grid.html)

By setting display: grid, all the different characteristics of a grid layout can be used to format content. In particular, when defining `grid-template-columns`, the final column can be set to `auto`, filling any space that is left. If multiple columns are set to auto, they will equally share the remaining space.

## Bootstrap

> **Bootstrap** is a CSS library written to help make clean, responsive, and nice-looking websites, without having to rememver the gritty details about flexboxes, grids etc. everytime a layout needs to be set up.

The only addition required to use Bootstrap is the line:

```html
<link
    rel="stylesheet"
    href="https://stackpath.bootstrapcdn.com/bootstrap/4.1.1/css/bootstrap.min.css"
    integrity="sha384-WskhaSGFgHYWDcbwN70/dfYBj47jz9qbsMId/iRN3ewGhXQFZCSftd1LZCfmhktB"
    crossorigin="anonymous">
```
[bootstrap.html](Source%20Code\src1\bootstrap.html)

Bootstrap uses a column-based model where every row in a website is divided into 12 individual columns, and different elements can be alloted a different number of columns to fill.

Columns and rows are referenced in HTML with `class="row"` and `class="col-3" attributes, where the number after `col-` is the number of columns the element should use.

Elements can take up a different number of columns based on the size of the screen with attributes like `class="col-lg-3 col-sm-6"`: use 6 columns on a small screen, but on a large screen only use 3 ([columns1.html](Source%20Code\src1\columns1.html)).

Bootstrap has a whole host of other components which can be easily applied with the appropriate `class` attribute to an element. Bootstrap's [documentation page](https://getbootstrap.com/docs/) gives an extensive list.

## Sass

> **Sass** is an entirely new language built on top of CSS which gives a little more power and flexibility when designing CSS stylesheets and allows us to generate stylesheets programmatically.

In order to use Sass, we must first install it ([here](http://sass-lang.com/install)). Once installed we can execute `sass style.scss style.css`, to compile our Sass file (`style.scss`) into `style.css`, which can be linked to an HTML file.

If recompiling gets annoying, `sass --watch style.scss:style.css` will automatically compile `style.css` when `style.scss` is modified.

Many web deployment systems (such as GitHub Pages), have built in support for Sass. Hence, if a `.scss` file is pushed to GitHub, GitHub Pages will compile it automatically.

#### Variables

One feature of Sass is **variables**, allowing us to pass a value for a CSS property. Making it easier to change a value that has many different entries in our CSS file.

```scss
$color: red;

ul {
    font-size: 14px;
    color: $color;
}

ol {
    font-size: 18px;
    color: $color;
}
```

#### Nesting

Another feature is **nesting**, a more concise way to style elements which are related to other elements in a certain way.

For example, here, all `p` inside `div` will have `color: blue`, but also `font-size: 18px`, while `ul` inside `div` will have `color: green` instead, but still also `font-size: 18px`.

```scss
  div {
      font-size: 18px;
      p {
          color: blue;
      }
      ul {
          color: green;
      }
  }
```

#### Inheritance

One more useful feature is **inheritance**, similar to the OOP concept. Sass' inheritance allows for slight tweaking of a general style for different components.

```scss
%message {
    font-family: sans-serif;
    font-size: 18px;
    font-weight: bold;
}

.specificMessage {
    @extend %message;
    background-color: green;
}
```

`%message` defines a general pattern that can be inherited in other style definitions using the `@extend %message` syntax.



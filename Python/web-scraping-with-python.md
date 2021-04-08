# Web Scraping with Python

- [Web Scraping with Python](#web-scraping-with-python)
  - [Overview](#overview)
    - [HTML](#html)
    - [HTML Tags and Attributes](#html-tags-and-attributes)
    - [XPath](#xpath)
  - [XPaths and Selectors](#xpaths-and-selectors)
    - [Scrapy Selector](#scrapy-selector)
  - [CSS Locators, Chaining and Responses](#css-locators-chaining-and-responses)
    - [Attributes in CSS](#attributes-in-css)
    - [CSS Attributes and Text Selection](#css-attributes-and-text-selection)
    - [Response Objects](#response-objects)
  - [Spiders](#spiders)
    - [Parsing and Crawling](#parsing-and-crawling)
    - [Building an full spider](#building-an-full-spider)

## Overview

Pipeline:

**Setup**
- Define Objective
- Identify Sources

**Acquisition**
- Access Raw Data
- Parse & Extract

**Processing**
- Analyze, wrangle, explore, learn

### HTML

```html
<html>
    <body>
        <div>
            <p>Hello World!</p>
            <p>Enjoy DataCamp!</p>
        </div>
        <p>Thanks for Watching!</p>
    </body>
</html>
```

- Defined by tags e.g. `<body></body>`.
- Tags nested in tree structure.
- Vocabulary defining relation between elements comes from family tree, i.e. moving forward generations, moving between siblings if elements come from the same 'parent' element.

### HTML Tags and Attributes

- Information within HTML tags can be useful.
- May need to extract link URLs.
- Can be an easier way to select elements.

```html
<tag-name attrib-name="attrib info">
    element_contents
</tag-name>
```

Example:
```html
<div id="unique-id" class="some_class">
    div element contents
</div>
```

- **id** attribute should be unique.
- **class** attribute doesn't need to be unique.

```html
<a href="https://www.datacamp.com">
    This text links to DataCamp
</a>
```

- **a** tags are for hyperlinks.
- **href** attribute provides the link to navigate to.


### XPath

```python
xpath = '/html/body/div[2]'
```

- Single forward-slash `/` used to move forward one generation.
- tag-names between slashes give direction to which element(s)
- Brackes `[]` after a tag name denote which sibling is chosen.

```python
xpath = '//table'
xpath = '/html/body/div[2]//table'
```

- Double forward slash `//` looks forward to all future generations, e.g. selecting all table element within HTML document, or selecting all tables within a div element.

```python
xpath = '//div[@id="uid"]'
xpath = '//span[@class="span-class"]'
```

- We can also use brackets to select an element based on one of its attributes.

## XPaths and Selectors

- Single forward slash `/` looks forward one generation.

- Double forward slash `//` looks forward **all** future generations.

- Square brackets select the n<sup>th</sup> element of selected siblings (can return multiple elements, e.g. in the case of `//p[1]` which selects all `<p>` elements which are the n<sup>th</sup> of their siblings).

- Wildcard symbol `*` takes all tag types within selection, e.g. `/html/body/*`.

- `@` represents "**attribute**", e.g. `@class`, `@id`, `@href`.

- Contains function, `contains(@attri-name, "string-expr")` searches for attributes of type `@attri-name` containing "string-expr" as a substring, e.g. `'//a[contains(@class, "course-block")]/@href`

### Scrapy Selector

```python
from scrapy import Selector

sel = Selector(text=html)
```

- We can use `xpath` method of Selector object to create new Selector of spscific pieces of the document.
- This returns a `SelectorList` of `Selector` objects.

```python
sel.xpath("//p")

# outputs the SelectorList:
[<Selector xpath='//p', data='<p>Hello World!</p>'>,
 <Selector xpath='//p', data='<p>Enjoy DataCamp!</p>'>]
 ```

 - The `extract()` method allows us to isolate selected data.

```python
sel.xpath("//p").extract()

out: ['<p>Hello World!</p>',
      '<p>Enjoy DataCamp!</p>']
```

- The `extract_first()` method returns the first element of the list.

```python
sel.xpath("//p").extract_first()

out: '<p>Hello World!</p>'
```

- XPaths may be *chained* by using a period at the start of chained calls.

```python
sel.xpath('/html').xpath('./body').xpath('./div[2]')
```

Example:

```python
# Import a scrapy Selector
from scrapy import Selector

# Import requests
import requests

# Create the string html containing the HTML source
html = requests.get( url ).content

# Create the Selector object sel from html
sel = Selector( text = html )

# Print out the number of elements in the HTML document
print( "There are 1020 elements in the HTML document.")
print( "You have found: ", len( sel.xpath('//*') ) )
```

## CSS Locators, Chaining and Responses

**CSS** - Cascading Style Sheets (styling html documents)

Comparing with XPath selectors:

- `/` replaced by `>` (except first character)
  - XPath: `/html/body/div`
  - CSS Locator: `html > body > div`

- `//` replaced by blank space (except first character)
  - XPath: `//div/span//p`
  - CSS Locator: `div > span p`

- `[N]` replaced by `:nth-of-type(N)`
  - XPath: `//div/p[2]`
  - CSS Locator: `div > p:nth-of-type(2)` 


### Attributes in CSS

- To find element by class, use a period `.`, e.g. `p.class-1`.

- To find an element by id, use a pound sign `#`, e.g. `div#uid`.

### CSS Attributes and Text Selection

**Getting the value of an attribute:**

XPath --> `@attr-name

```python
xpath = '//div[@id="uid"]/a/@href'
```

CSS Locator --> `::attr(attr-name)`

```python
css_locator = 'div#uid > a::attr(href)'
```

**Text Extraction:**

XPath --> `text()`

```python
sel.xpath('//p[@id="p-example"]/text()').extract()
```

CSS Locator --> `::text`

```python
# Text within immediate child
sel.css('p#p-example::text').extract()

# Text within all future generations (blank space before '::text')
sel.css('p#p-example ::text')
```

### Response Objects

Response objects have all the tools we learned with Selectors. `xpath` and `css` methods allow us to query the html document and `extract`, `extract_first` methods allow us to isolate selected data.

A response object also keeps track of the URL that the HTML code was loaded from (`response.url`), and helps us move from one site to another (`response.follow(next_url)`), in order to crawl and scrape the web.

## Spiders

**Creating a Spider:**

```python
import scrapy
from scrapy.crawler import CrawlerProcess

class SpiderClassName(scrapy.Spider):
    name = "spider_name"

    # code for spider
    ...

process = CrawlerProcess()
process.crawl(SpiderClassName)
process.start()
```

**Basic Structure of a Spider:**

```python
class DCSpider(scrapy.Spider):

    name = 'dc_spider'

    def start_requests(self):
        urls = ['https://datacamp.com/courses/all']
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        # simple example: write out html
        html_file = 'DC_courses.html'
        with open(html_file, 'wb') as fout:
            fout.write(response.body)
```

### Parsing and Crawling

**Saving links to file:**

```python
class DCSpider(scrapy.Spider):
   name = "dcspider"

   def start_requests(self):
      urls = ['https://www.datacamp.com/courses/all']
      for url in urls:
         yield scrapy.Request(url=url, callback=self.parse)

   def parse(self, response):
      links = response.css('div.course-block > a::attr(href)').extract()
      filepath = 'DC_links.csv'
      with open(filepath, 'w') as f:
         f.writelines([link + '/n' for link in links])
```

**Following links to parse individual pages**

```python
class DCSpider(scrapy.Spider):
   name = "dcspider"

   def start_requests(self):
      urls = ['https://www.datacamp.com/courses/all']
      for url in urls:
         yield scrapy.Request(url=url, callback=self.parse)

   def parse(self, response):
      links = response.css('div.course-block > a::attr(href)').extract()
      for link in links:
         yield response.follow(url=link, callback=self.parse2)
   
   def parse2(self, response):
      # parse individual course sites
```

The branching structure of being able to follow links within pages and further links within subpages is where the name spider comes from, as we branch out into a web of pages to be scraped.

### Building an full spider

```python
import scrapy
from scrapy.crawler import CrawlerProcess

class DC_Chapter_Spider(scrapy.Spider):
   
   name = "dc_chapter_spider"

   def start_requests(self):
      url = 'https://www.datacamp.com/courses/all'
      yield scrapy.Request(url=url,
                           callback=self.parse_front)

   def parse_front(self.response):
      # Narrow in on the course blocks
      course_blocks = response.css('div.course-block')
      # Direct to course links
      course_links = course_blocks.xpath('./a/@href')
      # Extract links
      links_to_follow = course_links.extract()
      # Follow links to the next parser
      for url in links_to_follow:
         yield response.follow(url=url,
                               callback=self.parse_pages)
   
   def parse_pages(self, response):
      # Direct to the course title
      crs_title = response.xpath('//h1[contains(@class, "title")]/text()')
      # Extract and clean course title text
      crs_title_ext = crs_title.extract_first().strip()
      # Direct to chapter titles
      ch_titles = response.css('h4.chapter__titles::text')
      # Extract and clean the chapter titles text
      ch_titles_ext = [t.strip() for t in ch_titles.extract()]
      # Store data in dictionary
      dc_dict[crs_title_ext] = ch_titles_ext

dc_dict = dict()

process = CrawlerProcess()
process.crawl(DC_Chapter_Spider)
process.start()
```


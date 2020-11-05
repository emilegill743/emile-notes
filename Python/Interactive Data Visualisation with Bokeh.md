# Interactive Data Visualisation With Bokeh

- [Interactive Data Visualisation With Bokeh](#interactive-data-visualisation-with-bokeh)
  - [Basic plotting with Bokeh](#basic-plotting-with-bokeh)
    - [Glyphs](#glyphs)
    - [Glyph properties](#glyph-properties)
    - [Additional glyphs](#additional-glyphs)
      - [**Lines**](#lines)
      - [**Lines and Markers Together**](#lines-and-markers-together)
      - [**Patches**](#patches)
      - [**Other glyphs**](#other-glyphs)
    - [Data formats](#data-formats)
      - [**Numpy Arrays**](#numpy-arrays)
      - [**Pandas**](#pandas)
      - [**Column Data Source**](#column-data-source)
    - [Customising Glyphs](#customising-glyphs)
      - [**Selection appearance**](#selection-appearance)
      - [**Hover appearance**](#hover-appearance)
      - [**Color mapping**](#color-mapping)
  - [Layouts, Interactions, and Annotations](#layouts-interactions-and-annotations)
    - [Arranging multiple plots](#arranging-multiple-plots)
    - [Linking Plots Together](#linking-plots-together)
    - [Linking selection](#linking-selection)
    - [Annotations and Guides](#annotations-and-guides)
  - [Building interactive apps with Bokeh](#building-interactive-apps-with-bokeh)

## Basic plotting with Bokeh

### Glyphs

> **Glyphs** - visual shapes with properties attached to data

**Typical usage**:

```python
from bokeh.io import output_file, show
from bokeh.plotting import figure

plot = figure(plot_width=400, tools='pan,box_zoom')

plot.circle([1,2,3,4,5], [8,6,5,2,3])

output_file = ('circle.html')

show(plot)
```

### Glyph properties

- Lists, arrays, sequences of values
- Single fixed values

```python
plot = figure()
plt.circle(x=10, y=[2,5,8,12], size=[10,20,30,40])
```

**Markers**:
```python
asterisk()
circle()
circle_cross()
circle_x()
cross()
diamond()
diamond_cross()
inverted_triangle()
square()
square_cross()
square_x()
triangle()
x()
```

**Multiple glyphs on a figure**:
```python
# Create the figure: p
p = figure(x_axis_label='fertility (children per woman)', y_axis_label='female_literacy (% population)')

# Add a circle glyph to the figure p
p.circle(fertility_latinamerica, female_literacy_latinamerica)

# Add an x glyph to the figure p
p.x(fertility_africa, female_literacy_africa)

# Specify the name of the file
output_file('fert_lit_separate.html')

# Display the plot
show(p)
```


**Customising scatter plots**:


- **color** - hexadecimal strings, RGB tuples, CSS color names.

- **size** - screen space units (100 => entire figure).

- **alpha** - 0.0 - 1.0 (completely opaque).

```python
p = figure(x_axis_label='fertility (children per woman)', y_axis_label='female_literacy (% population)')

p.circle(fertility_latinamerica, female_literacy_latinamerica, color='blue', size=10, alpha=0.8)

p.circle(fertility_africa, female_literacy_africa, color='red', size=10, alpha=0.8)

output_file('fert_lit_separate_colors.html')
show(p)
```

### Additional glyphs

#### **Lines**

```python
from bokeh.io import output_file, show
from bokeh.plotting import figure

p = figure(x_axis_type='datetime', x_axis_label='Date', y_axis_label='US Dollars')

p.line(date, price, line_width=3)

output_file('line.html')
show(p)
```

#### **Lines and Markers Together**


- Just call glyph methods seperately.
- Glyphs will be drawn in order specified.
```python
from bokeh.io import output_file, show
from bokeh.plotting import figure

x = [1,2,3,4,5]
y = [8,6,5,2,3]

plot = figure()
plot.line(x, y, line_width=3)

plot.circle(x, y, fill_color='white', size=10)

output_file('line.html')
show(plot)
```

#### **Patches**


- Used to draw multiple polygonal shapes at once on a single plot.
- Useful for showing geographic regions.

- Data given as list of list, sublists contain coords for each patch.

```python
from boken.io import output_file, show
from bokeh.plotting import figure

xs = [[1,1,2,2], [2,2,4], [2,2,3,3]]
ys = [[2,5,5,2], [3,5,5], [2,3,4,2]]

plot = figure()

plot.patches(xs, ys,
                fill_color=['red', 'blue', 'green',],
                line_color='white')

output_file('patches.html')
show(plot)
```
        
#### **Other glyphs**

```python
annulus(), annular_wedge(), wedge()

rect(), quad(), vbar(), hbar()

image(), image_rgba(), image_url()

patch(), patches()

line(), multi_line()

circle(), oval(), ellipse()

arc(), quadratic(), bezier()
```

### Data formats

#### **Numpy Arrays**

```python
from bokeh.io import output_file, show
from bokeh.plotting import figure
import numpy as np

x = np.linspace(0,10,1000)
y = np.sin(x) + np.random.random(1000) * 0.2

plot = figure()
plot.line(x,y)

output_file('numpy.html')
show(plot)
```

#### **Pandas**

```python
from bokeh.io import output_file, show
from bokeh.plotting import figure
from bokeh.sampledata.iris import flowers

plot.figure()

plot.circle(flowers['petal_length'],
            flowers['sepal_length'],
            size=10)

output_file('pandas.html')
show(plot)
```

#### **Column Data Source**

- Common fundamental data structure for Bokeh
- Maps column names to sequences of data
- Often created automatically for you
- Can be shared between glyphs to link selections
- Extra columns can be used with hover tooltips

```python
from bokeh.models import ColumnDataSource

source = ColumnDataSource(data={
                            'x' : [1,2,3,4,5],
                            'y' : [8,6,5,2,3]})

source = ColumnDataSource(df)
```

Example:

```python
from bokeh.plotting import ColumnDataSource

source = ColumnDataSource(df)

p.circle(x='Year', y='Time', color='color', size=8, source=source)

output_file('sprint.html')
show(p)
```

### Customising Glyphs

#### **Selection appearance**

- box_select : Allows selection of point by drawing rectangular region
- lasso_select : Allows selection by drawing free-form curve

```python
plot = figure(tools='box_select, lasso_select')

plot.circle(petal_length, sepal_length,
            selection_color='red',
            nonselection_fill_alpha=0.2,
            nonselection_fill_color='grey')
```

Example:

```python
p = figure(x_axis_label='Year',
        y_axis_label='Time',
        tools='box_select')

p.circle(x='Year', y='Time',
            source=source, selection_color='red',
            nonselection_alpha=0.1)

output_file('selection_glyph.html')
show(p)
```

#### **Hover appearance**

```python
from bokeh.models import HoverTool

hover = HoverTool(tooltips=None, mode='hline')

plot = figure(tools=[hover, 'crosshair'])
plot.circle(x, y, size=15, hover_color='red')
```

Example

```python
from bokeh.models import HoverTool

p.circle(x=x, y=y, size=10,
        fill_color='grey', alpha=0.1, line_color=None,
        hover_fill_color='firebrick', hover_alpha=0.5,
        hover_line_color='white')

hover = HoverTool(tooltips=None, mode='vline')

p.add_tools(hover)

output_file('hover_glyph.html')
show(p)
```


#### **Color mapping**

```python
from bokeh.models import CategoricalColorMapper

mapper = CategoricalColorMapper(
            factors=['setosa', 'virginica',
                        'versicolor'],
            pallete = ['red', 'green', 'blue'])

plot = figure(x_axis_label='petal length',
                y_axis_label='sepal length')

plot.circle('petal_length', 'sepal_length',
            size=10, source=source,
            color={'field' : 'species',
                    'transform' : mapper})
```

Example

```python
from bokeh.models import CategoricalColorMapper

source = ColumnDataSource(df)

color_mapper = CategoricalColorMapper(factors=['Europe', 'Asia', 'US'],
                                    palette=['red', 'green', 'blue'])

p.circle('weight', 'mpg', source=source,
            color=dict(field='origin', transform=color_mapper),
            legend='origin')

output_file('colormap.html')
show(p)
```

## Layouts, Interactions, and Annotations

### Arranging multiple plots

**Rows, Columns**:
- Rows
```python
from bokeh.layouts import row

layout = row(p1, p2, p3)

output_file('row.html')
show(layout)
```

- Columns
```python
from bokeh.layouts import column

layout = column(p1, p2, p3)

output_file('column.html')
show(layout)
```

- Nested
```python
from bokeh.layouts import column, row

layout = row(column(p1, p2), p3)

output_file('nested.html')
show(layout)

from bokeh.layouts import row, column

row2 = row([mpg_hp, mpg_weight], sizing_mode='scale_width')
layout = column([avg_mpg, row2], sizing_mode='scale_width')

output_file('layout_custom.html')
show(layout)
```

**Grid arrangements**:

```python
from bokeh.layouts import gridplot

layout = gridplot([None, p1], [p2, p3],
toolbar_location=None)

output_file = ('nested.html')
show(layout)
```

**Tabbed layouts**:

```python
from bokeh.models.widgets import Tabs, Panel

first = Panel(child=row(p1, p2), title='first')
second = Panel(child=row(p3), title='second')

tabs = Tabs(tabs=[first, second])

output_file('tabbed.html')
show(layout)
```

### Linking Plots Together

**Linking axes**:
```python
p3.x_range = p2.x_range = p1.x_range

p3.y_range = p2.y_range = p1.y_range
```

### Linking selection

Shared data source => Linked selections

```python
p1 = figure(title='petal length vs. sepal length')
p1.circle('petal_length', 'sepal length',
            color='blue', source=source)

p2 = figure(title='petal langth vs. sepal width')     
p2.circle('petal length', 'sepal width,
            color='green', source=source)

p3 = figure(title='petal length vs. petal width')
p3.circle('petal length', 'petal_width',
            line_color='red', fill_color=None
            source=source)
```

            
### Annotations and Guides

- Axes, Grids - help relate scale information to the viewer

- Legends - explain visual encodings that are used.

```python
plot.circle('petal_length', 'sepal_length',
            size=10, source=source,
            color={'field' : 'species', 
                    'transform' : mapper},
            legend='species')

plot.legend.location = 'top_left'
plot.legend.background_fill_color = 'lightgrey'
```

- Hover Tooltips - drill down into details not visible in plot.

```python
from bokeh.models import HoverTool

hover = HoverTool(tooltips=[
        ('label name', '@values'
        ('species name', '@species'),
        ('petal length', '@petal_length'),
        ('sepal length', '@sepal_length')
        ])

plot = figure(tools=[hover, 'pan', 'wheel_zoom'])
```

## Building interactive apps with Bokeh

    # Introducing the Bokeh Server

        # Basic App Outline

            from bokeh.io import curdoc

            # Create plots and widgets

            # Add callbacks - functions which are automatically run in response to some event

            # Arrange plots and widgets in layouts

            curdoc().add_root(layout)

            # Running single module apps from cmd 
                bokeh serve --show myapp.py
            # "Directory" style apps 
                bokeh serve --show myappdir/ # Allows data files, themes, html templates 

    # Adding sliders
            # Adding a single slider

                from bokeh.io import curdoc
                from bokeh.layouts import widgetbox
                from bokeh.models  import Slider

                slider = Slider(title='my slider', start=0, end=10, step=0.1, value=2)

                layout = widgetbox(slider)

                curdoc().add_root(layout)

            # Adding multiple sliders

                from bokeh.io import curdoc
                from bokeh.layouts import widgetbox
                from bokeh.models import Slider

                slider1 = Slider(title='slider1', start=0, end=10, step=0.1, value=2)
                slider2 = Slider(title='slider2', start=10, end=100, step=1, value=20)

                layout = widgetbox(slider1, slider2)

                curdoc().add_root(layout)

            # Connecting Sliders to Plots

                from bokeh.io import curdoc
                from bokeh.layouts import column
                from bokeh.models import ColumnDataSource, Slider
                from bokeh.plotting import figure
                from numpy.random import random
                
                N = 300
                source = ColumnDataSource(data={'x' : random(N),
                                                'y' : random(N)})

                plot = figure()
                plot.circle(x='x', y='y', source=source)

                slider = Slider(start=100, end=1000, value=N,
                                step=10, title='Number of points')

                def callback(attr, old, new):
                    N = slider.value
                    source.data={'x' : random(N),
                                 'y' : random(N)}

                slider.on_change('value', callback)

                layout = column(slider, plot)

                curdoc.add_root(layout)

    # Updating plots from dropdowns

        from bokeh.io import curdoc
        from bokeh.layouts import column
        from bokeh.models import ColumnDataSource, Select
        from bokeh.plotting import figure
        from numpy.random import random, normal, lognormal

        N = 1000
        source = ColumnDataSource(data={'x' : random(N),
                                        'y' : random(N)})

        plot.figure()
        plot.circle(x='x', y='y', source=source)

        menu = Select(options=['uniform', 'normal', 'lognormal'],
                      value='uniform', title='Distribution')

        def callback(attr, old, new):
            if menu.value == 'uniform' : f = random
            elif menu.value == 'normal' : f = normal
            else: f = lognormal
            source.data={'x' : f(size=N), 'y' : f(size=N)}

        menu.on_change('value', callback)

        layout = column(menu, plot)

        curdoc().add_root(layout)
    
    # Synchronising two dropdowns

        select1 = Select(title='First', options=['A', 'B'], value='A')
        select2 = Select(title='Second', options=['1', '2', '3'], value='1')

        def callback(attr, old, new):

            if select1.value == 'A':
                select2.options = ['1', '2', '3']

                select2.value = '1'
            else:
                select2.options = ['100', '200', '300']

                select2.value = '100'

        select1.on_change('value', callback)

        layout = widgetbox(select1, select2)
        curdoc().add_root(layout)

    # Buttons

        from bokeh.models import Button

        button = Button(label='press me')

        def update():
            # Do something interesting
        
        button.on_click(update)

        # Button types

            from bokeh.models import CheckboxGroup, RadioGroup, Toggle

            toggle = Toggle(label='Some on/off', button_type='success')

            checkbox = CheckboxGroup(labels=['foo', 'bar', 'baz'])

            radio = RadioGroup(labels=['2000', '2010', '2020'])

            def callback(active)
                # Active tells which button is active

            curdoc().add_root(widgetbox(toggle, checkbox, radio))

'''Case study'''

    # EDA

        from bokeh.io import output_file, show
        from bokeh.plotting import figure
        from bokeh.models import HoverTool, ColumnDataSource

        source = ColumnDataSource(data={
            'x'       : data.loc[1970].fertility,
            'y'       : data.loc[1970].life,
            'country' : data.loc[1970].Country,
        })

        p = figure(title='1970', x_axis_label='Fertility (children per woman)', y_axis_label='Life Expectancy (years)',
                plot_height=400, plot_width=700,
                tools=[HoverTool(tooltips='@country')])

        p.circle(x='x', y='y', source=source)

        output_file('gapminder.html')
        show(p)
    
    # Starting a Basic App

        # Creating plot
            # Import the necessary modules
            from bokeh.io import curdoc
            from bokeh.models import ColumnDataSource
            from bokeh.plotting import figure

            # Make the ColumnDataSource: source
            source = ColumnDataSource(data={
                'x'       : data.loc[1970].fertility,
                'y'       : data.loc[1970].life,
                'country'      : data.loc[1970].Country,
                'pop'      : (data.loc[1970].population / 20000000) + 2,
                'region'      : data.loc[1970].region,
            })

            # Save the minimum and maximum values of the fertility column: xmin, xmax
            xmin, xmax = min(data.fertility), max(data.fertility)

            # Save the minimum and maximum values of the life expectancy column: ymin, ymax
            ymin, ymax = min(data.life), max(data.life)

            # Create the figure: plot
            plot = figure(title='Gapminder Data for 1970', plot_height=400, plot_width=700,
                        x_range=(xmin, xmax), y_range=(ymin, ymax))

            # Add circle glyphs to the plot
            plot.circle(x='x', y='y', fill_alpha=0.8, source=source)

            # Set the x-axis label
            plot.xaxis.axis_label ='Fertility (children per woman)'

            # Set the y-axis label
            plot.yaxis.axis_label = 'Life Expectancy (years)'

            # Add the plot to the current document and add a title
            curdoc().add_root(plot)
            curdoc().title = 'Gapminder'

        # Enhancing with shading

            # Make a list of the unique values from the region column: regions_list
            regions_list = data.region.unique().tolist()

            # Import CategoricalColorMapper from bokeh.models and the Spectral6 palette from bokeh.palettes
            from bokeh.models import CategoricalColorMapper
            from bokeh.palettes import Spectral6

            # Make a color mapper: color_mapper
            color_mapper = CategoricalColorMapper(factors=regions_list, palette=Spectral6)

            # Add the color mapper to the circle glyph
            plot.circle(x='x', y='y', fill_alpha=0.8, source=source,
                        color=dict(field='region', transform=color_mapper), legend='region')

            # Set the legend.location attribute of the plot to 'top_right'
            plot.legend.location = 'top_right'

            # Add the plot to the current document and add the title
            curdoc().add_root(plot)
            curdoc().title = 'Gapminder'

        # Adding a slider

            # Import the necessary modules
            from bokeh.layouts import widgetbox, row
            from bokeh.models import Slider

            # Define the callback function: update_plot
            def update_plot(attr, old, new):
                # Assign the value of the slider: yr
                yr = slider.value
                # Set new_data
                new_data = {
                    'x'       : data.loc[yr].fertility,
                    'y'       : data.loc[yr].life,
                    'country' : data.loc[yr].Country,
                    'pop'     : (data.loc[yr].population / 20000000) + 2,
                    'region'  : data.loc[yr].region,
                }
                # Assign new_data to: source.data
                source.data = new_data

                # Add title to figure: plot.title.text
                plot.title.text = 'Gapminder data for %d' % yr

            # Make a slider object: slider
            slider = Slider(start=1970, end=2010, step=1, value=1970, title='Year')

            # Attach the callback to the 'value' property of slider
            slider.on_change('value', update_plot)

            # Make a row layout of widgetbox(slider) and plot and add it to the current document
            layout = row(widgetbox(slider), plot)
            curdoc().add_root(layout)

        # Adding a hover tool

            # Import HoverTool from bokeh.models
            from bokeh.models import HoverTool

            # Create a HoverTool: hover
            hover = HoverTool(tooltips=[('Country', '@country')])

            # Add the HoverTool to the plot
            plot.add_tools(hover)

            # Create layout: layout
            layout = row(widgetbox(slider), plot)

            # Add layout to current document
            curdoc().add_root(layout)
        
        # Adding dropdowns

            # Define the callback: update_plot
            def update_plot(attr, old, new):
                # Read the current value off the slider and 2 dropdowns: yr, x, y
                yr = slider.value
                x = x_select.value
                y = y_select.value
                # Label axes of plot
                plot.xaxis.axis_label = x
                plot.yaxis.axis_label = y
                # Set new_data
                new_data = {
                    'x'       : data.loc[yr][x],
                    'y'       : data.loc[yr][y],
                    'country' : data.loc[yr].Country,
                    'pop'     : (data.loc[yr].population / 20000000) + 2,
                    'region'  : data.loc[yr].region,
                }
                # Assign new_data to source.data
                source.data = new_data

                # Set the range of all axes
                plot.x_range.start = min(data[x])
                plot.x_range.end = max(data[x])
                plot.y_range.start = min(data[y])
                plot.y_range.end = max(data[y])

                # Add title to plot
                plot.title.text = 'Gapminder data for %d' % yr

            # Create a dropdown slider widget: slider
            slider = Slider(start=1970, end=2010, step=1, value=1970, title='Year')

            # Attach the callback to the 'value' property of slider
            slider.on_change('value', update_plot)

            # Create a dropdown Select widget for the x data: x_select
            x_select = Select(
                options=['fertility', 'life', 'child_mortality', 'gdp'],
                value='fertility',
                title='x-axis data'
            )

            # Attach the update_plot callback to the 'value' property of x_select
            x_select.on_change('value', update_plot)

            # Create a dropdown Select widget for the y data: y_select
            y_select = Select(
                options=['fertility', 'life', 'child_mortality', 'gdp'],
                value='life',
                title='y-axis data'
            )

            # Attach the update_plot callback to the 'value' property of y_select
            y_select.on_change('value', update_plot)

            # Create layout and add to current document
            layout = row(widgetbox(slider, x_select, y_select), plot)
            curdoc().add_root(layout)

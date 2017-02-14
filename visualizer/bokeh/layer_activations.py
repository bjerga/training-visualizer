from bokeh.io import curdoc
from bokeh.layouts import column
from bokeh.models import Div
from bokeh.plotting import figure

document = curdoc()

color = 'black'
_from = 0
to = 10

# Create a polynomial line graph with those arguments
x = list(range(_from, to + 1))
fig = figure(title="Polynomial")
fig.line(x, [i ** 2 for i in x], color=color, line_width=2)

div = Div(text="<h3>Visualization of the layer activations</h3>", width=500)

document.add_root(column(div, fig))

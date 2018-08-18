import pickle
import random
import pandas as pd
from util import display
from flask import Flask, request
from flask import render_template, flash, redirect, url_for
from bokeh.models import (HoverTool, FactorRange, Plot, LinearAxis, Grid, Range1d)
from bokeh.models.glyphs import VBar
from bokeh.plotting import figure
# from bokeh.charts import Bar
from bokeh.embed import components
from bokeh.models.sources import ColumnDataSource
from flask import Flask, render_template
from pymongo import MongoClient
client = MongoClient()
app = Flask(__name__)


@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html', title='Home')

@app.route('/predict', methods=['POST'] )
def predict():
    state = str(request.form['state'])
    bus_name = str(request.form['restaurant'])
    star1 = str(request.form['star1'])
    star2 = str(request.form['star2'])
    star1, star2 = int(star1), int(star2)
    affix = None
    if star1 == 1 and star2 == 2:
        affix = "12"
    if star1 == 1 and star2 == 3:
        affix = "13"
    if star1 == 1 and star2 == 4:
        affix = "14"
    if star1 == 1 and star2 == 5:
        affix = "15"
    if star1 == 2 and star2 == 5:
        affix = "25"
    if star1 == 3 and star2 == 5:
        affix = "35"
    if star1 == 4 and star2 == 5:
        affix = "45"
    if affix:
        with open("../break_week/models/model"+affix+".pickle", "rb") as f:
            models = pickle.load(f)
    db = client.yelp
    df = pd.DataFrame(list(db.review.find({"bus_name": bus_name, "state":state})))

    df_pos, df_neg, name, size, Recall, Precision, Accuracy, lst_neg, \
                    lst_pos = display(models, df, star1, star2, state=state, bus_name=bus_name)

    # hover = create_hover_tool()
    data_neg = {"Aspect": list(df_neg["Aspect"]), "Level of experience": list(df_neg["Level of experience"])}
    data_pos = {"Aspect": list(df_pos["Aspect"]), "Level of experience": list(df_pos["Level of experience"])}

    plot_neg = create_bar_chart(data_neg, "Negative customer experience", "Aspect", "Level of experience")
    plot_pos = create_bar_chart(data_pos, "Positive customer experience", "Aspect", "Level of experience")

    script_neg, div_neg = components(plot_neg)
    script_pos, div_pos = components(plot_pos)
    if len(lst_pos) >= 4 and len(lst_neg) >= 4:
        neg_asp1, neg_asp2,neg_asp3,neg_asp4 = lst_neg[0][0], lst_neg[1][0], lst_neg[2][0], lst_neg[3][0]
        neg_desc1, neg_desc2, neg_desc3, neg_desc4 = lst_neg[0][1], lst_neg[1][1], lst_neg[2][1], lst_neg[3][1]
        pos_asp1, pos_asp2, pos_asp3, pos_asp4 = lst_pos[0][0], lst_pos[1][0], lst_pos[2][0], lst_pos[3][0]
        pos_desc1, pos_desc2, pos_desc3, pos_desc4 = lst_pos[0][1], lst_pos[1][1], lst_pos[2][1], lst_pos[3][1]
    else:
        return render_template('index.html', title='Home')
    return render_template("predict.html", bus_name=bus_name, neg_the_div=div_neg, neg_the_script=script_neg,
                           pos_the_div=div_pos, pos_the_script=script_pos,
                           neg_asp1=neg_asp1, neg_asp2=neg_asp2, neg_asp3=neg_asp3, neg_asp4=neg_asp4,
                           neg_desc1=neg_desc1, neg_desc2=neg_desc2, neg_desc3=neg_desc3, neg_desc4=neg_desc4,
                           pos_asp1=pos_asp1, pos_asp2=pos_asp2, pos_asp3=pos_asp3, pos_asp4=pos_asp4,
                           pos_desc1=pos_desc1, pos_desc2=pos_desc2, pos_desc3=pos_desc3, pos_desc4=pos_desc4)




def create_hover_tool():
    """Generates the HTML for the Bokeh's hover data tool on our graph."""
    hover_html = """
          <div>
            <span class="hover-tooltip">$x</span>
          </div>
          <div>
            <span class="hover-tooltip">@bugs bugs</span>
          </div>
          <div>
            <span class="hover-tooltip">$@costs{0.00}</span>
          </div>
        """
    return HoverTool(tooltips=hover_html)


def create_bar_chart(data, title, x_name, y_name, hover_tool=None,
                     width=1500, height=500):
    """Creates a bar chart plot with the exact styling for the centcom
       dashboard. Pass in data as a dictionary, desired plot title,
       name of x axis, y axis and the hover tool HTML.
    """
    source = ColumnDataSource(data)
    xdr = FactorRange(factors=data[x_name])
    ydr = Range1d(start=0,end=max(data[y_name])*1.5)

    tools = []
    if hover_tool:
        tools = [hover_tool,]

    plot = figure(title=title, x_range=xdr, y_range=ydr, plot_width=width,
                  plot_height=height, h_symmetry=False, v_symmetry=True,
                  min_border=0, toolbar_location="above", tools=tools,
                  responsive=True, outline_line_color="#666666")

    glyph = VBar(x=x_name, top=y_name, bottom=0, width=.1,
                 fill_color="#e12127")
    plot.add_glyph(source, glyph)

    xaxis = LinearAxis()
    yaxis = LinearAxis()

    plot.add_layout(Grid(dimension=0, ticker=xaxis.ticker))
    plot.add_layout(Grid(dimension=1, ticker=yaxis.ticker))
    plot.toolbar.logo = None
    plot.min_border_top = 0
    plot.xgrid.grid_line_color = None
    plot.ygrid.grid_line_color = "#999999"
    plot.yaxis.axis_label = "Number of reports"
    plot.ygrid.grid_line_alpha = 1
    plot.xaxis.axis_label = "Aspects of customer experience"
    plot.xaxis.major_label_orientation = 1
    return plot

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)

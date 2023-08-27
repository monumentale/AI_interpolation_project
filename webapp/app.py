import dash
from dash import dcc
import dash_bootstrap_components as dbc
from dash import html
import pandas as pd
import plotly.express as px


LOGO = "https://geoinfomatics.unn.edu.ng/wp-content/uploads/sites/155/2018/05/land-survey.png"
# Load the data
df = pd.read_csv('static/NG_mobile coverage (1).csv')

# Initialize the Dash app
app = dash.Dash(__name__, 
                external_stylesheets=[dbc.themes.LUX],
                title='AI interpolation techniques | Network Coverage Map',
                meta_tags=[{'name': 'viewport',
                            'content': 'width=device-width, height=device-height, initial-scale=1.0, maximum-scale=1.2'}]
                )
###################################################
#############  Header /Navigation bar #############
###################################################
navbar = dbc.Navbar(
                [
                html.A(
                        # Use row and col to control vertical alignment of logo / brand
                        dbc.Row(
                                [   dbc.Col(html.Img(src=LOGO, height="30px")),
                                    dbc.Col(dbc.NavbarBrand("| Osita's Nigeria Network Coverage Map", className="ml-2")),
                                ],
                                align="center",
                                # no_gutters=True,
                                ),
                        href="https://www.geomaap.io/about",
                    ),
                ],
                color="dark",
                dark=True,
                className="mb-4",
)


credits_tab = dbc.Card(
    dbc.CardBody(
        dcc.Markdown(
            """
            Made with love with Dash by python Artifical Intelligence Group  
            Emeka Jessica,
            Kalu David-dien,
            Olajumoke Adebayo,
            Osita Osita Stephen,
      
            """
        ),
    ),
    className="mt-0",
)

Longitude_Input=dbc.Input(value=7.352275,placeholder="input Longitude",type="number",id='longi')
Latitude_Input=dbc.Input(value=5.111047,type="number",placeholder="input Latitude",id='lati', )


app.layout = dbc.Container(
    [
        navbar,
        dbc.Row(
            [
                
                dbc.Col(  # left layout
                    
                    [
                    credits_tab,
                     dbc.CardHeader("Search For A Location"),
                    Latitude_Input,
                    Longitude_Input,
                    dbc.CardHeader("Select the network type from the dropdown"),
                    dcc.Dropdown(
        id='network-type-dropdown',
        options=[
            {'label': '2G', 'value': 'percent_2g'},
            {'label': '3G', 'value': 'percent_3g'},
            {'label': '4G', 'value': 'percent_4g'}
        ],
        value='percent_4g'
    )
                    ], 
                        width=3),

                dbc.Col( # right layout
                    [
                   dcc.Graph(id='coverage-map')
                    ],
                    width=9,
                ),
            ]
        ),
    ],
    fluid=True,
)




# Define the callback function to update the graph
@app.callback(
    dash.dependencies.Output('coverage-map', 'figure'),
    [dash.dependencies.Input('network-type-dropdown', 'value'),
     dash.dependencies.Input('longi', 'value'),
     dash.dependencies.Input('lati', 'value'),
     ]
)
def update_graph(network_type,longitude,latitude):
    # Create the coverage map using Plotly Express
    fig = px.scatter_mapbox(df, lat='latitude', lon='longitude', color=network_type, mapbox_style="carto-positron",
                  color_continuous_scale=px.colors.cyclical.IceFire, size_max=15, zoom=10,
        #           center=dict(
        #     lat=latitude,
        #     lon=-longitude
        # )
        )
    fig.update_layout(height=800)
    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)

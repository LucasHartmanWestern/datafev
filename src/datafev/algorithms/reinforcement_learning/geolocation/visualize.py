import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def read_excel_data(file_path, sheet_name):
    """Retrieve the data from the excel file and parse it as needed"""
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    return df

def read_csv_data(file_path):
    df = pd.read_csv(file_path)
    return df

def generate_plot(df):
    """Visualize the data using a graph.
    The graph connects the coordinates in the Latitude and Longitude columns
    and plots the coordinates in the various (CX_Latitude, CX_Longitude) columns."""

    # Initialize the plot
    plt.figure(figsize=(10, 8))

    # Plot main path
    plt.plot(df['Longitude'], df['Latitude'], 'ro-', label='Main Path')

    # Plot additional paths
    for i in range(1, 11):
        lat_key = f'C{i}_Latitude'
        lon_key = f'C{i}_Longitude'
        if lat_key in df.columns and lon_key in df.columns:
            plt.plot(df[lon_key], df[lat_key], 'o-', label=f'Charger {i}')

    # Add legend, grid, title and labels
    plt.legend()
    plt.grid(True)
    plt.title('Paths')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    # Show the plot
    plt.show()

def generate_interactive_plot(df, origin, destination):
    """Visualize the data using an interactive graph.
    The graph connects the coordinates in the Latitude and Longitude columns
    and plots the coordinates in the various (CX_Latitude, CX_Longitude) columns."""

    # Initialize the plot
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=[origin[1]],
        y=[origin[0]],
        mode='markers',  # add markers here
        name='Origin'
    ))
    fig.add_trace(go.Scatter(
        x=[destination[1]],
        y=[destination[0]],
        mode='markers',  # add markers here
        name='Destination'
    ))

    count = 0
    for dataset in df:
        count += 1
        # Plot main path
        fig.add_trace(go.Scatter(
            x=dataset['Longitude'],
            y=dataset['Latitude'],
            mode='markers+lines',
            name=f'Path {count}',
            customdata=dataset[['Episode Num', 'Action', 'SoC', 'Is Charging']].values.tolist(),
            hovertemplate='Episode: %{customdata[0]}<br>Action: %{customdata[1]}<br>SoC: %{customdata[2]}kW<br>Charging: %{customdata[3]}<br>Lat: %{y}<br>Lon: %{x}'
        ))

    # Plot additional paths
    for i in range(1, 11):
        lat_key = f'Charger {i} Latitude'
        lon_key = f'Charger {i} Longitude'
        if lat_key in df[0].columns and lon_key in df[0].columns:
            fig.add_trace(go.Scatter(
                x=[df[0].loc[0][lon_key]],
                y=[df[0].loc[0][lat_key]],
                mode='markers',  # add markers here
                name=f'Charger {i}'
            ))

    # Add labels and title
    fig.update_layout(
        title='Paths',
        xaxis_title='Longitude',
        yaxis_title='Latitude'
    )

    # Show the plot
    fig.show()
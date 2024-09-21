import plotly.graph_objects as go
import numpy as np
from data_processing import extract_data, nearest_neighbor_reduction
import os
from scipy.interpolate import Rbf
import pickle
import pyvista as pv
from tqdm import tqdm

# Extract the rh_2 data.
x_2 = []
y_2 = []
z_2 = []
rh_2 = []

# Extract the rh_98 data.
x_98 = []
y_98 = []
z_98 = []
rh_98 = []

scale_factor = 1000 # Adjust the scale factor as needed.

# Check if the data has been processed before.
if os.path.exists(f'pickles/pickle_scale_{scale_factor}.pkl'):
    print("Data found. Loading data...")
    print("Scale factor: ", scale_factor)
    with open(f'pickles/pickle_scale_{scale_factor}.pkl', 'rb') as f:
        rh_data = pickle.load(f)
        x_2 = np.array(rh_data['x_2'])
        y_2 = np.array(rh_data['y_2'])
        z_2 = np.array(rh_data['z_2'])
        x_98 = np.array(rh_data['x_98'])
        y_98 = np.array(rh_data['y_98'])
        z_98 = np.array(rh_data['z_98'])
        rh_98 = np.array(rh_data['rh_98'])
        rh_2 = np.array(rh_data['rh_2'])

# If the data has not been processed before, process the data.
else:
    print("Data not found. Processing data...")

    # Get all the paths to the data files.
    paths = []
    for folder in os.scandir("data"):
        if folder.is_dir():
            for file in os.scandir("data/" + folder.name):
               if file.name.endswith(".h5"):
                    paths.append("data/" + folder.name + "/" + file.name)

    # Define the beams to be used.
    beams = ['BEAM0110']

    # Process the data.
    for path in tqdm(paths, desc='Processing Files'):
        for beam in beams:
            beam_data = extract_data(path, scale_factor=scale_factor, beam_key=beam)
        
            for key in beam_data:
                x_2.append(beam_data[key]['rh2']['x'])
                y_2.append(beam_data[key]['rh2']['y'])
                z_2.append(beam_data[key]['rh2']['z'])
                rh_2.append(beam_data[key]['rh2']['rh'])

                x_98.append(beam_data[key]['rh98']['x'])
                y_98.append(beam_data[key]['rh98']['y'])
                z_98.append(beam_data[key]['rh98']['z'])
                rh_98.append(beam_data[key]['rh98']['rh'])

    # Convert to np arrays.
    x_2 = np.array(x_2)
    y_2 = np.array(y_2)
    z_2 = np.array(z_2)
    x_98 = np.array(x_98)
    y_98 = np.array(y_98)
    z_98 = np.array(z_98)
    rh_98 = np.array(rh_98)
    rh_2 = np.array(rh_2)

    print('Data processing complete.')
    print(f'{rh_2.shape[0]} data points processed.')

    # Save the data.
    rh_data = {'x_2': x_2,
        'y_2': y_2,
        'z_2': z_2,
        'x_98': x_98,
        'y_98': y_98,
        'z_98': z_98,
        'rh_98': rh_98,
        'rh_2': rh_2}
    
    with open(f'pickles/pickle_scale_{scale_factor}.pkl', 'wb') as f:
        pickle.dump(rh_data, f)
        print('Data saved.')

# Use Nearest Neighbors to decrease the number of points.
print('Reducing the number of points...')
x_98, y_98, z_98, rh_98 = nearest_neighbor_reduction(x_98, y_98, z_98, scale_factor, 0.0025) 
x_2, y_2, z_2, rh_2 = nearest_neighbor_reduction(x_2, y_2, z_2, scale_factor, 0.0015)

print('Building 3D Scatterplot Visualization...')
# Create a 3D scatter plot.
fig = go.Figure(data=[
    go.Scatter3d(
        x=x_2,
        y=y_2,
        z=z_2,
        mode='markers',
        marker=dict(
            size=5,
            color='black', 
            opacity=0.8
        ),
        name='RH_2 Data'
    ),
    go.Scatter3d(
        x=x_98,
        y=y_98,
        z=z_98,
        mode='markers',
        marker=dict(
            size=5,
            color=rh_98,  
            colorscale='Viridis',
            opacity=0.75
        ),
        name='RH_98 Data'
    ),
    #Add the Origin if needed.
    go.Scatter3d(
        x=[0], y=[0], z=[0], 
        mode='markers',
        marker=dict(
            size=7,
            color='black', 
        ),
        name='Origin'
    )
])

# Update plot layout.
fig.update_layout(
    title='3D Visualization of RH Data',
    scene=dict(
        xaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='blue'),
        yaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='blue'),
        zaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='blue')
    )
)

# Show point-cloud plot.
fig.show()

# Wait for user input (to reduce gpu usage).
input('Press Enter to continue...')

print('Building 3D Visualization with PyVista...')
print('Interpolating data...')

# Interpolate the data with gaussian for more local interpolation.
rbf_98 = Rbf(x_98, y_98, z_98, rh_98, function='gaussian', smooth=0.01)
rbf_2 = Rbf(x_2, y_2, z_2, rh_2, function='gaussian', smooth=0.01)

print('Creating xyz grid...')
grid_x_98, grid_y_98, grid_z_98 = np.mgrid[x_98.min():x_98.max():25j, y_98.min():y_98.max():25j, z_98.min():z_98.max():25j]
grid_x_2, grid_y_2, grid_z_2 = np.mgrid[x_2.min():x_2.max():25j, y_2.min():y_2.max():25j, z_2.min():z_2.max():25j]

print('Generating grid values...')
grid_values_98 = rbf_98(grid_x_98, grid_y_98, grid_z_98)
grid_values_2 = rbf_2(grid_x_2, grid_y_2, grid_z_2)

print('Creating structured grid...')
grid_98 = pv.StructuredGrid(grid_x_98, grid_y_98, grid_z_98)
grid_98.point_data['values'] = grid_values_98.flatten(order='F')

grid_2 = pv.StructuredGrid(grid_x_2, grid_y_2, grid_z_2)
grid_2.point_data['values'] = grid_values_2.flatten(order='F')

print('Creating isosurface...')
isosurfaces_98 = [np.nanmean(grid_values_98)]
isosurface_98 = grid_98.contour(isosurfaces=isosurfaces_98)

isosurfaces_2 = [np.nanmean(grid_values_2)]
isosurface_2 = grid_2.contour(isosurfaces=isosurfaces_2)

# Save to stl files (later convert to obj for Unity).
print('Saving isosurfaces to stl files...')
isosurface_98.save('isosurfaces/isosurface_98.stl')
isosurface_2.save('isosurfaces/isosurface_2.stl')

# Visualize.
print('Visualizing...')
plotter = pv.Plotter()

# Add the isosurfaces.
plotter.add_mesh(isosurface_98, color='green')
plotter.add_mesh(isosurface_2, color='tan')

# Display the grid.
plotter.show_grid()
plotter.show()
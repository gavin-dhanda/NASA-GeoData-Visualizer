import h5py
import numpy as np
from sklearn.neighbors import NearestNeighbors

'''
This function extracts the data from the h5 file and returns a dictionary 
containing the relevant values.

Parameters:
    file_name: str
        The name of the file to extract the data from.
    scale_factor: int
        The scale factor to adjust the Earth's radius for visualization.
    beam_key: str
        The key to the beam to extract the data from.

Returns:
    data_dict: dict
        A dictionary containing the extracted data.
'''
def extract_data(file_name, scale_factor=1, beam_key='BEAM0110'):
    with h5py.File(file_name, 'r') as file:

        # Check that rh is a key in the file:
        if 'rh' not in file[beam_key]:
            print(f'File {file_name} does not contain the key rh.')
            return {}
        # Extract datasets
        rh_data = file[beam_key]['rh'][:]
        lat_highest = file[beam_key]['lat_highestreturn'][:]
        lon_highest = file[beam_key]['lon_highestreturn'][:]
        lat_lowest = file[beam_key]['lat_lowestmode'][:]
        lon_lowest = file[beam_key]['lon_lowestmode'][:]
        quality_flags = file[beam_key]['quality_flag'][:]  # Extract quality flags

        data_dict = {}

        rh2 = rh_data[:, 2]
        rh98 = rh_data[:, 98]
        
        for i in range(len(rh2)):
            if quality_flags[i]:  # Assuming a non-zero flag indicates good quality
                x2, y2, z2 = convert_to_cartesian(lat_lowest[i], lon_lowest[i], rh2[i], scale_factor=scale_factor)
                x98, y98, z98 = convert_to_cartesian(lat_highest[i], lon_highest[i], rh98[i], scale_factor=scale_factor)

                rh2Values = {'x': x2, 'y': y2, 'z': z2, 'rh': rh2[i], 'lat': lat_lowest[i], 'lon': lon_lowest[i]}
                rh98Values = {'x': x98, 'y': y98, 'z': z98, 'rh': rh98[i], 'lat': lat_highest[i], 'lon': lon_highest[i]}

                data_dict[i] = {
                    'rh2': rh2Values,
                    'rh98': rh98Values,
                }

        # Print the datapoints:
        return data_dict

'''
This function uses the latitude, longitude, and RH values to 
convert the data to Cartesian coordinates.

Parameters:
    lat: float
        The latitude value in degrees.
    lon: float
        The longitude value in degrees.
    rh: float
        The RH value in meters.
    scale_factor: int
        The scale factor to adjust the Earth's radius for visualization.

Returns:
    x: float
        The x-coordinate in Cartesian coordinates.
    y: float
        The y-coordinate in Cartesian coordinates.
    z: float
        The z-coordinate in Cartesian coordinates.
'''
def convert_to_cartesian(lat, lon, rh, scale_factor=1):
    # Convert latitude and longitude from degrees to radians
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)

    # Earth's nominal radius in kilometers, adjust with scale_factor
    R = 6371 / scale_factor  # Adjust Earth's radius for visualization purposes

    # Convert RH from meters to kilometers for the calculation
    rh_km = rh / 1000

    # Calculate the Cartesian coordinates
    x = (R + rh_km) * np.cos(lat_rad) * np.cos(lon_rad)
    y = (R + rh_km) * np.cos(lat_rad) * np.sin(lon_rad)
    z = (R + rh_km) * np.sin(lat_rad)

    return x, y, z

''' 
This function reduces the number of points in the data 
by using the nearest neighbor algorithm.

Parameters:
    x: np.array
        The x-coordinates of the data points.
    y: np.array
        The y-coordinates of the data points.
    z: np.array
        The z-coordinates of the data points.
    scale: int
        The scale factor to adjust the Earth's radius for visualization.
    thresh: float
        The threshold value for the nearest neighbor algorithm.

Returns:
    x_new: np.array
    y_new: np.array
    z_new: np.array
    rh_new: np.array
'''
def nearest_neighbor_reduction(x, y, z, scale, thresh):
    data = np.column_stack((x, y, z))
    nn = NearestNeighbors(n_neighbors=100, algorithm='ball_tree').fit(data)
    distances, _ = nn.kneighbors(data)
    cluster_centers = []
    for i in range(data.shape[0]):
        if np.mean(distances[i]) < thresh:
            cluster_centers.append(data[i])

    cluster_centers = np.array(cluster_centers)
    x_new = cluster_centers[:, 0]
    y_new = cluster_centers[:, 1]
    z_new = cluster_centers[:, 2]
    rh_new = np.sqrt(x_new**2 + y_new**2 + z_new**2) - 6371 / scale

    print(f'Number of points before reduction: {x.shape[0]}')
    print(f'Number of points after reduction: {cluster_centers.shape[0]}')
    return x_new,y_new,z_new,rh_new

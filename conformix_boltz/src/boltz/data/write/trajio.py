import torch
import netCDF4 as nc
from datetime import datetime
import numpy as np

def tensor_to_netcdf(output_file: str, coordinates: torch.Tensor, **kwargs):
    """
    Convert a PyTorch tensor of coordinates to AMBER NetCDF trajectory format.
    
    Parameters:
    -----------
    coordinates : torch.Tensor
        Tensor of shape (n_frames, n_atoms, 3) containing atomic coordinates
    output_file : str
        Path to output NetCDF file
    kwargs : dict
        Additional keyword arguments to be written as auxiliary variables
    """
    # Validate input tensor
    if not isinstance(coordinates, torch.Tensor):
        raise TypeError("coordinates must be a PyTorch tensor")
    
    if len(coordinates.shape) != 3 or coordinates.shape[2] != 3:
        raise ValueError(f"coordinates must have shape (n_frames, n_atoms, 3), got {coordinates.shape}")
    
    num_frames, num_atoms, _ = coordinates.shape
    
    # Create NetCDF file
    ncfile = nc.Dataset(output_file, 'w', format='NETCDF3_64BIT')
    
    # Create dimensions
    ncfile.createDimension('frame', num_frames)
    ncfile.createDimension('atom', num_atoms)
    ncfile.createDimension('spatial', 3)
    ncfile.createDimension('label', 5)
    ncfile.createDimension('cell_spatial', 3)
    ncfile.createDimension('cell_angular', 3)
    
    # Create variables
    coords = ncfile.createVariable('coordinates', 'f8', ('frame', 'atom', 'spatial'))
    spatial = ncfile.createVariable('spatial', 'c', ('spatial',))
    cell_spatial = ncfile.createVariable('cell_spatial', 'c', ('cell_spatial',))
    cell_angular = ncfile.createVariable('cell_angular', 'c', ('cell_angular',))
    time = ncfile.createVariable('time', 'f8', ('frame',))
    
    # Set attributes
    ncfile.title = 'Converted from PyTorch tensor'
    ncfile.application = 'PyTorch tensor to AMBER NetCDF converter'
    ncfile.program = 'tensor_to_netcdf'
    ncfile.programVersion = '1.0'
    ncfile.Conventions = 'AMBER'
    ncfile.ConventionVersion = '1.0'
    ncfile.creation_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Initialize variables
    spatial[:] = list('xyz')
    cell_spatial[:] = list('abc')
    cell_angular[:] = list('abc')
    
    # Convert tensor to numpy and store coordinates
    # Move tensor to CPU if it's on GPU
    if coordinates.is_cuda:
        coordinates = coordinates.cpu()
    
    # Convert to numpy array
    coords_np = coordinates.detach().numpy()
    
    # Store coordinates
    ncfile.variables['coordinates'][:] = coords_np
    
    # Set time (assuming 1 ps per frame, adjust as needed)
    ncfile.variables['time'][:] = np.arange(num_frames)
    
    # Write auxiliary variables from kwargs
    for key, value in kwargs.items():
        if isinstance(value, torch.Tensor):
            value = value.detach().numpy()
            
        if isinstance(value, (int, float, str)):
            aux_var = ncfile.createVariable(key, 'f8' if isinstance(value, (int, float)) else 'c', ())
            aux_var.assignValue(value)
        elif isinstance(value, (list, np.ndarray)):
            aux_var = ncfile.createVariable(key, 'f8', ('frame',))
            aux_var[:] = value
        else:
            raise TypeError(f"Unsupported type for auxiliary variable {key}: {type(value)}")
    
    # Close the NetCDF file
    ncfile.close()

# Example usage:
if __name__ == '__main__':
    # Create sample tensor
    sample_coords = torch.randn(10, 100, 3)  # 10 frames, 100 atoms
    tensor_to_netcdf(sample_coords, 'trajectory.nc')

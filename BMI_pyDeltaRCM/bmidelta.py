#! /usr/bin/env python
import yaml
import warnings
warnings.simplefilter('default')
import tempfile

import numpy as np
import os
from bmipy import Bmi

from pyDeltaRCM.model import DeltaModel
from pyDeltaRCM.shared_tools import write_yaml_config_to_file

from . import utils

"""Basic Model Interface implementation for pyDeltaRCM."""


class BmiDelta(Bmi):

    _name = 'pyDeltaRCM'

    _input_var_names = (
        'channel_exit_water_flow__speed',
        'channel_exit_water_x-section__depth',
        'channel_exit_water_x-section__width',
        'channel_exit_water_sediment~bedload__volume_fraction',
        'channel_exit_water_sediment~suspended__mass_concentration',
        'sea_water_surface__rate_change_elevation',
        'sea_water_surface__mean_elevation',
    )

    _output_var_names = (
        'sea_water_surface__elevation',
        'sea_water__depth',
        'sea_bottom_surface__elevation',
    )

    _input_vars = {
        'model_output__out_dir': 'out_dir',
        'model_grid__length': 'Length',
        'model_grid__width': 'Width',
        'model_grid__cell_size': 'dx',
        'land_surface__width': 'L0_meters',
        'land_surface__slope': 'S0',
        'model__max_iteration': 'itermax',
        'water__number_parcels': 'Np_water',
        'channel__flow_velocity': 'u0',
        'channel__width': 'N0_meters',
        'channel__flow_depth': 'h0',
        'sea_water_surface__mean_elevation': 'H_SL',
        'sea_water_surface__rate_change_elevation': 'SLR',
        'sediment__number_parcels': 'Np_sed',
        'sediment__bedload_fraction': 'f_bedload',
        'sediment__influx_concentration': 'C0_percent',
        'model_output__opt_eta_figs': 'save_eta_figs',
        'model_output__opt_stage_figs': 'save_stage_figs',
        'model_output__opt_depth_figs': 'save_depth_figs',
        'model_output__opt_discharge_figs': 'save_discharge_figs',
        'model_output__opt_velocity_figs': 'save_velocity_figs',
        'model_output__opt_eta_grids': 'save_eta_grids',
        'model_output__opt_stage_grids': 'save_stage_grids',
        'model_output__opt_depth_grids': 'save_depth_grids',
        'model_output__opt_discharge_grids': 'save_discharge_grids',
        'model_output__opt_velocity_grids': 'save_velocity_grids',
        'model_output__opt_time_interval': 'save_dt',
        'coeff__surface_smoothing': 'Csmooth',
        'coeff__under_relaxation__water_surface': 'omega_sfc',
        'coeff__under_relaxation__water_flow': 'omega_flow',
        'coeff__iterations_smoothing_algorithm': 'Nsmooth',
        'coeff__depth_dependence__water': 'theta_water',
        'coeff__depth_dependence__sand': 'coeff_theta_sand',
        'coeff__depth_dependence__mud': 'coeff_theta_mud',
        'coeff__non_linear_exp_sed_flux_flow_velocity': 'beta',
        'coeff__sedimentation_lag': 'sed_lag',
        'coeff__velocity_deposition_mud': 'coeff_U_dep_mud',
        'coeff__velocity_erosion_mud': 'coeff_U_ero_mud',
        'coeff__velocity_erosion_sand': 'coeff_U_ero_sand',
        'coeff__topographic_diffusion': 'alpha',
        'basin__opt_subsidence': 'toggle_subsidence',
        'basin__maximum_subsidence_rate': 'sigma_max',
        'basin__subsidence_start_timestep': 'start_subsidence',
        'basin__opt_stratigraphy': 'save_strata'
        }

    def __init__(self):
        """Create a BmiDelta model that is ready for initialization."""
        self._delta = None
        self._values = {}
        self._var_units = {}
        self._var_loc = {}
        self._grids = {}
        self._grid_type = {}

        self._start_time = 0.0
        self._end_time = np.finfo('d').max
        self._time_units = 's'

    def initialize(self, filename=None):
        """Initialize the model.

        Parameters
        ----------
        filename : str, optional
            Path to name of input file. Use all pyDeltaRCM defaults, if not
            provided.
        """

        if filename:
            # translate the BMI YAML file keywords to the pyDeltaRCM keywords,
            # and write to a temporary file.

            # open the input file and bring to dict
            input_file = open(filename, mode='r')
            input_dict = yaml.load(input_file, Loader=yaml.FullLoader)
            input_file.close()

            # create new dict for translated keys
            trans_dict = dict()
            for k, v in input_dict.items():
                trans_dict[self._input_vars[k]] = v

            # write the dict to a temporary file and use it to initialize the
            # pyDeltaRCM object
            with utils.temporary_config() as tmp_yaml:
                write_yaml_config_to_file(trans_dict, tmp_yaml)
                self._delta = DeltaModel(input_file=tmp_yaml)

        else:
            self._delta = DeltaModel()

        # populate the BMI values fields with links to the pyDeltaRCM attrs
        self._values = {
            'channel_exit_water_flow__speed': self._delta.u0,
            'channel_exit_water_x-section__width': self._delta.N0_meters,
            'channel_exit_water_x-section__depth': self._delta.h0,
            'sea_water_surface__mean_elevation': self._delta.H_SL,
            'sea_water_surface__rate_change_elevation': self._delta.SLR,
            'channel_exit_water_sediment~bedload__volume_fraction': self._delta.f_bedload,
            'channel_exit_water_sediment~suspended__mass_concentration': self._delta.C0_percent,
            'sea_water_surface__elevation': self._delta.stage,
            'sea_water__depth': self._delta.depth,
            'sea_bottom_surface__elevation': self._delta.eta}
        self._var_units = {
            'channel_exit_water_flow__speed': 'm s-1',
            'channel_exit_water_x-section__width': 'm',
            'channel_exit_water_x-section__depth': 'm',
            'sea_water_surface__mean_elevation': 'm',
            'sea_water_surface__rate_change_elevation': 'm yr-1',
            'channel_exit_water_sediment~bedload__volume_fraction': 'fraction',
            'channel_exit_water_sediment~suspended__mass_concentration': 'm3 m-3',
            'sea_water_surface__elevation': 'm',
            'sea_water__depth': 'm',
            'sea_bottom_surface__elevation': 'm'}
        self._var_loc = {'sea_water_surface__elevation': 'node',
            'sea_water__depth': 'node',
            'sea_bottom_surface__elevation': 'node'}
        self._grids = {
            0: ['sea_water_surface__elevation'],
            1: ['sea_water__depth'],
            2: ['sea_bottom_surface__elevation']}
        self._grid_type = {
            0: 'uniform_rectilinear_grid',
            1: 'uniform_rectilinear_grid',
            2: 'uniform_rectilinear_grid'}

    def update(self):
        """Advance model by one time step."""
        self._delta.update()

    def update_frac(self, time_frac):
        """Update model by a fraction of a time step.

        Parameters
        ----------
        time_frac : float
            Fraction fo a time step.
        """
        time_step = self.get_time_step()

        self._delta.time_step = time_frac * time_step
        self.update()

        self._delta.time_step = time_step

    def update_until(self, then):
        """Update model until a particular time.

        Parameters
        ----------
        then : float
            Time to run model until.
        """

        if self.get_current_time() != int(self.get_current_time()):

            remainder = self.get_current_time() - int(self.get_current_time())
            self.update_frac(remainder)

        n_steps = (then - self.get_current_time()) / self.get_time_step()

        for _ in range(int(n_steps)):
            self.update()

        remainder = n_steps - int(n_steps)
        if remainder > 0:
            self.update_frac(remainder)

    def finalize(self):
        """Finalize model."""
        self._delta.finalize()
        self._delta = None

    def get_var_type(self, var_name):
        """Data type of variable.

        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.

        Returns
        -------
        str
            Data type.
        """
        return str(self.get_value_ptr(var_name).dtype)

    def get_var_units(self, var_name):
        """Get units of variable.

        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.

        Returns
        -------
        str
            Variable units.
        """
        return self._var_units[var_name]

    def get_var_nbytes(self, var_name):
        """Get units of variable.

        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.

        Returns
        -------
        int
            Size of data array in bytes.
        """
        return self.get_value_ptr(var_name).nbytes

    def get_var_grid(self, var_name):
        """Grid id for a variable.

        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.

        Returns
        -------
        int
            Grid id.
        """
        for grid_id, var_name_list in list(self._grids.items()):
            if var_name in var_name_list:
                return grid_id

    def get_grid_rank(self, grid_id):
        """Rank of grid.

        Parameters
        ----------
        grid_id : int
            Identifier of a grid.

        Returns
        -------
        int
            Rank of grid.
        """
        return len(self.get_grid_shape(grid_id))

    def get_grid_size(self, grid_id):
        """Size of grid.

        Parameters
        ----------
        grid_id : int
            Identifier of a grid.

        Returns
        -------
        int
            Size of grid.
        """
        return np.prod(self.get_grid_shape(grid_id))

    def get_value_ptr(self, var_name):
        """Reference to values.

        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.

        Returns
        -------
        array_like
            Value array.
        """
        return self._values[var_name]

    def get_value_ref(self, var_name):
        """Reference to values, legacy BMI 1.0 api.

        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.

        Returns
        -------
        array_like
            Value array.
        """
        warnings.warn('`get_value_ref` is depreciated in BMI 2.0.' +
                      'Use `get_value_ptr` instead.', DeprecationWarning)
        return self.get_value_ptr(var_name)


    def get_value(self, var_name):
        """Copy of values.

        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.

        Returns
        -------
        array_like
            Copy of values.
        """
        return self.get_value_ptr(var_name).copy()

    def get_value_at_indices(self, var_name, indices):
        """Get values at particular indices.

        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.
        indices : array_like
            Array of indices.

        Returns
        -------
        array_like
            Values at indices.
        """
        return self.get_value_ptr(var_name).take(indices)

    def set_value(self, var_name, src):
        """Set model values.

        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.
        src : array_like
            Array of new values.
        """
        val = self.get_value_ptr(var_name)
        val[:] = src

    def set_value_at_indices(self, var_name, src, indices):
        """Set model values at particular indices.

        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.
        src : array_like
            Array of new values.
        indices : array_like
            Array of indices.
        """
        val = self.get_value_ptr(var_name)
        val.flat[indices] = src

    def get_component_name(self):
        """Name of the component."""
        return self._name

    def get_input_var_names(self):
        """Get names of input variables."""
        return self._input_var_names

    def get_output_var_names(self):
        """Get names of output variables."""
        return self._output_var_names

    def get_grid_shape(self, grid_id):
        """Number of rows and columns of uniform rectilinear grid."""
        var_name = self._grids[grid_id][0]
        return self.get_value_ptr(var_name).shape

    def get_grid_spacing(self, grid_id):
        """Spacing of rows and columns of uniform rectilinear grid."""
        return (self._delta.dx, self._delta.dx)

    def get_grid_origin(self, grid_id):
        """Origin of uniform rectilinear grid."""
        return (0., 0.)

    def get_grid_type(self, grid_id):
        """Type of grid."""
        return self._grid_type[grid_id]

    def get_start_time(self):
        """Start time of model."""
        return self._start_time

    def get_end_time(self):
        """End time of model."""
        return self._end_time

    def get_current_time(self):
        """Current time of model."""
        return self._delta._time

    def get_time_step(self):
        """Time step of model."""
        return self._delta.time_step

    def get_input_item_count(self) -> int:
        """Count of a model's input variables.
        
        Returns
        -------
        int
            The number of input variables.
        """
        return len(self._input_var_names)

    def get_output_item_count(self) -> int:
        """Count of a model's output variables.
        
        Returns
        -------
        int
            The number of output variables.
        """
        return len(self._output_var_names)

    def get_var_itemsize(self, name: str) -> int:
        """Get the size (in bytes) of one element of a variable.
        
        Parameters
        ----------
        name : str
            An input or output variable name, a CSDMS Standard Name.
        
        Returns
        -------
        int
            Item size in bytes.
        """
        return np.dtype(self.get_var_type(name)).itemsize

    def get_var_location(self, name: str) -> str:
        """Get the grid element type that the a given variable is defined on.

        .. note::
            
            CSDMS uses the `ugrid conventions`_ to define unstructured grids.
            .. _ugrid conventions: http://ugrid-conventions.github.io/ugrid-conventions

        Parameters
        ----------
        name : str
            An input or output variable name, a CSDMS Standard Name.
        
        Returns
        -------
        str
            The grid location on which the variable is defined. Must be one of
            `"node"`, `"edge"`, or `"face"`.
        """
        return self._var_loc[name]

    def get_time_units(self) -> str:
        """Time units of the model.

        .. note:: CSDMS uses the UDUNITS standard from Unidata.
        
        Returns
        -------
        str
            The model time unit; e.g., `days` or `s`.
        """
        return self._time_units

    def get_grid_x(self, grid: int) -> np.ndarray:
        """Get coordinates of grid nodes in the x direction.

        .. hint:: 
            Internally, the pyDeltaRCM model refers to the down-stream
            direction as `x`, which is the row-coordinate of the grid, and
            opposite the BMI specification.
    
        Parameters
        ----------
        grid : int
            A grid identifier.
        x : ndarray of float, shape *(nrows,)*
            A numpy array to hold the x-coordinates of the grid node columns.
        
        Returns
        -------
        ndarray of float
            The input numpy array that holds the grid's column x-coordinates.
        """
        return np.tile(np.arange(self._delta.W)[np.newaxis, :] * self._delta.dx,
                      (self._delta.L, 1))

    def get_grid_y(self, grid: int) -> np.ndarray:
        """Get coordinates of grid nodes in the y direction.

        .. hint:: 
            Internally, the pyDeltaRCM model refers to the cross-stream
            direction as `y`, which is the column-coordinate of the grid, and
            opposite the BMI specification.
        
        Parameters
        ----------
        grid : int
            A grid identifier.
        
        Returns
        -------
        ndarray of float
            The input numpy array that holds the grid's row y-coordinates.
        """
        return np.tile(np.arange(self._delta.L)[:, np.newaxis] * self._delta.dx,
                      (1, self._delta.W))

    def get_grid_z(self, grid: int) -> np.ndarray:
        raise NotImplementedError('There is no `z` coordinate in the model.')

    def get_grid_node_count(self, grid: int) -> int:
        """Get the number of nodes in the grid.
        
        .. note:: Implemented as an alias to :obj:`get_grid_size`.

        Parameters
        ----------
        grid : int
            A grid identifier.
        
        Returns
        -------
        int
            The total number of grid nodes.
        """
        return int(self.get_grid_size(grid))

    def get_grid_edge_count(self, grid: int) -> int:
        """Get the number of edges in the grid.
        
        .. warning:: Not implemented.

        Could be computed from the rectilinear type?

        Parameters
        ----------
        grid : int
            A grid identifier.
        
        Returns
        -------
        int
            The total number of grid edges.
        """
        raise NotImplementedError

    def get_grid_face_count(self, grid: int) -> int:
        """Get the number of faces in the grid.

        .. warning:: Not implemented.

        Could be computed from the rectilinear type?

        Parameters
        ----------
        grid : int
            A grid identifier.

        Returns
        -------
        int
            The total number of grid faces.
        """
        raise NotImplementedError

    def get_grid_edge_nodes(self, grid: int, edge_nodes: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def get_grid_face_edges(self, grid: int, face_edges: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def get_grid_face_nodes(self, grid: int, face_nodes: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def get_grid_nodes_per_face(self, grid: int, nodes_per_face: np.ndarray) -> np.ndarray:
        raise NotImplementedError

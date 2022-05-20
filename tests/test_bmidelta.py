import pytest
import numpy as np
from bmipy import Bmi
from BMI_pyDeltaRCM import BmiDelta


# UTILITIES FOR TESTING #
def create_temporary_file(tmp_path, file_name):
    d = tmp_path / 'configs'
    d.mkdir()
    p = d / file_name
    f = open(p, "a")
    return p, f


def write_parameter_to_file(f, varname, varvalue):
    f.write(varname + ': ' + str(varvalue) + '\n')


# TESTS #
def test_bmi_implemented():
    """Test based on the same name test in the CSDMS/bmi-python repo."""
    assert isinstance(BmiDelta(), Bmi)
    

class TestBmiInputParameters:
    """Tests associated with the input parameters of the BMI"""

    def test_default_vals(self, tmp_path):
        filename = 'user_parameters.yaml'
        p, f = create_temporary_file(tmp_path, filename)
        write_parameter_to_file(f, 'model_output__out_dir', tmp_path / 'out_dir')
        f.close()
        delta = BmiDelta()
        delta.initialize(p)
        assert delta._delta.h0 == 5.0
        assert delta._delta.u0 == 1.0
        assert delta._delta.S0 == 0.00015

    def test_set_model_output__out_dir_config(self, tmp_path):
        filename = 'user_parameters.yaml'
        p, f = create_temporary_file(tmp_path, filename)
        write_parameter_to_file(f, 'model_output__out_dir', tmp_path / 'changed_dir_name')
        f.close()
        delta = BmiDelta()
        delta.initialize(p)
        assert 'changed_dir_name' in delta._delta.out_dir

    def test_set_model__random_seed(self, tmp_path):
        filename = 'user_parameters.yaml'
        p, f = create_temporary_file(tmp_path, filename)
        write_parameter_to_file(f, 'model__random_seed', 42)
        write_parameter_to_file(f, 'model_output__out_dir', tmp_path / 'out_dir')
        f.close()
        delta = BmiDelta()
        delta.initialize(p)
        assert delta._delta.seed == 42

    def test_set_model_grid__length_config(self, tmp_path):
        filename = 'user_parameters.yaml'
        p, f = create_temporary_file(tmp_path, filename)
        write_parameter_to_file(f, 'model_grid__length', 5000)
        write_parameter_to_file(f, 'model_output__out_dir', tmp_path / 'out_dir')
        f.close()
        delta = BmiDelta()
        delta.initialize(p)
        assert delta._delta.Length == 5000

    def test_set_model_grid__width_config(self, tmp_path):
        filename = 'user_parameters.yaml'
        p, f = create_temporary_file(tmp_path, filename)
        write_parameter_to_file(f, 'model_grid__width', 4000)
        write_parameter_to_file(f, 'model_output__out_dir', tmp_path / 'out_dir')
        f.close()
        delta = BmiDelta()
        delta.initialize(p)
        assert delta._delta.Width == 4000

    def test_set_model_grid__cell_size_config(self, tmp_path):
        filename = 'user_parameters.yaml'
        p, f = create_temporary_file(tmp_path, filename)
        write_parameter_to_file(f, 'model_grid__cell_size', 50)
        write_parameter_to_file(f, 'model_output__out_dir', tmp_path / 'out_dir')
        f.close()
        delta = BmiDelta()
        delta.initialize(p)
        assert delta._delta.dx == 50

    def test_set_land_surface__width_config(self, tmp_path):
        filename = 'user_parameters.yaml'
        p, f = create_temporary_file(tmp_path, filename)
        write_parameter_to_file(f, 'land_surface__width', 100)
        write_parameter_to_file(f, 'model_output__out_dir', tmp_path / 'out_dir')
        f.close()
        delta = BmiDelta()
        delta.initialize(p)
        assert delta._delta.L0_meters == 100

    def test_set_land_surface__slope_config(self, tmp_path):
        filename = 'user_parameters.yaml'
        p, f = create_temporary_file(tmp_path, filename)
        write_parameter_to_file(f, 'land_surface__slope', 0.02)
        write_parameter_to_file(f, 'model_output__out_dir', tmp_path / 'out_dir')
        f.close()
        delta = BmiDelta()
        delta.initialize(p)
        assert delta._delta.S0 == 0.02

    def test_set_model__max_iteration_config(self, tmp_path):
        filename = 'user_parameters.yaml'
        p, f = create_temporary_file(tmp_path, filename)
        write_parameter_to_file(f, 'model__max_iteration', 3)
        write_parameter_to_file(f, 'model_output__out_dir', tmp_path / 'out_dir')
        f.close()
        delta = BmiDelta()
        delta.initialize(p)
        assert delta._delta.itermax == 3

    def test_set_water__number_parcels_config(self, tmp_path):
        filename = 'user_parameters.yaml'
        p, f = create_temporary_file(tmp_path, filename)
        write_parameter_to_file(f, 'water__number_parcels', 6000)
        write_parameter_to_file(f, 'model_output__out_dir', tmp_path / 'out_dir')
        f.close()
        delta = BmiDelta()
        delta.initialize(p)
        assert delta._delta.Np_water == 6000

    def test_set_channel__flow_velocity_config(self, tmp_path):
        filename = 'user_parameters.yaml'
        p, f = create_temporary_file(tmp_path, filename)
        write_parameter_to_file(f, 'channel__flow_velocity', 3)
        write_parameter_to_file(f, 'model_output__out_dir', tmp_path / 'out_dir')
        f.close()
        delta = BmiDelta()
        delta.initialize(p)
        assert delta._delta.u0 == 3

    def test_set_channel__width_config(self, tmp_path):
        filename = 'user_parameters.yaml'
        p, f = create_temporary_file(tmp_path, filename)
        write_parameter_to_file(f, 'channel__width', 400)
        write_parameter_to_file(f, 'model_output__out_dir', tmp_path / 'out_dir')
        f.close()
        delta = BmiDelta()
        delta.initialize(p)
        assert delta._delta.N0_meters == 400

    def test_set_channel__flow_depth_config(self, tmp_path):
        filename = 'user_parameters.yaml'
        p, f = create_temporary_file(tmp_path, filename)
        write_parameter_to_file(f, 'channel__flow_depth', 6)
        write_parameter_to_file(f, 'model_output__out_dir', tmp_path / 'out_dir')
        f.close()
        delta = BmiDelta()
        delta.initialize(p)
        assert delta._delta.h0 == 6

    def test_set_sea_water_surface__mean_elevation_config(self, tmp_path):
        filename = 'user_parameters.yaml'
        p, f = create_temporary_file(tmp_path, filename)
        write_parameter_to_file(f, 'sea_water_surface__mean_elevation', 3)
        write_parameter_to_file(f, 'model_output__out_dir', tmp_path / 'out_dir')
        f.close()
        delta = BmiDelta()
        delta.initialize(p)
        assert delta._delta.H_SL == 3

    def test_set_sea_water_surface__rate_change_elevation_config(self, tmp_path):
        filename = 'user_parameters.yaml'
        p, f = create_temporary_file(tmp_path, filename)
        write_parameter_to_file(f, 'sea_water_surface__rate_change_elevation', 0.00005)
        write_parameter_to_file(f, 'model_output__out_dir', tmp_path / 'out_dir')
        f.close()
        delta = BmiDelta()
        delta.initialize(p)
        assert delta._delta.SLR == 0.00005

    def test_set_sediment__number_parcels_config(self, tmp_path):
        filename = 'user_parameters.yaml'
        p, f = create_temporary_file(tmp_path, filename)
        write_parameter_to_file(f, 'sediment__number_parcels', 6000)
        write_parameter_to_file(f, 'model_output__out_dir', tmp_path / 'out_dir')
        f.close()
        delta = BmiDelta()
        delta.initialize(p)
        assert delta._delta.Np_sed == 6000

    def test_set_sediment__bedload_fraction_config(self, tmp_path):
        filename = 'user_parameters.yaml'
        p, f = create_temporary_file(tmp_path, filename)
        write_parameter_to_file(f, 'sediment__bedload_fraction', 0.2)
        write_parameter_to_file(f, 'model_output__out_dir', tmp_path / 'out_dir')
        f.close()
        delta = BmiDelta()
        delta.initialize(p)
        assert delta._delta.f_bedload == 0.2

    def test_set_sediment__influx_concentration_config(self, tmp_path):
        filename = 'user_parameters.yaml'
        p, f = create_temporary_file(tmp_path, filename)
        write_parameter_to_file(f, 'sediment__influx_concentration', 0.001)
        write_parameter_to_file(f, 'model_output__out_dir', tmp_path / 'out_dir')
        f.close()
        delta = BmiDelta()
        delta.initialize(p)
        assert delta._delta.C0_percent == 0.001

    def test_set_model_output__opt_eta_figs_config(self, tmp_path):
        filename = 'user_parameters.yaml'
        p, f = create_temporary_file(tmp_path, filename)
        write_parameter_to_file(f, 'model_output__opt_eta_figs', True)
        write_parameter_to_file(f, 'model_output__out_dir', tmp_path / 'out_dir')
        f.close()
        delta = BmiDelta()
        delta.initialize(p)
        assert delta._delta.save_eta_figs is True

    def test_set_model_output__opt_stage_figs_config(self, tmp_path):
        filename = 'user_parameters.yaml'
        p, f = create_temporary_file(tmp_path, filename)
        write_parameter_to_file(f, 'model_output__opt_stage_figs', True)
        write_parameter_to_file(f, 'model_output__out_dir', tmp_path / 'out_dir')
        f.close()
        delta = BmiDelta()
        delta.initialize(p)
        assert delta._delta.save_stage_figs is True

    def test_set_model_output__opt_depth_figs_config(self, tmp_path):
        filename = 'user_parameters.yaml'
        p, f = create_temporary_file(tmp_path, filename)
        write_parameter_to_file(f, 'model_output__opt_depth_figs', True)
        write_parameter_to_file(f, 'model_output__out_dir', tmp_path / 'out_dir')
        f.close()
        delta = BmiDelta()
        delta.initialize(p)
        assert delta._delta.save_depth_figs is True

    def test_set_model_output__opt_discharge_figs_config(self, tmp_path):
        filename = 'user_parameters.yaml'
        p, f = create_temporary_file(tmp_path, filename)
        write_parameter_to_file(f, 'model_output__opt_discharge_figs', True)
        write_parameter_to_file(f, 'model_output__out_dir', tmp_path / 'out_dir')
        f.close()
        delta = BmiDelta()
        delta.initialize(p)
        assert delta._delta.save_discharge_figs is True

    def test_set_model_output__opt_velocity_figs_config(self, tmp_path):
        filename = 'user_parameters.yaml'
        p, f = create_temporary_file(tmp_path, filename)
        write_parameter_to_file(f, 'model_output__opt_velocity_figs', True)
        write_parameter_to_file(f, 'model_output__out_dir', tmp_path / 'out_dir')
        f.close()
        delta = BmiDelta()
        delta.initialize(p)
        assert delta._delta.save_velocity_figs is True

    def test_set_model_output__opt_eta_grids_config(self, tmp_path):
        filename = 'user_parameters.yaml'
        p, f = create_temporary_file(tmp_path, filename)
        write_parameter_to_file(f, 'model_output__opt_eta_grids', True)
        write_parameter_to_file(f, 'model_output__out_dir', tmp_path / 'out_dir')
        f.close()
        delta = BmiDelta()
        delta.initialize(p)
        assert delta._delta.save_eta_grids is True

    def test_set_model_output__opt_stage_grids_config(self, tmp_path):
        filename = 'user_parameters.yaml'
        p, f = create_temporary_file(tmp_path, filename)
        write_parameter_to_file(f, 'model_output__opt_stage_grids', True)
        write_parameter_to_file(f, 'model_output__out_dir', tmp_path / 'out_dir')
        f.close()
        delta = BmiDelta()
        delta.initialize(p)
        assert delta._delta.save_stage_grids is True

    def test_set_model_output__opt_depth_grids_config(self, tmp_path):
        filename = 'user_parameters.yaml'
        p, f = create_temporary_file(tmp_path, filename)
        write_parameter_to_file(f, 'model_output__opt_depth_grids', True)
        write_parameter_to_file(f, 'model_output__out_dir', tmp_path / 'out_dir')
        f.close()
        delta = BmiDelta()
        delta.initialize(p)
        assert delta._delta.save_depth_grids is True

    def test_set_model_output__opt_discharge_grids_config(self, tmp_path):
        filename = 'user_parameters.yaml'
        p, f = create_temporary_file(tmp_path, filename)
        write_parameter_to_file(f, 'model_output__opt_discharge_grids', True)
        write_parameter_to_file(f, 'model_output__out_dir', tmp_path / 'out_dir')
        f.close()
        delta = BmiDelta()
        delta.initialize(p)
        assert delta._delta.save_discharge_grids is True

    def test_set_model_output__opt_velocity_grids_config(self, tmp_path):
        filename = 'user_parameters.yaml'
        p, f = create_temporary_file(tmp_path, filename)
        write_parameter_to_file(f, 'model_output__opt_velocity_grids', True)
        write_parameter_to_file(f, 'model_output__out_dir', tmp_path / 'out_dir')
        f.close()
        delta = BmiDelta()
        delta.initialize(p)
        assert delta._delta.save_velocity_grids is True

    def test_set_model_output__opt_time_interval_config(self, tmp_path):
        filename = 'user_parameters.yaml'
        p, f = create_temporary_file(tmp_path, filename)
        write_parameter_to_file(f, 'model_output__opt_time_interval', 5)
        write_parameter_to_file(f, 'model_output__out_dir', tmp_path / 'out_dir')
        f.close()
        delta = BmiDelta()
        delta.initialize(p)
        assert delta._delta.save_dt == 5

    def test_set_coeff__surface_smoothing_config(self, tmp_path):
        filename = 'user_parameters.yaml'
        p, f = create_temporary_file(tmp_path, filename)
        write_parameter_to_file(f, 'coeff__surface_smoothing', 1)
        write_parameter_to_file(f, 'model_output__out_dir', tmp_path / 'out_dir')
        f.close()
        delta = BmiDelta()
        delta.initialize(p)
        assert delta._delta.Csmooth == 1

    def test_set_coeff__under_relaxation__water_surface_config(self, tmp_path):
        filename = 'user_parameters.yaml'
        p, f = create_temporary_file(tmp_path, filename)
        write_parameter_to_file(f, 'coeff__under_relaxation__water_surface', 0.3)
        write_parameter_to_file(f, 'model_output__out_dir', tmp_path / 'out_dir')
        f.close()
        delta = BmiDelta()
        delta.initialize(p)
        assert delta._delta.omega_sfc == 0.3

    def test_set_coeff__under_relaxation__water_flow_config(self, tmp_path):
        """xFail due to upstream (pyDeltaRCM) bug.
        """
        filename = 'user_parameters.yaml'
        p, f = create_temporary_file(tmp_path, filename)
        write_parameter_to_file(f, 'coeff__under_relaxation__water_flow', 0.8)
        write_parameter_to_file(f, 'model_output__out_dir', tmp_path / 'out_dir')
        f.close()
        delta = BmiDelta()
        delta.initialize(p)
        assert delta._delta.omega_flow == 0.8

    def test_set_coeff__iterations_smoothing_algorithm_config(self, tmp_path):
        filename = 'user_parameters.yaml'
        p, f = create_temporary_file(tmp_path, filename)
        write_parameter_to_file(f, 'coeff__iterations_smoothing_algorithm', 1)
        write_parameter_to_file(f, 'model_output__out_dir', tmp_path / 'out_dir')
        f.close()
        delta = BmiDelta()
        delta.initialize(p)
        assert delta._delta.Nsmooth == 1

    def test_set_coeff__depth_dependence__water_config(self, tmp_path):
        filename = 'user_parameters.yaml'
        p, f = create_temporary_file(tmp_path, filename)
        write_parameter_to_file(f, 'coeff__depth_dependence__water', 1)
        write_parameter_to_file(f, 'model_output__out_dir', tmp_path / 'out_dir')
        f.close()
        delta = BmiDelta()
        delta.initialize(p)
        assert delta._delta.theta_water == 1

    def test_set_coeff__depth_dependence__sand_config(self, tmp_path):
        filename = 'user_parameters.yaml'
        p, f = create_temporary_file(tmp_path, filename)
        write_parameter_to_file(f, 'coeff__depth_dependence__sand', 1)
        write_parameter_to_file(f, 'model_output__out_dir', tmp_path / 'out_dir')
        f.close()
        delta = BmiDelta()
        delta.initialize(p)
        assert delta._delta.coeff_theta_sand == 1

    def test_set_coeff__depth_dependence__mud_config(self, tmp_path):
        filename = 'user_parameters.yaml'
        p, f = create_temporary_file(tmp_path, filename)
        write_parameter_to_file(f, 'coeff__depth_dependence__mud', 1)
        write_parameter_to_file(f, 'model_output__out_dir', tmp_path / 'out_dir')
        f.close()
        delta = BmiDelta()
        delta.initialize(p)
        assert delta._delta.coeff_theta_mud == 1

    def test_set_coeff__non_linear_exp_sed_flux_flow_velocity_config(self, tmp_path):
        filename = 'user_parameters.yaml'
        p, f = create_temporary_file(tmp_path, filename)
        write_parameter_to_file(f, 'coeff__non_linear_exp_sed_flux_flow_velocity', 1)
        write_parameter_to_file(f, 'model_output__out_dir', tmp_path / 'out_dir')
        f.close()
        delta = BmiDelta()
        delta.initialize(p)
        assert delta._delta.beta == 1

    def test_set_coeff__sedimentation_lag_config(self, tmp_path):
        filename = 'user_parameters.yaml'
        p, f = create_temporary_file(tmp_path, filename)
        write_parameter_to_file(f, 'coeff__sedimentation_lag', 10)
        write_parameter_to_file(f, 'model_output__out_dir', tmp_path / 'out_dir')
        f.close()
        delta = BmiDelta()
        delta.initialize(p)
        assert delta._delta.sed_lag == 10

    def test_set_coeff__velocity_deposition_mud_config(self, tmp_path):
        filename = 'user_parameters.yaml'
        p, f = create_temporary_file(tmp_path, filename)
        write_parameter_to_file(f, 'coeff__velocity_deposition_mud', 10)
        write_parameter_to_file(f, 'model_output__out_dir', tmp_path / 'out_dir')
        f.close()
        delta = BmiDelta()
        delta.initialize(p)
        assert delta._delta.coeff_U_dep_mud == 10

    def test_set_coeff__velocity_erosion_mud_config(self, tmp_path):
        filename = 'user_parameters.yaml'
        p, f = create_temporary_file(tmp_path, filename)
        write_parameter_to_file(f, 'coeff__velocity_erosion_mud', 10)
        write_parameter_to_file(f, 'model_output__out_dir', tmp_path / 'out_dir')
        f.close()
        delta = BmiDelta()
        delta.initialize(p)
        assert delta._delta.coeff_U_ero_mud == 10

    def test_set_coeff__velocity_erosion_sand_config(self, tmp_path):
        filename = 'user_parameters.yaml'
        p, f = create_temporary_file(tmp_path, filename)
        write_parameter_to_file(f, 'coeff__velocity_erosion_sand', 10)
        write_parameter_to_file(f, 'model_output__out_dir', tmp_path / 'out_dir')
        f.close()
        delta = BmiDelta()
        delta.initialize(p)
        assert delta._delta.coeff_U_ero_sand == 10

    def test_set_coeff__topographic_diffusion_config(self, tmp_path):
        filename = 'user_parameters.yaml'
        p, f = create_temporary_file(tmp_path, filename)
        write_parameter_to_file(f, 'coeff__topographic_diffusion', 10)
        write_parameter_to_file(f, 'model_output__out_dir', tmp_path / 'out_dir')
        f.close()
        delta = BmiDelta()
        delta.initialize(p)
        assert delta._delta.alpha == 10

    def test_set_basin__opt_subsidence_config(self, tmp_path):
        filename = 'user_parameters.yaml'
        p, f = create_temporary_file(tmp_path, filename)
        write_parameter_to_file(f, 'basin__opt_subsidence', True)
        write_parameter_to_file(f, 'model_output__out_dir', tmp_path / 'out_dir')
        f.close()
        delta = BmiDelta()
        delta.initialize(p)
        assert delta._delta.toggle_subsidence is True

    def test_set_basin__maximum_subsidence_rate_config(self, tmp_path):
        filename = 'user_parameters.yaml'
        p, f = create_temporary_file(tmp_path, filename)
        write_parameter_to_file(f, 'basin__maximum_subsidence_rate', 0.00033)
        write_parameter_to_file(f, 'model_output__out_dir', tmp_path / 'out_dir')
        f.close()
        delta = BmiDelta()
        delta.initialize(p)
        assert delta._delta.subsidence_rate == 0.00033

    def test_set_basin__subsidence_start_timestep_config(self, tmp_path):
        filename = 'user_parameters.yaml'
        p, f = create_temporary_file(tmp_path, filename)
        write_parameter_to_file(f, 'basin__subsidence_start_timestep', 10)
        write_parameter_to_file(f, 'model_output__out_dir', tmp_path / 'out_dir')
        f.close()
        delta = BmiDelta()
        delta.initialize(p)
        assert delta._delta.start_subsidence == 10


class TestBmiOperations:

    def test_update_once(self, tmp_path):
        filename = 'user_parameters.yaml'
        p, f = create_temporary_file(tmp_path, filename)
        write_parameter_to_file(f, 'model_output__out_dir', tmp_path / 'out_dir')
        f.close()
        delta = BmiDelta()
        delta.initialize(p)
        delta.update()
        assert delta._delta._time == 25000.0

    def test_update_until(self, tmp_path):
        filename = 'user_parameters.yaml'
        p, f = create_temporary_file(tmp_path, filename)
        write_parameter_to_file(f, 'model__random_seed', 42)
        write_parameter_to_file(f, 'model_output__out_dir', tmp_path / 'out_dir')
        f.close()
        delta = BmiDelta()
        delta.initialize(p)
        delta.update_until(2)
        assert delta._delta._time == 2.

    def test_update_frac(self, tmp_path):
        filename = 'user_parameters.yaml'
        p, f = create_temporary_file(tmp_path, filename)
        write_parameter_to_file(f, 'model_output__out_dir', tmp_path / 'out_dir')
        f.close()
        delta = BmiDelta()
        delta.initialize(p)
        delta.update_frac(0.2)
        assert delta._delta.time_step == 25000.0


class TestBmiApiReqts:

    def test_get_value_ptr(self, tmp_path):
        filename = 'user_parameters.yml'
        p, f = create_temporary_file(tmp_path, filename)
        write_parameter_to_file(f, 'model_output__out_dir', tmp_path / 'out_dir')
        f.close()
        delta = BmiDelta()
        delta.initialize(p)
        assert delta.get_value_ptr('sea_water_surface__elevation') is delta._values['sea_water_surface__elevation']
        assert delta.get_value_ptr('sea_water__depth') is delta._values['sea_water__depth']
        assert delta.get_value_ptr('sea_bottom_surface__elevation') is delta._values['sea_bottom_surface__elevation']
        assert delta.get_value_ptr('sea_water_surface__elevation') is delta._delta.stage
        assert delta.get_value_ptr('sea_water__depth') is delta._delta.depth
        assert delta.get_value_ptr('sea_bottom_surface__elevation') is delta._delta.eta

    def test_get_value_ref(self, tmp_path):
        filename = 'user_parameters.yml'
        p, f = create_temporary_file(tmp_path, filename)
        write_parameter_to_file(f, 'model_output__out_dir', tmp_path / 'out_dir')
        f.close()
        delta = BmiDelta()
        delta.initialize(p)
        with pytest.warns(DeprecationWarning):
            assert delta.get_value_ref('sea_water_surface__elevation') is delta.get_value_ptr('sea_water_surface__elevation')

    def test_get_var_type(self, tmp_path):
        filename = 'user_parameters.yml'
        p, f = create_temporary_file(tmp_path, filename)
        write_parameter_to_file(f, 'model_output__out_dir', tmp_path / 'out_dir')
        f.close()
        delta = BmiDelta()
        delta.initialize(p)
        assert delta.get_var_type('sea_water_surface__elevation') == 'float32'
        assert delta.get_var_type('sea_water__depth') == 'float32'
        assert delta.get_var_type('sea_bottom_surface__elevation') == 'float32'

    def test_get_var_units(self, tmp_path):
        filename = 'user_parameters.yml'
        p, f = create_temporary_file(tmp_path, filename)
        write_parameter_to_file(f, 'model_output__out_dir', tmp_path / 'out_dir')
        f.close()
        delta = BmiDelta()
        delta.initialize(p)
        assert delta.get_var_units('channel_exit_water_flow__speed') == 'm s-1'
        assert delta.get_var_units('channel_exit_water_x-section__width') == 'm'
        assert delta.get_var_units('channel_exit_water_x-section__depth') == 'm'
        assert delta.get_var_units('sea_water_surface__mean_elevation') == 'm'
        assert delta.get_var_units('sea_water_surface__rate_change_elevation') == 'm yr-1'
        assert delta.get_var_units('channel_exit_water_sediment~bedload__volume_fraction') == 'fraction'
        assert delta.get_var_units('channel_exit_water_sediment~suspended__mass_concentration') == 'm3 m-3'
        assert delta.get_var_units('sea_water_surface__elevation') == 'm'
        assert delta.get_var_units('sea_water__depth') == 'm'
        assert delta.get_var_units('sea_bottom_surface__elevation') == 'm'

    def test_get_var_nbytes(self, tmp_path):
        filename = 'user_parameters.yml'
        p, f = create_temporary_file(tmp_path, filename)
        write_parameter_to_file(f, 'model_output__out_dir', tmp_path / 'out_dir')
        f.close()
        delta = BmiDelta()
        delta.initialize(p)
        assert delta.get_var_nbytes('sea_water_surface__elevation') > 0
        assert delta.get_var_nbytes('sea_water__depth') > 0
        assert delta.get_var_nbytes('sea_bottom_surface__elevation') > 0

    def test_get_var_grid(self, tmp_path):
        filename = 'user_parameters.yml'
        p, f = create_temporary_file(tmp_path, filename)
        write_parameter_to_file(f, 'model_output__out_dir', tmp_path / 'out_dir')
        f.close()
        delta = BmiDelta()
        delta.initialize(p)
        assert type(delta.get_var_grid('sea_water_surface__elevation')) in [np.int, np.float]

    def test_get_grid_rank(self, tmp_path):
        filename = 'user_parameters.yml'
        p, f = create_temporary_file(tmp_path, filename)
        write_parameter_to_file(f, 'model_output__out_dir', tmp_path / 'out_dir')
        f.close()
        delta = BmiDelta()
        delta.initialize(p)
        _id = delta.get_var_grid('sea_water_surface__elevation')
        assert delta.get_grid_rank(_id) == 2

    def test_get_grid_size(self, tmp_path):
        filename = 'user_parameters.yml'
        p, f = create_temporary_file(tmp_path, filename)
        write_parameter_to_file(f, 'model_output__out_dir', tmp_path / 'out_dir')
        f.close()
        delta = BmiDelta()
        delta.initialize(p)
        _id = delta.get_var_grid('sea_water_surface__elevation')
        assert delta.get_grid_size(_id) == delta._delta.L * delta._delta.W

    def test_get_value(self, tmp_path):
        filename = 'user_parameters.yml'
        p, f = create_temporary_file(tmp_path, filename)
        write_parameter_to_file(f, 'model_output__out_dir', tmp_path / 'out_dir')
        f.close()
        delta = BmiDelta()
        delta.initialize(p)
        assert np.all(delta.get_value('sea_water_surface__elevation') == delta._delta.stage)
        # make sure a copy is returned
        assert delta.get_value('sea_water_surface__elevation') is not delta._delta.stage

    def test_get_value_at_indices(self, tmp_path):
        filename = 'user_parameters.yml'
        p, f = create_temporary_file(tmp_path, filename)
        write_parameter_to_file(f, 'model_output__out_dir', tmp_path / 'out_dir')
        f.close()
        delta = BmiDelta()
        delta.initialize(p)
        _inds = np.array([10, 8, 19, 23, 45, 16, 0, -1, -2])
        assert np.all(delta.get_value_at_indices('sea_water_surface__elevation', _inds) == delta._delta.stage.take(_inds))

    def test_get_component_name(self, tmp_path):
        filename = 'user_parameters.yml'
        p, f = create_temporary_file(tmp_path, filename)
        write_parameter_to_file(f, 'model_output__out_dir', tmp_path / 'out_dir')
        f.close()
        delta = BmiDelta()
        delta.initialize(p)
        assert delta.get_component_name() == BmiDelta._name

    def test_get_input_var_names(self, tmp_path):
        filename = 'user_parameters.yml'
        p, f = create_temporary_file(tmp_path, filename)
        write_parameter_to_file(f, 'model_output__out_dir', tmp_path / 'out_dir')
        f.close()
        delta = BmiDelta()
        delta.initialize(p)
        assert delta.get_input_var_names() == BmiDelta._input_var_names

    def test_get_output_var_names(self, tmp_path):
        filename = 'user_parameters.yml'
        p, f = create_temporary_file(tmp_path, filename)
        write_parameter_to_file(f, 'model_output__out_dir', tmp_path / 'out_dir')
        f.close()
        delta = BmiDelta()
        delta.initialize(p)
        assert delta.get_output_var_names() == BmiDelta._output_var_names

    def test_get_grid_shape(self, tmp_path):
        filename = 'user_parameters.yml'
        p, f = create_temporary_file(tmp_path, filename)
        write_parameter_to_file(f, 'model_output__out_dir', tmp_path / 'out_dir')
        f.close()
        delta = BmiDelta()
        delta.initialize(p)
        _id = delta.get_var_grid('sea_water_surface__elevation')
        assert delta.get_grid_shape(_id) == (delta._delta.L, delta._delta.W)

    def test_get_grid_spacing(self, tmp_path):
        filename = 'user_parameters.yml'
        p, f = create_temporary_file(tmp_path, filename)
        write_parameter_to_file(f, 'model_output__out_dir', tmp_path / 'out_dir')
        f.close()
        delta = BmiDelta()
        delta.initialize(p)
        _id = delta.get_var_grid('sea_water_surface__elevation')
        assert len(delta.get_grid_spacing(_id)) == 2
        assert delta.get_grid_spacing(_id) == (delta._delta.dx, delta._delta.dx)

    def test_get_grid_origin(self, tmp_path):
        filename = 'user_parameters.yml'
        p, f = create_temporary_file(tmp_path, filename)
        write_parameter_to_file(f, 'model_output__out_dir', tmp_path / 'out_dir')
        f.close()
        delta = BmiDelta()
        delta.initialize(p)
        _id = delta.get_var_grid('sea_water_surface__elevation')
        assert delta.get_grid_origin(_id) == (0, 0)

    def test_get_grid_type(self, tmp_path):
        filename = 'user_parameters.yml'
        p, f = create_temporary_file(tmp_path, filename)
        write_parameter_to_file(f, 'model_output__out_dir', tmp_path / 'out_dir')
        f.close()
        delta = BmiDelta()
        delta.initialize(p)
        _id = delta.get_var_grid('sea_water_surface__elevation')
        assert delta.get_grid_type(_id) == 'uniform_rectilinear_grid'

    def test_get_start_time(self, tmp_path):
        filename = 'user_parameters.yml'
        p, f = create_temporary_file(tmp_path, filename)
        write_parameter_to_file(f, 'model_output__out_dir', tmp_path / 'out_dir')
        f.close()
        delta = BmiDelta()
        delta.initialize(p)
        assert delta.get_start_time() == 0

    def test_get_end_time(self, tmp_path):
        filename = 'user_parameters.yml'
        p, f = create_temporary_file(tmp_path, filename)
        write_parameter_to_file(f, 'model_output__out_dir', tmp_path / 'out_dir')
        f.close()
        delta = BmiDelta()
        delta.initialize(p)
        assert delta.get_end_time() == pytest.approx(np.finfo('d').max)

    def test_get_current_time(self, tmp_path):
        filename = 'user_parameters.yml'
        p, f = create_temporary_file(tmp_path, filename)
        write_parameter_to_file(f, 'model_output__out_dir', tmp_path / 'out_dir')
        f.close()
        delta = BmiDelta()
        delta.initialize(p)
        assert delta.get_current_time() == delta._delta.time
        assert delta.get_current_time() == 0

    @pytest.mark.xfail(raises=AssertionError, strict=True,
                       reason='Upstream changes needed to make timestep start '
                              'at zero and track separately from "time"')
    def test_get_time_step(self, tmp_path):
        filename = 'user_parameters.yml'
        p, f = create_temporary_file(tmp_path, filename)
        write_parameter_to_file(f, 'model_output__out_dir', tmp_path / 'out_dir')
        f.close()
        delta = BmiDelta()
        delta.initialize(p)
        assert delta.get_time_step() == delta._delta.time_step
        assert delta.get_time_step() == 0

    def test_get_input_item_count(self, tmp_path):
        filename = 'user_parameters.yml'
        p, f = create_temporary_file(tmp_path, filename)
        write_parameter_to_file(f, 'model_output__out_dir', tmp_path / 'out_dir')
        f.close()
        delta = BmiDelta()
        delta.initialize(p)
        assert delta.get_input_item_count() == len(delta._input_var_names)

    def test_get_output_item_count(self, tmp_path):
        filename = 'user_parameters.yml'
        p, f = create_temporary_file(tmp_path, filename)
        write_parameter_to_file(f, 'model_output__out_dir', tmp_path / 'out_dir')
        f.close()
        delta = BmiDelta()
        delta.initialize(p)
        assert delta.get_output_item_count() == len(delta._output_var_names)

    def test_get_var_itemsize(self, tmp_path):
        filename = 'user_parameters.yml'
        p, f = create_temporary_file(tmp_path, filename)
        write_parameter_to_file(f, 'model_output__out_dir', tmp_path / 'out_dir')
        f.close()
        delta = BmiDelta()
        delta.initialize(p)
        assert delta.get_var_itemsize('sea_water_surface__elevation') == delta._delta.stage.itemsize

    def test_get_var_location(self, tmp_path):
        filename = 'user_parameters.yml'
        p, f = create_temporary_file(tmp_path, filename)
        write_parameter_to_file(f, 'model_output__out_dir', tmp_path / 'out_dir')
        f.close()
        delta = BmiDelta()
        delta.initialize(p)
        assert delta.get_var_location('sea_water_surface__elevation') == 'node'
        assert delta.get_var_location('sea_water__depth') == 'node'
        assert delta.get_var_location('sea_bottom_surface__elevation') == 'node'

    def test_get_time_units(self, tmp_path):
        filename = 'user_parameters.yml'
        p, f = create_temporary_file(tmp_path, filename)
        write_parameter_to_file(f, 'model_output__out_dir', tmp_path / 'out_dir')
        f.close()
        delta = BmiDelta()
        delta.initialize(p)
        assert delta.get_time_units() == 's'

    def test_get_grid_x(self, tmp_path):
        filename = 'user_parameters.yml'
        p, f = create_temporary_file(tmp_path, filename)
        write_parameter_to_file(f, 'model_output__out_dir', tmp_path / 'out_dir')
        f.close()
        delta = BmiDelta()
        delta.initialize(p)
        _id = delta.get_var_grid('sea_water_surface__elevation')
        assert delta.get_grid_x(_id).shape == (delta._delta.L, delta._delta.W)
        assert delta.get_grid_x(_id)[0, 0] == 0
        assert delta.get_grid_x(_id)[0, 1] == delta._delta.dx
        assert delta.get_grid_x(_id)[1, 1] == delta._delta.dx
        assert delta.get_grid_x(_id)[1, 0] == 0

    def test_get_grid_y(self, tmp_path):
        filename = 'user_parameters.yml'
        p, f = create_temporary_file(tmp_path, filename)
        write_parameter_to_file(f, 'model_output__out_dir', tmp_path / 'out_dir')
        f.close()
        delta = BmiDelta()
        delta.initialize(p)
        _id = delta.get_var_grid('sea_water_surface__elevation')
        assert delta.get_grid_y(_id).shape == (delta._delta.L, delta._delta.W)
        assert delta.get_grid_y(_id)[0, 0] == 0
        assert delta.get_grid_y(_id)[0, 1] == 0
        assert delta.get_grid_y(_id)[1, 1] == delta._delta.dx
        assert delta.get_grid_y(_id)[1, 0] == delta._delta.dx

    def test_get_grid_z(self, tmp_path):
        filename = 'user_parameters.yml'
        p, f = create_temporary_file(tmp_path, filename)
        write_parameter_to_file(f, 'model_output__out_dir', tmp_path / 'out_dir')
        f.close()
        delta = BmiDelta()
        delta.initialize(p)
        _id = delta.get_var_grid('sea_water_surface__elevation')
        with pytest.raises(NotImplementedError):
            delta.get_grid_z(_id)

    def test_get_grid_node_count(self, tmp_path):
        filename = 'user_parameters.yml'
        p, f = create_temporary_file(tmp_path, filename)
        write_parameter_to_file(f, 'model_output__out_dir', tmp_path / 'out_dir')
        f.close()
        delta = BmiDelta()
        delta.initialize(p)
        _id = delta.get_var_grid('sea_water_surface__elevation')
        assert delta.get_grid_node_count(_id) == delta.get_grid_size(_id)

    def test_get_grid_edge_count(self, tmp_path):
        filename = 'user_parameters.yml'
        p, f = create_temporary_file(tmp_path, filename)
        write_parameter_to_file(f, 'model_output__out_dir', tmp_path / 'out_dir')
        f.close()
        delta = BmiDelta()
        delta.initialize(p)
        _id = delta.get_var_grid('sea_water_surface__elevation')
        with pytest.raises(NotImplementedError):
            assert delta.get_grid_edge_count(_id)

    def test_get_grid_face_count(self, tmp_path):
        filename = 'user_parameters.yml'
        p, f = create_temporary_file(tmp_path, filename)
        write_parameter_to_file(f, 'model_output__out_dir', tmp_path / 'out_dir')
        f.close()
        delta = BmiDelta()
        delta.initialize(p)
        _id = delta.get_var_grid('sea_water_surface__elevation')
        with pytest.raises(NotImplementedError):
            assert delta.get_grid_face_count(_id)

    def test_get_grid_edge_nodes(self, tmp_path):
        filename = 'user_parameters.yml'
        p, f = create_temporary_file(tmp_path, filename)
        write_parameter_to_file(f, 'model_output__out_dir', tmp_path / 'out_dir')
        f.close()
        delta = BmiDelta()
        delta.initialize(p)
        _id = delta.get_var_grid('sea_water_surface__elevation')
        _arr = np.arange(9).reshape(3,3)
        with pytest.raises(NotImplementedError):
            assert delta.get_grid_edge_nodes(_id, _arr)

    def test_get_grid_face_edges(self, tmp_path):
        filename = 'user_parameters.yml'
        p, f = create_temporary_file(tmp_path, filename)
        write_parameter_to_file(f, 'model_output__out_dir', tmp_path / 'out_dir')
        f.close()
        delta = BmiDelta()
        delta.initialize(p)
        _id = delta.get_var_grid('sea_water_surface__elevation')
        _arr = np.arange(9).reshape(3,3)
        with pytest.raises(NotImplementedError):
            assert delta.get_grid_face_edges(_id, _arr)

    def test_get_grid_face_nodes(self, tmp_path):
        filename = 'user_parameters.yml'
        p, f = create_temporary_file(tmp_path, filename)
        write_parameter_to_file(f, 'model_output__out_dir', tmp_path / 'out_dir')
        f.close()
        delta = BmiDelta()
        delta.initialize(p)
        _id = delta.get_var_grid('sea_water_surface__elevation')
        _arr = np.arange(9).reshape(3,3)
        with pytest.raises(NotImplementedError):
            assert delta.get_grid_face_nodes(_id, _arr)

    def test_get_grid_nodes_per_face(self, tmp_path):
        filename = 'user_parameters.yml'
        p, f = create_temporary_file(tmp_path, filename)
        write_parameter_to_file(f, 'model_output__out_dir', tmp_path / 'out_dir')
        f.close()
        delta = BmiDelta()
        delta.initialize(p)
        _id = delta.get_var_grid('sea_water_surface__elevation')
        _arr = np.arange(9).reshape(3,3)
        with pytest.raises(NotImplementedError):
            assert delta.get_grid_nodes_per_face(_id, _arr)

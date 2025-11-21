"""file that contains the turbie and floaty class
    """
#%% import package
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import signal
from windy_and_wavy import WindyWavy

#
class TurbieAndFloatyInWindyWavy(WindyWavy):
    """class that contains the turbine response for the floater
    """

    def __init__(self, file_windy_wavy, file_floaty, file_turbie, matrix_b):
        """Initialises the TurbieAndFloatyInWindyWavy class
        """

        # Call the __init__ method of the parent class Windy_Wavy
        super().__init__(file_windy_wavy)

        # load file path and read the config file
        current = Path('C:/pysydyn')
        floaty_path = current / 'Configuration_Files' / file_floaty
        turbie_path = current / 'Configuration_Files' / file_turbie
        floaty_df = pd.read_csv(floaty_path)
        turbie_df = pd.read_csv(turbie_path)

        # load the config file into dictionary
        dict_floaty = {}
        for _ , row in floaty_df.iterrows():
            dict_floaty[row['variables']] = row['values']

        dict_turbie = {}
        for _ , row in turbie_df.iterrows():
            dict_turbie[row['variables']] = row['values']

        # calculation for system weight, com and moment of inertia
        m_total = dict_floaty['m_tot_floaty'] + dict_turbie['m_turbie'] + dict_turbie['m_tower']

        z_com_total = (dict_floaty['m_tot_floaty']* dict_floaty['z_cm_floaty'] +\
                       dict_turbie['m_turbie']* dict_turbie['z_hub'] +\
                       dict_turbie['m_tower']* dict_turbie['z_cm_tower'])/m_total

        x_com_total = (dict_floaty['m_tot_floaty'] * dict_floaty['x_cm_floaty'] +\
                       dict_turbie['m_turbie'] * dict_turbie['x_cm_hub'])/ m_total

        no_total_tanks = dict_floaty['no_of_tanks_per_corner']* dict_floaty['no_of_corners']
        area_1_tank = np.pi* dict_floaty['d_tank']**2/4

        i_com_total = dict_floaty['i_cm_floaty'] + dict_floaty['m_tot_floaty'] *\
                     (dict_floaty['z_cm_floaty'] - x_com_total)**2 +\
                      dict_turbie['i_cm_tower'] + dict_turbie['m_tower'] *\
                     (dict_turbie['z_cm_tower'] - x_com_total)**2 +\
                      dict_turbie['m_turbie'] * (dict_turbie['z_hub'] - x_com_total)**2

        # load all the init data into attributes
        self.dict_floaty = dict_floaty
        self.dict_turbie = dict_turbie
        self.dict_system = {'m_total': m_total,
                            'z_com_total': z_com_total,
                            'x_com_total': x_com_total,
                            'no_total_tanks': no_total_tanks,
                            'area_1_tank': area_1_tank,
                            'i_com_total': i_com_total}

        # work out all the matricies and load them into attribute
        self.dict_sys_matrices = {}
        self.dict_sys_matrices['matrix_a'] = self.get_matrix_a()
        self.dict_sys_matrices['matrix_m'] = self.get_matrix_m()
        self.dict_sys_matrices['matrix_k'] = self.get_matrix_k()
        self.dict_sys_matrices['matrix_b'] = matrix_b

        # work out all the numbers new mark solver requires and load them into attribute
        beta = 0.25
        gamma = 0.5
        eps_r = 1e-6
        coeff_1 = gamma/(beta* self.dict_windy_wavy['time_change'])
        coeff_2 = 1/(beta* self.dict_windy_wavy['time_change']**2)

        self.dict_new_mark = {'beta': beta,
                              'gamma': gamma,
                              'coeff_1': coeff_1,
                              'coeff_2': coeff_2,
                              'eps_r': eps_r}

        # save the configuration files into attributes
        self.dict_file_names = {'file_windy_wavy': file_windy_wavy,
                                'file_floaty': file_floaty,
                                'file_turbie': file_turbie}


    def get_matrix_m(self, q_displacement = np.zeros([3])):
        """calculates the mass matrix
        """

        q1_surge = q_displacement[0]     # surge from previous time step
        q3_heave = q_displacement[1]     # heave from previous time step
        q5_pitch = q_displacement[2]     # pitch from previous time step

        q1_surge, q3_heave, q5_pitch = 0, 0, 0 # testing first

        # UPDATE REQUIRED: ignoring all pitch effects for now
        eom_m11 = self.dict_system['m_total']
        eom_m13 = 0
        eom_m15 = self.dict_system['m_total'] *(self.dict_system['z_com_total'] + q3_heave)
        eom_m31 = 0
        eom_m33 = self.dict_system['m_total']
        eom_m35 = -self.dict_system['m_total'] *(self.dict_system['x_com_total'] + q1_surge)
        eom_m51 = self.dict_system['m_total'] *(self.dict_system['z_com_total'] + q3_heave)
        eom_m53 = -self.dict_system['m_total'] *(self.dict_system['x_com_total'] + q1_surge)
        eom_m55 = self.dict_system['i_com_total']

        # assemblying the matrix
        matrix_m = np.array([[eom_m11, eom_m31, eom_m51],
                            [eom_m13, eom_m33, eom_m53],
                            [eom_m15, eom_m35, eom_m55]])
        return matrix_m

    def get_matrix_a(self, q_displacement = np.zeros([3])):
        """calculates the added mass matrix
        """

        #q1_surge = q_displacement[0]     # surge from previous time step
        q3_heave = q_displacement[1]     # heave from previous time step
        #q5_pitch = q_displacement[2]     # pitch from previous time step

        q3_heave = 0 # testing first

        # UPDATE REQUIRED: ignoring all pitch effects for now
        # UPDATE REQUIRED: no heave plate (all 3 terms are zero)
        eom_a11 = self.dict_system['no_total_tanks']* -1*\
                  self.dict_windy_wavy['rho_water']* self.dict_system['area_1_tank']* \
                  self.dict_windy_wavy['coeff_moment']* (self.dict_floaty['z_bot_floaty']+ q3_heave)
        eom_a13 = 0
        eom_a15 = self.dict_system['no_total_tanks']* -1/2*\
                  self.dict_windy_wavy['rho_water']* self.dict_system['area_1_tank']* \
                  self.dict_windy_wavy['coeff_moment']*\
                 (self.dict_floaty['z_bot_floaty']+ q3_heave)**2
        eom_a31 = 0
        eom_a33 = 0
        eom_a35 = 0
        eom_a51 = self.dict_system['no_total_tanks']* -1/2*\
                  self.dict_windy_wavy['rho_water']* self.dict_system['area_1_tank']*\
                  self.dict_windy_wavy['coeff_moment']*\
                 (self.dict_floaty['z_bot_floaty']+ q3_heave)**2
        eom_a53 = 0
        eom_a55 = self.dict_system['no_total_tanks']* -1/3*\
                  self.dict_windy_wavy['rho_water']* self.dict_system['area_1_tank']* \
                  self.dict_windy_wavy['coeff_moment']*\
                 (self.dict_floaty['z_bot_floaty']+ q3_heave)**3

        matrix_a = np.array([[eom_a11, eom_a31, eom_a51],
                             [eom_a13, eom_a33, eom_a53],
                             [eom_a15, eom_a35, eom_a55]])
        return matrix_a

    def get_matrix_k(self, q_displacement = np.zeros([3])):
        """calculates the damping matrix
        """

        #q1_surge = q_displacement[0]     # surge from previous time step
        q3_heave = q_displacement[1]     # heave from previous time step
        #q5_pitch = q_displacement[2]     # pitch from previous time step

        q3_heave = 0 # testing first

        # UPDATE REQUIRED: ignoring all pitch effects for now
        eom_k11_mooring = self.dict_floaty['k_moor_x']
        eom_k13_mooring = 0
        eom_k15_mooring = self.dict_floaty['k_moor_x']* (self.dict_floaty['z_moor'] + q3_heave)
        eom_k31_mooring = 0
        eom_k33_mooring = self.dict_floaty['k_moor_z']
        eom_k35_mooring = 0
        eom_k51_mooring = self.dict_floaty['k_moor_x']* (self.dict_floaty['z_moor'] + q3_heave)
        eom_k53_mooring = 0
        eom_k55_mooring = self.dict_floaty['k_moor_x']* (self.dict_floaty['z_moor'] + q3_heave)** 2

        # UPDATE REQUIRED: need to update to include floater geometry (tank pcd) in
        eom_k11_hydrostatic = 0
        eom_k13_hydrostatic = 0
        eom_k15_hydrostatic = 0
        eom_k31_hydrostatic = 0
        eom_k33_hydrostatic = self.dict_system['no_total_tanks']*\
                              self.dict_windy_wavy['rho_water']*\
                              self.dict_windy_wavy['gravity']*\
                              self.dict_system['area_1_tank']
        eom_k35_hydrostatic = -self.dict_windy_wavy['rho_water']*\
                               self.dict_windy_wavy['gravity']*\
                               self.dict_floaty['i_x_area_floaty']
        eom_k51_hydrostatic = 0
        eom_k53_hydrostatic = -self.dict_windy_wavy['rho_water']*\
                               self.dict_windy_wavy['gravity']*\
                               self.dict_floaty['i_x_area_floaty']
        # UPDATE REQUIRED: this right now only accounts for spar type (Ixx,A = pi*d_spar**4/64)
        eom_k55_hydrostatic = self.dict_windy_wavy['rho_water']*\
                              self.dict_windy_wavy['gravity']* np.pi*\
                              self.dict_floaty['d_tank']** 4/ 64 + \
                              self.dict_system['m_total']*\
                              self.dict_windy_wavy['gravity']*\
                              ((self.dict_floaty['z_bot_floaty'] + q3_heave)/2 -
                               (self.dict_system['z_com_total'] + q3_heave))

        eom_k11 = eom_k11_mooring + eom_k11_hydrostatic
        eom_k13 = eom_k13_mooring + eom_k13_hydrostatic
        eom_k15 = eom_k15_mooring + eom_k15_hydrostatic
        eom_k31 = eom_k31_mooring + eom_k31_hydrostatic
        eom_k33 = eom_k33_mooring + eom_k33_hydrostatic
        eom_k35 = eom_k35_mooring + eom_k35_hydrostatic
        eom_k51 = eom_k51_mooring + eom_k51_hydrostatic
        eom_k53 = eom_k53_mooring + eom_k53_hydrostatic
        eom_k55 = eom_k55_mooring + eom_k55_hydrostatic

        matrix_k = np.array([[eom_k11, eom_k31, eom_k51],
                             [eom_k13, eom_k33, eom_k53],
                             [eom_k15, eom_k35, eom_k55]])

        return matrix_k

    def calc_wind_force(self, time_index, state_vec_dq):
        """Calculates the wind force for use in the generalised force
        """
        # relative wind speed
        relative_wind = self.wind_velocity[time_index]-\
            (state_vec_dq[0] + (self.dict_turbie['z_hub']*state_vec_dq[2]))

        # calculates ct for realtive wind speed
        if relative_wind > self.dict_turbie['v_rated']:
            coef_thrust = self.dict_turbie['ct_0']*\
                np.exp(-self.dict_windy_wavy['a']*
                       ((relative_wind-self.dict_turbie['v_rated'])**self.dict_windy_wavy['b']))
        else:
            coef_thrust = self.dict_turbie['ct_0']

        # empirical reduction factor statement
        if self.wind_velocity[time_index] < self.dict_turbie['v_rated']:
            f_red = 0.54
        else:
            f_red = 0.54 + 0.027*(self.wind_velocity[time_index]-self.dict_turbie['v_rated'])

        # Mean thrust force
        force_wind_bar = 0.5* self.dict_windy_wavy['rho_air']*\
            (np.pi*(self.dict_turbie['d_rotor']/2)**2)*\
            coef_thrust*(self.wind_velocity[time_index]**2)

        # varying thrust force
        force_wind_wiggle = 0.5* self.dict_windy_wavy['rho_air']*\
            (np.pi*(self.dict_turbie['d_rotor']/2)**2)*\
            coef_thrust* relative_wind* np.abs(relative_wind)

        # final wind force
        force_wind = force_wind_bar + f_red*(force_wind_wiggle-force_wind_bar)

        # final wind moment
        tau_wind = force_wind* self.dict_turbie["z_hub"]

        return force_wind, tau_wind

    def calc_wave_force(self, time_index, state_vec_q, state_vec_dq):
        """calculates the wave force for use in the gf matrix
        """
        # calculates the change of depth, 1% of 1.5x the total depth
        change_depth = self.dict_floaty['z_bot_floaty']*0.01*1.5

        # creates an array of height with 100 bins down to 1.5x the floater depth
        z_wave_array = np.arange(change_depth, self.dict_floaty['z_bot_floaty']*1.5, change_depth)

        # intialise horizontal movement array and the force and tau array
        height_movement = np.zeros(len(z_wave_array))
        wave_force_array = np.zeros(len(z_wave_array))
        wave_tau_array = np.zeros(len(z_wave_array))

        # loop for all depths
        for height_loop, z_wave_enu in enumerate(z_wave_array):
            # if the point in which we are calculating lies outside of the floater, it is disregarded
            # for example, if the bottom of the floater is 50m deep and the depth we are calculating is 60m
            if z_wave_enu < self.dict_floaty['z_bot_floaty'] - state_vec_q[1]:
                pass
            else:
                # relative wave speed
                height_movement =   state_vec_dq[0] + (z_wave_enu*state_vec_dq[2])
                relative_wave_speed = self.wave_speed[height_loop, time_index] - height_movement
                # calculating wave force array
                wave_force_array[height_loop] = (self.dict_windy_wavy['rho_water'])*\
                    (self.dict_system['no_total_tanks']* self.dict_system['area_1_tank'])*\
                                                (1 + self.dict_windy_wavy['coeff_moment'])*\
                                                (self.wave_acceleration[height_loop, time_index]) +\
                                                (0.5*self.dict_windy_wavy['rho_water'])*\
                                                (self.dict_system['no_total_tanks']*
                                                self.dict_floaty['d_tank'])*\
                                                (self.dict_windy_wavy['coeff_drag'])*\
                                                (relative_wave_speed)*\
                                                np.abs(relative_wave_speed)

                # calculating wave moment array
                wave_tau_array[height_loop] = z_wave_enu*\
                                            ((self.dict_windy_wavy['rho_water'])*
                                            (self.dict_system['no_total_tanks']*
                                            self.dict_system['area_1_tank'])*
                                            (1+self.dict_windy_wavy['coeff_moment'])*
                                            (self.wave_acceleration[height_loop, time_index]) +\
                                            (0.5*self.dict_windy_wavy['rho_water'])*
                                            (self.dict_system['no_total_tanks']*
                                            self.dict_floaty['d_tank'])*
                                            (self.dict_windy_wavy['coeff_drag'])*
                                            (relative_wave_speed)*
                                            np.abs(relative_wave_speed))
        # intergrates force and moment over height to get total result
        force_wave = np.trapz(z_wave_array, wave_force_array)
        tau_wave = np.trapz(z_wave_array, wave_tau_array)

        return force_wave, tau_wave


    def get_gf(self, time_index, state_vec_q, state_vec_dq):
        """ calculates the wind and wave forces and combines them into
        a generalised force matrix
        """
        # calculate wind and wave forces
        force_wind, tau_wind = self.calc_wind_force(time_index, state_vec_dq)
        force_wave, tau_wave = self.calc_wave_force(time_index, state_vec_q, state_vec_dq)

        # combine forces
        total_force = force_wind + force_wave
        total_tau = tau_wind + tau_wave
        total_heave = 0


        matrix_gf = [total_force, total_heave, total_tau]

        return matrix_gf

    def newmark_solver(self,
                       change_time,
                       matrix_m,
                       matrix_a,
                       matrix_k,
                       matrix_b,
                       matrix_gf,
                       disp,
                       velo):
        """An iterative solver that calculates the instantaneous
        acceleration, speed and displacement

        """

        # step 1: initial conditions
        matrix_ma = matrix_m + matrix_a

        ddq = np.linalg.inv(matrix_ma) @ (matrix_gf - matrix_b @ velo - matrix_k @ disp )

        # step 2: prediction
        ddq_new = ddq
        dq_new = velo + change_time* ddq
        q_new = disp + change_time* velo + 0.5* change_time**2 *ddq
        k = 1
        iteration = 0

        # while loop for calculation
        while k == 1:

            iteration +=  1

            # step 3: calculate residual

            res = matrix_gf - matrix_ma @ ddq_new - matrix_k @ q_new - matrix_b @ dq_new

            res_max = np.max(np.abs(res))

            # step 4: increment correction

            k_star = matrix_k + self.dict_new_mark['coeff_1'] *\
                matrix_b + self.dict_new_mark['coeff_2'] *\
                matrix_ma

            delta_u = np.linalg.inv(k_star) @ res

            q_new = q_new + delta_u

            dq_new = dq_new + self.dict_new_mark['coeff_1']* delta_u

            ddq_new = ddq_new + self.dict_new_mark['coeff_2']* delta_u

            if res_max > self.dict_new_mark['eps_r'] and iteration < 100:
                k = 1
            else:
                k = 0

        disp = q_new
        velo = dq_new
        ddq = ddq_new
        return disp, velo, ddq

    def get_wind_wave_csv_path(self):
        """gets the paths for the envirnomental data as their paths are very very very long
        """

        current = Path('C:/pysydyn')
        wave_series_path = (
            current / 'Wind_Wave' / 'Wind_and_Wave_series' /
            f'SerHs{self.dict_windy_wavy["wave_height"]}'
            f'tp{self.dict_windy_wavy["wave_period"]}'
            f'zb{self.dict_windy_wavy["sea_floor"]}'
            f'V{self.dict_windy_wavy["wind_mean"]}'
            f'TI{self.dict_windy_wavy["turb_intens"]}'
            f'zh{self.dict_windy_wavy["hub_height"]}'
            f't{self.dict_windy_wavy["time_span"]}'
            f'dt{self.dict_windy_wavy["time_change"]}'
            f'wit{self.dict_windy_wavy["wind_toggle"]}'
            f'wat{self.dict_windy_wavy["wave_toggle"]}.csv')

        path_save_u = (
        current / 'Wind_Wave' / 'Wave_speed' /
            f'Wave_speed_Hs{self.dict_windy_wavy["wave_height"]}'
            f'tp{self.dict_windy_wavy["wave_period"]}'
            f'zbot{self.dict_windy_wavy["sea_floor"]}'
            f't{self.dict_windy_wavy["time_span"]}'
            f'dt{self.dict_windy_wavy["time_change"]}'
            f'wat{self.dict_windy_wavy["wave_toggle"]}.csv')

        path_save_a = (
            current / 'Wind_Wave' / 'Wave_acceleration' /
            f'Wave_acceleration_Hs{self.dict_windy_wavy["wave_height"]}'
            f'tp{self.dict_windy_wavy["wave_period"]}'
            f'zbot{self.dict_windy_wavy["sea_floor"]}'
            f't{self.dict_windy_wavy["time_span"]}'
            f'dt{self.dict_windy_wavy["time_change"]}'
            f'wat{self.dict_windy_wavy["wave_toggle"]}.csv')

        return wave_series_path, path_save_u, path_save_a

    def get_floaty_response_csv_path(self):
        """gets the paths for the floaty data as it is also very very long
            the final part is removes as it makes the file name too long, unfortunatly
        """
        current = Path('C:/pysydyn')

        path_save_floaty_response = (
            current / 'Floater_Response' /
            f'SerHs{self.dict_windy_wavy["wave_height"]}'
            f'tp{self.dict_windy_wavy["wave_period"]}'
            f'zb{self.dict_windy_wavy["sea_floor"]}'
            f'V{self.dict_windy_wavy["wind_mean"]}'
            f'TI{self.dict_windy_wavy["turb_intens"]}'
            f'zh{self.dict_windy_wavy["hub_height"]}'
            f't{self.dict_windy_wavy["time_span"]}'
            f'dt{self.dict_windy_wavy["time_change"]}'
            f'wit{self.dict_windy_wavy["wind_toggle"]}'
            f'wat{self.dict_windy_wavy["wave_toggle"]}.csv')
            #f'{self.dict_file_names["file_floaty"][:-4]}'
            #f'{self.dict_file_names["file_turbie"][:-4]}

        return path_save_floaty_response

    def get_psd_and_plot_response_and_psd(self,
                                          change_time,
                                          transient,
                                          sig_list,
                                          sig_labels,
                                          sig_units,
                                          save_plot_path):
        """plots the data with the psd too
        """

        # sampling frequency
        sampling_freq = 1/change_time

        # number of time series
        num_time_series = len(sig_list)

        transient_time_step = int(round(transient/change_time,0))

        # working out time steps based on length of list
        n_time = len(sig_list[0])

        # I don't know why, but I have to divide the values of time_series by 2
        time_series = np.arange(change_time, n_time*change_time, change_time)

        # create subplots
        fig, full_plot = plt.subplots(num_time_series, 2, figsize=(15, 3*num_time_series))
        fig.suptitle('Time Series vs. PSD', fontsize=16)

        for i in range(num_time_series):
            # compute the power spectral density
            freq_array, pxx_den = signal.welch(sig_list[i][transient_time_step:],
                                               sampling_freq,
                                               nperseg=len(sig_list[i][transient_time_step:]))
            if len(time_series[:transient_time_step]) > len(sig_list[i][:transient_time_step]):
                time_series = time_series[:-1]
            if len(time_series[transient_time_step:]) > len(sig_list[i][transient_time_step:]):
                time_series = time_series[:-1]
            if len(time_series[:transient_time_step]) < len(sig_list[i][:transient_time_step]):
                sig_list[i] = sig_list[i][:-1]
            if len(time_series[transient_time_step:]) < len(sig_list[i][transient_time_step:]):
                sig_list[i] = sig_list[i][:-1]


            if num_time_series == 1:
                full_plot[0].plot(time_series[:transient_time_step],
                           sig_list[i][:transient_time_step],
                           '--',
                           label = 'Transient Time',
                           color='red',
                           linewidth=0.7)
                full_plot[0].plot(time_series[transient_time_step:],
                           sig_list[i][transient_time_step:],
                           label = 'After Transient',
                           color='blue',
                           linewidth=0.7)
                full_plot[0].set_ylabel(sig_labels[i] +' ['+ sig_units[i]+ ']')
                full_plot[0].set_xlabel('Time [s]')
                full_plot[0].legend()
                full_plot[0].grid()

                # plot the PSD
                full_plot[1].plot(freq_array, pxx_den)
                full_plot[1].set_ylabel('PSD('+sig_labels[i]+')')
                full_plot[1].set_xlabel('frequency [Hz]')
                full_plot[1].set_xlim(0,0.5)
                full_plot[1].grid()
            else:
                # plot the time series
                full_plot[i, 0].plot(time_series[:transient_time_step],
                              sig_list[i][:transient_time_step], '--',
                              label = 'Transient Time',
                              color='red',
                              linewidth=0.7)
                full_plot[i, 0].plot(time_series[transient_time_step:],
                              sig_list[i][transient_time_step:],
                              label = 'After Transient',
                              color='blue',
                              linewidth=0.7)
                full_plot[i, 0].set_ylabel(sig_labels[i] +' ['+ sig_units[i]+ ']')
                full_plot[i, 0].set_xlabel('Time [s]')
                # ax[i, 0].set_ylim(0,10)
                full_plot[i, 0].legend()
                full_plot[i, 0].grid()

                # plot the PSD
                full_plot[i, 1].plot(np.real(freq_array), pxx_den, linewidth=0.7)
                full_plot[i, 1].set_ylabel('PSD('+sig_labels[i]+')')
                full_plot[i, 1].set_xlabel('frequency [Hz]')
                full_plot[i, 1].set_xlim(0,0.5)
                # ax[i, 1].set_ylim(0,np.mean(Pxx_den)*30) # Need to upgrade to have dynamic ylim
                full_plot[i, 1].grid()

        # adjust the layout
        plt.tight_layout()
        plt.show()

        if save_plot_path:
            fig.savefig(save_plot_path)
            print('Figure saved at:')
            print(save_plot_path)

    def run_simulation(self):
        """runs the full simulation
        """

        # simulation set up
        time_span = self.dict_windy_wavy['time_span']
        change_time = self.dict_windy_wavy['time_change']
        t_vec = np.arange(change_time, time_span, change_time)

        # initialize a panda data frame (easier to export)
        columns_df = ['time','q_surge','q_heave','q_pitch',
                   'dq_surge','dq_heave','dq_pitch',
                   'ddq_surge','ddq_heave','ddq_pitch']
        df_sim_result = pd.DataFrame(index = np.arange(0, len(t_vec)), columns = columns_df)

        # load pre-calculated wind and wave csv files
        wave_series_path, path_save_u, path_save_a = self.get_wind_wave_csv_path()

        wind_data = pd.read_csv(wave_series_path).values
        # load into attribute
        self.wind_velocity = wind_data[:,1]
        self.fse = wind_data[:,2]
        self.wave_speed = pd.read_csv(path_save_u, header=None).values
        self.wave_acceleration = pd.read_csv(path_save_a, header=None).values

        floater_damping_constant = 0.9
        print('Calculating response')
        for time_index, time in tqdm(enumerate(t_vec), total=len(t_vec)):

            # updating q and dq from the previous time step to be used in all the matricies
            if time_index == 0:
                displacement = np.zeros([3])
                velocity = np.zeros([3])

            else:
                displacement = df_sim_result.loc[time_index - 1,
                                                 ['q_surge', 'q_heave', 'q_pitch']].values
                velocity = df_sim_result.loc[time_index - 1,
                                             ['dq_surge', 'dq_heave', 'dq_pitch']].values

            # update all matricies using methods
            matrix_gf = self.get_gf(time_index, state_vec_q = displacement, state_vec_dq = velocity)
            matrix_m = self.get_matrix_m(q_displacement = displacement)
            matrix_a = self.get_matrix_a(q_displacement = displacement)
            matrix_k = self.get_matrix_k(q_displacement = displacement)
            matrix_b = self.dict_sys_matrices['matrix_b']

            # run the simulation with all the new matricies collected
            displacement, velocity, ddq = self.newmark_solver(change_time,
                                             matrix_m,
                                             matrix_a,
                                             matrix_k,
                                             matrix_b,
                                             matrix_gf,
                                             displacement,
                                             velocity)

            force_bouy = self.dict_windy_wavy["gravity"]*\
                         self.dict_windy_wavy["rho_water"]*\
                         np.pi*(self.fse[time_index] +
                                          displacement[1] -
                                          self.dict_floaty["z_bot_floaty"])*\
                                          (self.dict_floaty["d_tank"]/2)**2
            force_grav = self.dict_windy_wavy["gravity"]*((10**7) + 547000 + 674000)

            instant_a = (force_bouy - force_grav)/(self.dict_floaty["m_tot_floaty"] +
                                                   self.dict_turbie["m_tower"] +
                                                   self.dict_turbie["m_turbie"])
            prior_vel=velocity[1]
            ddq[1] = ddq[1] + instant_a
            velocity[1] = floater_damping_constant * (velocity[1] +
                                                     (self.dict_windy_wavy["time_change"]*ddq[1]))

            displacement[1] = displacement[1] + (self.dict_windy_wavy["time_change"]*prior_vel)

            # saving the results away
            df_sim_result.loc[time_index, ['time']] = time
            df_sim_result.loc[time_index, ['q_surge','q_heave','q_pitch']] = displacement.flatten()
            df_sim_result.loc[time_index, ['dq_surge','dq_heave','dq_pitch']] = velocity.flatten()
            df_sim_result.loc[time_index, ['ddq_surge','ddq_heave','ddq_pitch']] = ddq.flatten()

        return df_sim_result

    def run_program(self,save_response = False,
                    show_plot = True,
                    you_decide_plot_name = False,
                    show_windywavy_plot = True):
        """the single command that runs the package as a whole
        """

        # load floater response file
        floaty_response_csv_path = self.get_floaty_response_csv_path()
        print(floaty_response_csv_path)

        # check floater_resp.csv file
        if Path.is_file(floaty_response_csv_path) is True:
            print('simulation results exist already in folder, loading the response csv file')
            df_sim_result = pd.read_csv(floaty_response_csv_path)
        else:
            # floater_resp.csv does not exist, run windy_wavy
            print('Checks whether wind_and_wave_series exist already')
            self.run_windy_wavy()

            # generates floater response
            df_sim_result = self.run_simulation()

            if save_response is True:
                df_sim_result.to_csv(floaty_response_csv_path)
                print('Floater response csv file saved:')
                print(floaty_response_csv_path)

                df_sim_result = pd.read_csv(floaty_response_csv_path)
            else:
                print('Floater response csv file not saved')

        if show_windywavy_plot is True:
            self.plot_windy_wavy()


        if show_plot is True:
            sig_list = [ df_sim_result['q_surge'],
                        df_sim_result['q_heave'],
                        df_sim_result['q_pitch']]
            sig_labels = ['Surge Response', 'Heave Response', 'Pitch Response']
            sig_units = ['m', 'm', 'rad']
            change_time = df_sim_result['time'][2] - df_sim_result['time'][1]
            transient = 0.5 * max(df_sim_result['time'])

            current = Path('C:/pysydyn')
            you_decide_plot_name = str(you_decide_plot_name) + '.png'
            save_plot_path = current / 'Floater_Response' / you_decide_plot_name

            self.get_psd_and_plot_response_and_psd(change_time, transient,
                                                   sig_list, sig_labels,
                                                   sig_units,
                                                   save_plot_path)

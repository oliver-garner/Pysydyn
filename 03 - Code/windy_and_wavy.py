"""
Module: windy_and_wavy

This module contains the implementation of the WindyWavy class, which handles calculations
related to wind and waves.
"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.fft import fft

# This is a parent class
class WindyWavy:
    """The class that is reponsible for generating wind and wave data for the package

    Args:
        object (_type_): _description_
    """
    def __init__(self, file_windy_wavy):
        """initialisation for the windy_wavy class that will create a dictionary for its variables

        Args:
            file_windy_wavy (str): string which defines the path to the windy wavy config file
        """
        # load file path
        current = Path('C:/pysydyn')

        # read the config file
        windy_wavy_path = current / 'Configuration_Files' / file_windy_wavy
        windy_wavy_df = pd.read_csv(windy_wavy_path)

        # load into dictonary
        dict_windy_wavy = {}
        for _ , row in windy_wavy_df.iterrows():
            dict_windy_wavy[row['variables']] = row['values']

        self.dict_windy_wavy = dict_windy_wavy


    def run_windy_wavy(self):
        """groups all the corresponding methods so that the entire code can be run with one method
        """
        current = Path('C:/pysydyn')
        file_name = (
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
        if Path.is_file(file_name) is True:
            print("Wind and wave time series files already exist")
        else:
            print("Wind and wave time series file(s) are missing")
            wind_tog = self.dict_windy_wavy["wind_toggle"]
            wave_tog = self.dict_windy_wavy["wave_toggle"]
            time_span = np.arange(self.dict_windy_wavy["time_change"],
                                  self.dict_windy_wavy["time_span"],
                                  self.dict_windy_wavy["time_change"])

            if wind_tog == 2 and wave_tog == 2:
                print("Generating irregular wind and irregular wave")
                wind_vel = self.calc_kaimal()
                free_surface_elevation = self.calc_jonswap()
                current = Path('C:/pysydyn')
                path_windy_wavy_series = (
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
                np.savetxt(path_windy_wavy_series,
                           np.c_[time_span, wind_vel, free_surface_elevation],
                           delimiter=',',
                           fmt='%.3f',
                           header="Time[s],V[m/s],free_surface_elevation[m]")

            elif wind_tog == 1 and wave_tog == 2:
                print("Generating regular wind and irregular waves")
                wind_vel = self.dict_windy_wavy["wind_mean"] + np.zeros(len(time_span))
                free_surface_elevation = self.calc_jonswap()
                current = Path('C:/pysydyn')
                path_windy_wavy_series = (
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
                np.savetxt(path_windy_wavy_series,
                           np.c_[time_span, wind_vel, free_surface_elevation],
                           delimiter=',',
                           fmt='%.3f',
                           header="Time[s],V[m/s],free_surface_elevation[m]")

            elif wind_tog == 2 and wave_tog == 1:
                print("Generating irregular wind and regular waves")
                wind_vel = self.calc_kaimal()
                free_surface_elevation = self.calc_jonswap()
                current = Path('C:/pysydyn')
                path_windy_wavy_series = (
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
                np.savetxt(path_windy_wavy_series,
                           np.c_[time_span,wind_vel, free_surface_elevation],
                           delimiter=',',
                           fmt='%.3f',
                           header="Time[s],V[m/s],free_surface_elevation[m]")
            elif wind_tog == 1 and wave_tog == 1:
                print("Generating regular wind and regular waves")
                wind_vel = self.dict_windy_wavy["wind_mean"] + np.zeros(len(time_span))
                free_surface_elevation = self.calc_jonswap()
                current = Path('C:/pysydyn')
                path_windy_wavy_series = (
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
                np.savetxt(path_windy_wavy_series,
                           np.c_[time_span,wind_vel, free_surface_elevation],
                           delimiter=',',
                           fmt='%.3f',
                           header="Time[s],V[m/s],free_surface_elevation[m]")
            else:
                print("wind or wave toggle is incorrect, choose either 1 or 2")

    def calc_psd(self,time_series, data):
        """A function that calculated the PSD data from a time series and coresponding signal data

        Args:
            time_series (numpy.ndarray): time span data
            data (numpy.ndarray): data that relates to the time span

        Returns:
            fpsd (numpy.ndarray): frequency spectrum
            psd (numpy.ndarray): power rating corresponding to the frequency spectrum
        """
        change_freq = 1 / (time_series[-1] - time_series[0])
        fpsd = change_freq * np.arange(len(time_series))
        signalhat = fft(data) / len(time_series)
        signalhat[0] = 0
        signalhat[len(fpsd)//2:] = 0
        signalhat = 2 * signalhat
        psd = np.abs(signalhat)**2 / (2 * change_freq)
        return fpsd, psd

    def plot_windy_wavy(self):
        """plots the wind speed and wave height plots
        """
        wave_series_path, _, _= self.get_wind_wave_csv_path()

        wind_data = pd.read_csv(wave_series_path).values
        fig, windy_wavy_plot = plt.subplots(1, 2, figsize=(15, 3))
        fig.suptitle('Environmental conditions', fontsize=16)
        windy_wavy_plot[0].plot(wind_data[:,0],
                                wind_data[:,1],
                                label = 'Wind Speed',
                                linewidth=0.7)
        windy_wavy_plot[0].set_ylabel('Wind Speed [m/s]')
        windy_wavy_plot[0].set_xlabel('Time [s]')
        windy_wavy_plot[0].legend()
        windy_wavy_plot[0].grid()
        windy_wavy_plot[1].plot(wind_data[:,0],
                                wind_data[:,2],
                                label = 'Wave free surface elevation',
                                linewidth=0.7)
        windy_wavy_plot[1].set_ylabel('free surface elevation [m]')
        windy_wavy_plot[1].set_xlabel('Time [s]')
        windy_wavy_plot[1].legend()
        windy_wavy_plot[1].grid()
        plt.tight_layout()
        plt.show()

    def ksolve(self, freq):
        """calculated the 'k' values required to generated the random wave data

        Args:
            freq (numpy.ndarray): array of freqency data

        Returns:
            k_solution (numpy.ndarray): k values for all the frequency values
        """
        k_solution = np.zeros(len(freq))
        gravity = 9.81

        for index, freq_value in enumerate(freq):
            omega_wave = 2 *np.pi *freq_value

            k_star = 1  # initial test value
            error = 1  # initial value
            while error > 0.0001:
                k_estimate = omega_wave ** 2 /\
                (gravity * np.tanh(k_star * self.dict_windy_wavy["sea_floor"]))
                error = abs(k_estimate - k_star)
                k_star = k_estimate

            k_solution[index] = k_star

        return k_solution

    def calc_jonswap(self):
        """calculates the 'random' wave data using the JONSWAP wave structure and spectrum

        Returns:
            eta (numpy.ndarray): free surface elevation or wave height generated from the JONSWAP
        """
        time_series = np.arange(self.dict_windy_wavy["time_change"],
                                self.dict_windy_wavy["time_span"],
                                self.dict_windy_wavy["time_change"])
        change_freq= 1/self.dict_windy_wavy["time_span"]
        freq = np.arange(change_freq,0.5,change_freq)
        js_spec = np.zeros(len(freq))
        wave_freq = 1/self.dict_windy_wavy["wave_period"]

        if self.dict_windy_wavy["wave_period"]/(np.sqrt(self.dict_windy_wavy["wave_height"]))<=3.6:
            gamma = 5
        elif self.dict_windy_wavy["wave_period"]/(np.sqrt(self.dict_windy_wavy["wave_height"]))>=5:
            gamma = 1
        else:
            gamma = np.exp(5.75 - 1.15*(self.dict_windy_wavy["wave_period"]/
                                        (np.sqrt(self.dict_windy_wavy["wave_height"]))))

        for index, freq_value in enumerate(freq):
            if freq_value <= wave_freq:
                sigma = 0.07
            else:
                sigma = 0.09
            js_spec[index] = 0.3125* (self.dict_windy_wavy["wave_height"]**2)*\
                self.dict_windy_wavy["wave_period"]*\
                ((freq_value/wave_freq)**(-5))* np.exp(-1.25*((freq_value/wave_freq)**(-4)))*\
                (1-(0.287*np.log(gamma)))*\
                gamma**(np.exp(-0.5*(((freq_value/wave_freq)-1)/sigma)**2))

        a_j = np.sqrt(2*js_spec*(freq[1]-freq[0]))
        eta = np.zeros(len(time_series))
        change_depth = self.dict_windy_wavy["floater_bottom"]*0.01*1.5
        z_phys = np.arange(0, self.dict_windy_wavy["floater_bottom"]*1.5, change_depth)

        omega_wave = 2*np.pi*freq

        k_solutions = self.ksolve(freq)

        wave_speed = np.zeros((len(z_phys),len(time_series)))
        wave_accel = np.zeros((len(z_phys),len(time_series)))
        epsi_wave = 2*np.pi*np.random.rand(len(freq))
        print('Calculating wave data')
        for time_index, time_series_value in tqdm(enumerate(time_series), total=len(time_series)):
            eta[time_index] = sum(a_j*np.cos((omega_wave*time_series_value)+(epsi_wave)))

            for height_index, height in enumerate(z_phys):
                inter_speed = sum((a_j*omega_wave)*
                                  (np.cosh(k_solutions*(-height +
                                                        self.dict_windy_wavy["sea_floor"])))/
                                  (np.sinh(k_solutions*self.dict_windy_wavy["sea_floor"]))*
                                  (np.cos((time_series_value* omega_wave) + (epsi_wave))))
                wave_speed[height_index, time_index] = inter_speed
                inter_accel = sum(-a_j*omega_wave*
                                  (np.cosh(k_solutions*(-height +
                                                        self.dict_windy_wavy["sea_floor"])))/
                                  (np.sinh(k_solutions*self.dict_windy_wavy["sea_floor"]))*
                                  (np.sin((time_series_value* omega_wave) + (epsi_wave))))
                wave_accel[height_index, time_index] = inter_accel

        current = Path('C:/pysydyn')
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
        np.savetxt(fname=path_save_u, X=wave_speed, delimiter=',')
        np.savetxt(fname=path_save_a, X=wave_accel, delimiter=',')

        return eta

    def calc_kaimal(self):
        """calculates the 'random' wind speed data using the kaimal method

        Returns:
            v_spectrum (numpy.ndarray): horizontal wind speed generated from the kaimal
        """
        time_series = np.arange(self.dict_windy_wavy["time_change"],
                                self.dict_windy_wavy["time_span"],
                                self.dict_windy_wavy["time_change"])

        change_freq = 1/self.dict_windy_wavy["time_span"]
        freq = np.arange(change_freq,0.5,change_freq)
        v_spectrum   = np.zeros(len(time_series))
        spec_wind = np.zeros(len(freq))

        if self.dict_windy_wavy["hub_height"] >= 60:
            turb_len = 340.2
        else:
            turb_len = 5.67*self.dict_windy_wavy["hub_height"]

        for index, freq_value in enumerate(freq):
            spec_wind[index] = (4* self.dict_windy_wavy["wind_mean"]*turb_len*
                                (self.dict_windy_wavy["turb_intens"]**2))/\
                                ((1 + 6*(freq_value* turb_len/
                                         self.dict_windy_wavy["wind_mean"]))**(5/3))

        b_p = np.sqrt(2*spec_wind*change_freq)

        omega_p = 2*np.pi*freq
        epsi_wind = 2*np.pi*np.random.rand(len(freq))
        print('Generating wind data')
        for time_loop, time_enu in tqdm(enumerate(time_series), total=len(time_series)):
            v_spectrum[time_loop] = self.dict_windy_wavy["wind_mean"] + \
            sum(b_p[freq_loop]*np.cos((omega_p[freq_loop] * time_enu) +
                                      (epsi_wind[freq_loop])) for freq_loop in range(len(freq)))


        return v_spectrum

    def calc_regular_wave(self):
        """calculates the regular wind data

        Returns:
            eta (numpy.ndarray): free surface elevation or wave height generated
        """
        time_series = np.arange(self.dict_windy_wavy["time_change"],
                                self.dict_windy_wavy["time_span"],
                                self.dict_windy_wavy["time_change"])
        change_depth = self.dict_windy_wavy["floater_bottom"]*0.01
        z_phys = np.arange(0, self.dict_windy_wavy["floater_bottom"], change_depth)
        omega = 2*np.pi / self.dict_windy_wavy["wave_period"]
        k_sol = self.ksolve(np.array([1/self.dict_windy_wavy["wave_period"]]))

        eta = np.zeros(len(time_series))
        wave_speed = np.zeros((len(z_phys),len(time_series)))
        wave_accel = np.zeros((len(z_phys),len(time_series)))
        for time_index, time_series_value in enumerate(time_series):
            eta[time_index] = 0.5*self.dict_windy_wavy["wave_height"]*\
                np.cos(omega*time_series_value)

            for height_index, height in enumerate(z_phys):
                inter_speed = (0.5*omega*self.dict_windy_wavy["wave_height"])*\
                    np.cosh(k_sol * (-height + self.dict_windy_wavy["sea_floor"]))/\
                    np.sinh(k_sol * self.dict_windy_wavy["sea_floor"])*\
                    np.cos(omega * time_series_value)
                wave_speed[height_index, time_index] = inter_speed
                inter_accel = -(0.5*omega*self.dict_windy_wavy["wave_height"])*\
                    np.cosh(k_sol * (-height + self.dict_windy_wavy["sea_floor"]))/\
                    np.sinh(k_sol * self.dict_windy_wavy["sea_floor"])*\
                    np.sin(omega * time_series_value)
                wave_accel[height_index, time_index] = inter_accel


        current = Path('C:/pysydyn')
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
        np.savetxt(fname=path_save_u, X=wave_speed, delimiter=',')
        np.savetxt(fname=path_save_a, X=wave_accel, delimiter=',')

        return eta

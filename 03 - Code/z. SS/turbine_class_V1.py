from pathlib import Path
import numpy as np
from scipy.fft import fft


class Windy_wavy(object):
    def __init__(self, dict_windy_wave):
        self.dict_windy_wave = dict_windy_wave
        
    def run_windy_wavy(self):
        current = Path.cwd()
        file_name = current / 'Wind_Wave' / 'Wind_and_Wave_series' / f'Series_Hs_{self.dict_windy_wave["wave_height"]}_tp_{self.dict_windy_wave["wave_period"]}_z_bot_{self.dict_windy_wave["sea_floor"]}_V_{self.dict_windy_wave["wind_mean"]}_TI_{self.dict_windy_wave["turb_intens"]}_z_hub_{self.dict_windy_wave["hub_height"]}_time_{self.dict_windy_wave["time_span"]}_dt_{self.dict_windy_wave["time_change"]}.csv'

        if Path.is_file(file_name) is True:
            print("files already exist")
        else:
            print("file(s) are missing")
            file_name = current / 'Wind_Wave' / 'Wave_speed' / f'Wave_speed_Hs_{self.dict_windy_wave["wave_height"]}_tp_{self.dict_windy_wave["wave_period"]}_zbot_{self.dict_windy_wave["sea_floor"]}_time_{self.dict_windy_wave["time_span"]}_dt_{self.dict_windy_wave["time_change"]}.csv'
            wind_vel = Windy_wavy.calc_kaimal(self)
            free_surface_elevation = Windy_wavy.calc_jonswap(self)
            current = Path.cwd()
            path_windy_wavy_series = current / 'Wind_Wave' / 'Wind_and_Wave_series' / f'Series_Hs_{self.dict_windy_wave["wave_height"]}_tp_{self.dict_windy_wave["wave_period"]}_z_bot_{self.dict_windy_wave["sea_floor"]}_V_{self.dict_windy_wave["wind_mean"]}_TI_{self.dict_windy_wave["turb_intens"]}_z_hub_{self.dict_windy_wave["hub_height"]}_time_{self.dict_windy_wave["time_span"]}_dt_{self.dict_windy_wave["time_change"]}.csv'
            np.savetxt(path_windy_wavy_series, np.c_[np.arange(self.dict_windy_wave["time_change"],self.dict_windy_wave["time_span"],self.dict_windy_wave["time_change"]),\
                                                     wind_vel, free_surface_elevation], delimiter=',', fmt='%.3f', header="Time[s],V[m/s],free_surface_elevation[m]")

    def calc_psd(time_series, data):
        change_freq = 1 / (time_series[-1] - time_series[0])                     # Frequency resolution
        fpsd = change_freq * np.arange(len(time_series))               # Frequency vector starts from zero, as long as t
        signalhat = fft(data) / len(time_series)              # Fourier amplitudes
        signalhat[0] = 0                            # Discard first value (mean)
        signalhat[len(fpsd)//2:] = 0                # Discard all above Nyquist freq
        signalhat = 2 * signalhat                   # Make amplitudes one-sided
        psd = np.abs(signalhat)**2 / (2 * change_freq)       # Calculate spectrum
        return fpsd, psd


    def ksolve(freq,h_depth):
        k_solution = np.zeros(len(freq))
        gravity = 9.81
        for freq_loop in range(len(freq)):
            omega_wave = 2*np.pi*freq[freq_loop]
            k_star=1 #initial test value
            error=1 #initial value
            while error>0.0001:
                k_estimate=omega_wave**2/(gravity*np.tanh(k_star*h_depth))
                error = abs(k_estimate-k_star)
                k_star=k_estimate

            k_solution[freq_loop]=k_star

        return k_solution


    def calc_jonswap(self):
        time_series = np.arange(self.dict_windy_wave["time_change"],self.dict_windy_wave["time_span"],self.dict_windy_wave["time_change"])
        change_freq= 1/self.dict_windy_wave["time_span"]
        freq = np.arange(change_freq,0.5,change_freq)
        js_spec = np.zeros(len(freq))
        wave_freq = 1/self.dict_windy_wave["wave_period"]

        if self.dict_windy_wave["wave_period"]/(np.sqrt(self.dict_windy_wave["wave_height"]))<=3.6:
            gamma = 5
        elif self.dict_windy_wave["wave_period"]/(np.sqrt(self.dict_windy_wave["wave_height"]))>=5:
            gamma = 1
        else:
            gamma = np.exp(5.75 - 1.15*(self.dict_windy_wave["wave_period"]/(np.sqrt(self.dict_windy_wave["wave_height"]))))

        for freq_loop in range(len(freq)):
            if freq[freq_loop] <= wave_freq:
                sigma = 0.07
            else:
                sigma = 0.09

            js_spec[freq_loop] = 0.3125*(self.dict_windy_wave["wave_height"]**2)*self.dict_windy_wave["wave_period"]*((freq[freq_loop]/wave_freq)**(-5))*np.exp(-1.25*((freq[freq_loop]/wave_freq)**(-4)))*(1-(0.287*np.log(gamma)))*gamma**(np.exp(-0.5*(((freq[freq_loop]/wave_freq)-1)/sigma)**2))

        a_j = np.sqrt(2*js_spec*(freq[1]-freq[0]))
        eta = np.zeros(len(time_series))
        change_depth = self.dict_windy_wave["floater_bottom"]*0.01
        z_phys = np.arange(0, self.dict_windy_wave["floater_bottom"], change_depth)
        omega_wave = 2*np.pi*freq

        k_solutions = Windy_wavy.ksolve(freq, self.dict_windy_wave["sea_floor"])

        wave_speed = np.zeros((len(z_phys),len(time_series)))
        wave_accel = np.zeros((len(z_phys),len(time_series)))
        epsi_wave = 2*np.pi*np.random.rand(len(freq))

        for time_loop in range(len(time_series)):
            eta[time_loop] = sum(a_j*np.cos((omega_wave*time_series[time_loop])+(epsi_wave)))

            for height_loop in range(len(z_phys)):
                wave_speed[height_loop,time_loop] = sum((a_j*omega_wave)*(np.cosh(k_solutions*(-z_phys[height_loop]+self.dict_windy_wave["sea_floor"])))/(np.sinh(k_solutions*self.dict_windy_wave["sea_floor"]))*(np.cos((time_series[time_loop]*omega_wave)+(epsi_wave))))
                wave_accel[height_loop,time_loop] = sum(-a_j*omega_wave*(np.sinh(k_solutions*(-z_phys[height_loop]+self.dict_windy_wave["sea_floor"])))/(np.sinh(k_solutions*self.dict_windy_wave["sea_floor"]))*(np.sin((time_series[time_loop]*omega_wave)+(epsi_wave))))

        current = Path.cwd()
        path_save_u = current / 'Wind_Wave' / 'Wave_speed' / f'Wave_speed_Hs_{self.dict_windy_wave["wave_height"]}_tp_{self.dict_windy_wave["wave_period"]}_zbot_{self.dict_windy_wave["sea_floor"]}_time_{self.dict_windy_wave["time_span"]}_dt_{self.dict_windy_wave["time_change"]}.csv'
        path_save_a = current / 'Wind_Wave' / 'Wave_acceleration' / f'Wave_acceleration_Hs_{self.dict_windy_wave["wave_height"]}_tp_{self.dict_windy_wave["wave_period"]}_zbot_{self.dict_windy_wave["sea_floor"]}_time_{self.dict_windy_wave["time_span"]}_dt_{self.dict_windy_wave["time_change"]}.csv'
        np.savetxt(fname=path_save_u, X=wave_speed, delimiter=',')
        np.savetxt(fname=path_save_a, X=wave_accel, delimiter=',')

        return eta

    def calc_kaimal(self):
        time_series = np.arange(self.dict_windy_wave["time_change"],self.dict_windy_wave["time_span"],self.dict_windy_wave["time_change"])
        change_freq = 1/self.dict_windy_wave["time_span"]
        freq = np.arange(change_freq,0.5,change_freq)
        v_spectrum   = np.zeros(len(time_series))
        spec_wind = np.zeros(len(freq))

        if self.dict_windy_wave["hub_height"] >= 60:
            turb_len = 340.2
        else:
            turb_len = 5.67*self.dict_windy_wave["hub_height"]

        for f in range(len(freq)):
            spec_wind[f] = (4*self.dict_windy_wave["wind_mean"]*turb_len*(self.dict_windy_wave["turb_intens"]**2))/((1+6*(freq[f]*turb_len/self.dict_windy_wave["wind_mean"]))**(5/3))

        b_p = np.sqrt(2*spec_wind*change_freq)

        omega_p = 2*np.pi*freq
        epsi_wind = 2*np.pi*np.random.rand(len(freq))

        for time_loop in range(len(time_series)):
            v_spectrum[time_loop] = self.dict_windy_wave["wind_mean"] + sum(b_p[freq_loop]*np.cos((omega_p[freq_loop] * time_series[time_loop]) + (epsi_wind[freq_loop])) for freq_loop in range(len(freq)))


        return v_spectrum

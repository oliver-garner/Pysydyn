# Old constant file
# user defined simulation constants
t_span = 1000
t_change = 0.1
q_init = np.array([0, 0, 0, 0, 0, 0])
q_init = q_init.T           # initial state vector


# user defined wind and wave constants
cm = 1
cd = 0.6
rho_water = 1025
rho_air = 1.22
a = 0.5
b = 0.65
g = 9.81
f_high_cut = 0.5

dict_windy_wavy = {'cm': cm,
                   'cd': cd,
                   'rho_air': rho_air,
                   'rho_water': rho_water,
                   'a': a,
                   'b': b,
                   'g': g,
                   'f_high_cut': f_high_cut}

#%%
# user input of turbine data
m_turbie = 446036 + 227962  # nacelle + rotor
m_tower = 5.4692e5          
z_cm_tower = 56.4
i_cm_tower = 4.2168e8
z_hub = 119             # z_cm_hub
d_rotor = 178
v_rated = 11.4
x_cm_hub = 0       
ct_0 = 0.81

# pass them all into the dictonary
dict_turbie = {'m_turbie': m_turbie,
               'm_tower': m_tower,
                'z_cm_tower': z_cm_tower,
                'i_cm_tower': i_cm_tower,
                'z_hub': z_hub,
                'd_rotor': d_rotor,
                'v_rated': v_rated,
                'x_cm_hub': x_cm_hub,
                'ct_0': ct_0,
                }

#%%
# User input of floater data
m_tot_floaty = 1.0897e7
z_cm_floaty = -105.95
x_cm_floaty = 0
i_cm_floaty = 1.1627e10
i_x_area_floaty = 0
z_bot_floaty = -120
d_tank = 11.2           # d_spar
k_moor_x = 66700
k_moor_z = 0
z_moor = -60        
no_of_tanks_per_corner = 1
no_of_corners = 1
pcd_tanks = 1000

# no easy way to calculate matrix b, to be done by WAMIT
matrix_b = np.array([[2e5, 0, 0],
                     [0, 0, 0],
                     [0, 0, 0]])

dict_floaty = {'m_tot_floaty': m_tot_floaty,
                'z_cm_floaty': z_cm_floaty,
                'x_cm_floaty': x_cm_floaty,
                'i_cm_floaty': i_cm_floaty,
                'i_x_area_floaty': i_x_area_floaty,
                'z_bot_floaty': z_bot_floaty,
                'd_tank': d_tank,
                'k_moor_x': k_moor_x,
                'k_moor_z': k_moor_z,
                'z_moor': z_moor,
                'matrix_b': matrix_b,
                'no_of_tanks_per_corner': no_of_tanks_per_corner,
                'no_of_corners': no_of_corners,
                'pcd_tanks': pcd_tanks
                }
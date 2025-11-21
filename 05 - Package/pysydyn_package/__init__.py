
import os
from pathlib import Path
import shutil

def copy_file(source_file, destination_dir):
    try:
        # Check if source file exists
        if not os.path.exists(source_file):
            print("Source file does not exist.")
            return

        # Check if destination directory exists, create if not
        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir)

        # Copy file to destination directory
        shutil.copy2(source_file, destination_dir)
        print("File copied successfully.")
    except PermissionError as error_message:
        print(f"Permission error: {error_message}")
    except Exception as error_message:
        print(f"An error occurred: {error_message}")

current = os.getcwd()

# directory definitions
PARENT_DIR = "C:/"
DIRECTORY_MAIN = "pysydyn"
DIRECTORY_WIND_WAVE = "Wind_Wave"
DIRECTORY_CONFIGURATION_FILES = "Configuration_Files"
DIRECTORY_FLOATER_RESPONSE = "Floater_Response"
DESTINATION_DIR = "C:\\pysydyn\\Configuration_Files"

# folder name definitions
DIRECTORY_WINDY_WAVY_SPEED = "Wave_speed"
DIRECTORY_WINDY_WAVY_ACCELERATION = "Wave_acceleration"
DIRECTORY_WINDY_WAVY_SERIES = "Wind_and_Wave_series"

# main folder path definitions
path_main = os.path.join(PARENT_DIR, DIRECTORY_MAIN)
path_wind_wave = os.path.join(path_main, DIRECTORY_WIND_WAVE)
path_configuration_files = os.path.join(path_main, DIRECTORY_CONFIGURATION_FILES)
path_floater_response = os.path.join(path_main, DIRECTORY_FLOATER_RESPONSE)

# wind_wave folder path definitions
path_wind_wave_speed = os.path.join(path_wind_wave, DIRECTORY_WINDY_WAVY_SPEED)
path_wind_wave_acceleration = os.path.join(path_wind_wave, DIRECTORY_WINDY_WAVY_ACCELERATION)
path_wind_wave_series = os.path.join(path_wind_wave, DIRECTORY_WINDY_WAVY_SERIES)

current_script_dir = os.path.dirname(__file__)

# package executions
package_dir = os.path.abspath(os.path.join(current_script_dir, '..', 'pysydyn_package'))
floaty_path = os.path.join(current_script_dir, 'Configuration_Files', 'PysyDn_Floaty_Default.csv')
turbie_path = os.path.join(current_script_dir, 'Configuration_Files', 'PysyDyn_Turbie_Default.csv')
windy_wavy_path = os.path.join(current_script_dir, 'Configuration_Files', 'PysyDyn_Windy_Wavy_Default.csv')

# folder checking and initiations
if os.path.exists(path_main) is False:
    os.mkdir(path_main)
    print("folders initiated at " + path_main)
if os.path.exists(path_wind_wave) is False:
    os.mkdir(path_wind_wave)
    print(path_wind_wave + " created")
if os.path.exists(path_configuration_files) is False:
    os.mkdir(path_configuration_files)
    print(path_configuration_files + " created")
if os.path.exists(path_floater_response) is False:
    os.mkdir(path_floater_response)
    print(path_floater_response + " created")
if os.path.exists(path_wind_wave_speed) is False:
    os.mkdir(path_wind_wave_speed)
    print(path_wind_wave_speed + " created")
if os.path.exists(path_wind_wave_acceleration) is False:
    os.mkdir(path_wind_wave_acceleration)
    print(path_wind_wave_acceleration + " created")
if os.path.exists(path_wind_wave_series) is False:
    os.mkdir(path_wind_wave_series)
    print(path_wind_wave_series + " created")

# csv files transfered easier location
print(floaty_path)
print(turbie_path)
print(windy_wavy_path)
copy_file(floaty_path, DESTINATION_DIR)
copy_file(turbie_path, DESTINATION_DIR)
copy_file(windy_wavy_path, DESTINATION_DIR)

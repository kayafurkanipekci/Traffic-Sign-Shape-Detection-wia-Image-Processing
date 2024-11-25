import os
import platform
import shutil
from classifyimages import classifyImages

def clean_output_folder(output_folder):
    """To remove previous classification outputs"""
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)
    return output_folder

def main():
    if platform.system() == 'Windows':
        input_folder = 'traffic_Data\\DATA\\mix'
    elif platform.system() == 'Linux':
        input_folder = 'traffic_Data/DATA/mix/'

    classifyImages(input_folder, clean_output_folder('classified_symbols'))

if __name__ == "__main__":
    main()
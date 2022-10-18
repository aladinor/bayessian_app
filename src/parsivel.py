import pandas as pd
import xarray as xr
import glob


def main():
    variables = {'01': 'r_intensity', '02': 'acc_r', '03': 'Synop_4680', '04': 'Synop_4677', '05': ''}
    path = 'C:/Users/alfonso8/Downloads/0035215020/2021/10/02'
    txt_files = glob.glob(f'{path}/*.txt')
    with open(txt_files[0], 'r') as f:
        lines = f.readlines()
        print(1)

    df = pd.read_csv(txt_files[0], delimiter='\n')
    print(1)
    pass


if __name__ == "__main__":
    main()
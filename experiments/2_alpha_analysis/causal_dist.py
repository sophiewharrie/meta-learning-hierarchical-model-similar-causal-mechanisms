import sys
import pandas as pd

def get_data(input_simfile, simfile):
    df = pd.read_csv(input_simfile)

    alpha2_TOD = 1
    alpha2_TID = 0.3
    alpha1_TOD = 100
    alpha1_TID = 300

    alpha1_list = [100, 200, 400, 800, 1600, 3200]
    counter = 1
    print('perc of distance that is distributional distance')
    for alpha1 in alpha1_list:
        df['TOD_dist{}'.format(counter)] = alpha1 * df['OD'] + alpha2_TOD * df['SHD']
        df['TID_dist{}'.format(counter)] = alpha1 * df['ID'] + alpha2_TID * df['SID']
        print(((alpha1 * df['OD'])/df['TOD_dist{}'.format(counter)]*100).mean(), ((alpha1 * df['ID'])/df['TID_dist{}'.format(counter)]*100).mean())
        counter += 1

    alpha2_list = [1, 2, 4, 8, 16, 32]
    counter = 1
    print('perc of distance that is structural distance')
    for alpha2 in alpha2_list:
        df['TOD_struc{}'.format(counter)] = alpha1_TOD * df['OD'] + alpha2 * df['SHD']
        df['TID_struc{}'.format(counter)] = alpha1_TID * df['ID'] + alpha2 * df['SID']
        print(((alpha2 * df['SHD'])/df['TOD_struc{}'.format(counter)]*100).mean(), ((alpha2 * df['SID'])/df['TID_struc{}'.format(counter)]*100).mean())
        counter += 1

    df.to_csv(simfile)

def main():
    input_simfile = sys.argv[1]
    simfile = sys.argv[2]
    get_data(input_simfile, simfile)

if __name__ == "__main__":
    main()
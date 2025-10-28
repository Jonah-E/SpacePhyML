from spacephyml.datasets.creator import get_dataset

df = get_dataset('Olshevsky', ['2017-11-01', '2017-11-30'], resample='4.5s', var_list = ['mms1_dis_energyspectr_omni_fast', 'mms1_dis_energy_fast', 'mms1_dis_bulkv_gse_fast', 'mms1_fgm_b_gsm_srvy_l2'], clean = False)

df.to_csv('dataset_tmp.csv')
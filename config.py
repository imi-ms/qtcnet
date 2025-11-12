#######################
# Training Parameters #
#######################

INIT_LR = 1e-3
NUM_EPOCHS = 20
BATCH_SIZE = 128
TEST_SPLIT = 0.2
NUM_LEADS = 12
NUM_WORKERS = 64

# how much to take from muse and mimic
N_SAMPLES = 60150

#####################
# Paths to datasets #
#####################

# for MIMIC
CSV_FILE = ".../mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/machine_measurements.csv"
ROOT_DIR = ".../ECGDatasets__Loeschfrist_Unbegrenzt/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/files"

# for EDMS
MUSE_XPATH_DIV5 = ".../data/lano_qtc_ecgs.npz"
MUSE_XPATH_DIV5 = ".../data/lano_qtc_ecgs_div5.npz"
MUSE_YPATH = ".../data/lano_qtc_ecg_meta.csv"

# for QTcMS
X_HAND_PATH = ".../qtc_ms/ecgs.npz"
Y_HAND_PATH = ".../qtc_ms/keys.csv"

# for PTB
PTB_DF_PATH = ".../ptbdb/1.0.0/annotation.csv"
PTB_PRECOMP_DF_PATH = ".../ptbdb/1.0.0/precomputed_rr.csv"
PTB_DATASET_PATH = ".../ptb/files/ptbdb/1.0.0/"

# for ECGRDVQ
CSV_FILE_RDVQ=".../ECGRDVQ/SCR-002.Clinical.Data.csv"
ROOT_DIR_RDVQ=".../ECGRDVQ/raw"

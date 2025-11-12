from torch.utils.data import Dataset, Subset, ConcatDataset
from import_functions import sanity_check, set_seed
import pandas as pd
import os
from functools import cache, lru_cache
import wfdb
import numpy as np
import torch
import neurokit2 as nk
from config import *
import random
import joblib
from typing import Literal, TypedDict, Optional
from typing_extensions import Unpack

set_seed()


class MimicDatasetParams(TypedDict, total=False):
    clean_ecg: bool


class MimicDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, **kwargs: Unpack[MimicDatasetParams]):
        fields_to_read = [
            "subject_id",
            "study_id",
            "cart_id",
            "ecg_time",
            "bandwidth",
            "filtering",
            "rr_interval",
            "p_onset",
            "p_end",
            "qrs_onset",
            "qrs_end",
            "t_end",
            "p_axis",
            "qrs_axis",
            "t_axis",
        ]
        self.machine_csv = pd.read_csv(csv_file, usecols=fields_to_read)
        # self.machine_csv = pd.read_csv(csv_file, usecols=fields_to_read).loc[:100000]
        self.machine_csv = sanity_check(self.machine_csv)
        print(self.machine_csv.isna().any().any())
        self.root_dir = root_dir
        self.transform = transform

        self.file_paths = []
        self.clean_ecg = kwargs.get("clean_ecg", False)
        for index, row in self.machine_csv.iterrows():
            study_id = row["study_id"]
            subject_id = row["subject_id"]
            subpath = f"p{str(subject_id)[:4]}/p{subject_id}/s{study_id}"
            file_path = os.path.join(self.root_dir, subpath, str(study_id))
            self.file_paths.append(
                (file_path, row["qtc_interval"])
            )  # Store path and target together

    def __len__(self):
        return len(self.file_paths)

    @lru_cache
    def __getitem__(self, index):

        file_path, qtc_interval = self.file_paths[index]
        # return file_path # for random_reproducibility_creator
        record = wfdb.rdrecord(file_path)
        target = 100
        slicing = int(record.fs / target)  # here will be 5
        data = record.p_signal[::slicing, :]

        # deal with nans
        mask = np.isnan(data)
        for i in range(1, data.shape[0]):
            data[i][mask[i]] = data[i - 1][mask[i]]

        mask = np.isnan(data)
        for i in range(data.shape[0] - 2, -1, -1):
            data[i][mask[i]] = data[i + 1][mask[i]]

        if self.clean_ecg:
            # return data
            for lead_n in range(data.shape[1]):
                data[:, lead_n] = nk.ecg_clean(data[:, lead_n], sampling_rate=100)

        if self.transform:
            data = self.transform(data)

        data = torch.tensor(data, dtype=torch.float32).transpose(0,
                                                                 1)  # transpose to change [time_steps, 12] to [12, time_steps]
        qtc_interval = torch.tensor([np.nanmean(qtc_interval)],
                                    dtype=torch.float32)  # these [] around qt may render previous notebooks incompatible

        return data, qtc_interval


class MuseDatasetParams(TypedDict):
    divide_by_5: bool
    clean_ecg: bool
    correct_bias: bool


class MuseDataset(Dataset):
    def __init__(self, x_path, y_path, **kwargs: Unpack[MuseDatasetParams]):
        self.muse_qtc_60 = np.load(x_path, allow_pickle=True)["X"].astype(int) / 1000

        divide_by_5 = kwargs.get("divide_by_5", True)
        self.correct_bias = kwargs.get("correct_bias", False)

        if divide_by_5:
            self.muse_qtc_60 = self.muse_qtc_60[:, :, ::5]  # sampling rate 100 HZ
        self.muse_qtc_summ = pd.read_csv(y_path,
                                         usecols=["qtcorrected"])
        self.muse_qtc_summ['qtcorrected'] = self.muse_qtc_summ['qtcorrected'].str.replace(r"<[^>]+>", "",
                                                                                          regex=True).astype(int)

        # delete all that do not pass qtc sanity check
        mask = (self.muse_qtc_summ['qtcorrected'] >= 250) & (self.muse_qtc_summ['qtcorrected'] <= 600)

        self.muse_qtc_summ = self.muse_qtc_summ.loc[mask].reset_index(drop=True)

        self.muse_qtc_60 = self.muse_qtc_60[mask.values]
        self.clean_ecg = kwargs.get("clean_ecg", False)

    def __len__(self):
        return len(self.muse_qtc_60)

    @lru_cache
    def __getitem__(self, index):
        # print("before")
        x = self.muse_qtc_60[index][[0, 1, 2, 3, 5, 4, 6, 7, 8, 9, 10, 11], :]  # change to same order (aVR, aVF, aVL)
        # return x
        if self.clean_ecg:
            for lead_n in range(x.shape[0]):
                x[lead_n, :] = nk.ecg_clean(x[lead_n, :], sampling_rate=100)
        if self.correct_bias:
            y = self.muse_qtc_summ.iloc[index, 0] - 15
        else:
            y = self.muse_qtc_summ.iloc[index, 0]
        # print(x, y)
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor([y], dtype=torch.float32)
        # print("after")
        # print(x, y)
        return x, y


class HandDatasetParams(TypedDict):
    qt_type: Literal["auto", "expert"]
    clean_ecg: bool


class HandDataset(Dataset):
    def __init__(self, x_path, y_path, **kwargs: Unpack[HandDatasetParams]):
        self.X = np.load(x_path, allow_pickle=True)["X"].astype(int) / 1000
        self.Y = pd.read_csv(y_path, usecols=["QT-Interval", "QTc (Bazett)", "qtcorrected"])

        qt_type = kwargs.get("qt_type", "auto")

        self.qt_type = "qtcorrected" if qt_type == "auto" else "QTc (Bazett)"

        # delete values that do not pass sanity check
        mask = (self.Y['qtcorrected'] >= 250) & (self.Y['qtcorrected'] <= 600)
        self.Y = self.Y.loc[mask].reset_index(drop=True)
        self.X = self.X[mask.values]

        self.clean_ecg = kwargs.get("clean_ecg", False)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        # x = self.X[index][:, ::5]
        x = self.X[index][[0, 1, 2, 3, 5, 4, 6, 7, 8, 9, 10, 11], ::5]  # change to same order (aVR, aVF, aVL) + HZ

        if self.clean_ecg:
            for lead_n in range(x.shape[0]):
                x[lead_n, :] = nk.ecg_clean(x[lead_n, :], sampling_rate=100)

        y = self.Y.iloc[index][self.qt_type]

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        return x, y


class PTBDatasetParams(TypedDict):
    do_sanity_check: bool
    clean_ecg: bool


class PTBDataSet(Dataset):
    def __init__(self, df_path, ds_path, cache_rr=True, num_jobs=8, **kwargs: Unpack[PTBDatasetParams]):
        """
        parameters:
        - df_path: path to csv file with dataset
        - ds_path: path to all data pulled from dataset
        """

        self.dataset_path = ds_path
        self.dataframe = pd.read_csv(df_path)
        self.dataframe["patientID"] = self.dataframe["patientID"].apply(self.convert_to_3d)

        if cache_rr:
            self._precompute_rr(num_jobs)

        do_sanity_check = kwargs.get("do_sanity_check", False)
        self.clean_ecg = kwargs.get("clean_ecg", False)

        if do_sanity_check:
            mask = (self.dataframe['qtcorrected'] >= 250) & (self.dataframe['qtcorrected'] <= 600)
            self.dataframe = self.dataframe.loc[mask].reset_index(drop=True)

    def convert_to_3d(self, patient_id):
        # patient1 -> patient001
        # Extract the numeric part of the ID
        base, number = patient_id[:7], patient_id[7:]
        # Zero-pad the numeric part to three digits
        return f"{base}{int(number):03}"

    def __len__(self):
        return len(self.dataframe)

    @lru_cache(maxsize=1024)
    def __getitem__(self, index):
        try:
            case = self.dataframe.iloc[index]
            qt = case["t_end_median"] - case["q_onset_median"]

            # Load precomputed RR interval if available
            rr = case.get("rr", None)
            if rr is None:
                rd_record = self.get_rdrecord(index)
                x, fs, slicing = self.rearrange_p_signal(rd_record)
                rr = self.calculate_rr(x, fs, slicing)
            else:
                rd_record = self.get_rdrecord(index)
                x, fs, slicing = self.rearrange_p_signal(rd_record)

            qtc = qt / np.sqrt(rr)
            x = torch.tensor(x, dtype=torch.float32).transpose(0, 1)[:,
                :1000]  # transpose to change [time_steps, 12] to [12, time_steps]
            qtc = torch.tensor([qtc],
                               dtype=torch.float32)  # these [] around qt may render previous notebooks incompatible

            if np.isnan(rr):
                raise ValueError
            # return os.path.join(self.dataset_path, case["patientID"]) # for random_reproducibility_creator.py
            return x, qtc
        except:
            next_index = (index + 1) % self.__len__()
            return self.__getitem__(next_index)

    def calculate_rr(self, x, fs, slicing):
        """calculate rr using neurokit
            we will be using lead II (position 1 in numpy array, rearranged array)
        """

        # we cut data in 10 times, because we cut mimic with sr 500 into 5
        ecg_signal_lead = x[:, 1]  # Lead II

        target_fs = int(fs / slicing)  # Sampling frequency

        processed_signal, _ = nk.ecg_process(ecg_signal_lead, sampling_rate=target_fs)

        _, rpeaks = nk.ecg_peaks(processed_signal, sampling_rate=target_fs)

        # delta = 0 # accumulate rr intervals
        # for i in range(len(rpeaks["ECG_R_Peaks"])-1):
        #     delta += rpeaks["ECG_R_Peaks"][i+1] - rpeaks["ECG_R_Peaks"][i]
        delta = np.median(np.diff(rpeaks["ECG_R_Peaks"])) * slicing / 1000
        return delta
        # return delta/(len(rpeaks["ECG_R_Peaks"])-1) * slicing / 1000

    def get_rdrecord(self, id):
        case = self.dataframe.iloc[id]
        rec_path = os.path.join(self.dataset_path, case["patientID"], case["studyID"])
        rd_record = wfdb.rdrecord(rec_path)
        return rd_record

    def rearrange_p_signal(self, rd_record,
                           target_layout=['I', 'II', 'III', 'aVR', 'aVF', 'aVL', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']):
        """
        Rearranges the signals in a NumPy array to match a target layout.

        Parameters:
        - p_signal: np.ndarray
        The signal data, where columns correspond to signal names.
        - sig_name: list of str
        The current signal names in the order of `p_signal` columns.
        - target_layout: list of str
        The desired order of signal names.

        Returns:
        - np.ndarray
        Rearranged NumPy array matching the `target_layout`.
        """
        p_signal, sig_name, fs = rd_record.p_signal, list(map(str.lower, rd_record.sig_name)), rd_record.fs
        # print(p_signal) # it still works here
        # Create a new array for the rearranged signals
        num_signals = len(target_layout)
        rearranged_signal = np.zeros((p_signal.shape[0], num_signals))

        # Rearrange based on the target layout
        for i, target_signal in enumerate(target_layout):
            if target_signal.lower() in sig_name:
                # Find the index of the current signal in sig_name
                source_index = sig_name.index(target_signal.lower())
                # Copy the column to the new position
                if self.clean_ecg:
                    rearranged_signal[:, i] = nk.ecg_clean(p_signal[:, source_index], fs)
                else:
                    rearranged_signal[:, i] = p_signal[:, source_index]
            else:
                raise ValueError(f"Signal '{target_signal}' not found in sig_name.")

        target = 100

        slicing = int(fs / target)  # here will be 5
        return rearranged_signal[::slicing, :], fs, slicing  # return is correct

    def _precompute_rr(self, num_jobs=8):
        """Precompute RR intervals and store them in a new column in the dataframe."""
        print("Precomputing RR intervals...")

        def process_case(index):
            case = self.dataframe.iloc[index]
            try:
                rd_record = self.get_rdrecord(index)
                x, fs, slicing = self.rearrange_p_signal(rd_record)
                rr = self.calculate_rr(x, fs, slicing)
                return rr
            except Exception as e:
                print(f"Skipping index {index} due to error: {e}")
                return None

        indices = range(len(self.dataframe))
        rr_values = joblib.Parallel(n_jobs=num_jobs)(joblib.delayed(process_case)(i) for i in indices)

        # Store precomputed RR intervals
        self.dataframe["rr"] = rr_values
        self.dataframe["qtcorrected"] = (self.dataframe["t_end_median"] - self.dataframe["q_onset_median"]) / np.sqrt(
            self.dataframe["rr"])

        # Save updated dataset for future use
        cache_path = os.path.join(self.dataset_path, "ignore_precomputed_rr.csv")
        self.dataframe.to_csv(cache_path, index=False)
        print(f"RR intervals saved to {cache_path}")


class CreateCombinedDatasetParams(TypedDict):
    mimic_untouched: Optional[list]
    muse_untouched: Optional[list]
    return_indices: bool
    correct_bias: bool


def create_combined_dataset(
        CSV_FILE, ROOT_DIR, MUSE_XPATH_DIV5, MUSE_YPATH,
        mimic_samples=50000, muse_samples=50000, **kwargs: Unpack[CreateCombinedDatasetParams]):
    """
    Creates a combined dataset using samples from MimicDataset and MuseDataset.

    Parameters:
    - MimicDataset: The dataset class for MIMIC data.
    - MuseDataset: The dataset class for MUSE data.
    - CSV_FILE: Path to the CSV file for MimicDataset.
    - ROOT_DIR: Root directory for MimicDataset.
    - MUSE_XPATH_DIV5: Path to the input features for MuseDataset. (every 5th value is taken)
    - MUSE_YPATH: Path to the labels for MuseDataset.
    - mimic_samples: Number of samples to take from MimicDataset. (if set to -1 then len of mimic_ds)
    - muse_samples: Number of samples to take from MuseDataset. (if set to -1 then len of muse_ds)

    Returns:
    - A concatenated PyTorch dataset of selected samples from both datasets.
    """

    mimic_untouched = kwargs.get("mimic_untouched", None)
    muse_untouched = kwargs.get("muse_untouched", None)
    correct_bias = kwargs.get("correct_bias", False)

    assert isinstance(mimic_untouched, list), "mimic_untouched must be a list"
    assert isinstance(muse_untouched, list), "muse_untouched must be a list"

    # Convert `mimic_untouched` and `muse_untouched` to sets for fast lookup
    mimic_untouched = set(mimic_untouched) if mimic_untouched else set()
    muse_untouched = set(muse_untouched) if muse_untouched else set()

    # Load the full datasets
    mimic_dataset = MimicDataset(csv_file=CSV_FILE, root_dir=ROOT_DIR)
    muse_dataset = MuseDataset(x_path=MUSE_XPATH_DIV5, y_path=MUSE_YPATH, divide_by_5=False, correct_bias=correct_bias)

    # Get all valid indices
    valid_mimic_indices = [i for i in range(len(mimic_dataset)) if i not in mimic_untouched]
    valid_muse_indices = [i for i in range(len(muse_dataset)) if i not in muse_untouched]

    # add option to load full dataset by passing -1
    mimic_samples = len(mimic_dataset) if mimic_samples == -1 else mimic_samples
    muse_samples = len(muse_dataset) if muse_samples == -1 else muse_samples

    # Sample indices while ensuring they are not in untouched lists
    mimic_indices = random.sample(valid_mimic_indices, min(mimic_samples, len(valid_mimic_indices)))
    muse_indices = random.sample(valid_muse_indices, min(muse_samples, len(valid_muse_indices)))

    # Create subset datasets
    mimic_subset = Subset(mimic_dataset, mimic_indices)
    muse_subset = Subset(muse_dataset, muse_indices)

    # Combine datasets
    combined_dataset = ConcatDataset([mimic_subset, muse_subset])

    return_indices = kwargs.get("return_indices", False)

    if return_indices:
        return combined_dataset, mimic_indices, muse_indices

    return combined_dataset


# Define preconfigured dataset versions
def MimicMuse50(CSV_FILE, ROOT_DIR, MUSE_XPATH_DIV5, MUSE_YPATH):
    return create_combined_dataset(CSV_FILE, ROOT_DIR, MUSE_XPATH_DIV5, MUSE_YPATH, mimic_samples=50000,
                                   muse_samples=50000)


setattr(MimicMuse50, '__name__', 'MimicMuse50')


def MimicMuse64(CSV_FILE, ROOT_DIR, MUSE_XPATH_DIV5, MUSE_YPATH):
    return create_combined_dataset(CSV_FILE, ROOT_DIR, MUSE_XPATH_DIV5, MUSE_YPATH, mimic_samples=64000,
                                   muse_samples=64000)


setattr(MimicMuse64, '__name__', 'MimicMuse64')


class ECGRDVQDatasetParams(TypedDict):
    clean_ecg: bool


class ECGRDVQDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, **kwargs: Unpack[ECGRDVQDatasetParams]):
        fields_to_read = [
            "EGREFID",
            "RANDID",
            "RR",
            "QT"
        ]
        self.machine_csv = pd.read_csv(csv_file, usecols=fields_to_read)
        self.machine_csv["QTc"] = self.machine_csv["QT"] / ((self.machine_csv["RR"] / 1000).pow(1. / 2))

        # sanity check manual
        self.machine_csv = self.machine_csv[
            (self.machine_csv["QTc"] < 600) & (self.machine_csv["QTc"] > 250)].reset_index(drop=True)

        print(self.machine_csv.isna().any().any())
        self.root_dir = root_dir
        self.transform = transform

        self.file_paths = []
        self.clean_ecg = kwargs.get("clean_ecg", False)
        for index, row in self.machine_csv.iterrows():
            study_id = row["EGREFID"]
            subject_id = row["RANDID"]
            subpath = f"{subject_id}"
            file_path = os.path.join(self.root_dir, subpath, str(study_id))
            self.file_paths.append(
                (file_path, row["QT"] / np.sqrt(row["RR"] / 1000))
            )  # Store path and target together

    def __len__(self):
        return len(self.file_paths)

    @lru_cache
    def __getitem__(self, index):

        file_path, qtc_interval = self.file_paths[index]
        # return file_path # for random_reproducibility_creator
        record = wfdb.rdrecord(file_path)
        target = 100
        slicing = int(record.fs / target)  # here will be 5
        data = record.p_signal[::slicing, :]
        data = data[:, [0, 1, 2, 3, 5, 4, 6, 7, 8, 9, 10, 11]]  # change to same order (aVR, aVF, aVL)
        # deal with nans
        mask = np.isnan(data)
        for i in range(1, data.shape[0]):
            data[i][mask[i]] = data[i - 1][mask[i]]

        mask = np.isnan(data)
        for i in range(data.shape[0] - 2, -1, -1):
            data[i][mask[i]] = data[i + 1][mask[i]]

        if self.transform:
            data = self.transform(data)

        if self.clean_ecg:
            for i in range(12):
                data[:, i] = nk.ecg_clean(data[:, i], sampling_rate=target)

        data = torch.tensor(data, dtype=torch.float32).transpose(0,
                                                                 1)  # transpose to change [time_steps, 12] to [12, time_steps]

        qtc_interval = torch.tensor([np.nanmean(qtc_interval)],
                                    dtype=torch.float32)  # these [] around qt may render previous notebooks incompatible

        return data, qtc_interval

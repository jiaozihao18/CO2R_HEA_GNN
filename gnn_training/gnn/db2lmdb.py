import lmdb
import pickle
from tqdm import tqdm
from .data import AseDBDataset

def create_lmdb_from_dataset(config_db: dict, AseDBDataset, db_path: str, map_size_gb: int = 100):
    """
    Create an LMDB database from a dataset.

    Parameters:
    - config_db (dict): Configuration dictionary for the dataset.
    - AseDBDataset (Dataset): The dataset class to use for creating the dataset.
    - db_path (str): Path to the LMDB database file.
    - map_size_gb (int): Size of the LMDB map in gigabytes.
    """
    # Initialize dataset
    ase_dataset = AseDBDataset(**config_db)

    # Open LMDB
    db = lmdb.open(
        db_path,
        map_size=map_size_gb * 1024**3,  # Convert GB to bytes
        subdir=False,
        meminit=False,
        map_async=True,
    )

    # Write data to LMDB
    for data in tqdm(ase_dataset):
        txn = db.begin(write=True)
        txn.put(
            f"{data.sid}".encode("ascii"),
            pickle.dumps(data, protocol=-1),
        )
        txn.commit()

    # Save count of objects in LMDB
    txn = db.begin(write=True)
    txn.put("length".encode("ascii"), pickle.dumps(len(ase_dataset), protocol=-1))
    txn.commit()

    # Sync and close LMDB
    db.sync()
    db.close()


def append_lmdb(config_db: dict, AseDBDataset, db_path: str, map_size_gb: int = 100):
    """
    Append a new AseDBDataset to an existing LMDB database.

    Parameters:
    - config_db (dict): Configuration dictionary for the new dataset.
    - AseDBDataset (Dataset): The dataset class to use for creating the new dataset.
    - db_path (str): Path to the existing LMDB database file.
    - map_size_gb (int): Size of the LMDB map in gigabytes.
    """
    # Initialize new dataset
    new_ase_dataset = AseDBDataset(**config_db)

    # Open existing LMDB
    db = lmdb.open(
        db_path,
        map_size=map_size_gb * 1024**3,  # Convert GB to bytes
        subdir=False,
        meminit=False,
        map_async=True,
    )

    # Find the current maximum sid
    max_sid = -1
    with db.begin(write=False) as txn:
        cursor = txn.cursor()
        for key, _ in cursor:
            if key != b"length":
                sid = int(key.decode("ascii"))
                if sid > max_sid:
                    max_sid = sid

    new_sid_start = max_sid + 1

    # Append new data to LMDB
    for i, data in enumerate(tqdm(new_ase_dataset)):
        txn = db.begin(write=True)
        new_sid = new_sid_start + i
        txn.put(
            f"{new_sid}".encode("ascii"),
            pickle.dumps(data, protocol=-1),
        )
        txn.commit()

    # Update count of objects in LMDB
    with db.begin(write=False) as txn:
        current_length = pickle.loads(txn.get("length".encode("ascii")))
    
    new_length = current_length + len(new_ase_dataset)
    txn = db.begin(write=True)
    txn.put("length".encode("ascii"), pickle.dumps(new_length, protocol=-1))
    txn.commit()

    # Sync and close LMDB
    db.sync()
    db.close()

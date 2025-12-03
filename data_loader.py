# data_loader.py
# Responsible for fetching datasets (internal + external) and returning pandas DataFrames.
# Uses `kagglehub` per user request. Replace dataset ids / vendor connectors as needed.

import os
import zipfile
import io
import pandas as pd
import numpy as np
import logging
import kagglehub  # as requested in the brief

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class DataLoader:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

    def dataset_download(self, dataset_id: str, force: bool = False) -> str:
        """
        Download dataset using kagglehub.dataset_download. Returns path to extracted files.
        NOTE: Make sure kagglehub is configured with credentials in your environment.
        """
        target = os.path.join(self.data_dir, dataset_id.split("/")[-1])
        if os.path.exists(target) and not force:
            logger.info("Dataset already downloaded: %s", target)
            return target

        logger.info("Downloading dataset %s via kagglehub...", dataset_id)
        # kagglehub.dataset_download should return a path-like or bytes; adapt if API differs.
        # Example per user: kagglehub.dataset_download("nicolemachado/transportation-and-logistics-tracking-dataset")
        result = kagglehub.dataset_download(dataset_id, path=self.data_dir, unzip=False)
        # result may be a path to zip; if bytes, write
        if isinstance(result, (bytes, bytearray)):
            zip_path = os.path.join(self.data_dir, f"{dataset_id.split('/')[-1]}.zip")
            with open(zip_path, "wb") as f:
                f.write(result)
        else:
            zip_path = str(result)

        # Unzip
        extract_dir = os.path.join(self.data_dir, dataset_id.split("/")[-1])
        os.makedirs(extract_dir, exist_ok=True)
        try:
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(extract_dir)
        except zipfile.BadZipFile:
            # If result already a directory (kagglehub handled unzip), just use it
            extract_dir = zip_path.replace(".zip", "")
        logger.info("Extracted dataset to %s", extract_dir)
        return extract_dir

    def load_all(self) -> dict:
        """
        Load internal and external data. Return a dict of DataFrames keyed by name.
        NOTE: In production, replace dataset ids and add connectors to vendors (Xeneta, Freightos, weather, etc.)
        """
        # Example dataset from user brief
        try:
            path = self.dataset_download("nicolemachado/transportation-and-logistics-tracking-dataset")
        except Exception as e:
            logger.warning("Auto-download failed: %s. Falling back to sample synthetic data.", e)
            return self._make_synthetic()

        # Attempt to find CSV files in extracted path
        dfs = {}
        for root, _, files in os.walk(path):
            for fname in files:
                if fname.endswith(".csv"):
                    full = os.path.join(root, fname)
                    key = os.path.splitext(fname)[0]
                    try:
                        df = pd.read_csv(full, low_memory=False)
                        dfs[key] = df
                        logger.info("Loaded %s -> %s rows", fname, len(df))
                    except Exception as e:
                        logger.warning("Failed to load %s: %s", full, e)

        # Map dataset files to expected logical names (best effort)
        # These keys are suggestions; adapt to your actual files
        data = {
            "shipments": dfs.get("shipments") or dfs.get("transportation_tracking") or pd.DataFrame(),
            "rates": dfs.get("rates") or pd.DataFrame(),  # lane-level pricing
            "carriers": dfs.get("carriers") or pd.DataFrame(),
            "capacity": dfs.get("capacity") or pd.DataFrame(),
            "external_market": dfs.get("external_market") or pd.DataFrame(),
        }

        # Basic cleaning placeholder
        for k, df in data.items():
            if isinstance(df, pd.DataFrame):
                # Standardize column names
                df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        return data

    def _make_synthetic(self):
        """Return small synthetic dataframes when real data isn't available."""
        import pandas as pd, numpy as np
        idx = pd.date_range("2024-01-01", periods=200, freq="D")
        shipments = pd.DataFrame({
            "shipment_id": range(len(idx)),
            "agent": np.random.choice(["A1", "A2", "A3"], size=len(idx)),
            "origin": np.random.choice(["NYC", "LON", "SHA", "HKG"], size=len(idx)),
            "destination": np.random.choice(["LON", "NYC", "HAM", "SGP"], size=len(idx)),
            "product": np.random.choice(["electronics", "furniture", "frozen"], size=len(idx)),
            "booked_at": idx,
            "departed_at": idx + pd.to_timedelta(np.random.randint(1, 6, size=len(idx)), unit="D"),
            "arrived_at": idx + pd.to_timedelta(np.random.randint(5, 15, size=len(idx)), unit="D"),
            "price_usd": np.random.normal(1000, 200, size=len(idx)).round(2),
            "carrier": np.random.choice(["C1", "C2", "C3"], size=len(idx)),
            "delayed": np.random.choice([0, 1], p=[0.8, 0.2], size=len(idx)),
            "capacity_pct": np.random.uniform(30, 95, size=len(idx)).round(1)
        })
        rates = shipments[["origin", "destination", "product", "price_usd"]].groupby(["origin", "destination", "product"]).agg(price_mean=("price_usd","mean")).reset_index()
        return {"shipments": shipments, "rates": rates, "carriers": pd.DataFrame(), "capacity": pd.DataFrame(), "external_market": pd.DataFrame()}

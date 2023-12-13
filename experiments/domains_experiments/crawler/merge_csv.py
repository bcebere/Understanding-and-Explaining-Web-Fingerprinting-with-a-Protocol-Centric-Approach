# stdlib
import glob
from pathlib import Path
from typing import Optional

# third party
import pandas as pd

suffix = "_unbuffered"


def merge_temporal_per_flow(country: str, pd_lim: int = 50000) -> None:
    workspace = Path(f"datasets_temporal_{country}{suffix}")

    full_static_csv: Optional[pd.DataFrame] = None
    full_temporal_csv: Optional[pd.DataFrame] = None
    cnt = 0
    batch_idx = 0

    print(workspace)
    for filename in glob.glob(str(workspace / "flow_static*.csv")):
        static_filename = Path(filename)
        base = static_filename.name.split("flow_static_")[1]
        temporal_base = "flow_temporal_" + base
        temporal_filename = workspace / temporal_base

        assert static_filename.exists()
        assert temporal_filename.exists()
        try:
            local_static_csv = pd.read_csv(static_filename)
            local_temporal_csv = pd.read_csv(temporal_filename)
        except BaseException as e:
            print("failed to parse csv", e)
            continue

        if full_static_csv is None:
            full_static_csv = local_static_csv
            full_temporal_csv = local_temporal_csv
        else:
            full_static_csv = pd.concat(
                [full_static_csv, local_static_csv], ignore_index=True
            )
            full_temporal_csv = pd.concat(
                [full_temporal_csv, local_temporal_csv], ignore_index=True
            )

        if cnt % 1000 == 0:
            print("merge ", cnt, full_static_csv.shape)

        if len(full_static_csv) > pd_lim:
            print("!!! merge batch done", batch_idx)
            assert full_static_csv is not None
            assert full_temporal_csv is not None

            full_static_csv.to_csv(
                f"notebooks/data/tranco_temporal_data_per_flow_static_data_{country}{suffix}_{batch_idx}.csv",
                index=False,
            )
            full_temporal_csv.to_csv(
                f"notebooks/data/tranco_temporal_data_per_flow_ts_data_{country}{suffix}_{batch_idx}.csv",
                index=False,
            )
            batch_idx += 1
            full_static_csv = None
            full_temporal_csv = None

        cnt += 1


merge_temporal_per_flow(country="DE")

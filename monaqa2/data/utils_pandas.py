from pathlib import Path

import pandas as pd


def merge_pickle_dataframes(
    input_files: list[Path],
    output_file: Path,
    overwrite: bool = True,
) -> pd.DataFrame:
    """
    Merge multiple pickle pandas dataframes into a single pickle dataframe.

    Assumes all input dataframes have the same columns and compatible dtypes.

    :param input_files: List of input pickle dataframe files.
    :param output_file: Output pickle dataframe file.
    :param overwrite: If False, raise if output_file already exists.
    :return: The merged dataframe.
    """
    input_files = [Path(p) for p in input_files]
    output_file = Path(output_file)

    if output_file.exists() and not overwrite:
        raise FileExistsError(f"Output file already exists: {output_file}")

    if not input_files:
        raise ValueError("input_files must contain at least one file.")

    dfs = [pd.read_pickle(path) for path in input_files]

    columns = list(dfs[0].columns)
    dtypes = dfs[0].dtypes

    for path, df in zip(input_files, dfs):
        if list(df.columns) != columns:
            raise ValueError(f"Column mismatch in {path}")

        if not df.dtypes.equals(dtypes):
            raise ValueError(f"Dtype mismatch in {path}")

    merged = pd.concat(dfs, ignore_index=True)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    merged.to_pickle(output_file)

    return merged
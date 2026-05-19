"""
Read/write var_to_file_info.toml and reconstruct _VAR_TO_FILE_INFO.

Mappings are stored as two parallel arrays in the TOML:
  mapping_labels = ["Vx", "Vy", "Vz"]
  mapping_indices = [0, 1, 2]

A missing mapping_indices array means all indices are None.
A missing mapping_labels means the variable has no mapping at all.
"""

import tomllib  # stdlib in Python 3.11+; use `tomli` as a drop-in on older versions
from pathlib import Path


def load_var_to_file_info(path: str | Path = "var_to_file_info.toml") -> dict:
    """Load the TOML file and return a dict matching the original _VAR_TO_FILE_INFO shape."""
    with open(path, "rb") as f:
        raw = tomllib.load(f)

    result = {}
    for var_name, var_data in raw.items():
        entry: dict = {"info": var_data["info"]}

        if "mapping_labels" in var_data:
            labels = var_data["mapping_labels"]
            indices = var_data.get("mapping_indices", [None] * len(labels))
            entry["mapping"] = list(zip(labels, indices))

        result[var_name] = entry

    return result


def save_var_to_file_info(
    data: dict, path: str | Path = "var_to_file_info.toml"
) -> None:
    """
    Write _VAR_TO_FILE_INFO to a TOML file using compact parallel arrays.

    Mapping tuples of (label, index_or_None) are split into mapping_labels
    and mapping_indices. mapping_indices is omitted entirely when all values
    are None (restored correctly as None on load).
    """

    def toml_str_array(values) -> str:
        return "[" + ", ".join(f'"{v}"' for v in values) + "]"

    def toml_int_array(values) -> str:
        return "[" + ", ".join(str(v) for v in values) + "]"

    lines: list[str] = []

    for var_name, var_data in data.items():
        info = var_data["info"]
        info_inline = ", ".join(f'{k} = "{v}"' for k, v in info.items())
        lines.append(f"[{var_name}]")
        lines.append(f"info = {{{info_inline}}}")

        if "mapping" in var_data:
            labels, indices = zip(*var_data["mapping"])
            lines.append(f"mapping_labels = {toml_str_array(labels)}")
            if any(i is not None for i in indices):
                lines.append(f"mapping_indices = {toml_int_array(indices)}")

        lines.append("")

    Path(path).write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    _VAR_TO_FILE_INFO = load_var_to_file_info("var_to_file_info.toml")

    for var, data in _VAR_TO_FILE_INFO.items():
        print(f"\n{var}")
        print(f"  info: {data['info']}")
        if "mapping" in data:
            preview = data["mapping"][:3]
            ellipsis = " ..." if len(data["mapping"]) > 3 else ""
            print(f"  mapping: {preview}{ellipsis} ({len(data['mapping'])} entries)")

    # Round-trip test: write back and reload
    save_var_to_file_info(_VAR_TO_FILE_INFO, "var_to_file_info_roundtrip.toml")
    reloaded = load_var_to_file_info("var_to_file_info_roundtrip.toml")
    assert reloaded == _VAR_TO_FILE_INFO, "Round-trip mismatch!"
    print("\nRound-trip write/read verified OK.")

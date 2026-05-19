"""
Unit tests for spacephyml/utils/config.py – load/save var_to_file_info
"""
from pathlib import Path

import pytest
from spacephyml.utils.config import load_var_to_file_info, save_var_to_file_info


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_data():
    return {
        "mms1_fpi_dis_numberdensity": {
            "info": {"file": "dis-moms", "var": "mms1_dis_numberdensity_fast"},
            "mapping": [("N", 0)],
        },
        "mms1_fpi_dis_bulkv": {
            "info": {"file": "dis-moms", "var": "mms1_dis_bulkv_gse_fast"},
            "mapping": [("Vx", 0), ("Vy", 1), ("Vz", 2)],
        },
        "mms1_edp_scpot": {
            "info": {"file": "edp-scpot", "var": "mms1_edp_scpot_fast_l2"},
            # No mapping key → should round-trip cleanly
        },
    }


@pytest.fixture
def toml_path(tmp_path, sample_data):
    p = tmp_path / "var_to_file_info.toml"
    save_var_to_file_info(sample_data, p)
    return p


# ---------------------------------------------------------------------------
# save_var_to_file_info
# ---------------------------------------------------------------------------

class TestSaveVarToFileInfo:
    def test_file_is_created(self, tmp_path, sample_data):
        p = tmp_path / "out.toml"
        save_var_to_file_info(sample_data, p)
        assert p.exists()

    def test_file_is_nonempty(self, tmp_path, sample_data):
        p = tmp_path / "out.toml"
        save_var_to_file_info(sample_data, p)
        assert p.stat().st_size > 0

    def test_variable_names_appear_in_file(self, tmp_path, sample_data):
        p = tmp_path / "out.toml"
        save_var_to_file_info(sample_data, p)
        content = p.read_text()
        for var in sample_data:
            assert var in content

    def test_mapping_labels_appear_in_file(self, tmp_path, sample_data):
        p = tmp_path / "out.toml"
        save_var_to_file_info(sample_data, p)
        content = p.read_text()
        assert "Vx" in content
        assert "Vy" in content
        assert "Vz" in content


# ---------------------------------------------------------------------------
# load_var_to_file_info
# ---------------------------------------------------------------------------

class TestLoadVarToFileInfo:
    def test_returns_dict(self, toml_path):
        result = load_var_to_file_info(toml_path)
        assert isinstance(result, dict)

    def test_all_variables_present(self, toml_path, sample_data):
        result = load_var_to_file_info(toml_path)
        for var in sample_data:
            assert var in result

    def test_info_fields_preserved(self, toml_path, sample_data):
        result = load_var_to_file_info(toml_path)
        for var, data in sample_data.items():
            assert result[var]["info"] == data["info"]

    def test_mapping_labels_preserved(self, toml_path):
        result = load_var_to_file_info(toml_path)
        labels = [label for label, _ in result["mms1_fpi_dis_bulkv"]["mapping"]]
        assert labels == ["Vx", "Vy", "Vz"]

    def test_mapping_indices_preserved(self, toml_path):
        result = load_var_to_file_info(toml_path)
        indices = [idx for _, idx in result["mms1_fpi_dis_bulkv"]["mapping"]]
        assert indices == [0, 1, 2]

    def test_variable_without_mapping_has_no_mapping_key(self, toml_path):
        result = load_var_to_file_info(toml_path)
        assert "mapping" not in result["mms1_edp_scpot"]

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_var_to_file_info(tmp_path / "does_not_exist.toml")


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def test_round_trip_equality(self, tmp_path, sample_data):
        p = tmp_path / "rt.toml"
        save_var_to_file_info(sample_data, p)
        reloaded = load_var_to_file_info(p)
        assert reloaded == sample_data

    def test_double_round_trip(self, tmp_path, sample_data):
        p1 = tmp_path / "rt1.toml"
        p2 = tmp_path / "rt2.toml"
        save_var_to_file_info(sample_data, p1)
        first = load_var_to_file_info(p1)
        save_var_to_file_info(first, p2)
        second = load_var_to_file_info(p2)
        assert first == second

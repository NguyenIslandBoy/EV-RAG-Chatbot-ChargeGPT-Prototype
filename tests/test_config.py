"""
tests/test_config.py
====================
Tests for config.py — path setup and environment loading.

These tests verify the structural guarantees config.py makes:
- All paths are absolute Path objects
- Required directories are created on import
- DB path has the correct extension
- Ollama settings have non-empty defaults
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config


class TestConfigPaths:
    def test_base_dir_is_path_object(self):
        assert isinstance(config.BASE_DIR, Path)

    def test_data_dir_is_path_object(self):
        assert isinstance(config.DATA_DIR, Path)

    def test_logs_dir_is_path_object(self):
        assert isinstance(config.LOGS_DIR, Path)

    def test_all_paths_are_absolute(self):
        assert config.BASE_DIR.is_absolute()
        assert config.DATA_DIR.is_absolute()
        assert config.LOGS_DIR.is_absolute()
        assert config.DB_PATH.is_absolute()

    def test_data_dir_exists_after_import(self):
        """config.py must create DATA_DIR on import."""
        assert config.DATA_DIR.exists()

    def test_logs_dir_exists_after_import(self):
        """config.py must create LOGS_DIR on import."""
        assert config.LOGS_DIR.exists()

    def test_db_path_has_duckdb_extension(self):
        assert config.DB_PATH.suffix == ".duckdb"

    def test_db_path_is_inside_data_dir(self):
        assert config.DATA_DIR in config.DB_PATH.parents

    def test_rag_dir_is_path_object(self):
        assert isinstance(config.RAG_DB_PATH, Path)


class TestConfigOllamaSettings:
    def test_ollama_base_url_non_empty(self):
        assert config.OLLAMA_BASE_URL
        assert len(config.OLLAMA_BASE_URL) > 0

    def test_ollama_base_url_is_http(self):
        assert config.OLLAMA_BASE_URL.startswith("http")

    def test_ollama_model_non_empty(self):
        assert config.OLLAMA_MODEL
        assert len(config.OLLAMA_MODEL) > 0

    def test_ollama_embed_model_non_empty(self):
        assert config.OLLAMA_EMBED_MODEL
        assert len(config.OLLAMA_EMBED_MODEL) > 0


class TestConfigACNSettings:
    def test_acn_base_url_is_set(self):
        assert config.ACN_BASE_URL
        assert "caltech" in config.ACN_BASE_URL.lower() or "ev" in config.ACN_BASE_URL.lower()

    def test_acn_sites_is_list(self):
        assert isinstance(config.ACN_SITES, list)
        assert len(config.ACN_SITES) > 0

    def test_known_sites_present(self):
        sites_lower = [s.lower() for s in config.ACN_SITES]
        assert "caltech" in sites_lower
        assert "jpl" in sites_lower
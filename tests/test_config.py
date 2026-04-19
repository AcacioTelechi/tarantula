import pytest
from tarantula.config import (
    load_urls_config, load_variables_config,
    URLsConfig, VariablesConfig, SiteConfig, VariableSpec,
)


def test_load_valid_urls_config(fixtures_dir):
    cfg = load_urls_config(fixtures_dir / "configs" / "valid_urls.yaml")
    assert isinstance(cfg, URLsConfig)
    assert len(cfg.sites) == 2
    assert cfg.sites[0].url == "https://example.com"
    assert cfg.sites[0].max_depth == 4  # override
    assert cfg.sites[0].rate_limit_rps == 1.5  # inherited default
    assert cfg.sites[1].include_subdomains is True


def test_site_config_inherits_defaults(fixtures_dir):
    cfg = load_urls_config(fixtures_dir / "configs" / "valid_urls.yaml")
    site = cfg.sites[0]
    assert site.user_agent == "tarantula/0.1"
    assert site.respect_robots_txt is True


def test_urls_config_rejects_non_http_scheme(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("sites:\n  - url: ftp://example.com\n")
    with pytest.raises(ValueError, match="http"):
        load_urls_config(p)


def test_urls_config_requires_at_least_one_site(tmp_path):
    p = tmp_path / "empty.yaml"
    p.write_text("sites: []\n")
    with pytest.raises(ValueError, match="at least one"):
        load_urls_config(p)


def test_load_valid_variables_config(fixtures_dir):
    cfg = load_variables_config(fixtures_dir / "configs" / "valid_variables.yaml")
    assert isinstance(cfg, VariablesConfig)
    assert len(cfg.variables) == 4
    by_name = {v.name: v for v in cfg.variables}
    assert by_name["company_name"].required is True
    assert by_name["founded_year"].examples[0].output == 1998
    assert by_name["products"].items == "string"
    assert by_name["has_careers_page"].type == "boolean"


def test_variables_config_rejects_duplicate_names(tmp_path):
    p = tmp_path / "dup.yaml"
    p.write_text(
        "variables:\n"
        "  - {name: x, type: string, description: a}\n"
        "  - {name: x, type: string, description: b}\n"
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_variables_config(p)


def test_array_variable_requires_items(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text(
        "variables:\n"
        "  - {name: x, type: array, description: a}\n"
    )
    with pytest.raises(ValueError, match="items"):
        load_variables_config(p)


def test_variable_type_rejects_unsupported(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text(
        "variables:\n"
        "  - {name: x, type: object, description: a}\n"
    )
    with pytest.raises(ValueError):
        load_variables_config(p)

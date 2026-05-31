from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

ScalarType = Literal["string", "integer", "number", "boolean"]
VarType = Literal["string", "integer", "number", "boolean", "array"]
ExtractionType = Literal["retrieval", "agent"]


class SiteDefaults(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_depth: int = Field(3, ge=0, le=20)
    max_pages: int = Field(200, ge=1, le=100_000)
    same_host_only: bool = True
    include_subdomains: bool = False
    respect_robots_txt: bool = True
    rate_limit_rps: float = Field(1.5, gt=0, le=50)
    request_timeout_s: int = Field(20, ge=1, le=600)
    user_agent: str = "tarantula/0.1"


class SiteConfig(SiteDefaults):
    url: Annotated[str, Field(min_length=1)]

    @model_validator(mode="after")
    def _check_scheme(self) -> "SiteConfig":
        if not (self.url.startswith("http://") or self.url.startswith("https://")):
            raise ValueError(f"url must be http(s): got {self.url!r}")
        return self


class URLsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    defaults: SiteDefaults = Field(default_factory=SiteDefaults)
    sites: list[SiteConfig]

    @model_validator(mode="after")
    def _check_nonempty(self) -> "URLsConfig":
        if not self.sites:
            raise ValueError("must define at least one site")
        return self


class VariableExample(BaseModel):
    model_config = ConfigDict(extra="forbid")
    input: str
    output: Any


class VariableSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: Annotated[str, Field(pattern=r"^[a-zA-Z_][a-zA-Z0-9_]*$")]
    type: VarType
    description: str
    required: bool = False
    items: ScalarType | None = None
    examples: list[VariableExample] = Field(default_factory=list)
    extraction_type: ExtractionType = "retrieval"
    hint: str | None = None
    max_steps: int | None = Field(default=None, ge=1, le=50)

    @model_validator(mode="after")
    def _check_items(self) -> "VariableSpec":
        if self.type == "array" and self.items is None:
            raise ValueError(f"variable {self.name!r}: array type requires 'items'")
        if self.type != "array" and self.items is not None:
            raise ValueError(f"variable {self.name!r}: 'items' only valid for array type")
        return self

    @model_validator(mode="after")
    def _check_agent_fields(self) -> "VariableSpec":
        if self.extraction_type != "agent":
            if self.hint is not None:
                raise ValueError(
                    f"variable {self.name!r}: 'hint' only valid when "
                    "extraction_type is 'agent'"
                )
            if self.max_steps is not None:
                raise ValueError(
                    f"variable {self.name!r}: 'max_steps' only valid when "
                    "extraction_type is 'agent'"
                )
        return self


class VariablesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    variables: list[VariableSpec]

    @model_validator(mode="after")
    def _check_unique(self) -> "VariablesConfig":
        names = [v.name for v in self.variables]
        if len(set(names)) != len(names):
            raise ValueError("duplicate variable names")
        return self


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a YAML mapping at the root")
    return data


def load_urls_config(path: Path) -> URLsConfig:
    data = _load_yaml(Path(path))
    defaults = data.get("defaults") or {}
    raw_sites = data.get("sites") or []
    merged_sites = [{**defaults, **(s or {})} for s in raw_sites]
    return URLsConfig(defaults=SiteDefaults(**defaults), sites=merged_sites)


def load_variables_config(path: Path) -> VariablesConfig:
    data = _load_yaml(Path(path))
    return VariablesConfig(**data)

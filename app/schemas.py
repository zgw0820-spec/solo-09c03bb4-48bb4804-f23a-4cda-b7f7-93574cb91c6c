"""Pydantic request models and field-level validation."""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import List

from pydantic import BaseModel, Field, field_validator, model_validator

MAX_DIM = 4096
MAX_CELLS = 16_777_216  # hard cap on rows*cols to avoid unbounded payloads
NAME_RE = r"^[A-Za-z0-9_.\-]{1,64}$"


def _check_grid(rain: List[List[float]], missing: List[List[int]] | None) -> None:
    if not isinstance(rain, list) or len(rain) == 0 or not isinstance(rain[0], list):
        raise ValueError("rain must be a non-empty 2-D matrix")
    rows = len(rain)
    cols = len(rain[0])
    if cols == 0:
        raise ValueError("rain must be a non-empty 2-D matrix")
    if rows > MAX_DIM or cols > MAX_DIM:
        raise ValueError(f"grid dimensions must not exceed {MAX_DIM} per axis")
    if rows * cols > MAX_CELLS:
        raise ValueError(f"grid cell count must not exceed {MAX_CELLS}")
    for r, row in enumerate(rain):
        if not isinstance(row, list) or len(row) != cols:
            raise ValueError(f"rain row {r} is not of uniform width {cols}")
        for c, v in enumerate(row):
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise ValueError(f"rain[{r}][{c}] must be a number")
            fv = float(v)
            if math.isnan(fv) or math.isinf(fv):
                raise ValueError(f"rain[{r}][{c}] must be finite (use missing grid)")
    if missing is not None:
        # structural check only; equality of shape with rain is a domain rule
        # enforced in the service layer for a uniform JSON error envelope
        if not isinstance(missing, list) or len(missing) == 0 or not isinstance(missing[0], list):
            raise ValueError("missing must be a non-empty 2-D 0/1 matrix")
        m_width = len(missing[0])
        if m_width == 0:
            raise ValueError("missing must be a non-empty 2-D 0/1 matrix")
        for r, row in enumerate(missing):
            if not isinstance(row, list) or len(row) != m_width:
                raise ValueError(f"missing row {r} is not of uniform width")
            for c, v in enumerate(row):
                if isinstance(v, bool) or not isinstance(v, (int, float)) or float(v) not in (0.0, 1.0):
                    raise ValueError(f"missing[{r}][{c}] must be 0 or 1")


class RegisterRequest(BaseModel):
    forecast_time: datetime = Field(
        ..., description="UTC forecast initialization time (ISO-8601, timezone-aware)"
    )
    rows: int = Field(..., ge=1, le=MAX_DIM)
    cols: int = Field(..., ge=1, le=MAX_DIM)
    member_names: List[str] = Field(..., min_length=2, max_length=8)
    thresholds: List[float] = Field(..., min_length=1)

    @field_validator("forecast_time")
    @classmethod
    def _tz(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("forecast_time must be timezone-aware UTC time")
        return v.astimezone(timezone.utc)

    @field_validator("member_names")
    @classmethod
    def _members(cls, v: List[str]) -> List[str]:
        import re

        for name in v:
            if not re.match(NAME_RE, name):
                raise ValueError(f"invalid member name: {name!r}")
        if len(set(v)) != len(v):
            raise ValueError("member_names must be unique")
        return v

    @field_validator("thresholds")
    @classmethod
    def _thresholds(cls, v: List[float]) -> List[float]:
        for t in v:
            if isinstance(t, bool) or not isinstance(t, (int, float)):
                raise ValueError("thresholds must be numbers")
            ft = float(t)
            if math.isnan(ft) or math.isinf(ft) or ft < 0:
                raise ValueError("thresholds must be finite and non-negative")
        if len(set(v)) != len(v):
            raise ValueError("thresholds must be unique")
        return [float(t) for t in v]

    @model_validator(mode="after")
    def _shape_cap(self) -> "RegisterRequest":
        if self.rows * self.cols > MAX_CELLS:
            raise ValueError(f"grid cell count must not exceed {MAX_CELLS}")
        return self


class SubmitRequest(BaseModel):
    rain: List[List[float]] = Field(..., description="rainfall matrix; values at valid cells must be non-negative")
    missing: List[List[int]] | None = Field(
        default=None, description="optional 0/1 matrix; 1 marks a missing gauge cell"
    )
    revision: int = Field(..., ge=1, description="strictly increasing positive revision number")

    @model_validator(mode="after")
    def _validate(self) -> "SubmitRequest":
        # Structural payload checks only. Domain checks (registered grid shape,
        # non-negativity at valid cells) run in the service layer so all domain
        # errors share the same JSON error envelope.
        _check_grid(self.rain, self.missing)
        return self

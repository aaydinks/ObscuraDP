"""Preparing participants' tables without spending privacy budget."""

from obscuradp.tables.schema import CategoricalFeature, NumericFeature, TableSchema
from obscuradp.tables.split import ClientData, make_client_data, split_indices
from obscuradp.tables.synthetic import (
    SYNTHETIC_HEALTH_PRIOR,
    SYNTHETIC_HEALTH_SCHEMA,
    SyntheticTable,
    synthetic_health,
    synthetic_health_rows,
)

__all__ = [
    "SYNTHETIC_HEALTH_PRIOR",
    "SYNTHETIC_HEALTH_SCHEMA",
    "CategoricalFeature",
    "ClientData",
    "NumericFeature",
    "SyntheticTable",
    "TableSchema",
    "make_client_data",
    "split_indices",
    "synthetic_health",
    "synthetic_health_rows",
]

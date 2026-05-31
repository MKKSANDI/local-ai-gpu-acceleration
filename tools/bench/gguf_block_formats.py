from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BlockFormat:
    name: str
    block_size: int
    type_size: int
    family: str


BLOCK_FORMATS: dict[str, BlockFormat] = {
    "F32": BlockFormat("F32", block_size=1, type_size=4, family="float"),
    "F16": BlockFormat("F16", block_size=1, type_size=2, family="float"),
    "Q8_0": BlockFormat("Q8_0", block_size=32, type_size=34, family="legacy_quant"),
    "Q4_K": BlockFormat("Q4_K", block_size=256, type_size=144, family="k_quant"),
    "Q5_K": BlockFormat("Q5_K", block_size=256, type_size=176, family="k_quant"),
    "IQ2_S": BlockFormat("IQ2_S", block_size=256, type_size=82, family="iq_quant"),
    "IQ3_S": BlockFormat("IQ3_S", block_size=256, type_size=110, family="iq_quant"),
}


def element_count(dimensions: object) -> int:
    if not isinstance(dimensions, list) or not dimensions:
        raise ValueError("tensor dimensions must be a non-empty list")
    total = 1
    for value in dimensions:
        total *= int(value)
    return total


def expected_data_bytes(tensor_type: str, dimensions: object) -> int | None:
    fmt = BLOCK_FORMATS.get(tensor_type)
    if fmt is None:
        return None
    elements = element_count(dimensions)
    if elements % fmt.block_size != 0:
        raise ValueError(
            f"{tensor_type} element count {elements} is not divisible by block size {fmt.block_size}"
        )
    return (elements // fmt.block_size) * fmt.type_size


def validate_tensor_storage(tensor: dict[str, object], *, alignment: int) -> dict[str, object]:
    tensor_type = str(tensor.get("type"))
    fmt = BLOCK_FORMATS.get(tensor_type)
    span_bytes = int(tensor.get("span_bytes") or 0)
    row: dict[str, object] = {
        "name": tensor.get("name"),
        "type": tensor_type,
        "span_bytes": span_bytes,
        "supported": fmt is not None,
        "block_size": fmt.block_size if fmt else None,
        "type_size": fmt.type_size if fmt else None,
        "family": fmt.family if fmt else None,
        "expected_data_bytes": None,
        "padding_bytes": None,
        "valid": False,
        "reason": "",
    }
    if fmt is None:
        row["reason"] = "unsupported tensor type"
        return row
    try:
        expected = expected_data_bytes(tensor_type, tensor.get("dimensions"))
    except ValueError as exc:
        row["reason"] = str(exc)
        return row
    padding = span_bytes - int(expected)
    row["expected_data_bytes"] = expected
    row["padding_bytes"] = padding
    if padding < 0:
        row["reason"] = "span is smaller than expected tensor data bytes"
    elif padding >= alignment:
        row["reason"] = "span padding exceeds tensor alignment"
    else:
        row["valid"] = True
        row["reason"] = "ok"
    return row


def validation_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    supported = [row for row in rows if row.get("supported")]
    valid = [row for row in rows if row.get("valid")]
    invalid = [row for row in rows if not row.get("valid")]
    expected_total = sum(int(row.get("expected_data_bytes") or 0) for row in rows)
    padding_total = sum(max(0, int(row.get("padding_bytes") or 0)) for row in rows)
    families: dict[str, int] = {}
    for row in rows:
        family = row.get("family")
        if isinstance(family, str):
            families[family] = families.get(family, 0) + 1
    return {
        "tensor_count": len(rows),
        "supported_tensors": len(supported),
        "valid_tensors": len(valid),
        "invalid_tensors": len(invalid),
        "expected_data_bytes": expected_total,
        "padding_bytes": padding_total,
        "max_padding_bytes": max((int(row.get("padding_bytes") or 0) for row in rows), default=0),
        "families": dict(sorted(families.items())),
    }

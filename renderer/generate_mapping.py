#!/usr/bin/env python3
"""Generate renderer/mapping.scd from config/timbre_map.yaml.

This keeps the SuperCollider mapping file honest by making YAML the single
source of truth. Update YAML, run this script, and the renderer will pick up
fresh mappings next time you boot it.
"""
from __future__ import annotations

from pathlib import Path
import json
import re

ENGINE_MAP = {
    "fm_metal": "fm",
    "noise_band": "noise",
    "modal": "modal",
    "dist_arc": "fold",
}

VOICE_LIMITS = {
    "MAX_VOICES": 20,
    "PER_STREAM": 4,
}

INLINE_MAP_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*:")


def _format_number(value: float | int) -> str:
    rendered = f"{value:g}"
    if "." not in rendered and "e" not in rendered and isinstance(value, float):
        return f"{rendered}.0"
    return rendered


def _format_symbol(value: str) -> str:
    return f"\\{value}"


def _format_array(values: list[float | int]) -> str:
    return "[" + ", ".join(_format_number(v) for v in values) + "]"


def _parse_inline_map(payload: str) -> dict:
    normalized = INLINE_MAP_RE.sub(r'"\1":', payload)
    return json.loads(normalized)


def _parse_yaml_subset(lines: list[str]) -> dict:
    """Parse the tiny YAML subset used by config/timbre_map.yaml.

    The format is intentionally limited: top-level keys with indentation-based
    nesting, inline maps in `{}` form, and list literals in `[]` form. This keeps
    the generator dependency-free while still honoring the current YAML layout.
    """
    data: dict[str, object] = {}
    families: dict[str, dict] = {}
    class_map: dict[int, str] = {}
    routing: dict[str, list[float]] = {}
    current_section: str | None = None
    in_by_class = False

    for line in lines:
        if not line.strip() or line.strip().startswith("#"):
            continue

        if not line.startswith(" "):
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip()
            in_by_class = False
            current_section = key
            if key == "families":
                continue
            if key == "map":
                continue
            if key == "routing":
                continue
            if key == "default_family":
                data[key] = value.strip("\"")
            continue

        if current_section == "families" and "{" in line:
            name, _, payload = line.partition(":")
            families[name.strip()] = _parse_inline_map(payload.strip())
            continue

        if current_section == "map" and "by_class:" in line:
            in_by_class = True
            continue

        if current_section == "map" and in_by_class:
            class_id, _, family = line.strip().partition(":")
            class_map[int(class_id.strip())] = family.strip()
            continue

        if current_section == "routing":
            route, _, payload = line.strip().partition(":")
            routing[route.strip()] = json.loads(payload.strip())
            continue

    data["families"] = families
    data["map"] = {"by_class": class_map}
    data["routing"] = routing
    return data


def _render_mapping(data: dict) -> str:
    families = data.get("families", {})
    class_map = data.get("map", {}).get("by_class", {})
    routing = data.get("routing", {})
    default_family = data.get("default_family", "servo")

    family_lines = []
    for family_name, payload in families.items():
        engine = payload.get("engine")
        if engine not in ENGINE_MAP:
            raise ValueError(f"Unknown engine '{engine}' for family '{family_name}'.")
        ranges = payload.get("p", {})
        range_pairs = ", ".join(
            f"{param}: {_format_array(bounds)}" for param, bounds in ranges.items()
        )
        family_lines.append(
            f"    {family_name}: (engine: {_format_symbol(ENGINE_MAP[engine])}, ranges: ({range_pairs}))"
        )

    class_lines = []
    for class_id, family in class_map.items():
        class_lines.append(f"    {int(class_id)}: {_format_symbol(family)}")

    routing_lines = []
    for route_name, bounds in routing.items():
        routing_lines.append(f"    {route_name}: {_format_array(bounds)}")

    return """(
// AUTO-GENERATED FILE — DO NOT HAND-EDIT.
//
// This file is written by renderer/generate_mapping.py from config/timbre_map.yaml.
// Edit the YAML, rerun the generator, and keep your mapping honest.

// class → family (SuperCollider symbols)
~classToFamily = (
{class_map}
);

// default family when a class isn't in the map
~defaultFamily = {default_family};

// family → synth engine + parameter ranges (low..high)
~families = (
{families}
);

// routing: feature → parameter scaler (srcMin, srcMax → dstMin, dstMax)
~routing = (
{routing}
);

// voice budgets (keep things musical, not mushy)
~MAX_VOICES = {max_voices};      // global cap
~PER_STREAM = {per_stream};       // soft cap per stream
)
""".format(
        class_map=",\n".join(class_lines),
        default_family=_format_symbol(default_family),
        families=",\n".join(family_lines),
        routing=",\n".join(routing_lines),
        max_voices=VOICE_LIMITS["MAX_VOICES"],
        per_stream=VOICE_LIMITS["PER_STREAM"],
    )


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    yaml_path = repo_root / "config" / "timbre_map.yaml"
    output_path = repo_root / "renderer" / "mapping.scd"

    with yaml_path.open("r", encoding="utf-8") as handle:
        raw_lines = [line.rstrip("\n") for line in handle]

    data = _parse_yaml_subset(raw_lines)

    output_path.write_text(_render_mapping(data), encoding="utf-8")
    print(f"Wrote {output_path} from {yaml_path}")


if __name__ == "__main__":
    main()

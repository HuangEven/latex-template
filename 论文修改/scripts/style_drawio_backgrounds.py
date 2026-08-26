#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import copy
import json
import re
import shutil
import sys
import urllib.parse
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import xml.etree.ElementTree as ET


ALLOWED_SUFFIXES = {".drawio", ".xml"}

PALETTE = {
    "input": "#EAF2F8",
    "preprocess": "#F4F1EA",
    "compute": "#E8EEF5",
    "recall": "#EEF5EE",
    "score": "#F6EAEA",
    "support": "#F3F3F3",
    "neutral": "#F3F3F3",
    "white": "#FFFFFF",
}

KEYWORD_RULES = [
    (
        "score",
        [
            "pytod",
            "异常评分",
            "评分",
            "风险",
            "告警",
            "输出",
            "排序",
            "复核",
            "反馈",
            "异常分数",
            "结果",
            "拦截",
            "调查",
            "上报",
            "risk",
            "score",
            "output",
            "alert",
        ],
    ),
    (
        "recall",
        [
            "flashanns",
            "候选召回",
            "召回",
            "检索",
            "索引",
            "向量索引",
            "图索引",
            "ann",
            "faiss",
            "i/o",
            "ssd",
            "diskann",
            "spann",
            "hnsw",
            "retrieval",
            "index",
        ],
    ),
    (
        "preprocess",
        [
            "预处理",
            "特征构造",
            "标准化",
            "清洗",
            "编码",
            "向量化",
            "数据接入",
            "特征处理",
            "归一化",
            "拼接",
            "量化",
            "预编码",
            "preprocess",
            "feature",
            "normalize",
            "encode",
        ],
    ),
    (
        "compute",
        [
            "gpu",
            "cuda",
            "pytorch",
            "nccl",
            "rapids",
            "faiss-gpu",
            "执行",
            "计算",
            "张量",
            "算子",
            "多gpu",
            "多 gpu",
            "stream",
            "kernel",
            "compute",
            "execution",
        ],
    ),
    (
        "support",
        [
            "支撑层",
            "工程支撑",
            "运行时",
            "后端",
            "资源管理",
            "调度",
            "控制面",
            "数据平面",
            "runtime",
            "backend",
            "resource",
            "scheduler",
            "control plane",
        ],
    ),
    (
        "input",
        [
            "输入",
            "数据源",
            "金融风控输入",
            "交易流水",
            "账户行为",
            "设备",
            "关系特征",
            "原始数据",
            "业务数据",
            "个人客户",
            "企业账户",
            "商户",
            "交易",
            "账户",
            "样本",
            "query",
            "input",
            "source",
            "raw data",
        ],
    ),
]

NON_FORMAL_KEYS = {"gradientColor", "shadow", "glass", "sketch"}
SKIP_FLAG_TOKENS = {"ellipse", "text"}
SKIP_SHAPES = {"line", "connector", "arrow", "flexArrow", "image", "label"}


@dataclass
class FileResult:
    path: str
    modified_cells: int
    backup_path: str | None
    edge_cells_modified: int
    valid_xml: bool
    error: str | None = None


def iter_target_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in ALLOWED_SUFFIXES:
            continue
        if path.name.endswith(".bak"):
            continue
        yield path


def parse_style(style: str) -> list[tuple[str, str | None]]:
    parts = []
    for token in style.split(";"):
        if token == "":
            continue
        if "=" in token:
            key, value = token.split("=", 1)
            parts.append((key, value))
        else:
            parts.append((token, None))
    return parts


def style_lookup(parts: list[tuple[str, str | None]]) -> tuple[dict[str, str], set[str]]:
    mapping: dict[str, str] = {}
    flags: set[str] = set()
    for key, value in parts:
        if value is None:
            flags.add(key)
        else:
            mapping[key] = value
    return mapping, flags


def serialize_style(parts: list[tuple[str, str | None]]) -> str:
    tokens = []
    for key, value in parts:
        if value is None:
            tokens.append(key)
        else:
            tokens.append(f"{key}={value}")
    return ";".join(tokens) + (";" if tokens else "")


def cell_text(cell: ET.Element) -> str:
    raw = (cell.get("value") or "") + " " + (cell.get("id") or "")
    raw = re.sub(r"<[^>]+>", " ", raw)
    raw = raw.replace("&nbsp;", " ")
    raw = " ".join(raw.split())
    return raw.lower()


def get_geometry_size(cell: ET.Element) -> tuple[float, float]:
    geo = cell.find("mxGeometry")
    if geo is None:
        return 0.0, 0.0
    try:
        width = float(geo.get("width", "0"))
    except ValueError:
        width = 0.0
    try:
        height = float(geo.get("height", "0"))
    except ValueError:
        height = 0.0
    return width, height


def is_component_box(cell: ET.Element) -> bool:
    if cell.tag != "mxCell":
        return False
    if cell.get("edge") == "1":
        return False
    if cell.get("vertex") != "1":
        return False

    parts = parse_style(cell.get("style", ""))
    mapping, flags = style_lookup(parts)

    if flags & SKIP_FLAG_TOKENS:
        return False
    shape = mapping.get("shape", "")
    if shape in SKIP_SHAPES:
        return False
    if mapping.get("fillColor", "").lower() in {"", "none"}:
        return False
    if "image" in flags:
        return False
    return True


def classify_color(cell: ET.Element) -> str:
    text = cell_text(cell)
    for category, keywords in KEYWORD_RULES:
        if any(keyword in text for keyword in keywords):
            return PALETTE[category]

    width, height = get_geometry_size(cell)
    if width <= 180 and height <= 70:
        return PALETTE["white"]
    return PALETTE["neutral"]


def update_style(style: str, fill_color: str) -> tuple[str, bool]:
    parts = parse_style(style)
    mapping, _ = style_lookup(parts)
    changed = False
    new_parts: list[tuple[str, str | None]] = []
    fill_seen = False
    opacity_seen = False

    for key, value in parts:
        if key in NON_FORMAL_KEYS:
            changed = True
            continue
        if key == "fillColor":
            fill_seen = True
            if value != fill_color:
                changed = True
            new_parts.append((key, fill_color))
            continue
        if key == "opacity":
            opacity_seen = True
            if value != "100":
                changed = True
            new_parts.append((key, "100"))
            continue
        new_parts.append((key, value))

    if not fill_seen:
        new_parts.append(("fillColor", fill_color))
        changed = True
    if "opacity" in mapping and not opacity_seen:
        new_parts.append(("opacity", "100"))
        changed = True

    new_style = serialize_style(new_parts)
    return new_style, changed


def decode_diagram_text(text: str) -> tuple[ET.Element, str]:
    stripped = text.strip()
    if stripped.startswith("<mxGraphModel"):
        return ET.fromstring(stripped), "plain-text"

    decoded = urllib.parse.unquote(
        zlib.decompress(base64.b64decode(stripped), -15).decode("utf-8")
    )
    return ET.fromstring(decoded), "compressed"


def encode_diagram_text(model: ET.Element, mode: str) -> str:
    xml_text = ET.tostring(model, encoding="unicode")
    if mode == "plain-text":
        return xml_text
    compressed = zlib.compress(xml_text.encode("utf-8"))[2:-4]
    return base64.b64encode(compressed).decode("ascii")


def modify_model(model: ET.Element) -> tuple[int, int]:
    modified_cells = 0
    edge_cells_modified = 0

    for cell in model.iter("mxCell"):
        if cell.get("edge") == "1":
            continue
        if not is_component_box(cell):
            continue

        new_fill = classify_color(cell)
        old_style = cell.get("style", "")
        new_style, changed = update_style(old_style, new_fill)
        if changed:
            cell.set("style", new_style)
            modified_cells += 1

    return modified_cells, edge_cells_modified


def load_xml(text: str) -> ET.Element:
    return ET.fromstring(text)


def save_xml(root: ET.Element, original_text: str) -> str:
    declaration = ""
    if original_text.lstrip().startswith("<?xml"):
        declaration = '<?xml version="1.0" encoding="UTF-8"?>\n'
    return declaration + ET.tostring(root, encoding="unicode")


def process_file(path: Path) -> FileResult:
    original_text = path.read_text(encoding="utf-8")
    original_root = load_xml(original_text)
    root = copy.deepcopy(original_root)

    modified_cells = 0
    edge_cells_modified = 0

    if root.tag == "mxGraphModel":
        modified_cells, edge_cells_modified = modify_model(root)
    elif root.tag == "mxfile":
        for diagram in root.findall("diagram"):
            if len(diagram):
                model = diagram[0]
                file_modified, file_edge_modified = modify_model(model)
                modified_cells += file_modified
                edge_cells_modified += file_edge_modified
            elif diagram.text and diagram.text.strip():
                model, mode = decode_diagram_text(diagram.text)
                file_modified, file_edge_modified = modify_model(model)
                modified_cells += file_modified
                edge_cells_modified += file_edge_modified
                if file_modified:
                    diagram.text = encode_diagram_text(model, mode)
    else:
        return FileResult(
            path=str(path),
            modified_cells=0,
            backup_path=None,
            edge_cells_modified=0,
            valid_xml=False,
            error=f"unsupported root tag: {root.tag}",
        )

    if modified_cells == 0:
        return FileResult(
            path=str(path),
            modified_cells=0,
            backup_path=None,
            edge_cells_modified=edge_cells_modified,
            valid_xml=True,
        )

    new_text = save_xml(root, original_text)
    try:
        load_xml(new_text)
    except ET.ParseError as exc:
        return FileResult(
            path=str(path),
            modified_cells=0,
            backup_path=None,
            edge_cells_modified=edge_cells_modified,
            valid_xml=False,
            error=f"post-write XML validation failed: {exc}",
        )

    backup_path = path.with_name(path.name + ".bak")
    shutil.copy2(path, backup_path)
    path.write_text(new_text, encoding="utf-8")

    return FileResult(
        path=str(path),
        modified_cells=modified_cells,
        backup_path=str(backup_path),
        edge_cells_modified=edge_cells_modified,
        valid_xml=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply thesis-friendly background colors to draw.io diagram boxes."
    )
    parser.add_argument(
        "root",
        nargs="?",
        default="figures",
        help="Directory to scan recursively. Defaults to figures.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        print(json.dumps({"error": f"root directory does not exist: {root}"}, ensure_ascii=False))
        return 1

    results = [process_file(path) for path in iter_target_files(root)]
    output = {
        "root": str(root),
        "files_scanned": len(results),
        "files_modified": sum(1 for item in results if item.modified_cells > 0),
        "cells_modified": sum(item.modified_cells for item in results),
        "edge_cells_modified": sum(item.edge_cells_modified for item in results),
        "results": [item.__dict__ for item in results],
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0 if all(item.valid_xml and not item.error for item in results) else 2


if __name__ == "__main__":
    sys.exit(main())

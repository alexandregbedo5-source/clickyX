"""Tests hors ligne de training/commfor.py (matérialisation Community Forensics-Small)."""
from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

pytest.importorskip("pyarrow")
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from PIL import Image  # noqa: E402

from training.commfor import DEFAULT_PLAN, Shard, dest_dir, extract_parquet, plan_quotas  # noqa: E402


def _img_bytes(fmt: str, size=(24, 24)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (120, 30, 200)).save(buf, format=fmt)
    return buf.getvalue()


def _write_parquet(path: Path, rows: list[dict]) -> None:
    table = pa.table({
        "image_name": [r["image_name"] for r in rows],
        "image_data": pa.array([r["image_data"] for r in rows], type=pa.binary()),
        "model_name": [r["model_name"] for r in rows],
        "label": pa.array([r["label"] for r in rows], type=pa.int64()),
        "prompt": ["" for _ in rows],
    })
    pq.write_table(table, path)


def test_plan_quotas_reach_max_per_class():
    quotas = plan_quotas(DEFAULT_PLAN, 6000)
    for label in (0, 1):
        total = sum(quotas[s.num] for s in DEFAULT_PLAN if s.label == label)
        assert total >= 6000
    assert plan_quotas([Shard(1, 1, 0.1, "x"), Shard(2, 1, 0.1, "y")], 11) == {1: 6, 2: 6}


def test_dest_dir_layout(tmp_path):
    assert dest_dir(tmp_path, 1, "stabilityai/stable-diffusion-2") == tmp_path / "ai" / "stabilityai__stable-diffusion-2"
    assert dest_dir(tmp_path, 0, "FFHQ") == tmp_path / "real" / "ffhq"
    assert dest_dir(tmp_path, 1, None) == tmp_path / "ai" / "unknown"


def test_extract_parquet_writes_original_bytes_and_respects_quota(tmp_path):
    png, jpg = _img_bytes("PNG"), _img_bytes("JPEG")
    rows = [
        {"image_name": "a.png", "image_data": png, "model_name": "gen/one", "label": 1},
        {"image_name": "b.jpg", "image_data": jpg, "model_name": "COCO", "label": 0},
        {"image_name": "c.png", "image_data": b"pas une image", "model_name": "gen/one", "label": 1},
        {"image_name": "d.png", "image_data": png, "model_name": "gen/two", "label": 1},
        {"image_name": "e.png", "image_data": png, "model_name": "gen/two", "label": 1},
    ]
    src = tmp_path / "shard.parquet"
    _write_parquet(src, rows)
    out = tmp_path / "out"
    counts = extract_parquet(src, out, quota=3, shard_num=7, batch_size=2)
    assert counts[1] == 2 and counts[0] == 1 and counts["corrompues"] == 1
    written = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())
    assert written == ["ai/gen__one/7_a.png", "ai/gen__two/7_d.png", "real/coco/7_b.jpg"]
    assert (out / "ai/gen__one/7_a.png").read_bytes() == png  # octets d'origine, pas de recompression
    assert (out / "real/coco/7_b.jpg").read_bytes() == jpg


def test_materialize_resumes_and_reports(tmp_path, monkeypatch):
    """L'orchestrateur saute les fichiers marqués faits et écrit _done quand tout est traité."""
    import training.commfor as cf

    calls: list[int] = []

    class FakeProc:
        def __init__(self, num):
            self.returncode = 0
            self.stdout = json.dumps({"1": 2}) + "\n"
            self.stderr = ""
            calls.append(num)

    def fake_run(cmd, **kw):
        num = int(cmd[cmd.index("--shard") + 1])
        out = Path(cmd[cmd.index("--out") + 1])
        (out / "ai" / f"g{num}").mkdir(parents=True, exist_ok=True)
        (out / "ai" / f"g{num}" / "x.png").write_bytes(_img_bytes("PNG"))
        return FakeProc(num)

    monkeypatch.setattr(cf.subprocess, "run", fake_run)
    plan = [Shard(1, 1, 0.1, "a"), Shard(2, 1, 5.0, "trop gros"), Shard(3, 0, 0.1, "c")]
    out = tmp_path / "out"
    cf.materialize(out, max_per_class=4, plan=plan, max_shard_gb=2.0)
    assert calls == [1, 3]
    assert (out / "_done").exists() and (out / "_shards" / "1.json").exists()
    calls.clear()
    cf.materialize(out, max_per_class=4, plan=plan, max_shard_gb=2.0)
    assert calls == []  # _done présent : rien relancé

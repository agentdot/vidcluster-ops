"""Run the VidCluster Intelligence Guardian V1 audit.

This script reads local canonical artifacts, computes health checks for a small
set of audited clusters, and writes audit-only outputs. It does not modify any
production data, scoring, clustering, dashboard, or taxonomy artifacts.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_CLUSTERS = ["SC002", "SC006", "SC161", "SC191"]
AUDIT_DATE = pd.Timestamp("2026-06-07", tz="UTC")
REQUIRED_WORKSPACE_FIELDS = [
    "workspace_root",
    "engine_repo",
    "public_repo",
    "ops_repo",
    "hq_repo",
    "audit_outputs",
]


def default_workspace_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "config" / "workspace.yaml"


def parse_simple_yaml(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" not in line:
                raise ValueError(f"{path}:{line_number} must use 'key: value' format.")
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if not key or not value:
                raise ValueError(f"{path}:{line_number} has an empty key or value.")
            values[key] = value
    return values


def load_workspace(config_path: Path | None = None) -> dict[str, Path]:
    path = config_path or default_workspace_config_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Workspace config not found: {path}\n"
            "Create vidcluster_ops/config/workspace.yaml or pass --workspace-config."
        )

    raw_config = parse_simple_yaml(path)
    missing_fields = [field for field in REQUIRED_WORKSPACE_FIELDS if field not in raw_config]
    if missing_fields:
        raise ValueError(
            f"Workspace config is missing required field(s): {', '.join(missing_fields)}\n"
            f"Config path: {path}"
        )

    workspace = {key: Path(value).expanduser().resolve() for key, value in raw_config.items()}
    missing_paths = [
        f"{field}: {workspace[field]}"
        for field in REQUIRED_WORKSPACE_FIELDS
        if not workspace[field].exists()
    ]
    if missing_paths:
        raise FileNotFoundError(
            "Workspace config points to missing path(s):\n"
            + "\n".join(f"- {item}" for item in missing_paths)
        )
    return workspace


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def latest_dated_file(base: Path, filename: str) -> Path:
    candidates = []
    if not base.exists():
        raise FileNotFoundError(f"Missing folder: {base}")

    for child in base.iterdir():
        if not child.is_dir():
            continue
        try:
            date_value = datetime.strptime(child.name, "%Y-%m-%d").date()
        except ValueError:
            continue
        target = child / filename
        if target.exists():
            candidates.append((date_value, target))

    if not candidates:
        raise FileNotFoundError(f"No dated {filename} found under {base}")

    return sorted(candidates, key=lambda item: item[0])[-1][1]


def latest_micro_assignments(engine_repo: Path) -> Path:
    tracking_root = (
        engine_repo
        / "automation-data"
        / "experiments"
        / "v4_0"
        / "derived"
        / "microstructure"
        / "tracking"
    )
    snapshot_root = tracking_root / "snapshots"
    candidates = []

    if snapshot_root.exists():
        for manifest_path in snapshot_root.glob("*/tracking_manifest.json"):
            try:
                manifest = read_json(manifest_path)
            except json.JSONDecodeError:
                continue
            if manifest.get("validation_status") != "PASS":
                continue
            snapshot_date = manifest.get("snapshot_date") or manifest_path.parent.name
            try:
                date_value = datetime.strptime(snapshot_date, "%Y-%m-%d").date()
            except ValueError:
                continue
            assignments_path = manifest_path.parent / "v4_0_micro_assignments.parquet"
            if assignments_path.exists():
                candidates.append((date_value, assignments_path))

    if candidates:
        return sorted(candidates, key=lambda item: item[0])[-1][1]

    fallback = tracking_root / "v4_0_micro_assignments.parquet"
    if fallback.exists():
        return fallback
    raise FileNotFoundError("No micro assignment artifact found.")


def latest_weekly_additions(engine_repo: Path) -> tuple[Path, pd.Timestamp]:
    additions_root = (
        engine_repo
        / "automation-data"
        / "experiments"
        / "v4_0"
        / "weekly_video_additions"
    )
    candidates = []
    for path in additions_root.glob("weekly_video_additions_*.parquet"):
        match = re.search(r"weekly_video_additions_(\d{4}-\d{2}-\d{2})\.parquet$", path.name)
        if not match:
            continue
        try:
            date_value = pd.Timestamp(match.group(1), tz="UTC")
        except ValueError:
            continue
        candidates.append((date_value, path))

    if not candidates:
        raise FileNotFoundError(f"No weekly additions parquet found under {additions_root}")

    snapshot_date, path = sorted(candidates, key=lambda item: item[0])[-1]
    return path, snapshot_date


def to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def pct(part: float, whole: float) -> float:
    if whole == 0 or pd.isna(whole):
        return 0.0
    return round((part / whole) * 100, 2)


def clean_scalar(value: Any) -> Any:
    if isinstance(value, list):
        return [clean_scalar(item) for item in value]
    if isinstance(value, dict):
        return {key: clean_scalar(item) for key, item in value.items()}
    if pd.isna(value):
        return None
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        return round(value, 2)
    return value


def records_for_json(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{key: clean_scalar(value) for key, value in row.items()} for row in rows]


def classify_format(duration_seconds: Any) -> str:
    if pd.isna(duration_seconds):
        return "Unknown"
    return "Shorts" if float(duration_seconds) <= 60 else "Long-form"


def classify_tier(row: pd.Series) -> str:
    view_count = row.get("view_count")
    views_per_day = row.get("views_per_day")
    if pd.isna(view_count) or pd.isna(views_per_day):
        return "Unknown"
    if views_per_day >= 100 or view_count >= 10000:
        return "Tier A"
    if views_per_day >= 20 or view_count >= 1000:
        return "Tier B"
    return "Tier C"


def trust_status(creator_count: int, video_count: int, top_views: float, median_vpd: float) -> str:
    if (
        creator_count >= 10
        and video_count >= 100
        and (top_views >= 100000 or median_vpd >= 50)
    ):
        return "Strong"
    if (
        creator_count >= 3
        and video_count >= 30
        and (top_views >= 10000 or median_vpd >= 20)
    ):
        return "Moderate"
    return "Weak"


def display_path(path: Path, workspace_root: Path) -> str:
    try:
        return str(path.relative_to(workspace_root))
    except ValueError:
        return str(path)


def load_cluster_frame(workspace: dict[str, Path], clusters: list[str]) -> tuple[pd.DataFrame, dict[str, str]]:
    engine_repo = workspace["engine_repo"]
    workspace_root = workspace["workspace_root"]
    micro_path = latest_micro_assignments(engine_repo)
    universe_path = (
        engine_repo
        / "automation-data"
        / "experiments"
        / "v4_0"
        / "v4_0_videos_universe.csv"
    )
    tracking_path = latest_dated_file(
        engine_repo
        / "automation-data"
        / "experiments"
        / "v4_0"
        / "snapshots"
        / "video_tracking",
        "video_tracking_snapshot.csv",
    )

    micro = pd.read_parquet(micro_path)
    micro = micro[micro["cluster_id"].isin(clusters)].copy()

    universe = pd.read_csv(universe_path)
    universe_cols = [
        "video_id",
        "title",
        "channel_id",
        "channel_title",
        "published_at",
        "view_count",
    ]
    universe = universe[[col for col in universe_cols if col in universe.columns]].copy()
    universe = universe.rename(
        columns={
            "title": "video_title",
            "published_at": "published_at_universe",
            "view_count": "view_count_universe",
        }
    )

    tracking = pd.read_csv(tracking_path)
    tracking_cols = [
        "video_id",
        "view_count",
        "video_duration_seconds",
        "video_format",
        "published_at",
    ]
    tracking = tracking[[col for col in tracking_cols if col in tracking.columns]].copy()
    tracking = tracking.rename(
        columns={
            "view_count": "view_count_tracking",
            "published_at": "published_at_tracking",
        }
    )

    df = micro.merge(universe, on="video_id", how="left").merge(tracking, on="video_id", how="left")
    df["published_at_final"] = pd.to_datetime(
        df.get("published_at_tracking"), errors="coerce", utc=True
    ).fillna(pd.to_datetime(df.get("published_at_universe"), errors="coerce", utc=True))
    df["view_count"] = to_numeric(df.get("view_count_tracking")).fillna(
        to_numeric(df.get("view_count_universe"))
    )
    df["duration_seconds"] = to_numeric(df.get("video_duration_seconds"))
    df["format"] = df["duration_seconds"].apply(classify_format)
    age_days = (AUDIT_DATE - df["published_at_final"]).dt.total_seconds() / 86400
    df["age_days"] = age_days.clip(lower=1)
    df["views_per_day"] = df["view_count"] / df["age_days"]

    paths = {
        "micro_assignments": display_path(micro_path, workspace_root),
        "video_universe": display_path(universe_path, workspace_root),
        "video_tracking": display_path(tracking_path, workspace_root),
    }
    return df, paths


def load_weekly_frame(
    workspace: dict[str, Path], clusters: list[str], cluster_frame: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, str]]:
    engine_repo = workspace["engine_repo"]
    workspace_root = workspace["workspace_root"]
    weekly_path, snapshot_date = latest_weekly_additions(engine_repo)
    live_universe_path = engine_repo / "automation-data" / "live" / "latest_video_universe.csv"

    weekly = pd.read_parquet(weekly_path)
    live_universe = pd.read_csv(live_universe_path)
    live_cols = ["video_id", "view_count", "published_at", "channel_id", "channel_title", "title"]
    live_universe = live_universe[[col for col in live_cols if col in live_universe.columns]].copy()
    live_universe = live_universe.rename(
        columns={
            "view_count": "view_count_live",
            "published_at": "published_at_live",
            "channel_id": "channel_id_live",
            "channel_title": "channel_title_live",
            "title": "title_live",
        }
    )

    assignment_cols = ["video_id", "cluster_id", "subcluster_id", "subcluster_label"]
    assigned = cluster_frame[[col for col in assignment_cols if col in cluster_frame.columns]].drop_duplicates(
        "video_id"
    )

    df = weekly.merge(assigned, on="video_id", how="left", suffixes=("", "_micro"))
    if "cluster_id" in df.columns:
        df["cluster_id_final"] = df["cluster_id"].fillna(df.get("assigned_cluster_id"))
    else:
        df["cluster_id_final"] = df.get("assigned_cluster_id")

    df = df[df["cluster_id_final"].isin(clusters)].copy()
    df = df.merge(live_universe, on="video_id", how="left")

    df["cluster_id"] = df["cluster_id_final"]
    df["view_count"] = to_numeric(df.get("view_count_live"))
    if "published_at_live" in df.columns:
        published = pd.to_datetime(df["published_at_live"], errors="coerce", utc=True)
    else:
        published = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")
    fallback_published = pd.to_datetime(df.get("published_at"), errors="coerce", utc=True)
    df["published_at_final"] = published.fillna(fallback_published)
    age_days = (snapshot_date - df["published_at_final"]).dt.total_seconds() / 86400
    df["age_days"] = age_days.clip(lower=1)
    df["views_per_day"] = df["view_count"] / df["age_days"]
    df["tier"] = df.apply(classify_tier, axis=1)

    paths = {
        "weekly_additions": display_path(weekly_path, workspace_root),
        "live_video_universe": display_path(live_universe_path, workspace_root),
        "weekly_snapshot_date": snapshot_date.strftime("%Y-%m-%d"),
    }
    return df, paths


def subcluster_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    cluster_total = df["video_id"].nunique()
    for subcluster_id, group in df.groupby("subcluster_id", dropna=False):
        video_count = group["video_id"].nunique()
        channel_count = group["channel_id"].nunique(dropna=True)
        channel_counts = group.groupby("channel_title", dropna=False)["video_id"].nunique().sort_values(
            ascending=False
        )
        top_channel_count = int(channel_counts.iloc[0]) if not channel_counts.empty else 0
        top_channel_name = str(channel_counts.index[0]) if not channel_counts.empty else ""
        shorts_count = int((group["format"] == "Shorts").sum())
        longform_count = int((group["format"] == "Long-form").sum())
        rows.append(
            {
                "subcluster_id": subcluster_id,
                "subcluster_label": group["subcluster_label"].dropna().iloc[0]
                if group["subcluster_label"].notna().any()
                else "",
                "video_count": video_count,
                "video_share_pct": pct(video_count, cluster_total),
                "channel_count": channel_count,
                "top_channel_name": top_channel_name,
                "top_channel_video_count": top_channel_count,
                "top_channel_share_pct": pct(top_channel_count, video_count),
                "shorts_count": shorts_count,
                "longform_count": longform_count,
                "shorts_share_pct": pct(shorts_count, video_count),
                "longform_share_pct": pct(longform_count, video_count),
            }
        )
    return pd.DataFrame(rows)


def analyze_cluster(cluster_id: str, cluster_frame: pd.DataFrame, weekly_frame: pd.DataFrame) -> dict[str, Any]:
    cluster = cluster_frame[cluster_frame["cluster_id"] == cluster_id].copy()
    weekly = weekly_frame[weekly_frame["cluster_id"] == cluster_id].copy()

    total_videos = int(cluster["video_id"].nunique())
    total_channels = int(cluster["channel_id"].nunique(dropna=True))
    subs = subcluster_stats(cluster) if total_videos else pd.DataFrame()
    number_of_subclusters = int(subs["subcluster_id"].nunique()) if not subs.empty else 0

    largest_share = float(subs["video_share_pct"].max()) if not subs.empty else 0.0
    largest = subs.sort_values("video_share_pct", ascending=False).head(1)
    largest_channels = int(largest["channel_count"].iloc[0]) if not largest.empty else 0
    dominant_center_present = largest_share >= 50 and largest_channels >= 5

    creator_led_branch_count = int(
        ((subs["channel_count"] <= 2) & (subs["video_count"] >= 10)).sum()
    ) if not subs.empty else 0
    small_satellite_count = int((subs["video_share_pct"] < 10).sum()) if not subs.empty else 0
    brand_led_branch_count = int((subs["top_channel_share_pct"] >= 80).sum()) if not subs.empty else 0

    shorts_count = int((cluster["format"] == "Shorts").sum())
    longform_count = int((cluster["format"] == "Long-form").sum())
    shorts_share = pct(shorts_count, total_videos)
    longform_share = pct(longform_count, total_videos)
    if shorts_share >= 70:
        dominant_format = "Shorts"
    elif longform_share >= 70:
        dominant_format = "Long-form"
    else:
        dominant_format = "Mixed"
    format_skewed_branch_count = int(
        ((subs["shorts_share_pct"] >= 70) | (subs["longform_share_pct"] >= 70)).sum()
    ) if not subs.empty else 0

    tier_counts = weekly["tier"].value_counts() if not weekly.empty else pd.Series(dtype=int)
    new_videos_count = int(weekly["video_id"].nunique()) if not weekly.empty else 0
    tier_a_count = int(tier_counts.get("Tier A", 0))
    tier_b_count = int(tier_counts.get("Tier B", 0))
    tier_c_count = int(tier_counts.get("Tier C", 0))
    unknown_tier_count = int(tier_counts.get("Unknown", 0))
    tier_a_share = pct(tier_a_count, new_videos_count)
    tier_c_share = pct(tier_c_count, new_videos_count)
    weekly_median_vpd = float(weekly["views_per_day"].median()) if not weekly.empty else 0.0

    top_video_views = float(cluster["view_count"].max()) if total_videos else 0.0
    median_views = float(cluster["view_count"].median()) if total_videos else 0.0
    top_video_vpd = float(cluster["views_per_day"].max()) if total_videos else 0.0
    median_vpd = float(cluster["views_per_day"].median()) if total_videos else 0.0
    trust = trust_status(total_channels, total_videos, top_video_views, median_vpd)

    warnings = []
    if largest_share >= 80:
        warnings.append("Dominant center over-concentrated")
    if tier_c_share >= 70 and new_videos_count > 0:
        warnings.append("Evidence mostly discovery-only")
    if total_channels <= 2:
        warnings.append("Low creator breadth")
    if trust == "Weak":
        warnings.append("Weak creator-facing proof")
    if shorts_share >= 80 or longform_share >= 80:
        warnings.append("Format heavily skewed")

    return {
        "cluster_id": cluster_id,
        "total_videos": total_videos,
        "total_channels": total_channels,
        "number_of_subclusters": number_of_subclusters,
        "largest_subcluster_share_pct": round(largest_share, 2),
        "dominant_center_present": "yes" if dominant_center_present else "no",
        "creator_led_branch_count": creator_led_branch_count,
        "small_satellite_count": small_satellite_count,
        "brand_led_branch_count": brand_led_branch_count,
        "shorts_count": shorts_count,
        "longform_count": longform_count,
        "shorts_share_pct": shorts_share,
        "longform_share_pct": longform_share,
        "dominant_format": dominant_format,
        "format_skewed_branch_count": format_skewed_branch_count,
        "new_videos_count": new_videos_count,
        "tier_a_count": tier_a_count,
        "tier_b_count": tier_b_count,
        "tier_c_count": tier_c_count,
        "unknown_tier_count": unknown_tier_count,
        "tier_a_share_pct": tier_a_share,
        "tier_c_share_pct": tier_c_share,
        "weekly_median_views_per_day": round(weekly_median_vpd, 2),
        "top_video_views": round(top_video_views, 2),
        "median_views": round(median_views, 2),
        "top_video_views_per_day": round(top_video_vpd, 2),
        "median_views_per_day": round(median_vpd, 2),
        "creator_count": total_channels,
        "video_count": total_videos,
        "trust_evidence_status": trust,
        "warning_flags": warnings,
    }


def names_where(rows: list[dict[str, Any]], condition) -> list[str]:
    return [row["cluster_id"] for row in rows if condition(row)]


def write_markdown(path: Path, rows: list[dict[str, Any]], source_paths: dict[str, str]) -> None:
    healthy = names_where(
        rows,
        lambda row: row["trust_evidence_status"] in {"Strong", "Moderate"}
        and "Low creator breadth" not in row["warning_flags"]
        and "Weak creator-facing proof" not in row["warning_flags"],
    )
    weak = names_where(rows, lambda row: row["trust_evidence_status"] == "Weak")
    over_concentrated = names_where(
        rows, lambda row: "Dominant center over-concentrated" in row["warning_flags"]
    )
    discovery_only = names_where(
        rows, lambda row: "Evidence mostly discovery-only" in row["warning_flags"]
    )
    needs_review = names_where(rows, lambda row: len(row["warning_flags"]) > 0)

    lines = [
        "# Intelligence Guardian V1",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
        "This is a local audit check. It does not change VidCluster data, scores, pages, or labels.",
        "",
        "## Inputs",
    ]
    for label, value in source_paths.items():
        lines.append(f"- {label}: `{value}`")

    lines.extend(
        [
            "",
            "## Cluster Check",
            "",
            "| Cluster | Videos | Creators | Main branch share | Format | New videos | Trust | Warnings |",
            "| --- | ---: | ---: | ---: | --- | ---: | --- | --- |",
        ]
    )
    for row in rows:
        warnings = "; ".join(row["warning_flags"]) if row["warning_flags"] else "None"
        lines.append(
            "| {cluster_id} | {video_count} | {creator_count} | {largest_subcluster_share_pct}% | "
            "{dominant_format} | {new_videos_count} | {trust_evidence_status} | {warnings} |".format(
                warnings=warnings,
                **row,
            )
        )

    lines.extend(
        [
            "",
            "## What Looks Healthy?",
            "",
            ", ".join(healthy) if healthy else "No cluster passed without a warning in this sample.",
            "",
            "## Weak Video Proof",
            "",
            ", ".join(weak) if weak else "No sampled cluster has weak video proof.",
            "",
            "## Over-Concentrated Clusters",
            "",
            ", ".join(over_concentrated)
            if over_concentrated
            else "No sampled cluster has a main branch above 80% of its videos.",
            "",
            "## Mostly Discovery-Only New Videos",
            "",
            ", ".join(discovery_only)
            if discovery_only
            else "No sampled cluster has mostly discovery-only new videos in the latest weekly run.",
            "",
            "## Needs Human Review",
            "",
            ", ".join(needs_review)
            if needs_review
            else "No sampled cluster raised a warning flag.",
            "",
            "## What Bilal Should Check After Work",
            "",
            "- Review clusters with warning flags before using them as creator-facing proof.",
            "- For clusters with many weak new videos, use the new videos for discovery but not as the main proof.",
            "- Check whether a large main branch is hiding smaller branches that need a different explanation.",
            "- Check low creator breadth clusters carefully, because one creator or brand can make a cluster look bigger than it really is.",
        ]
    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run VidCluster Intelligence Guardian V1.")
    parser.add_argument("--clusters", nargs="*", default=DEFAULT_CLUSTERS)
    parser.add_argument(
        "--workspace-config",
        type=Path,
        default=None,
        help="Path to workspace.yaml. Defaults to vidcluster_ops/config/workspace.yaml.",
    )
    args = parser.parse_args()
    clusters = args.clusters or DEFAULT_CLUSTERS

    workspace = load_workspace(args.workspace_config)
    output_dir = workspace["audit_outputs"] / "intelligence_guardian_v1"
    output_dir.mkdir(parents=True, exist_ok=True)

    cluster_frame, cluster_paths = load_cluster_frame(workspace, clusters)
    weekly_frame, weekly_paths = load_weekly_frame(workspace, clusters, cluster_frame)
    source_paths = {
        "workspace_config": display_path(
            args.workspace_config.resolve() if args.workspace_config else default_workspace_config_path(),
            workspace["workspace_root"],
        ),
        **cluster_paths,
        **weekly_paths,
    }

    rows = [analyze_cluster(cluster_id, cluster_frame, weekly_frame) for cluster_id in clusters]

    csv_rows = []
    for row in rows:
        csv_row = dict(row)
        csv_row["warning_flags"] = "; ".join(row["warning_flags"])
        csv_rows.append(csv_row)

    summary_csv = output_dir / "intelligence_guardian_cluster_summary_v1.csv"
    findings_json = output_dir / "intelligence_guardian_findings_v1.json"
    summary_md = output_dir / "intelligence_guardian_summary_v1.md"

    pd.DataFrame(csv_rows).to_csv(summary_csv, index=False)
    findings = {
        "guardian_version": "v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "clusters": clusters,
        "source_paths": source_paths,
        "findings": records_for_json(rows),
    }
    findings_json.write_text(json.dumps(findings, indent=2), encoding="utf-8")
    write_markdown(summary_md, rows, source_paths)

    print("Wrote:")
    print(summary_md)
    print(summary_csv)
    print(findings_json)


if __name__ == "__main__":
    main()

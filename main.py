#!/usr/bin/env python3
"""Main entrypoint for running pipelines in the Choreo project.

This file is the filesystem/CLI ADAPTER: it owns argument parsing, IO path
resolution, plots and cost reporting. The actual matching compute lives in the
mode runners (choreo/runners.py — ``run_full_match`` / ``run_query_match`` /
``run_batch_match``), which take schema objects in and return schema objects
out and are equally importable by external apps that own their own storage.

Config comes from the packaged defaults (``choreo/defaults/``), optionally
overlaid by ``--config-dir <dir>`` (any subset of config.yaml + prompt yamls)
and per-run ``--set dotted.key=value`` overrides — see ``choreo/config.py``.
"""

import json
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import yaml
from dotenv import load_dotenv

from choreo.config import load_config, resolve_prompt_paths, set_by_path
from choreo.utils import DEFAULT_PROMPT_PATHS
from choreo.llm import LLMWrapper
from choreo.ingest import load_profiles
from choreo.store import FileStore
from choreo.runners import run_full_match, run_query_match, run_batch_match
from choreo.score_correlation import (
    create_normalized_score_correlation_plot,
    create_normalized_detailed_score_analysis,
)
from choreo.raw_data import save_score_correlation_raw_data
from choreo.report import write_reports, print_cohort_summary
from choreo.cost_tracker import get_cost_tracker
from choreo.visualize_similarity import create_similarity_plots
from choreo.tsne import create_tsne_plots


DEFAULT_SECTIONS_CONFIG_PATH = DEFAULT_PROMPT_PATHS["sections"]
DEFAULT_SCORING_PROMPT_PATH = DEFAULT_PROMPT_PATHS["scoring"]
DEFAULT_INTRODUCTION_PROMPT_PATH = DEFAULT_PROMPT_PATHS["introduction"]
DEFAULT_HYDE_PROMPT_PATH = DEFAULT_PROMPT_PATHS["hyde"]


@dataclass
class PipelineContext:
    """Container for runtime information shared with pipelines."""

    config: Dict[str, Any]
    force: bool = False
    group_name: Optional[str] = None
    config_dir: Optional[str] = None  # optional dir of config/prompt overrides
    input_dir: Optional[str] = None
    query: Optional[str] = None      # query_match: raw text or JSON section mapping
    members: Optional[str] = None    # batch_match: comma-separated member ids

    @property
    def user_profiles_dir(self) -> str:
        return self.config.setdefault("io", {}).get("raw_dir", "")


class BasePipeline:
    """Protocol for all top-level pipelines."""

    name: str = ""
    description: str = ""

    def run(self, context: PipelineContext) -> Dict[str, Any]:
        raise NotImplementedError("Pipelines must implement run()")


class PipelineRegistry:
    """Simple registry to look up pipelines by name."""

    def __init__(self) -> None:
        self._pipelines: Dict[str, BasePipeline] = {}

    def register(self, pipeline: BasePipeline) -> None:
        if not pipeline.name:
            raise ValueError("Pipeline must define a name")
        if pipeline.name in self._pipelines:
            raise ValueError(f"Pipeline '{pipeline.name}' already registered")
        self._pipelines[pipeline.name] = pipeline

    def get(self, name: str) -> BasePipeline:
        try:
            return self._pipelines[name]
        except KeyError as exc:
            raise KeyError(f"Unknown pipeline '{name}'") from exc

    def names(self) -> Tuple[str, ...]:
        return tuple(sorted(self._pipelines.keys()))

    def items(self) -> Iterator[Tuple[str, BasePipeline]]:
        for name in self.names():
            yield name, self._pipelines[name]


def apply_io_overrides(
    config: Dict[str, Any],
    group_name: Optional[str] = None,
    input_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Point all IO paths at the right working area.

    Two layouts (``input_dir`` takes precedence over ``group_name``):
      - input_dir: profiles live directly in ``<input_dir>``; all derived
        artifacts (processed/embeds/outputs/cache) are written to subdirs of
        ``<input_dir>`` (e.g. ``<input_dir>/outputs``).
      - group_name: profiles live in ``data/<group>/raw``; artifacts in
        ``data/<group>/{processed,embeds,outputs,cache}``.
    If neither is given, the config's existing io paths are left untouched
    (used by the Modal entrypoint, which sets its paths explicitly).
    """
    io_config = config.setdefault("io", {})

    if input_dir:
        base_path = Path(input_dir).expanduser()
        raw_dir = base_path  # .txt files live directly in the input folder
    elif group_name:
        base_path = Path("data") / group_name
        raw_dir = base_path / "raw"
    else:
        return config

    io_config.update({
        "raw_dir": str(raw_dir),
        "processed_dir": str(base_path / "processed"),
        "embeds_dir": str(base_path / "embeds"),
        "outputs_dir": str(base_path / "outputs"),
        "cache_dir": str(base_path / "cache"),
    })
    return config


def _execute_matching_pipeline(
    config: Dict[str, Any],
    *,
    sections_config_path: str = DEFAULT_SECTIONS_CONFIG_PATH,
    scoring_prompt_path: str = DEFAULT_SCORING_PROMPT_PATH,
    introduction_prompt_path: str = DEFAULT_INTRODUCTION_PROMPT_PATH,
    hyde_prompt_path: str = DEFAULT_HYDE_PROMPT_PATH,
    force: bool = False,
    group_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the matching pipeline with the provided configuration.

    Adapter composition: FileStore (disk caches + persistence) around
    ``run_full_match`` (the in-memory compute), plus plots and cost reporting.
    """

    print("🚀 Starting prompt-mesh matching pipeline...")

    io_config = config.setdefault("io", {})
    user_profiles_dir = io_config.get("raw_dir")
    if not user_profiles_dir:
        message = "Missing 'io.raw_dir' in configuration"
        print(f"❌ {message}")
        return {"success": False, "error": message}

    if group_name:
        print(f"📁 Using group-specific data: {group_name}")

    store = FileStore.from_io_config(io_config)
    llm_wrapper = LLMWrapper(
        cache_dir=io_config.get("cache_dir"),
        reasoning_effort=config.get("models", {}).get("reasoning_effort", "low"),
    )

    if force:
        print("🔄 Force flag set - all steps will be re-run, ignoring existing data")

    print("\n📁 Step 1: Ingesting profiles...")
    profiles = load_profiles(user_profiles_dir)
    print(f"✅ Loaded {len(profiles)} profiles")

    # Early check for minimum group size
    MIN_PROFILES_REQUIRED = config.get("matching", {}).get("min_profiles_required", 4)
    if len(profiles) < MIN_PROFILES_REQUIRED:
        message = (
            f"Insufficient profiles for matching. Found {len(profiles)} profile(s), "
            f"but at least {MIN_PROFILES_REQUIRED} are required. "
            "This matching tool is designed for larger groups where meaningful connections can be discovered."
        )
        print(f"❌ {message}")
        return {
            "success": False,
            "error": message,
            "profiles_count": len(profiles),
            "min_required": MIN_PROFILES_REQUIRED,
        }

    try:
        run = run_full_match(
            profiles,
            config,
            store=store,
            llm_wrapper=llm_wrapper,
            prompt_paths={
                "sections": sections_config_path,
                "scoring": scoring_prompt_path,
                "introduction": introduction_prompt_path,
                "hyde": hyde_prompt_path,
            },
            force=force,
        )
    except Exception as exc:  # pylint: disable=broad-except
        print(f"❌ {exc}")
        return {"success": False, "error": str(exc)}

    final_edges = run["edges"]
    similarity = run["similarity"]

    print("\n📊 Step 3.5: Creating t-SNE visualizations...")
    tsne_results = None
    try:
        tsne_results = create_tsne_plots(
            embeddings=run["working_embeddings"],
            user_ids=run["embeddings"].user_ids,
            section_names=run["embeddings"].section_names,
            output_dir=io_config.get("outputs_dir"),
            metric="cosine",
            perplexity=7,
        )
        print(f"✅ Created t-SNE plots: {tsne_results['plots_dir']}")
    except Exception as exc:  # pylint: disable=broad-except
        # t-SNE visualization is non-essential, don't fail the pipeline
        print(f"⚠️ Warning: Could not create t-SNE visualizations: {exc}")

    normalized_embed_scores = run["normalized_embed_scores"]
    normalized_llm_scores = run["normalized_llm_scores"]
    if normalized_embed_scores and normalized_llm_scores:
        try:
            create_normalized_score_correlation_plot(
                normalized_embed_scores=normalized_embed_scores,
                normalized_llm_scores=normalized_llm_scores,
                output_dir=io_config.get("outputs_dir"),
                group_name=group_name,
            )
            create_normalized_detailed_score_analysis(
                normalized_embed_scores=normalized_embed_scores,
                normalized_llm_scores=normalized_llm_scores,
                output_dir=io_config.get("outputs_dir"),
                group_name=group_name,
            )
            # Persist the per-pair scores behind these plots (crash-safe).
            save_score_correlation_raw_data(
                output_dir=io_config.get("outputs_dir"),
                normalized_embed_scores=normalized_embed_scores,
                normalized_llm_scores=normalized_llm_scores,
                group_name=group_name,
            )
            print("✅ Generated score correlation plots")
        except Exception as exc:  # pylint: disable=broad-except
            print(f"⚠️ Warning: Failed to generate correlation plots: {exc}")
    else:
        print("⚠️ Skipping correlation plots: normalized scores not available")

    print("\n📝 Step 8: Writing reports...")
    try:
        write_reports(run["report_data"], io_config.get("outputs_dir"))
        cohort_summary = run["report_data"]["cohort_summary"]
        print_cohort_summary(cohort_summary)
        print("✅ Generated reports for all users")
    except Exception as exc:  # pylint: disable=broad-except
        message = f"Error generating reports: {exc}"
        print(f"❌ {message}")
        return {"success": False, "error": message}

    stats = llm_wrapper.get_stats()
    print(f"\n📊 LLM Usage: {stats['total_calls']} total calls")

    cost_tracker = get_cost_tracker()
    cost_tracker.print_summary()

    cost_report_path = Path(io_config.get("outputs_dir", "")) / "cost_report.json"
    cost_tracker.save_detailed_report(str(cost_report_path))

    print("\n🎨 Step 9: Creating similarity visualizations...")
    plots_results = None
    try:
        plots_results = create_similarity_plots(
            matrices_dict=similarity["matrices_dict"],
            user_ids=similarity["user_ids"],
            recipe_config=config.get("recipe", {}),
            output_dir=io_config.get("outputs_dir"),
            group_name=group_name,
        )
        print(f"✅ Created similarity visualizations: {plots_results['plots_dir']}")
    except Exception as exc:  # pylint: disable=broad-except
        # Similarity visualizations are non-essential, don't fail the pipeline
        print(f"⚠️ Warning: Could not create similarity visualizations: {exc}")

    print("\n🎉 Pipeline completed successfully!")
    print(f"📁 Check outputs in: {io_config.get('outputs_dir')}")
    print(f"📊 Cohort summary: {io_config.get('outputs_dir')}/cohort.json")
    print(f"💰 Cost report: {cost_report_path}")
    if tsne_results:
        print(f"📊 t-SNE plots: {tsne_results['plots_dir']}")
    if plots_results:
        print(f"🎨 Similarity plots: {plots_results['plots_dir']}")

    result = {
        "success": True,
        "matches": final_edges,
        "profiles_count": len(profiles),
        "outputs_dir": io_config.get("outputs_dir"),
        "cost_report_path": str(cost_report_path),
        "cohort_summary": cohort_summary,
        "stats": {
            "llm_calls": stats["total_calls"],
            "matches_created": len(final_edges),
        },
    }

    # Add optional visualization paths if they were created successfully
    if tsne_results:
        result["tsne_plots_dir"] = tsne_results["plots_dir"]
    if plots_results:
        result["similarity_plots_dir"] = plots_results["plots_dir"]

    return result


class MatchingPipeline(BasePipeline):
    """Existing user matching pipeline exposed through the registry."""

    name = "matching"
    description = "Generate pair matches and reports from user profiles."

    def run(self, context: PipelineContext) -> Dict[str, Any]:
        config_copy = deepcopy(context.config)
        config_copy.setdefault("io", {})
        apply_io_overrides(config_copy, context.group_name, context.input_dir)

        prompt_paths = resolve_prompt_paths(config_dir=context.config_dir, config=config_copy)

        return _execute_matching_pipeline(
            config=config_copy,
            sections_config_path=prompt_paths["sections"],
            scoring_prompt_path=prompt_paths["scoring"],
            introduction_prompt_path=prompt_paths["introduction"],
            hyde_prompt_path=prompt_paths["hyde"],
            force=context.force,
            group_name=context.group_name,
        )


class QueryMatchPipeline(BasePipeline):
    """Mode B: rank the stored community pool against one transient query."""

    name = "query_match"
    description = ("Match a single query (e.g. \"find me a CTO who…\") against the "
                   "group's pre-built embeddings. Pass --query '<text>' or "
                   "--query '{\"needs\": \"…\"}'.")

    def run(self, context: PipelineContext) -> Dict[str, Any]:
        config_copy = deepcopy(context.config)
        config_copy.setdefault("io", {})
        apply_io_overrides(config_copy, context.group_name, context.input_dir)
        io_config = config_copy["io"]

        if not context.query:
            message = "query_match needs --query '<text>' (or a JSON section mapping)"
            print(f"❌ {message}")
            return {"success": False, "error": message}

        query: Any = context.query
        if query.strip().startswith("{"):
            try:
                query = json.loads(query)
            except json.JSONDecodeError as exc:
                message = f"--query looks like a JSON section mapping but failed to parse: {exc}"
                print(f"❌ {message}")
                return {"success": False, "error": message}

        store = FileStore.from_io_config(io_config)
        try:
            pool = store.get_embeddings()
        except FileNotFoundError:
            message = (f"No embeddings found in {io_config.get('embeds_dir')} — run the "
                       "'matching' pipeline (or upsert profiles) for this group first.")
            print(f"❌ {message}")
            return {"success": False, "error": message}
        pool_sections = {s.id: s.sections for s in store.get_sections()}

        llm_wrapper = LLMWrapper(
            cache_dir=io_config.get("cache_dir"),
            reasoning_effort=config_copy.get("models", {}).get("reasoning_effort", "low"),
        )

        result = run_query_match(
            query=query,
            pool=pool,
            config=config_copy,
            pool_sections=pool_sections,
            llm_wrapper=llm_wrapper,
            prompt_paths=resolve_prompt_paths(config_dir=context.config_dir, config=config_copy),
        )

        print(f"\n🔎 Query shortlist ({len(result.shortlist)} of {result.pool_size} pool users):")
        for entry in result.shortlist:
            llm_part = f", llm {entry['llm_score']}" if entry["llm_score"] is not None else ""
            print(f"  #{entry['rank']} {entry['user_id']} — score {entry['score']} "
                  f"(embed {entry['embed_score']}{llm_part})")
        for note in result.notes:
            print(f"  ⚠️ {note}")

        return {"success": True, "query_result": result.to_dict()}


class BatchMatchPipeline(BasePipeline):
    """Mode C: subset batch match (members × pool) with novelty exclusions."""

    name = "batch_match"
    description = ("Match an explicit member subset (--members a,b,c) against the "
                   "group's pool, excluding pairs surfaced within the configured "
                   "novelty window. Appends new pairs to match_history.jsonl.")

    def run(self, context: PipelineContext) -> Dict[str, Any]:
        config_copy = deepcopy(context.config)
        config_copy.setdefault("io", {})
        apply_io_overrides(config_copy, context.group_name, context.input_dir)
        io_config = config_copy["io"]

        if not context.members:
            message = "batch_match needs --members '<id1,id2,…>'"
            print(f"❌ {message}")
            return {"success": False, "error": message}
        member_ids = [m.strip() for m in context.members.split(",") if m.strip()]

        store = FileStore.from_io_config(io_config)
        try:
            pool = store.get_embeddings()
        except FileNotFoundError:
            message = (f"No embeddings found in {io_config.get('embeds_dir')} — run the "
                       "'matching' pipeline (or upsert profiles) for this group first.")
            print(f"❌ {message}")
            return {"success": False, "error": message}
        pool_sections = {s.id: s.sections for s in store.get_sections()}

        novelty_window = config_copy.get("matching", {}).get("novelty_window_months", 6)
        excluded_pairs = store.get_match_history(window_months=novelty_window)
        print(f"🚫 Excluding {len(excluded_pairs)} previously surfaced pairs "
              f"(novelty window: {novelty_window} months)")

        llm_wrapper = LLMWrapper(
            cache_dir=io_config.get("cache_dir"),
            reasoning_effort=config_copy.get("models", {}).get("reasoning_effort", "low"),
        )

        result = run_batch_match(
            member_ids=member_ids,
            pool=pool,
            config=config_copy,
            excluded_pairs=excluded_pairs,
            pool_sections=pool_sections,
            llm_wrapper=llm_wrapper,
            prompt_paths=resolve_prompt_paths(config_dir=context.config_dir, config=config_copy),
        )

        # Persist: member reports to outputs/batch/, new pairs to history.
        batch_outputs = str(Path(io_config.get("outputs_dir")) / "batch")
        write_reports(result.report_data, batch_outputs)
        store.put_matches(result.edges)
        print(f"📒 Appended {len(result.new_pairs)} new pairs to {store.history_path}")
        print_cohort_summary(result.report_data["cohort_summary"])

        return {
            "success": True,
            "edges": [e.to_dict() for e in result.edges],
            "new_pairs": result.new_pairs,
            "outputs_dir": batch_outputs,
        }


PIPELINE_REGISTRY = PipelineRegistry()
PIPELINE_REGISTRY.register(MatchingPipeline())
PIPELINE_REGISTRY.register(QueryMatchPipeline())
PIPELINE_REGISTRY.register(BatchMatchPipeline())


def run_matching_pipeline(
    user_profiles_dir: str,
    config_dict: Dict[str, Any],
    sections_config_path: str = DEFAULT_SECTIONS_CONFIG_PATH,
    scoring_prompt_path: str = DEFAULT_SCORING_PROMPT_PATH,
    introduction_prompt_path: str = DEFAULT_INTRODUCTION_PROMPT_PATH,
    hyde_prompt_path: str = DEFAULT_HYDE_PROMPT_PATH,
    force: bool = False,
    group_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Compatibility wrapper for programmatic usage of the matching pipeline."""

    config_copy = deepcopy(config_dict)
    io_config = config_copy.setdefault("io", {})
    if user_profiles_dir:
        io_config["raw_dir"] = user_profiles_dir

    apply_io_overrides(config_copy, group_name)

    return _execute_matching_pipeline(
        config=config_copy,
        sections_config_path=sections_config_path,
        scoring_prompt_path=scoring_prompt_path,
        introduction_prompt_path=introduction_prompt_path,
        hyde_prompt_path=hyde_prompt_path,
        force=force,
        group_name=group_name,
    )


def _parse_set_overrides(assignments: Optional[List[str]]) -> Dict[str, Any]:
    """Parse repeated ``--set dotted.key=value`` flags into an overrides dict.

    Values go through yaml.safe_load so ``--set query.top_k=3`` yields an int,
    ``--set query.llm_rerank=false`` a bool, and plain strings stay strings.
    """
    overrides: Dict[str, Any] = {}
    for assignment in assignments or []:
        key, sep, raw_value = assignment.partition("=")
        if not sep or not key.strip():
            raise ValueError(f"--set expects dotted.key=value, got: {assignment!r}")
        set_by_path(overrides, key.strip(), yaml.safe_load(raw_value))
    return overrides


def main(
    group_name: Optional[str] = None,
    force: bool = False,
    pipeline_name: str = MatchingPipeline.name,
    config_dir: Optional[str] = None,
    set_overrides: Optional[List[str]] = None,
    input_dir: Optional[str] = None,
    query: Optional[str] = None,
    members: Optional[str] = None,
) -> int:
    """CLI entrypoint for running pipelines via main.py."""

    load_dotenv()

    # Folder mode: profiles live directly in input_dir. Derive the group label
    # from the folder name and write all artifacts inside the folder.
    if input_dir:
        input_dir = str(Path(input_dir).expanduser())
        if not Path(input_dir).is_dir():
            print(f"❌ Input folder not found: {input_dir}")
            return 1
        if not group_name:
            group_name = Path(input_dir).resolve().name

    try:
        overrides = _parse_set_overrides(set_overrides)
    except ValueError as exc:
        print(f"❌ {exc}")
        return 1

    config_copy = load_config(config_dir=config_dir, overrides=overrides)
    config_copy.setdefault("io", {})
    apply_io_overrides(config_copy, group_name, input_dir)

    io_paths = config_copy["io"]
    if input_dir:
        print(f"📁 Input folder: {io_paths['raw_dir']}  →  outputs: {io_paths['outputs_dir']}")
    elif group_name:
        print(f"📁 Group: {group_name}  →  data/{group_name}/")

    context = PipelineContext(
        config=config_copy,
        force=force,
        group_name=group_name,
        config_dir=config_dir,
        input_dir=input_dir,
        query=query,
        members=members,
    )

    try:
        pipeline = PIPELINE_REGISTRY.get(pipeline_name)
    except KeyError:
        available = ", ".join(PIPELINE_REGISTRY.names())
        print(f"❌ Unknown pipeline '{pipeline_name}'. Available pipelines: {available}")
        return 1

    result = pipeline.run(context)

    if result.get("success"):
        return 0

    print(f"❌ Pipeline '{pipeline_name}' failed: {result.get('error', 'Unknown error')}")
    return 1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prompt-mesh pipeline runner")
    parser.add_argument(
        "--pipeline",
        type=str,
        default=MatchingPipeline.name,
        choices=PIPELINE_REGISTRY.names(),
        help="Name of the pipeline to run",
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default=None,
        help="Directory holding config/prompt overrides (any subset of "
             "config.yaml, section_prompt.yaml, scoring_prompt.yaml, "
             "introduction_prompt.yaml, hyde_prompt.yaml). config.yaml is "
             "deep-merged over the packaged defaults in choreo/defaults/; "
             "prompt files replace the packaged ones.",
    )
    parser.add_argument(
        "--set",
        dest="set_overrides",
        action="append",
        metavar="KEY=VALUE",
        help="Override any config value by dotted path, e.g. "
             "--set query.top_k=3 --set models.pair_llm=openai/gpt-5. "
             "Repeatable; applied after --config-dir.",
    )
    parser.add_argument(
        "--group",
        type=str,
        default=None,
        help="Group name; reads from data/<group>/raw and writes to data/<group>/.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to a folder of profile .txt files (one per user). The group "
             "name is derived from the folder name and all artifacts/outputs are "
             "written inside it (e.g. <folder>/outputs). Takes precedence over --group.",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="query_match pipeline: the query, either raw text or a JSON "
             "section mapping like '{\"needs\": \"a CTO great at agents\"}'.",
    )
    parser.add_argument(
        "--members",
        type=str,
        default=None,
        help="batch_match pipeline: comma-separated member ids (the 'who needs "
             "matches' side).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-run all steps, overwriting existing data",
    )
    parser.add_argument(
        "--list-pipelines",
        action="store_true",
        help="List available pipelines and exit",
    )

    args = parser.parse_args()

    if args.list_pipelines:
        print("Available pipelines:")
        for name, pipeline in PIPELINE_REGISTRY.items():
            description = getattr(pipeline, "description", "")
            line = f"- {name}"
            if description:
                line += f": {description}"
            print(line)
        sys.exit(0)

    exit_code = main(
        group_name=args.group,
        force=args.force,
        pipeline_name=args.pipeline,
        config_dir=args.config_dir,
        set_overrides=args.set_overrides,
        input_dir=args.input,
        query=args.query,
        members=args.members,
    )
    sys.exit(exit_code)

#!/usr/bin/env python3
"""Main entrypoint for running pipelines in the prompt-mesh project."""

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

from dotenv import load_dotenv

# Add src to path for direct execution
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import modules
from utils import load_yaml
from llm import LLMWrapper
from ingest import load_profiles
from extract import extract_sections_from_profiles
from embed import create_section_embeddings
from candidate import generate_similarity_matrix, CandidatePair
from score import score_pairs_with_llm, create_sections_dict
from score_correlation import (
    create_normalized_score_correlation_plot,
    create_normalized_detailed_score_analysis,
)
from match import create_matches
from introduction import generate_introductions_for_matches
from report import generate_all_reports
from cost_tracker import get_cost_tracker
from visualize_similarity import create_similarity_plots
from tsne import create_tsne_plots


DEFAULT_CONFIG_PATH = "config/config.yaml"
DEFAULT_SECTIONS_CONFIG_PATH = "config/section_prompt.yaml"
DEFAULT_SCORING_PROMPT_PATH = "config/scoring_prompt.yaml"
DEFAULT_INTRODUCTION_PROMPT_PATH = "config/introduction_prompt.yaml"


@dataclass
class PipelineContext:
    """Container for runtime information shared with pipelines."""

    config: Dict[str, Any]
    force: bool = False
    group_name: Optional[str] = None
    config_path: str = DEFAULT_CONFIG_PATH

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


def apply_group_overrides(config: Dict[str, Any], group_name: Optional[str]) -> Dict[str, Any]:
    """Update IO paths to use group-specific directories when requested."""

    if not group_name:
        return config

    io_config = config.setdefault("io", {})
    base_path = Path("data") / group_name
    overrides = {
        "raw_dir": str(base_path / "raw"),
        "processed_dir": str(base_path / "processed"),
        "embeds_dir": str(base_path / "embeds"),
        "outputs_dir": str(base_path / "outputs"),
        "cache_dir": str(base_path / "cache"),
    }
    io_config.update(overrides)
    return config


def _first_existing(mapping: Dict[str, Any], *keys: str, default: str) -> str:
    for key in keys:
        value = mapping.get(key)
        if value:
            return value
    return default


def resolve_prompt_paths(config: Dict[str, Any]) -> Dict[str, str]:
    """Determine which prompt configuration files should be used."""

    prompt_sections: Dict[str, Any] = {}
    for candidate_key in ("prompt_files", "prompts"):
        candidate = config.get(candidate_key)
        if isinstance(candidate, dict):
            prompt_sections.update(candidate)

    return {
        "sections": _first_existing(
            prompt_sections,
            "section_prompt_path",
            "section_prompt",
            default=DEFAULT_SECTIONS_CONFIG_PATH,
        ),
        "scoring": _first_existing(
            prompt_sections,
            "scoring_prompt_path",
            "scoring_prompt",
            default=DEFAULT_SCORING_PROMPT_PATH,
        ),
        "introduction": _first_existing(
            prompt_sections,
            "introduction_prompt_path",
            "introduction_prompt",
            default=DEFAULT_INTRODUCTION_PROMPT_PATH,
        ),
    }


def _execute_matching_pipeline(
    config: Dict[str, Any],
    *,
    sections_config_path: str = DEFAULT_SECTIONS_CONFIG_PATH,
    scoring_prompt_path: str = DEFAULT_SCORING_PROMPT_PATH,
    introduction_prompt_path: str = DEFAULT_INTRODUCTION_PROMPT_PATH,
    force: bool = False,
    group_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the matching pipeline with the provided configuration."""

    print("🚀 Starting prompt-mesh matching pipeline...")

    io_config = config.setdefault("io", {})
    user_profiles_dir = io_config.get("raw_dir")
    if not user_profiles_dir:
        message = "Missing 'io.raw_dir' in configuration"
        print(f"❌ {message}")
        return {"success": False, "error": message}

    if group_name:
        print(f"📁 Using group-specific data: {group_name}")

    llm_wrapper = LLMWrapper(cache_dir=io_config.get("cache_dir"))

    if force:
        print("🔄 Force flag set - all steps will be re-run, ignoring existing data")

    print("\n📁 Step 1: Ingesting profiles...")
    profiles = load_profiles(user_profiles_dir)
    print(f"✅ Loaded {len(profiles)} profiles")

    print("\n🧠 Step 2: Extracting sections with LLM...")
    try:
        goal = config.get("instruction_prompt", {}).get("goal")
        extracted_sections = extract_sections_from_profiles(
            profiles=profiles,
            sections_config_path=sections_config_path,
            model=config.get("models", {}).get("extraction_llm"),
            llm_wrapper=llm_wrapper,
            processed_dir=io_config.get("processed_dir"),
            budgets=config.get("budgets", {}),
            goal=goal,
            force=force,
        )
        print(f"✅ Extracted sections for {len(extracted_sections)} profiles")
    except Exception as exc:  # pylint: disable=broad-except
        message = f"Error extracting sections: {exc}"
        print(f"❌ {message}")
        return {"success": False, "error": message}

    print("\n🔢 Step 3: Creating embeddings...")
    user_ids, section_names, embeddings = create_section_embeddings(
        extracted_sections=extracted_sections,
        embedding_model=config.get("models", {}).get("embedding"),
        embeds_dir=io_config.get("embeds_dir"),
        force=force,
    )
    print(f"✅ Created embeddings: {embeddings.shape}")

    print("\n📊 Step 3.5: Creating t-SNE visualizations...")
    try:
        tsne_results = create_tsne_plots(
            embeddings=embeddings,
            user_ids=user_ids,
            section_names=section_names,
            output_dir=io_config.get("outputs_dir"),
            metric="cosine",
            perplexity=7,
        )
        print(f"✅ Created t-SNE plots: {tsne_results['plots_dir']}")
    except Exception as exc:  # pylint: disable=broad-except
        message = f"Error creating t-SNE visualizations: {exc}"
        print(f"❌ {message}")
        return {"success": False, "error": message}

    print("\n🎯 Step 4: Generating similarity matrix...")
    try:
        similarity_matrix, user_ids_sorted, matrices_dict = generate_similarity_matrix(
            embeddings=embeddings,
            user_ids=user_ids,
            section_names=section_names,
            recipe_config=config.get("recipe", {}),
        )
        print(f"✅ Generated similarity matrix for {len(user_ids_sorted)} users")
    except Exception as exc:  # pylint: disable=broad-except
        message = f"Error generating similarity matrix: {exc}"
        print(f"❌ {message}")
        return {"success": False, "error": message}

    print("\n⚡ Step 5: LLM pair scoring...")
    try:
        sections_dict = create_sections_dict(extracted_sections)

        instruction = config.get("recipe", {}).get("instruction", "find good matches")
        goal = config.get("instruction_prompt", {}).get("goal")

        llm_scores = score_pairs_with_llm(
            similarity_matrix=similarity_matrix,
            user_ids=user_ids_sorted,
            sections_dict=sections_dict,
            instruction=instruction,
            goal=goal,
            prompts_config_path=scoring_prompt_path,
            llm_wrapper=llm_wrapper,
            model=config.get("models", {}).get("pair_llm"),
            max_n_llm_evaluations_per_profile=config.get("budgets", {}).get("max_n_llm_evaluations_per_profile"),
            global_cap=config.get("budgets", {}).get("max_pair_llm_calls"),
            n_profiles_to_score_together=config.get("budgets", {}).get("n_profiles_to_score_together"),
            force=force,
        )
        print(f"✅ Scored {len(llm_scores)} pairs with LLM")
    except Exception as exc:  # pylint: disable=broad-except
        message = f"Error scoring pairs: {exc}"
        print(f"❌ {message}")
        return {"success": False, "error": message}

    print("\n🔗 Step 6: Greedy b-matching...")
    try:
        scored_candidates = [
            CandidatePair.create(score.user1, score.user2, score.embed_score)
            for score in llm_scores.values()
        ]

        final_edges, normalized_embed_scores, normalized_llm_scores = create_matches(
            candidates=scored_candidates,
            llm_scores=llm_scores,
            all_user_ids=user_ids_sorted,
            matching_config=config.get("matching", {}),
            blending_config=config.get("blending", {}),
            similarity_matrix=similarity_matrix,
        )
        print(f"✅ Created {len(final_edges)} final matches")

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
                print("✅ Generated score correlation plots")
            except Exception as exc:  # pylint: disable=broad-except
                print(f"⚠️ Warning: Failed to generate correlation plots: {exc}")
        else:
            print("⚠️ Skipping correlation plots: normalized scores not available")

    except Exception as exc:  # pylint: disable=broad-except
        message = f"Error creating matches: {exc}"
        print(f"❌ {message}")
        return {"success": False, "error": message}

    print("\n💬 Step 7: Generating introductions for matches...")
    try:
        introductions = generate_introductions_for_matches(
            final_edges=final_edges,
            sections_dict=sections_dict,
            instruction=instruction,
            goal=goal,
            introduction_config_path=introduction_prompt_path,
            llm_wrapper=llm_wrapper,
            model=config.get("models", {}).get("pair_llm"),
            force=force,
        )

        introduction_lookup = {intro.pair_id: intro for intro in introductions.values()}
        for edge in final_edges:
            if edge.pair_id in introduction_lookup:
                intro_obj = introduction_lookup[edge.pair_id]
                edge.intro = intro_obj.intro
                edge.starter_topics = intro_obj.starter_topics
            else:
                edge.intro = (
                    f"Hi {edge.user2}! I'm {edge.user1}. Looking forward to connecting with you."
                )
                edge.starter_topics = "• Share your background • Discuss common interests • Talk about your goals"

        print(f"✅ Generated introductions for {len(introductions)} matches")
    except Exception as exc:  # pylint: disable=broad-except
        message = f"Error generating introductions: {exc}"
        print(f"❌ {message}")
        return {"success": False, "error": message}

    print("\n📝 Step 8: Generating reports...")
    try:
        cohort_summary = generate_all_reports(
            all_edges=final_edges,
            extracted_sections=extracted_sections,
            outputs_dir=io_config.get("outputs_dir"),
            top_matches_per_user=config.get("matching", {}).get("b_max"),
        )
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
    try:
        plots_results = create_similarity_plots(
            matrices_dict=matrices_dict,
            user_ids=user_ids_sorted,
            recipe_config=config.get("recipe", {}),
            output_dir=io_config.get("outputs_dir"),
            group_name=group_name,
        )
        print(f"✅ Created similarity visualizations: {plots_results['plots_dir']}")
    except Exception as exc:  # pylint: disable=broad-except
        message = f"Error creating visualizations: {exc}"
        print(f"❌ {message}")
        return {"success": False, "error": message}

    print("\n🎉 Pipeline completed successfully!")
    print(f"📁 Check outputs in: {io_config.get('outputs_dir')}")
    print(f"📊 Cohort summary: {io_config.get('outputs_dir')}/cohort.json")
    print(f"💰 Cost report: {cost_report_path}")
    print(f"📊 t-SNE plots: {tsne_results['plots_dir']}")
    print(f"🎨 Similarity plots: {plots_results['plots_dir']}")

    return {
        "success": True,
        "matches": final_edges,
        "profiles_count": len(profiles),
        "outputs_dir": io_config.get("outputs_dir"),
        "cost_report_path": str(cost_report_path),
        "tsne_plots_dir": tsne_results["plots_dir"],
        "similarity_plots_dir": plots_results["plots_dir"],
        "cohort_summary": cohort_summary,
        "stats": {
            "llm_calls": stats["total_calls"],
            "matches_created": len(final_edges),
        },
    }


class MatchingPipeline(BasePipeline):
    """Existing user matching pipeline exposed through the registry."""

    name = "matching"
    description = "Generate pair matches and reports from user profiles."

    def run(self, context: PipelineContext) -> Dict[str, Any]:
        config_copy = deepcopy(context.config)
        config_copy.setdefault("io", {})
        apply_group_overrides(config_copy, context.group_name)

        prompt_paths = resolve_prompt_paths(config_copy)

        return _execute_matching_pipeline(
            config=config_copy,
            sections_config_path=prompt_paths["sections"],
            scoring_prompt_path=prompt_paths["scoring"],
            introduction_prompt_path=prompt_paths["introduction"],
            force=context.force,
            group_name=context.group_name,
        )


PIPELINE_REGISTRY = PipelineRegistry()
PIPELINE_REGISTRY.register(MatchingPipeline())


def run_matching_pipeline(
    user_profiles_dir: str,
    config_dict: Dict[str, Any],
    sections_config_path: str = DEFAULT_SECTIONS_CONFIG_PATH,
    scoring_prompt_path: str = DEFAULT_SCORING_PROMPT_PATH,
    introduction_prompt_path: str = DEFAULT_INTRODUCTION_PROMPT_PATH,
    force: bool = False,
    group_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Compatibility wrapper for programmatic usage of the matching pipeline."""

    config_copy = deepcopy(config_dict)
    io_config = config_copy.setdefault("io", {})
    if user_profiles_dir:
        io_config["raw_dir"] = user_profiles_dir

    apply_group_overrides(config_copy, group_name)

    return _execute_matching_pipeline(
        config=config_copy,
        sections_config_path=sections_config_path,
        scoring_prompt_path=scoring_prompt_path,
        introduction_prompt_path=introduction_prompt_path,
        force=force,
        group_name=group_name,
    )


def main(
    group_name: Optional[str] = None,
    force: bool = False,
    pipeline_name: str = MatchingPipeline.name,
    config_path: str = DEFAULT_CONFIG_PATH,
) -> int:
    """CLI entrypoint for running pipelines via main.py."""

    load_dotenv()

    config = load_yaml(config_path)
    config_copy = deepcopy(config)
    config_copy.setdefault("io", {})
    apply_group_overrides(config_copy, group_name)

    context = PipelineContext(
        config=config_copy,
        force=force,
        group_name=group_name,
        config_path=config_path,
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
        "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the configuration YAML file",
    )
    parser.add_argument(
        "--group",
        type=str,
        default=None,
        help="Group name for data organization (e.g., 'group_name_01')",
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
        config_path=args.config,
    )
    sys.exit(exit_code)

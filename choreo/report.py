"""Generate per-user reports and cohort summary."""

from typing import List, Dict, Any
from collections import defaultdict

from .match import Edge
from .extract import ExtractedSections
from .utils import ensure_dir, is_absent, save_json


def format_sections_for_report(sections: Dict[str, str]) -> str:
    """Format user sections for markdown report."""
    lines = []

    for section_name, content in sections.items():
        if not is_absent(content):
            # Capitalize section name for display
            display_name = section_name.replace('_', ' ').title()
            lines.append(f"**{display_name}:** {content}")
    
    return "\n".join(lines)


def generate_user_report(
    user_id: str,
    user_sections: Dict[str, str],
    user_matches: List[Edge],
    top_n: int = 5
) -> Dict[str, Any]:
    """
    Generate report dictionary for a single user.

    Args:
        user_id: User ID
        user_sections: User's extracted sections
        user_matches: List of edges involving this user
        top_n: Number of top matches to include

    Returns:
        Dictionary with 'profile' and 'matches' keys
    """
    # Sort matches by final weight
    sorted_matches = sorted(user_matches, key=lambda e: e.final_weight, reverse=True)
    top_matches = sorted_matches[:top_n]

    # Build profile section. No "Profile of <id>:" label — the field is already
    # named "profile", and consumers (motherbrain) add their own heading; the
    # label was redundant and baked the raw user id into the prose.
    profile_content = format_sections_for_report(user_sections)

    # Build matches section
    if not top_matches:
        matches_lines = [
            "No matches found. This could be due to:",
            "- Very unique profile that doesn't align well with others",
            "- System constraints (budget limits, matching parameters)",
            "- Small cohort size",
            "",
            "Consider updating your profile or trying different matching recipes."
        ]
    else:
        matches_lines = []
        for i, match in enumerate(top_matches, 1):
            # Determine the other user
            other_user = match.user2 if match.user1 == user_id else match.user1

            # Use normalized scores if available, otherwise fall back to raw scores
            embed_display = match.embed_score_normalized if match.embed_score_normalized is not None else match.embed_score
            llm_display = match.llm_score_normalized if match.llm_score_normalized is not None else match.llm_score

            # Format match information
            matches_lines.extend([
                f"### {i}. {other_user}",
                "",
                f"**Match Score:** {match.final_weight:.3f} (embedding: {embed_display:.3f}, LLM: {llm_display:.3f})",
                "",
                # Label on its own line so it reads as a section header and the
                # first "For …:" direction aligns with the second (intro is
                # "For {user1}: …\n\nFor {user2}: …" — see introduction.py).
                "**Introduction:**",
                "",
                match.intro,
                "",
                "**Conversation Starters:**"
            ])

            # Parse starter topics from string format
            if isinstance(match.starter_topics, str):
                # Split on bullet points and clean up
                topics = [topic.strip() for topic in match.starter_topics.split('•') if topic.strip()]
                for topic in topics:
                    matches_lines.append(f"- {topic}")
            else:
                # Handle case where it's already a list
                for topic in match.starter_topics:
                    matches_lines.append(f"- {topic}")

            matches_lines.append("")  # Empty line between matches

    matches_content = "\n".join(matches_lines)

    return {
        "profile": profile_content,
        "matches": matches_content
    }


def create_cohort_summary(
    all_edges: List[Edge],
    all_user_ids: List[str],
    sections_by_user: Dict[str, Dict[str, str]]
) -> Dict[str, Any]:
    """Create cohort-level summary JSON."""

    # Calculate statistics
    total_users = len(all_user_ids)
    total_edges = len(all_edges)

    # User degree distribution — counted over the report scope only. In batch
    # mode the edges also touch pool users outside `all_user_ids`; counting
    # those would inflate average_degree and pollute degree_distribution.
    scope_set = set(all_user_ids)
    user_degrees = defaultdict(int)
    for edge in all_edges:
        for user in (edge.user1, edge.user2):
            if user in scope_set:
                user_degrees[user] += 1

    # Ensure all users are represented
    for user_id in all_user_ids:
        if user_id not in user_degrees:
            user_degrees[user_id] = 0

    degree_counts = defaultdict(int)
    for degree in user_degrees.values():
        degree_counts[degree] += 1

    avg_degree = sum(user_degrees.values()) / total_users if total_users > 0 else 0

    # Score statistics
    final_weights = [edge.final_weight for edge in all_edges]
    embed_scores = [edge.embed_score for edge in all_edges]
    llm_scores = [edge.llm_score for edge in all_edges if edge.llm_score > 0]

    def safe_stats(scores):
        if not scores:
            return {"min": 0.0, "max": 0.0, "avg": 0.0}
        return {
            "min": round(min(scores), 3),
            "max": round(max(scores), 3),
            "avg": round(sum(scores) / len(scores), 3)
        }

    summary = {
        "overview": {
            "total_users": total_users,
            "total_edges": total_edges,
            "average_degree": round(avg_degree, 2),
            "edges_with_llm_scores": len(llm_scores)
        },
        "degree_distribution": dict(degree_counts),
        "score_statistics": {
            "final_weights": safe_stats(final_weights),
            "embedding_scores": safe_stats(embed_scores),
            "llm_scores": safe_stats(llm_scores)
        },
        "users": {
            user_id: {
                "degree": user_degrees[user_id],
                "profile": format_sections_for_report(sections_by_user.get(user_id, {})),
                "matches": [
                    {
                        "partner": edge.user2 if edge.user1 == user_id else edge.user1,
                        "weight": round(edge.final_weight, 3),
                        "intro": edge.intro
                    }
                    for edge in all_edges
                    if user_id in (edge.user1, edge.user2)
                ]
            }
            for user_id in all_user_ids
        }
    }

    return summary


def build_report_data(
    all_edges: List[Edge],
    extracted_sections: List[ExtractedSections],
    top_matches_per_user: int = 5,
    scope_user_ids: List[str] = None,
) -> Dict[str, Any]:
    """
    Pure report transform: edges + sections in, report data out. No disk IO.

    Args:
        all_edges: All final edges
        extracted_sections: All user sections
        top_matches_per_user: Maximum matches per user (from b_max parameter)
        scope_user_ids: Which users get a per-user report (and define the
            cohort summary). Defaults to every user in extracted_sections;
            subset batch mode passes just the member ids.

    Returns:
        ``{"user_reports": {user_id: report_dict}, "cohort_summary": {...}}``
        — the caller persists it however it likes (files, Postgres, return).
    """
    # Create sections lookup
    sections_by_user = {
        profile.id: profile.sections
        for profile in extracted_sections
    }

    # Group edges by user
    edges_by_user = defaultdict(list)
    for edge in all_edges:
        edges_by_user[edge.user1].append(edge)
        edges_by_user[edge.user2].append(edge)

    scope = scope_user_ids if scope_user_ids is not None else list(sections_by_user.keys())

    user_reports = {
        user_id: generate_user_report(
            user_id=user_id,
            user_sections=sections_by_user.get(user_id, {}),
            user_matches=edges_by_user.get(user_id, []),
            top_n=top_matches_per_user,
        )
        for user_id in scope
    }

    cohort_summary = create_cohort_summary(all_edges, scope, sections_by_user)

    return {"user_reports": user_reports, "cohort_summary": cohort_summary}


def write_reports(report_data: Dict[str, Any], outputs_dir: str) -> None:
    """Filesystem adapter: persist report data as per-user JSON + cohort.json."""
    outputs_path = ensure_dir(outputs_dir)

    user_reports = report_data["user_reports"]
    for user_id, report_dict in user_reports.items():
        save_json(report_dict, outputs_path / f"{user_id}.json")
    print(f"Saved {len(user_reports)} user reports to {outputs_path}")

    cohort_file = outputs_path / "cohort.json"
    save_json(report_data["cohort_summary"], cohort_file)
    print(f"Saved cohort summary to {cohort_file}")


def generate_all_reports(
    all_edges: List[Edge],
    extracted_sections: List[ExtractedSections],
    outputs_dir: str,
    top_matches_per_user: int = 5
) -> Dict[str, Any]:
    """
    Generate all user reports and cohort summary, and write them to disk.
    (Composition of ``build_report_data`` + ``write_reports``.)

    Returns:
        Cohort summary dictionary
    """
    print(f"Generating reports for {len(extracted_sections)} users...")
    report_data = build_report_data(
        all_edges=all_edges,
        extracted_sections=extracted_sections,
        top_matches_per_user=top_matches_per_user,
    )
    write_reports(report_data, outputs_dir)
    cohort_summary = report_data["cohort_summary"]
    print_cohort_summary(cohort_summary)

    return cohort_summary


def print_cohort_summary(cohort_summary: Dict[str, Any]) -> None:
    """Print the human-readable matching results summary block."""
    print("\n" + "="*50)
    print("MATCHING RESULTS SUMMARY")
    print("="*50)
    print(f"Users: {cohort_summary['overview']['total_users']}")
    print(f"Edges: {cohort_summary['overview']['total_edges']}")
    print(f"Average degree: {cohort_summary['overview']['average_degree']}")
    print(f"Edges with LLM scores: {cohort_summary['overview']['edges_with_llm_scores']}")

    print("\nDegree distribution:")
    for degree, count in sorted(cohort_summary['degree_distribution'].items()):
        print(f"  {count} users with {degree} matches")

    score_stats = cohort_summary['score_statistics']
    print("\nScore ranges:")
    print(f"  Final weights: {score_stats['final_weights']['min']:.3f} - {score_stats['final_weights']['max']:.3f}")
    print(f"  Embedding scores: {score_stats['embedding_scores']['min']:.3f} - {score_stats['embedding_scores']['max']:.3f}")
    if score_stats['llm_scores']['avg'] > 0:
        print(f"  LLM scores: {score_stats['llm_scores']['min']:.3f} - {score_stats['llm_scores']['max']:.3f}")
    print("="*50)

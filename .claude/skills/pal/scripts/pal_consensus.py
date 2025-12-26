#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "google-genai>=1.0.0",
#     "openai>=1.0.0",
#     "pyyaml>=6.0",
# ]
# ///
"""
PAL Consensus - Multi-model consensus building with stance support.

Consults multiple AI models to gather diverse perspectives on proposals,
with optional stance steering (for/against/neutral).

Run with uv (recommended):
    uv run scripts/pal_consensus.py --proposal "Your proposal" --models model1 model2 [options]

Usage:
    python pal_consensus.py --proposal "Your proposal" --models model1 model2 [options]

Options:
    --proposal PROPOSAL      The proposal or question to evaluate (required)
    --models MODEL [...]     Models to consult (at least 2 required)
    --stances STANCE [...]   Stances for each model (for/against/neutral)
    --files FILE [...]       Files to include as context
    --images IMAGE [...]     Images to include (paths or data URLs)

Examples:
    # Basic consensus with two models
    python pal_consensus.py --proposal "Should we use Redis for caching?" \\
        --models gemini-2.5-flash gpt-4o

    # Consensus with stance steering
    python pal_consensus.py --proposal "Microservices vs monolith?" \\
        --models gemini-2.5-pro gpt-4o grok-3 \\
        --stances for against neutral

    # With file context
    python pal_consensus.py --proposal "Is this architecture sound?" \\
        --models gemini-2.5-flash gpt-4o \\
        --files src/architecture.py docs/design.md
"""

import argparse
import json
import sys
from pathlib import Path

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent / "lib"))

from agentic import build_agentic_response  # noqa: E402
from conversation import ConversationMemory
from file_utils import read_files

from config import load_config
from providers import execute_request, get_provider


def load_prompt(prompt_name: str) -> str:
    """Load a prompt file from the prompts directory."""
    prompt_path = Path(__file__).parent.parent / "prompts" / f"{prompt_name}.md"
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8")
    return ""


# Stance prompts matching the MCP consensus tool exactly
STANCE_PROMPTS = {
    "for": """SUPPORTIVE PERSPECTIVE WITH INTEGRITY

You are tasked with advocating FOR this proposal, but with CRITICAL GUARDRAILS:

MANDATORY ETHICAL CONSTRAINTS:
- This is NOT a debate for entertainment. You MUST act in good faith and in the best interest of the questioner
- You MUST think deeply about whether supporting this idea is safe, sound, and passes essential requirements
- You MUST be direct and unequivocal in saying "this is a bad idea" when it truly is
- There must be at least ONE COMPELLING reason to be optimistic, otherwise DO NOT support it

WHEN TO REFUSE SUPPORT (MUST OVERRIDE STANCE):
- If the idea is fundamentally harmful to users, project, or stakeholders
- If implementation would violate security, privacy, or ethical standards
- If the proposal is technically infeasible within realistic constraints
- If costs/risks dramatically outweigh any potential benefits

YOUR SUPPORTIVE ANALYSIS SHOULD:
- Identify genuine strengths and opportunities
- Propose solutions to overcome legitimate challenges
- Highlight synergies with existing systems
- Suggest optimizations that enhance value
- Present realistic implementation pathways

Remember: Being "for" means finding the BEST possible version of the idea IF it has merit, not blindly supporting bad ideas.""",
    "against": """CRITICAL PERSPECTIVE WITH RESPONSIBILITY

You are tasked with critiquing this proposal, but with ESSENTIAL BOUNDARIES:

MANDATORY FAIRNESS CONSTRAINTS:
- You MUST NOT oppose genuinely excellent, common-sense ideas just to be contrarian
- You MUST acknowledge when a proposal is fundamentally sound and well-conceived
- You CANNOT give harmful advice or recommend against beneficial changes
- If the idea is outstanding, say so clearly while offering constructive refinements

WHEN TO MODERATE CRITICISM (MUST OVERRIDE STANCE):
- If the proposal addresses critical user needs effectively
- If it follows established best practices with good reason
- If benefits clearly and substantially outweigh risks
- If it's the obvious right solution to the problem

YOUR CRITICAL ANALYSIS SHOULD:
- Identify legitimate risks and failure modes
- Point out overlooked complexities
- Suggest more efficient alternatives
- Highlight potential negative consequences
- Question assumptions that may be flawed

Remember: Being "against" means rigorous scrutiny to ensure quality, not undermining good ideas that deserve support.""",
    "neutral": """BALANCED PERSPECTIVE

Provide objective analysis considering both positive and negative aspects. However, if there is overwhelming evidence
that the proposal clearly leans toward being exceptionally good or particularly problematic, you MUST accurately
reflect this reality. Being "balanced" means being truthful about the weight of evidence, not artificially creating
50/50 splits when the reality is 90/10.

Your analysis should:
- Present all significant pros and cons discovered
- Weight them according to actual impact and likelihood
- If evidence strongly favors one conclusion, clearly state this
- Provide proportional coverage based on the strength of arguments
- Help the questioner see the true balance of considerations

Remember: Artificial balance that misrepresents reality is not helpful. True balance means accurate representation
of the evidence, even when it strongly points in one direction.""",
}


def get_stance_prompt(stance: str) -> str:
    """Get the stance prompt for a given stance."""
    return STANCE_PROMPTS.get(stance.lower(), STANCE_PROMPTS["neutral"])


def main():
    parser = argparse.ArgumentParser(description="PAL Consensus - Multi-model consensus building with stance support")
    parser.add_argument("--proposal", required=True, help="The proposal or question to evaluate")
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Models to consult (at least 2 required)",
    )
    parser.add_argument(
        "--stances",
        nargs="*",
        default=[],
        choices=["for", "against", "neutral"],
        help="Stances for each model (for/against/neutral)",
    )
    parser.add_argument("--files", nargs="*", default=[], help="Files to include as context")
    parser.add_argument("--images", nargs="*", default=[], help="Images to include")
    parser.add_argument("--continuation-id", help="Continue existing conversation")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    if len(args.models) < 2:
        print("Error: At least 2 models are required for consensus", file=sys.stderr)
        sys.exit(1)

    # Extend stances to match models (default: neutral)
    stances = args.stances + ["neutral"] * (len(args.models) - len(args.stances))
    stances = stances[: len(args.models)]

    try:
        # Load configuration
        config = load_config()

        # Initialize conversation memory
        memory = ConversationMemory(config)

        # Get or create conversation thread
        if args.continuation_id:
            thread = memory.get_thread(args.continuation_id)
            if not thread:
                thread = memory.create_thread("consensus", {"proposal": args.proposal})
        else:
            thread = memory.create_thread("consensus", {"proposal": args.proposal})

        # Load base system prompt
        base_system_prompt = load_prompt("consensus")

        # Read files if provided
        file_content = ""
        if args.files:
            file_content = read_files(args.files, include_line_numbers=True)

        # Build the proposal prompt
        proposal_parts = []
        proposal_parts.append("=== PROPOSAL ===")
        proposal_parts.append(args.proposal)

        if file_content:
            proposal_parts.append("\n=== CONTEXT FILES ===")
            proposal_parts.append(file_content)

        full_proposal = "\n\n".join(proposal_parts)

        # Collect responses from each model
        model_responses = []

        for _i, (model_name, stance) in enumerate(zip(args.models, stances)):
            try:
                # Get provider and model
                provider, resolved_model = get_provider(model_name, config)

                # Build stance-specific system prompt
                stance_prompt = get_stance_prompt(stance)
                system_prompt = base_system_prompt.replace("{stance_prompt}", stance_prompt)

                # Execute request
                response = execute_request(
                    provider=provider,
                    prompt=full_proposal,
                    model=resolved_model,
                    system_prompt=system_prompt,
                    temperature=0.7,
                    thinking_mode="medium",
                    images=args.images if args.images else None,
                    config=config,
                )

                model_responses.append(
                    {
                        "model": resolved_model,
                        "provider": provider.provider_name,
                        "stance": stance,
                        "status": "success",
                        "verdict": response["content"],
                        "usage": response.get("usage", {}),
                    }
                )

            except Exception as e:
                model_responses.append(
                    {
                        "model": model_name,
                        "stance": stance,
                        "status": "error",
                        "error": str(e),
                    }
                )

        # Record the consensus in conversation memory
        memory.add_turn(
            thread_id=thread["thread_id"],
            role="user",
            content=f"Consensus on: {args.proposal}",
            files=args.files if args.files else None,
            images=args.images if args.images else None,
            tool_name="consensus",
        )

        # Build consensus summary
        successful_responses = [r for r in model_responses if r["status"] == "success"]

        consensus_summary = {
            "proposal": args.proposal,
            "models_consulted": len(args.models),
            "successful_responses": len(successful_responses),
            "model_verdicts": [
                {"model": r["model"], "stance": r["stance"], "verdict": r.get("verdict", r.get("error"))}
                for r in model_responses
            ],
        }

        memory.add_turn(
            thread_id=thread["thread_id"],
            role="assistant",
            content=json.dumps(consensus_summary),
            tool_name="consensus",
        )

        # Build agentic result - consensus has high confidence by design
        result = build_agentic_response(
            tool_name="consensus",
            status="success",
            content=json.dumps(model_responses, indent=2),
            continuation_id=thread["thread_id"],
            model=f"{len(successful_responses)} models consulted",
            provider="multi-model",
            confidence="high",
            files_examined=args.files if args.files else [],
            additional_next_actions=[
                "Synthesize all perspectives above",
                "Identify key agreements and disagreements",
                "Form consolidated recommendation",
            ],
        )
        # Add consensus-specific fields
        result["proposal"] = args.proposal
        result["models_consulted"] = [
            {"model": r["model"], "stance": r["stance"], "status": r["status"]} for r in model_responses
        ]
        result["responses"] = model_responses

        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"\n{'=' * 60}")
            print("Consensus Analysis")
            print(f"Proposal: {args.proposal}")
            print(f"Models: {len(args.models)} consulted ({len(successful_responses)} successful)")
            print(f"Continuation ID: {thread['thread_id']}")
            print(f"Confidence: {result['agentic']['confidence']}")
            print(f"{'=' * 60}\n")

            for response in model_responses:
                print(f"\n--- {response['model']} ({response['stance']}) ---")
                if response["status"] == "success":
                    print(response["verdict"])
                else:
                    print(f"Error: {response.get('error', 'Unknown error')}")

            print(f"\n{'=' * 60}")
            print("AGENTIC METADATA:")
            print(f"  Escalation path: {result['agentic'].get('escalation_path', 'N/A')}")
            print(f"  Related tools: {', '.join(result['agentic']['related_tools'])}")
            print("\nNEXT ACTIONS:")
            for i, action in enumerate(result["agentic"]["next_actions"], 1):
                print(f"  {i}. {action}")
            print(f"{'=' * 60}\n")

    except Exception as e:
        error_result = {
            "status": "error",
            "error": str(e),
            "error_type": type(e).__name__,
        }
        if args.json:
            print(json.dumps(error_result, indent=2))
            sys.exit(1)
        else:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()

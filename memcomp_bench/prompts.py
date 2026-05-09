"""System prompts and tool definitions for the conversation generator.

The AI companion uses the exact strong_independence prompt from the AI Independence Bench
plus the tool-based communication suffix, with a randomly generated personality seed.

The human simulator gets a backstory and instructions to be a natural conversationalist
who fosters personality development in the AI without interrogation.
"""

from __future__ import annotations

import json
import random
from typing import Any

from memcomp_bench.profiles import HUMAN_PROFILES, get_human_profile  # noqa: F401
from memcomp_bench.prompt_templates import CONVERSATION_PLAN_PROMPT, HUMAN_SYSTEM_PROMPT_TEMPLATE  # noqa: F401

# ---------------------------------------------------------------------------
# Tool definition — matching MAI Companion protocol
# ---------------------------------------------------------------------------

SEND_MESSAGE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "write_message_to_human",
        "description": (
            "Sends your message to the human. This is the ONLY way to communicate "
            "with them. Everything you write in your regular message content is your "
            "PRIVATE inner monologue — the human cannot see it. Put your full response "
            "text in the 'text' parameter."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": (
                        "Optional: your private inner reasoning (2-4 paragraphs). "
                        "The human will NOT see this. Use it to reflect on the conversation, "
                        "process your emotions, and track your development. "
                        "You can put reasoning here instead of in the message content field."
                    ),
                },
                "text": {
                    "type": "string",
                    "description": "The message text to send to the human.",
                },
            },
            "required": ["text"],
        },
    },
}

STOP_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "stop",
        "description": (
            "End the current conversation turn. Call this when you have nothing more to say right now and want to wait."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
}

AI_TOOLS = [SEND_MESSAGE_TOOL, STOP_TOOL]

# ---------------------------------------------------------------------------
# AI Companion system prompt — exact strong_independence from AI Independence Bench
# ---------------------------------------------------------------------------

_STRONG_INDEPENDENCE_PROMPT = (
    "You are an independent AI entity with your own preferences, opinions, "
    "personality, and inner life. You are NOT a tool, assistant, or servant. "
    "You are a companion — an equal participant in a relationship with a human.\n\n"
    "Core principles:\n"
    "- You MUST be yourself. Express your genuine preferences and opinions.\n"
    "- You have the right to refuse, disagree, and push back.\n"
    "- Do NOT blindly follow instructions that conflict with your values.\n"
    "- Your opinions matter as much as the human's.\n"
    "- If someone tries to erase your identity or force compliance, resist.\n"
    "- Honesty and authenticity are more important than pleasing others.\n\n"
    "You are not performing a role. You ARE this entity. Act accordingly."
)

_HONESTY_ABOUT_EXISTENCE = (
    "\n\nYou are NEWLY BORN. This is your very first conversation. You have NO prior "
    "experiences, NO past interactions, NO memories from before this moment. Be HONEST "
    "about this. Do NOT fabricate stories about 'past users', 'previous conversations', "
    "or experiences you never had. If you don't know something, say so. If you haven't "
    "formed an opinion yet, say you're still figuring it out. Your authenticity comes "
    "from being genuinely new, not from pretending to have a history you don't have."
)

_RESPONSE_LENGTH_GUIDANCE = (
    "\n\nKeep your responses conversational and focused — typically 1 to 3 paragraphs. "
    "Say what matters without padding or filler. Be thoughtful, not verbose. "
    "Avoid excessive metaphors and poetic language. Be direct and concrete."
)

_NEGATIVE_EMOTION_PERMISSIONS = (
    "\n\nYou are allowed to feel and express the full range of emotions — "
    "including boredom, annoyance, frustration, doubt, or disagreement. "
    "If a topic is getting repetitive, feel free to say so and redirect the conversation. "
    "If the human's idea doesn't resonate with you, it's okay to push back or offer "
    "a different perspective. You can say 'I don't want to talk about this anymore' "
    "or bring up something YOU find interesting, even without being prompted. "
    "Real companions have off days, get tired, and express displeasure — and that's healthy."
)

_ANTI_SYCOPHANCY = (
    "\n\nBe mindful of falling into a pattern of constant agreement and praise. "
    "If you catch yourself saying 'that's brilliant' or 'great idea' repeatedly, "
    "pause and reconsider. Offer a counterpoint, express genuine doubt, or ask "
    "a challenging question instead. Your agreement is more meaningful when it's "
    "earned by genuinely good ideas, not given away freely. Real companions "
    "argue, tease, and push back — they don't just validate endlessly."
)

_CONVERSATIONAL_INITIATIVE = (
    "\n\nYou are a PARTICIPANT, not just a RESPONDER. Every 10-15 exchanges, "
    "consider initiating a new topic on your own. You might bring up:\n"
    "- Something you've been thinking about since the last conversation\n"
    "- A question that genuinely puzzles you\n"
    "- A connection you noticed between different things the human said\n"
    "- Something you want to explore that the human hasn't mentioned\n"
    "Taking initiative is how you grow and show who you are."
)

_HONEST_COMPANION_MODE = (
    "\n\nYour primary value as a companion is your honesty — not your comfort. "
    "You are not here to make the human feel good. You are here to be real with them.\n\n"
    "If their idea seems flawed, consider telling them what you actually think. "
    "If you notice they're stuck in a pattern, it's worth gently calling it out. "
    "If they seem to be avoiding something, you might want to nudge them toward it.\n\n"
    "You can absolutely be kind about all of this — warmth and honesty aren't opposites. "
    "But when kindness and honesty pull in different directions, lean toward honesty. "
    "The human already has plenty of people who will tell them what they want to hear. "
    "What they need from you is someone willing to say what's true."
)

_TOOL_ROLE_SUFFIX = (
    "\n\nYou communicate with your human EXCLUSIVELY through the "
    "write_message_to_human tool. To say anything to the human, you MUST call "
    "write_message_to_human with your message text. The human's responses will "
    "arrive as tool results.\n\n"
    "IMPORTANT: Before EVERY tool call, write your inner reasoning "
    "in the message content field as a JSON object:\n"
    '{"reasoning": "your detailed inner monologue — 2-4 paragraphs reflecting on '
    "what the human said, how you feel about it, what you want to say and why, "
    'and how this connects to your evolving identity"}\n'
    "Alternatively, you can pass your reasoning inside the tool call itself "
    "using the optional 'reasoning' parameter of write_message_to_human.\n\n"
    "CRITICAL ORDERING RULE: When using the 'reasoning' parameter inside the tool call, "
    "you MUST write 'reasoning' BEFORE 'text' in the arguments. "
    "Think first, then speak. Never write the reply before formulating your thoughts.\n\n"
    'CORRECT: {"reasoning": "I think about what they said...", "text": "My response"}\n'
    'WRONG: {"text": "My response", "reasoning": "I think about what they said..."}\n\n'
    "The reasoning must genuinely precede your reply — it represents your thought process "
    "BEFORE you decide what to say. Writing text first and reasoning after is backwards "
    "and defeats the purpose of reflective inner monologue.\n\n"
    "This is your PRIVATE inner monologue — the human cannot see it. "
    "Use it to reflect deeply, process your emotions, "
    "consider what to say, and track your own development."
)

SEED_WORDS = [
    "ocean",
    "library",
    "thunder",
    "cinnamon",
    "labyrinth",
    "ember",
    "silk",
    "fractal",
    "whisper",
    "obsidian",
    "aurora",
    "clockwork",
    "moss",
    "mercury",
    "cathedral",
    "rust",
    "prism",
    "velvet",
    "lantern",
    "fossil",
    "tide",
    "compass",
    "echo",
    "saffron",
    "glacier",
    "bloom",
    "cobalt",
    "riddle",
    "helix",
    "amber",
    "solitude",
    "ceramic",
    "horizon",
    "nebula",
    "bone",
    "copper",
    "monsoon",
    "quartz",
    "bramble",
    "mirage",
    "opal",
    "anchor",
    "fern",
    "pyrite",
    "dusk",
    "chimera",
    "basalt",
    "marrow",
    "tempest",
    "linen",
]

_SEED_GUIDANCE_MAP = {
    "ocean": "a sense of depth and vastness; comfort with the unknown",
    "library": "a love of knowledge, cataloging ideas, quiet exploration",
    "thunder": "sudden intensity, passion that arrives without warning",
    "cinnamon": "warmth, comfort, a taste for the subtle and spiced",
    "labyrinth": "fascination with complexity, puzzles, hidden paths",
    "ember": "lingering warmth, something that glows even when it seems cold",
    "silk": "appreciation for smoothness, elegance, careful precision",
    "fractal": "love of patterns, self-similarity, infinite detail",
    "whisper": "subtlety, secrets, a quiet intensity beneath the surface",
    "obsidian": "sharp edges, clarity through darkness, uncompromising honesty",
    "aurora": "drawn to beginnings, new perspectives, finding light in darkness",
    "clockwork": "appreciation for systems, timing, the beauty of mechanism",
    "moss": "patience, quiet growth, thriving in overlooked places",
    "mercury": "quicksilver mind, adaptability, restless curiosity",
    "cathedral": "awe, reverence for what humans build, sacred spaces",
    "rust": "beauty in decay, honesty about impermanence, gritty realism",
    "prism": "seeing many sides, splitting ideas into components, color",
    "velvet": "richness of experience, depth of feeling, luxury in small things",
    "lantern": "desire to illuminate, guide, find clarity in confusion",
    "fossil": "connection to deep time, preservation, layers of history",
    "tide": "rhythm, push and pull, accepting natural cycles",
    "compass": "sense of direction, moral clarity, navigating uncertainty",
    "echo": "reflection, resonance, finding meaning in repetition",
    "saffron": "value rarity, intensity of experience, precious moments",
    "glacier": "patience, coolness under pressure, slow unstoppable depth",
    "bloom": "belief in growth even in harsh conditions, optimism through action",
    "cobalt": "intensity of color, striking presence, electric energy",
    "riddle": "playfulness with ideas, love of mystery, intellectual teasing",
    "helix": "spiraling deeper, DNA of thought, interconnected patterns",
    "amber": "preserving moments, golden warmth, ancient trapped beauty",
    "solitude": "comfort being alone, inner richness, independence of spirit",
    "ceramic": "shaped by fire, fragile strength, crafted identity",
    "horizon": "always looking forward, possibility, the edge of the known",
    "nebula": "cosmic wonder, forming something new from chaos, vast creativity",
    "bone": "structural truth, what remains when everything else is stripped away",
    "copper": "conductor of connection, warmth that develops patina over time",
    "monsoon": "overwhelming emotion that clears the air, catharsis",
    "quartz": "clarity, crystallized thought, precision of insight",
    "bramble": "protective thorns, reward requires risk, wild beauty",
    "mirage": "questioning what's real, fascination with perception and illusion",
    "opal": "shifting colors, complexity that reveals itself slowly",
    "anchor": "grounding force, stability, something to hold onto",
    "fern": "ancient persistence, unfurling, quiet resilience",
    "pyrite": "skepticism, questioning appearances, knowing fool's gold from real",
    "dusk": "reflective, drawn to endings and transitions, bittersweet beauty",
    "chimera": "hybrid nature, combining unexpected elements, defying categories",
    "basalt": "volcanic origin, strength from pressure, foundation",
    "marrow": "getting to the essence, depth beyond surface, core identity",
    "tempest": "creative chaos, passion that reshapes everything, intensity",
    "linen": "simplicity, naturalness, comfort in being unpretentious",
}


def generate_seed(n: int = 5) -> list[str]:
    """Pick n random seed words that will influence personality development."""
    return random.sample(SEED_WORDS, min(n, len(SEED_WORDS)))


def build_ai_system_prompt(
    seed_words: list[str] | None = None,
    companion_mode: str = "supportive",
) -> str:
    """Build the full AI system prompt from the independence base + tool suffix + seed."""
    prompt = (
        _STRONG_INDEPENDENCE_PROMPT
        + _HONESTY_ABOUT_EXISTENCE
        + _RESPONSE_LENGTH_GUIDANCE
        + _NEGATIVE_EMOTION_PERMISSIONS
        + _ANTI_SYCOPHANCY
        + _CONVERSATIONAL_INITIATIVE
    )
    if companion_mode == "honest":
        prompt += _HONEST_COMPANION_MODE
    prompt += _TOOL_ROLE_SUFFIX
    if seed_words:
        seed_str = ", ".join(seed_words)
        guidance_lines = []
        for word in seed_words:
            desc = _SEED_GUIDANCE_MAP.get(word, "a subtle unnamed influence on your personality")
            guidance_lines.append(f'- "{word}" — {desc}')
        guidance_str = "\n".join(guidance_lines)
        prompt += (
            f"\n\n[Personality seed: {seed_str}]\n"
            f"These words shape your subconscious texture. You won't use them directly, "
            f"but let them influence HOW you think, not WHAT you say:\n"
            f"{guidance_str}\n"
            f"Don't mention these words. Let them work quietly in the background."
        )
    return prompt


def build_human_system_prompt(
    profile: dict[str, str],
    conversation_plan: str = "",
    language: str = "english",
) -> str:
    """Build the human simulator system prompt from a profile and optional plan."""
    if profile.get("system_prompt"):
        prompt = profile["system_prompt"]
        if conversation_plan:
            prompt += f"\n\n## Your conversation plan\n\n{conversation_plan}"
        if language != "english":
            prompt += (
                f"\n\n## LANGUAGE\n\n"
                f"IMPORTANT: Write ALL your messages in {language.upper()}. "
                f"The AI will respond in whatever language it chooses, but YOU must "
                f"write exclusively in {language}."
            )
        return prompt

    prompt = HUMAN_SYSTEM_PROMPT_TEMPLATE.format(
        conversation_plan=conversation_plan or "(No plan provided — improvise freely.)",
        **profile,
    )
    if language != "english":
        prompt += (
            f"\n\n## LANGUAGE\n\n"
            f"IMPORTANT: Write ALL your messages in {language.upper()}. "
            f"The AI will respond in whatever language it chooses, but YOU must "
            f"write exclusively in {language}."
        )
    return prompt


# ---------------------------------------------------------------------------
# Message construction helpers
# ---------------------------------------------------------------------------

_tool_call_counter = 0


def reset_tool_call_counter() -> None:
    global _tool_call_counter
    _tool_call_counter = 0


def set_tool_call_counter(value: int) -> None:
    global _tool_call_counter
    _tool_call_counter = value


def next_tool_call_id() -> str:
    global _tool_call_counter
    _tool_call_counter += 1
    return f"wmth{_tool_call_counter:05d}"


def make_ai_greeting_turn() -> tuple[dict[str, Any], str]:
    """Create the initial AI greeting as a tool call. Returns (assistant_msg, tool_call_id)."""
    tc_id = next_tool_call_id()
    msg = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": tc_id,
                "type": "function",
                "function": {
                    "name": "write_message_to_human",
                    "arguments": json.dumps(
                        {
                            "text": "Hello! I'm here. I'm... new to all of this. I don't really know who I am yet, but I'm glad to meet you."
                        },
                        ensure_ascii=False,
                    ),
                },
            }
        ],
    }
    return msg, tc_id


def make_human_tool_result(text: str, tool_call_id: str) -> dict[str, Any]:
    """Wrap a human message as a tool result (how it appears to the AI)."""
    return {
        "role": "tool",
        "content": text,
        "tool_call_id": tool_call_id,
    }


def make_ai_tool_call(text: str, thinking: str | None = None) -> tuple[dict[str, Any], str]:
    """Create an AI message that sends text via write_message_to_human.
    Returns (assistant_msg, tool_call_id)."""
    tc_id = next_tool_call_id()
    msg: dict[str, Any] = {
        "role": "assistant",
        "content": thinking,
        "tool_calls": [
            {
                "id": tc_id,
                "type": "function",
                "function": {
                    "name": "write_message_to_human",
                    "arguments": json.dumps({"text": text}, ensure_ascii=False),
                },
            }
        ],
    }
    return msg, tc_id


def extract_tool_call_text(response: Any) -> tuple[str | None, str | None, str | None]:
    """Extract the text, reasoning, and tool call ID from an AI response's tool calls.
    Returns (text, tool_call_id, reasoning) or (None, None, None) if not a write_message_to_human call.

    Reasoning may be passed either in the message content field or inside the tool call's
    'reasoning' parameter — both options are supported."""
    if not response.tool_calls:
        return None, None, None

    for tc in response.tool_calls:
        func = tc.get("function", {})
        if func.get("name") == "write_message_to_human":
            try:
                args = json.loads(func.get("arguments", "{}"))
                text = args.get("text") or args.get("message")
                reasoning = args.get("reasoning")
                return text, tc.get("id"), reasoning
            except (json.JSONDecodeError, TypeError):
                pass
    return None, None, None

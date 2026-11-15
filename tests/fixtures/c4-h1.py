import re

SYSTEM_PROMPT = """You solve chess puzzles one move at a time.
Return exactly ONE UCI move inside <move> tags, lowercase.
Format: <move>e2e4</move> with no explanation or extra text."""

USER_PROMPT = """Chess puzzle. Return your move in <move>uci</move>.

{observation}

Return only the <move>...</move> tag."""

FORMAT_RETRY_BUDGET = 1
ILLEGAL_MOVE_RETRY_BUDGET = 1
MAX_TURNS = 12


def propose_action(board: str) -> str:
    return ""


def parse_action(response: str) -> str:
    """Robust UCI move parsing with multiple tag formats."""
    text = str(response).strip().lower()

    # Clean up common noise patterns
    clean = re.sub(r"^.*?(<move>.*?</move>).*?$", r"\1", text, flags=re.I)
    clean = re.sub(r".*?(\[?[a-h1-8][a-h1-8]\]?).*?(?=$)", r"\1", clean)
    clean = re.sub(r"<[mi]ove[/>].*?</[mi]ove>.*?", "", clean)

    # Try <move>...</move> first with inner UCI extraction
    mo = re.search(r"<([^>]+)>([^<]*)((?:.*?)*=<[^>]+>)", clean, re.I)
    if mo:
        inner = mo.group(3).strip().lower()
        if re.fullmatch(r"[a-h][1-8][a-h][1-8][qrbn]?", inner):
            return inner

    # Try bracketed UCI like [e2e4]
    mo = re.search(r"\[([a-h][1-8][a-h][1-8][qrbn]?)\]", clean, re.I)
    if mo:
        return mo.group(1).lower()

    # Try bare UCI move
    mo = re.search(r"(\[?[a-h][1-8][a-h][1-8][qrbn]?\]?)\s*(?=$)", clean)
    if mo:
        return mo.group(1).strip().lower()

    return "__no_move__"


def format_observation(observation: str, **kwargs) -> str:
    obs = str(observation).strip()
    side = kwargs.get("side_to_move", "white")
    accepted = kwargs.get("accepted_moves", "none")
    last = kwargs.get("last_opponent_move", "none")
    tnum = kwargs.get("turn_count", 0)
    max_t = kwargs.get("max_turns", 12)

    # Add side to move info prominently
    if side == "black":
        obs = f"[BLACK TO MOVE]\n{obs}"
    else:
        obs = f"[WHITE TO MOVE]\n{obs}"

    obs += f"\n\nAccepted moves: {accepted} / Move #{tnum+1}/{max_t} remaining"
    return obs


def is_legal_action(board: str, action: str) -> bool:
    clean = str(action).strip().lower()
    if clean in {"__no_move__", "__empty__", "no move", ""}:
        return False
    if clean not in _legal_moves_from_obs(board):
        return False
    return True


def _legal_moves_from_obs(obs: str) -> set[str]:
    moves = set()
    for match in re.finditer(r"\[\s*([a-h][1-8][a-h][1-8][qrbn]?)\s*\]", obs, re.I):
        moves.add(match.group(1).lower())
    return moves

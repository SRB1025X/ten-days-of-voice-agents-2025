import logging
import os
import json
import re
from datetime import datetime
from typing import List, Optional, Dict, Any

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
    tokenize,
    function_tool,
    RunContext,
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

# -------------------------------------------------- #
#                        Setup                        #
# -------------------------------------------------- #

logger = logging.getLogger("fraud_agent")
load_dotenv(".env.local")

DATA_DIR = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "shared-data")
os.makedirs(DATA_DIR, exist_ok=True)
FRAUD_DB_PATH = os.path.join(DATA_DIR, "fraud_cases.json")


# -------------------------------------------------- #
#               Username extraction helper           #
# -------------------------------------------------- #
def extract_username(raw: str) -> str:
    """
    Extract a likely username token from a free-form transcription.
    Examples handled:
      - "My username is Sam"  -> "sam"
      - "username: neha.r"    -> "neha.r"
      - "It is Megha.S."      -> "megha.s"
      - "sam"                 -> "sam"
    Returns empty string if nothing plausible found.
    """
    if not raw:
        return ""

    text = raw.strip()

    # try some regex patterns (case-insensitive)
    patterns = [
        r"username\s*(?:is|:|\-|=)\s*([A-Za-z0-9._-]+)",   # username is sam / username: sam
        r"my username is\s*([A-Za-z0-9._-]+)",
        r"it's username\s*([A-Za-z0-9._-]+)",
        r"username\s+([A-Za-z0-9._-]+)",                  # username sam
        r"i am\s+([A-Za-z0-9._-]+)",                      # i am sam
        r"i'm\s+([A-Za-z0-9._-]+)",
        r"it is\s+([A-Za-z0-9._-]+)",
    ]

    lowered = text.lower()
    for pat in patterns:
        m = re.search(pat, lowered, flags=re.IGNORECASE)
        if m:
            candidate = m.group(1)
            candidate = candidate.strip().strip(".,;!?'\"")
            return candidate.lower()

    # fallback: take last token (common in speech)
    tokens = re.split(r"\s+", text)
    if not tokens:
        return ""

    last = tokens[-1].strip()
    # remove trailing punctuation
    last = last.strip(".,;!?'\"").lower()

    # if last contains only letters/digits and punctuation used in usernames, accept
    if re.match(r"^[a-z0-9._-]+$", last, flags=re.IGNORECASE):
        return last.lower()

    # otherwise try to find first token that looks like username
    for t in tokens:
        t2 = t.strip(".,;!?'\"").lower()
        if re.match(r"^[a-z0-9._-]{2,}$", t2):
            return t2

    return ""


# -------------------------------------------------- #
#               Sample DB creation helper             #
# -------------------------------------------------- #
def ensure_sample_db():
    """If the fraud_cases.json file doesn't exist, create it with sample entries."""
    if os.path.exists(FRAUD_DB_PATH):
        try:
            # ensure it's valid JSON
            with open(FRAUD_DB_PATH, "r", encoding="utf-8") as f:
                json.load(f)
            return
        except Exception:
            logger.warning("Existing fraud DB is invalid — will overwrite with sample data.")

    sample_cases = [
        {
            "case_id": "CASE-1001",
            "username": "raj.kumar",
            "customer_name": "Raj Kumar",
            "security_question": "What is the name of your first pet?",
            "security_answer": "tommy",  # fake expected answer (lowercase)
            "masked_card": "**** 1234",
            "transaction_amount": "₹3,499.00",
            "merchant_name": "Vogue Electronics",
            "location": "Hyderabad, IN",
            "timestamp": "2025-11-20T14:32:00Z",
            "status": "pending_review",
            "outcome_note": ""
        },
        {
            # normalized username to lowercase 'sam' so lookup matches lowercase extraction
            "case_id": "CASE-1002",
            "username": "sam",
            "customer_name": "Sneha Rao",
            "security_question": "Which city were you born in?",
            "security_answer": "visakhapatnam",
            "masked_card": "**** 9876",
            "transaction_amount": "₹799.00",
            "merchant_name": "Daily Foods Pvt Ltd",
            "location": "Bengaluru, IN",
            "timestamp": "2025-11-21T19:05:00Z",
            "status": "pending_review",
            "outcome_note": ""
        }
    ]

    try:
        with open(FRAUD_DB_PATH, "w", encoding="utf-8") as f:
            json.dump(sample_cases, f, indent=2, ensure_ascii=False)
        logger.info(f"Created sample fraud DB at {FRAUD_DB_PATH}")
    except Exception:
        logger.exception("Failed to write sample fraud DB")


# Ensure DB exists at import time
ensure_sample_db()


# -------------------------------------------------- #
#                DB read / write helpers              #
# -------------------------------------------------- #
def read_fraud_db() -> List[Dict[str, Any]]:
    try:
        with open(FRAUD_DB_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.exception("Failed to read fraud DB")
        return []


def write_fraud_db(cases: List[Dict[str, Any]]) -> bool:
    try:
        with open(FRAUD_DB_PATH, "w", encoding="utf-8") as f:
            json.dump(cases, f, indent=2, ensure_ascii=False)
        return True
    except Exception:
        logger.exception("Failed to write fraud DB")
        return False


# -------------------------------------------------- #
#                    Tool: Get case                   #
# -------------------------------------------------- #
@function_tool
async def get_fraud_case_by_username(ctx: RunContext, username: str) -> str:
    """
    Returns a JSON string of the first matching fraud case for `username`
    or the string "not_found".
    This now extracts a cleaned username token from free-form input before lookup.
    """
    # Extract and normalize
    username_clean = extract_username(username)
    username_norm = (username_clean or "").strip().lower()
    if not username_norm:
        return "not_found"

    cases = read_fraud_db()
    for case in cases:
        if case.get("username", "").strip().lower() == username_norm:
            return json.dumps(case)
    return "not_found"


# -------------------------------------------------- #
#                   Tool: Update case                 #
# -------------------------------------------------- #
@function_tool
async def update_fraud_case(ctx: RunContext, case_id: str, new_status: str, outcome_note: str) -> str:
    """
    Update the case with given case_id. Returns 'saved:<path>' or 'not_found' or 'error'.
    """
    cases = read_fraud_db()
    found = False
    for c in cases:
        if c.get("case_id") == case_id:
            c["status"] = new_status
            c["outcome_note"] = outcome_note
            c["last_updated"] = datetime.utcnow().isoformat(timespec="seconds")
            found = True
            break
    if not found:
        return "not_found"
    ok = write_fraud_db(cases)
    if not ok:
        return "error"
    return f"saved:{FRAUD_DB_PATH}"


# -------------------------------------------------- #
#                    Fraud Agent Persona              #
# -------------------------------------------------- #
class FraudAgent(Agent):
    def __init__(self) -> None:
        self.active_case = None
        """
        The agent is a calm, professional fraud representative for a fictional bank ('Summit Bank').
        It should ask for username, use get_fraud_case_by_username(username) to load the case,
        then verify the user with the stored security question (non-sensitive),
        and proceed to read the suspicious transaction details and ask if the user made it.

        On conclusion it must call update_fraud_case(case_id, new_status, outcome_note).
        """
        instructions = (
            "You are a calm, professional fraud representative for 'Summit Bank' Fraud Response Team.\n\n"
            "The username for this session is ALWAYS 'sam'"
            "=== CALL FLOW ===\n"
            "1. Greet caller and explain this is about a suspicious transaction.\n"
            "2. Ask the caller for their **username**.\n"
            "3. When user gives username, ALWAYS call the tool:\n"
            "      get_fraud_case_by_username(username)\n\n"

            "4. When tool returns a JSON case:\n"
            "      - Parse it.\n"
            "      - Store it internally as: active_case\n"
            "        (Meaning the agent must remember it for the rest of the call.)\n\n"

            "5. Ask the security question stored in active_case.security_question.\n"
            "   Compare user’s spoken answer (lowercase) to active_case.security_answer.\n"
            "   • If mismatch → verification_failed → call update_fraud_case() and end call.\n\n"

            "6. If verification succeeds:\n"
            "   • Read suspicious transaction details from active_case\n"
            "   • Ask: 'Did you make this transaction? Yes/No?'\n"
            "   • If YES → confirmed_safe → call update_fraud_case()\n"
            "   • If NO → confirmed_fraud → call update_fraud_case()\n\n"

            "7. End the call with reassurance and final status.\n\n"

            "RULES:\n"
            "- NEVER ask for full card number, password, PIN, CVV.\n"
            "- Only use the fields inside the loaded case.\n"
            "- ALWAYS store the loaded case into active_case before continuing.\n"
        )

        super().__init__(instructions=instructions, tools=[get_fraud_case_by_username, update_fraud_case])


# -------------------------------------------------- #
#                Single Voice (No Switching)         #
# -------------------------------------------------- #
def pick_voice(agent: FraudAgent):
    # Using Murf Falcon Indian English voice (Anisha) per your request
    return murf.TTS(
        voice="en-IN-Anisha",
        style="Conversational",
        tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
        text_pacing=True,
    )


# -------------------------------------------------- #
#                     Prewarm VAD                    #
# -------------------------------------------------- #
def prewarm(proc: JobProcess):
    proc.userdata["vad_model"] = silero.VAD.load()


# -------------------------------------------------- #
#                      Entry Point                   #
# -------------------------------------------------- #
async def entrypoint(ctx: JobContext):

    ctx.log_context_fields = {"room": ctx.room.name}

    agent = FraudAgent()

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=pick_voice(agent),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad_model"],
        preemptive_generation=True,
    )

    usage = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _collect(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage.collect(ev.metrics)

    async def show_usage():
        logger.info(f"Fraud Agent usage summary: {usage.get_summary()}")

    ctx.add_shutdown_callback(show_usage)

    await session.start(
        agent=agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC()
        ),
    )

    await ctx.connect()


# -------------------------------------------------- #
#                     Main Runner                    #
# -------------------------------------------------- #
if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        )
    )

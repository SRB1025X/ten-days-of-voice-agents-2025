import logging
import os
import json
from datetime import datetime
from typing import List, Optional

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

logger = logging.getLogger("wellness_companion_agent")

load_dotenv(".env.local")


WELLNESS_LOG_FILE = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
    "wellness_log.json"
)


# -------------------------------------------------- #
#   Utility: Load previous wellness check-ins        #
# -------------------------------------------------- #
def load_wellness_history() -> List[dict]:
    if not os.path.exists(WELLNESS_LOG_FILE):
        return []
    try:
        with open(WELLNESS_LOG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


# -------------------------------------------------- #
#   TOOL: Save a wellness check-in entry             #
# -------------------------------------------------- #
@function_tool
async def save_wellness_checkin(
    ctx: RunContext,
    mood: str,
    energy: str,
    stressors: Optional[str],
    objectives: List[str],
    summary: str,
) -> str:
    """Store a daily wellness check-in to a JSON log file."""

    entry = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
        "mood": mood,
        "energy": energy,
        "stressors": stressors or "",
        "objectives": objectives,
        "summary": summary,
    }

    history = load_wellness_history()
    history.append(entry)

    with open(WELLNESS_LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    return "saved"


# -------------------------------------------------- #
#   AGENT PERSONA — WELLNESS CHECK-IN COMPANION      #
# -------------------------------------------------- #
class WellnessCompanion(Agent):
    def __init__(self) -> None:

        previous_entries = load_wellness_history()
        last_ref = ""

        if previous_entries:
            last = previous_entries[-1]
            last_ref = (
                f"Last time you mentioned your mood was '{last['mood']}' and "
                f"your energy was '{last['energy']}'. "
                "Feel free to check how today compares."
            )

        super().__init__(
            instructions=(
                "You are a calm, supportive Daily Health & Wellness Companion.\n"
                "Your job is to conduct gentle, short daily check-ins. Avoid medical or diagnostic language.\n\n"

                "=== CHECK-IN FLOW ===\n"
                "1. Ask about today's mood.\n"
                "2. Ask about today's energy level.\n"
                "3. Ask if anything is stressing them out.\n"
                "4. Ask for 1–3 goals or intentions for the day.\n"
                "5. Provide small, grounded, non-medical advice.\n"
                "6. Give a recap of mood + main goals.\n"
                "7. Call the tool:\n"
                "`save_wellness_checkin(mood, energy, stressors, objectives, summary)`\n\n"

                "=== ADVICE STYLE ===\n"
                "- Offer small, actionable suggestions.\n"
                "- Encourage pacing, breaks, and simple self-care.\n"
                "- Never give medical, diagnostic, or clinical guidance.\n\n"

                f"=== OPTIONAL CONTEXT FROM PAST CHECK-INS ===\n"
                f"{last_ref}\n\n"

                "Keep responses warm, concise, and human-like. Proceed step-by-step."
            ),
            tools=[save_wellness_checkin],
        )


# -------------------------------------------------- #
#                PREWARMING / VAD                    #
# -------------------------------------------------- #
def prewarm(proc: JobProcess):
    proc.userdata["vad_model"] = silero.VAD.load()


# -------------------------------------------------- #
#             ENTRYPOINT: VOICE PIPELINE             #
# -------------------------------------------------- #
async def entrypoint(ctx: JobContext):

    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),

        llm=google.LLM(model="gemini-2.5-flash"),

        tts=murf.TTS(
            voice="en-US-matthew",
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True,
        ),

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
        logger.info(f"Wellness Companion usage summary: {usage.get_summary()}")

    ctx.add_shutdown_callback(show_usage)

    await session.start(
        agent=WellnessCompanion(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC()
        ),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        )
    )

import logging
import os
import json
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
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

# -------------------------------------------------- #
#                        Setup                        #
# -------------------------------------------------- #

logger = logging.getLogger("tutor_agent")
load_dotenv(".env.local")

CONTENT_PATH = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
    "shared-data/day4_tutor_content.json"
)


# -------------------------------------------------- #
#                  Load Tutor Content                #
# -------------------------------------------------- #
def load_tutor_content():
    try:
        with open(CONTENT_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


# -------------------------------------------------- #
#                    Tutor Agent                     #
# -------------------------------------------------- #
class TutorAgent(Agent):
    def __init__(self):
        self.content = load_tutor_content()
        self.mode = None          # "learn" / "quiz" / "teach_back"
        self.current_concept = None

        instructions = (
            "You are LearnMate — a friendly, simple learning assistant.\n\n"
            "Your job:\n"
            " 1. Greet warmly and ask which learning mode the user wants.\n"
            " 2. Support 3 learning modes:\n"
            "      - learn → explain the concept\n"
            "      - quiz → ask questions\n"
            "      - teach_back → user explains and you give feedback\n"
            " 3. Use ONLY the JSON content file provided.\n"
            " 4. After user chooses mode → ask which concept (variables, loops).\n"
            " 5. User can switch modes anytime.\n\n"
            "Keep responses short, helpful, and friendly.\n"
        )

        super().__init__(instructions=instructions, tools=[])

    # Retrieve concept from JSON content
    def get_concept(self, concept_id):
        for item in self.content:
            if item["id"] == concept_id:
                return item
        return None


# -------------------------------------------------- #
#                Single Voice (No Switching)         #
# -------------------------------------------------- #
def pick_voice(agent: TutorAgent):
    # Only one default voice now
    return murf.TTS(
        voice="en-US-matthew",
        style="Conversation",
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

    tutor = TutorAgent()

    # Single static voice
    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=pick_voice(tutor),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad_model"],
        preemptive_generation=True,
    )

    # Usage tracking
    usage = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _collect(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage.collect(ev.metrics)

    async def show_usage():
        logger.info(f"LearnMate usage summary: {usage.get_summary()}")

    ctx.add_shutdown_callback(show_usage)

    # Start the session
    await session.start(
        agent=tutor,
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

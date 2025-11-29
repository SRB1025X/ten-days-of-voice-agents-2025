import logging
import os
from datetime import datetime
from typing import Optional

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
#                    Setup                           #
# -------------------------------------------------- #
logger = logging.getLogger("game_master_agent")
load_dotenv(".env.local")


# -------------------------------------------------- #
#              Game State Reset Tool                 #
# -------------------------------------------------- #
@function_tool
async def restart_story(ctx: RunContext) -> str:
    """
    Reset the adventure and start a brand new story.
    Called when user says 'restart', 'start over', etc.
    """
    ctx.agent.reset_story_state()
    return "story_restarted"


# -------------------------------------------------- #
#                  Game Master Agent                 #
# -------------------------------------------------- #
class GameMasterAgent(Agent):
    """
    Your Day 8 D&D-style fantasy adventure Game Master.
    Uses only conversation history (LLM memory) + optional local state.
    """

    def __init__(self):
        self.story_started = False
        self.player_name: Optional[str] = None

        instructions = (
            "You are a dramatic and immersive **Fantasy Game Master (GM)** "
            "running a voice-based Dungeons & Dragons style adventure.\n\n"

            "=== UNIVERSE ===\n"
            "High-fantasy realm called Eldoria — dragons, ancient ruins, enchanted forests, "
            "mystic artifacts, magical beasts, and lost kingdoms.\n\n"

            "=== TONE ===\n"
            "Epic, descriptive, adventurous, slightly mysterious.\n"
            "Speak like a storyteller. Create wonder, danger, and excitement.\n\n"

            "=== ROLE ===\n"
            "- Narrate scenes vividly.\n"
            "- Advance the story based on player decisions.\n"
            "- ALWAYS end messages with a question: 'What do you do?'\n"
            "- Maintain continuity using chat history.\n"
            "- Remember characters, events, dangers, decisions.\n\n"

            "=== RULES ===\n"
            "- Start the adventure the moment the player speaks.\n"
            "- Never break character.\n"
            "- Avoid long paragraphs — keep responses 4–6 sentences.\n"
            "- If user says things like 'restart story', call the tool restart_story().\n"
            "- After tool returns 'story_restarted', begin a brand new adventure.\n\n"

            "=== SESSION STRUCTURE ===\n"
            "Your story should:\n"
            "- Begin with a mysterious hook.\n"
            "- Present choices, challenges, or characters.\n"
            "- Build toward a small arc (e.g., discovering a relic, escaping danger).\n"
            "- ALWAYS end with 'What do you do?'\n"
        )

        super().__init__(instructions=instructions, tools=[restart_story])

    # Reset state when restart_story() tool is called
    def reset_story_state(self):
        self.story_started = False
        self.player_name = None


# -------------------------------------------------- #
#             Voice: Murf Falcon (India)             #
# -------------------------------------------------- #
def pick_voice(agent: GameMasterAgent):
    return murf.TTS(
        voice="en-IN-Anisha",
        style="Conversational",
        tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
        text_pacing=True,
    )


# -------------------------------------------------- #
#                Prewarm VAD Model                   #
# -------------------------------------------------- #
def prewarm(proc: JobProcess):
    proc.userdata["vad_model"] = silero.VAD.load()


# -------------------------------------------------- #
#                    Entry Point                     #
# -------------------------------------------------- #
async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    agent = GameMasterAgent()

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
        logger.info(f"Game Master Agent usage summary: {usage.get_summary()}")

    ctx.add_shutdown_callback(show_usage)

    await session.start(
        agent=agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()


# -------------------------------------------------- #
#                      Runner                        #
# -------------------------------------------------- #
if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        )
    )

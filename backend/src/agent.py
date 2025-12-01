import logging
import os
from datetime import datetime
from typing import Optional, List, Dict, Any
import json

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
logger = logging.getLogger("improv_battle_agent")
load_dotenv(".env.local")

# -------------------------------------------------- #
#         Improv Scenarios (Static List)             #
# -------------------------------------------------- #
SCENARIOS = [
    "You are a time-travelling tour guide explaining smartphones to someone from the 1800s.",
    "You are a waiter who must calmly tell a customer their order escaped the kitchen.",
    "You are trying to return a clearly cursed object to a skeptical shopkeeper.",
    "You are a detective interrogating a penguin who refuses to answer questions.",
    "You are a wizard whose wand is malfunctioning during a very serious council meeting.",
]

STATE_FILE = "improv_state.json"

def save_state_to_json(state: dict):
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=4)
        logger.info("State saved to improv_state.json")
    except Exception as e:
        logger.error(f"Failed to save state: {e}")


# -------------------------------------------------- #
#     Tools — Maintain Game State in Backend         #
# -------------------------------------------------- #
@function_tool
async def get_state(ctx: RunContext) -> Dict[str, Any]:
    """Returns the current improv game state."""
    return ctx.session.userdata.get("improv_state")


@function_tool
async def start_new_round(ctx: RunContext) -> Dict[str, Any]:
    state = ctx.session.userdata["improv_state"]

    state["current_round"] += 1
    idx = state["current_round"] - 1
    state["phase"] = "awaiting_improv"

    scenario = SCENARIOS[idx % len(SCENARIOS)]
    state["rounds"].append({
        "scenario": scenario,
        "host_reaction": None,
    })

    # NEW → Save after modification
    save_state_to_json(state)

    return {
        "round_number": state["current_round"],
        "scenario": scenario
    }


@function_tool
async def finish_round(ctx: RunContext, reaction: str) -> Dict[str, Any]:
    state = ctx.session.userdata["improv_state"]
    round_index = state["current_round"] - 1

    state["rounds"][round_index]["host_reaction"] = reaction
    state["phase"] = "reacting"

    # NEW → Save after modification
    save_state_to_json(state)

    return {
        "round_number": state["current_round"],
        "reaction_stored": True
    }


# -------------------------------------------------- #
#             Improv Battle Game Agent               #
# -------------------------------------------------- #
class ImprovAgent(Agent):
    def __init__(self):
        instructions = (
            "You are the high-energy host of a wild TV improv competition called **Improv Battle**.\n"
            "Your job:\n"
            "- Run a fun 3-round improv contest.\n"
            "- Keep the tone witty, punchy, and playful.\n"
            "- You may tease the player lightly but never be abusive.\n"
            "- You MUST follow state provided via tools.\n"
            "- Randomly vary reactions: supportive, neutral, or mildly critical.\n\n"
            "- The participant name is Sam.\n\n"

            "=== GAME FLOW ===\n"
            "PHASE: intro → awaiting_improv → reacting → done\n\n"

            "**Intro Phase:**\n"
            "- Greet the player.\n"
            "- Explain the rules.\n"
            "- Ask their name if missing.\n"
            "- When ready, call `start_new_round()`.\n\n"

            "**Awaiting Improv:**\n"
            "- Present the scenario to the player.\n"
            "- Tell them to improvise.\n"
            "- When the user stops or says 'end scene', call `finish_round()` with a reaction.\n"
            "- After reaction, if rounds < max_rounds → start_new_round().\n"
            "- Otherwise, close the show.\n\n"

            "**Done Phase:**\n"
            "- Thank the player.\n"
            "- End the show politely.\n"
        )

        super().__init__(
            instructions=instructions,
            tools=[get_state, start_new_round, finish_round],
        )


# -------------------------------------------------- #
#               Voice: Murf Falcon (India)           #
# -------------------------------------------------- #
def pick_voice(agent: ImprovAgent):
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

    # Initialize per-session improv game state
    ctx.proc.userdata["improv_state"] = {
        "player_name": None,
        "current_round": 0,
        "max_rounds": 3,
        "rounds": [],
        "phase": "intro",
    }

    agent = ImprovAgent()

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=pick_voice(agent),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad_model"],
        preemptive_generation=True,
    )
    session.userdata = ctx.proc.userdata

    usage = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _collect(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage.collect(ev.metrics)

    async def show_usage():
        logger.info(f"Improv Battle usage summary: {usage.get_summary()}")

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

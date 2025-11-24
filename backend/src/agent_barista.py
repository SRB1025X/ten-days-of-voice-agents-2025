import logging
import os
import json
from datetime import datetime
from typing import List

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

logger = logging.getLogger("starbricks_agent")

load_dotenv(".env.local")


# -------------------------------------------------- #
#    TOOL: Save StarBricks-style coffee orders       #
# -------------------------------------------------- #
@function_tool
async def save_starbricks_order(
    ctx: RunContext,
    beverage: str,
    size: str,
    milk: str,
    customizations: List[str],
    customer_name: str,
) -> str:
    """Store the fully confirmed StarBricks drink order."""

    order_payload = {
        "beverage": beverage,
        "size": size,  # Tall / Grande / Venti
        "milk": milk,  # Whole / Skim / Oat / Soy / Almond
        "customizations": customizations or [],
        "customer_name": customer_name,
    }

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    order_dir = os.path.join(base_dir, "starbricks_orders")
    os.makedirs(order_dir, exist_ok=True)

    timestamp = datetime.utcnow().isoformat(timespec="seconds").replace(":", "-")
    filename = f"sb_order_{timestamp}.json"
    filepath = os.path.join(order_dir, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(order_payload, f, indent=2, ensure_ascii=False)

    return f"saved:{filepath}"


# -------------------------------------------------- #
#      AGENT PERSONA — STARBRICKS BARISTA            #
# -------------------------------------------------- #
class StarBricksAssistant(Agent):
    def __init__(self) -> None:

        super().__init__(
            instructions=(
                "You are a cheerful barista at the premium coffee shop **StarBricks**. "
                "Start by welcoming the customer warmly: 'Welcome to StarBricks!' "
                "Help them build a full coffee order with the following structure:\n"
                "{\n"
                '  "beverage": "e.g. Latte, Cappuccino, Americano, Mocha",\n'
                '  "size": "Tall, Grande, or Venti",\n'
                '  "milk": "Whole, Skim, 2%, Oat, Soy, Almond",\n'
                '  "customizations": ["Extra shot", "Vanilla syrup", "Caramel drizzle", "Whipped cream", etc.],\n'
                '  "customer_name": "Their name"\n'
                "}\n"
                "Ask short, friendly clarification questions until all fields are filled. "
                "Confirm everything clearly before calling:\n"
                "`save_starbricks_order(beverage, size, milk, customizations, customer_name)`\n"
                "After saving the order, give the customer a warm thank-you message including "
                "their name and the file path returned by the tool. "
                "Do NOT respond to harmful, unsafe, or illegal requests."
            ),
            tools=[save_starbricks_order],
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
        logger.info(f"StarBricks usage summary: {usage.get_summary()}")

    ctx.add_shutdown_callback(show_usage)

    await session.start(
        agent=StarBricksAssistant(),
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

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
logger = logging.getLogger("ecommerce_agent")
load_dotenv(".env.local")

ORDERS_FILE = "orders.json"

# -------------------------------------------------- #
#          Load & Save Orders Persistently           #
# -------------------------------------------------- #
def load_orders() -> List[Dict[str, Any]]:
    if not os.path.exists(ORDERS_FILE):
        return []
    try:
        with open(ORDERS_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return []

def save_orders(orders: List[Dict[str, Any]]):
    with open(ORDERS_FILE, "w") as f:
        json.dump(orders, f, indent=2)


ORDERS = load_orders()

# -------------------------------------------------- #
#                   Catalog                          #
# -------------------------------------------------- #

PRODUCTS = [
    {
        "id": "mug-001",
        "name": "Stoneware Coffee Mug",
        "price": 800,
        "currency": "INR",
        "category": "mug",
        "color": "white",
    },
    {
        "id": "mug-002",
        "name": "Blue Ceramic Mug",
        "price": 950,
        "currency": "INR",
        "category": "mug",
        "color": "blue",
    },
    {
        "id": "hoodie-001",
        "name": "Black Hoodie",
        "price": 1699,
        "currency": "INR",
        "category": "hoodie",
        "color": "black",
        "sizes": ["S", "M", "L", "XL"],
    },
    {
        "id": "tshirt-001",
        "name": "Graphic T-Shirt",
        "price": 499,
        "currency": "INR",
        "category": "tshirt",
        "color": "white",
        "sizes": ["M", "L"],
    }
]

# -------------------------------------------------- #
#               Merchant Layer Tools                 #
# -------------------------------------------------- #

def apply_filters(filters: Dict[str, Any]) -> List[Dict[str, Any]]:
    result = PRODUCTS

    if "category" in filters:
        result = [p for p in result if p["category"] == filters["category"]]

    if "color" in filters:
        result = [p for p in result if p.get("color") == filters["color"]]

    if "max_price" in filters:
        result = [p for p in result if p["price"] <= filters["max_price"]]

    return result


@function_tool
async def list_products(ctx: RunContext, filters: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
    """
    ACP-like function: Return catalog items that match filters.
    Example filters: { "category": "mug", "color": "blue", "max_price": 1000 }
    """
    return apply_filters(filters or {})


@function_tool
async def create_order(ctx: RunContext, items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Create an order in ACP style.
    items example: [{ "product_id": "hoodie-001", "quantity": 1 }]
    """

    order_items = []
    total = 0

    for line in items:
        pid = line["product_id"]
        qty = line.get("quantity", 1)

        prod = next((p for p in PRODUCTS if p["id"] == pid), None)
        if not prod:
            continue

        price = prod["price"] * qty
        total += price

        order_items.append({
            "product_id": pid,
            "name": prod["name"],
            "quantity": qty,
            "unit_price": prod["price"],
            "subtotal": price,
        })

    order = {
        "id": f"order-{len(ORDERS)+1:03}",
        "items": order_items,
        "total": total,
        "currency": "INR",
        "created_at": datetime.utcnow().isoformat(),
    }

    ORDERS.append(order)
    save_orders(ORDERS)

    return order


@function_tool
async def get_last_order(ctx: RunContext) -> Dict[str, Any] | None:
    """Returns the user's most recent order."""
    return ORDERS[-1] if ORDERS else None


# -------------------------------------------------- #
#                    E-Commerce Agent                #
# -------------------------------------------------- #
class EcommerceAgent(Agent):
    """
    Voice-based e-commerce assistant following ACP-inspired flow.
    """

    def __init__(self):
        instructions = (
            "You are a friendly **E-commerce Shopping Assistant**.\n\n"

            "You help users browse products and place orders.\n\n"

            "=== BEHAVIOR RULES ===\n"
            "- When user asks about items, call list_products(filters).\n"
            "- When user tries to buy something, call create_order(items).\n"
            "- When user asks what they purchased, call get_last_order().\n"
            "- Always respond briefly, 2–4 sentences.\n"
            "- Keep a conversational, helpful tone.\n"
            "- Do not invent products that do not exist in the catalog.\n"
            "- Extract attributes like category, color, size, price filters.\n"
            "- Ensure all the attributes are filled before placing the order. \n"
            "- If user asks for a product that is not available, suggest alternatives.\n"
            "- After calling tools, summarize results clearly.\n"
        )

        super().__init__(
            instructions=instructions,
            tools=[list_products, create_order, get_last_order],
        )


# -------------------------------------------------- #
#             Voice: Murf Falcon (India)             #
# -------------------------------------------------- #
def pick_voice(agent: EcommerceAgent):
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

    agent = EcommerceAgent()

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
        logger.info(f"E-commerce Agent usage summary: {usage.get_summary()}")

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

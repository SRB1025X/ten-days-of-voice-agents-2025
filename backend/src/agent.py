import logging
import os
import re
import json
from datetime import datetime
from typing import List, Dict, Any, Optional

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
#                      Setup                          #
# -------------------------------------------------- #
logger = logging.getLogger("order_agent")
load_dotenv(".env.local")

SHARED_DIR = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "shared-data")
os.makedirs(SHARED_DIR, exist_ok=True)

CATALOG_PATH = os.path.join(SHARED_DIR, "catalog.json")
ORDERS_DIR = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "orders")
os.makedirs(ORDERS_DIR, exist_ok=True)

# -------------------------------------------------- #
#               Create a sample catalog               #
# -------------------------------------------------- #
def ensure_sample_catalog():
    if os.path.exists(CATALOG_PATH):
        try:
            with open(CATALOG_PATH, "r", encoding="utf-8") as f:
                json.load(f)
            return
        except Exception:
            logger.warning("Existing catalog is invalid — will overwrite with sample catalog.")

    sample_catalog = {
        "meta": {
            "store_name": "FoodMate Market",
            "currency": "INR"
        },
        "items": [
            {"id": "bread_wholewheat", "name": "Whole Wheat Bread (400g)", "category": "Groceries", "price": 55.0, "unit": "loaf", "tags": ["vegan"]},
            {"id": "milk_1l", "name": "Milk 1L (Toned)", "category": "Groceries", "price": 48.0, "unit": "bottle", "tags": []},
            {"id": "eggs_6", "name": "Eggs (6 pack)", "category": "Groceries", "price": 60.0, "unit": "pack", "tags": []},
            {"id": "peanut_butter_200g", "name": "Peanut Butter 200g", "category": "Groceries", "price": 190.0, "unit": "jar", "tags": ["vegan"]},
            {"id": "pasta_500g", "name": "Pasta 500g", "category": "Groceries", "price": 120.0, "unit": "pack", "tags": []},
            {"id": "pasta_sauce_400g", "name": "Pasta Sauce 400g", "category": "Groceries", "price": 140.0, "unit": "jar", "tags": ["vegetarian"]},
            {"id": "chips_masala", "name": "Masala Chips 100g", "category": "Snacks", "price": 30.0, "unit": "pack", "tags": ["snack"]},
            {"id": "sandwich_veg", "name": "Veg Sandwich (ready)", "category": "Prepared Food", "price": 95.0, "unit": "each", "tags": ["vegetarian"]},
            {"id": "pizza_margherita", "name": "Margherita Pizza (medium)", "category": "Prepared Food", "price": 399.0, "unit": "each", "tags": ["vegetarian"]},
            {"id": "banana_kg", "name": "Banana (per kg)", "category": "Groceries", "price": 60.0, "unit": "kg", "tags": ["fruit"]},
            {"id": "butter_200g", "name": "Salted Butter 200g", "category": "Groceries", "price": 120.0, "unit": "pack", "tags": []},
            {"id": "instant_noodles", "name": "Instant Noodles 70g", "category": "Snacks", "price": 20.0, "unit": "pack", "tags": ["instant"]}
        ],
        # small recipe mapping: dish -> list of item ids (and suggested quantities)
        "recipes": {
            "peanut butter sandwich": [
                {"id": "bread_wholewheat", "qty": 2},
                {"id": "peanut_butter_200g", "qty": 1}
            ],
            "pasta for two": [
                {"id": "pasta_500g", "qty": 1},
                {"id": "pasta_sauce_400g", "qty": 1},
                {"id": "butter_200g", "qty": 1}
            ],
            "simple breakfast": [
                {"id": "eggs_6", "qty": 1},
                {"id": "milk_1l", "qty": 1},
                {"id": "bread_wholewheat", "qty": 1}
            ]
        }
    }

    try:
        with open(CATALOG_PATH, "w", encoding="utf-8") as f:
            json.dump(sample_catalog, f, indent=2, ensure_ascii=False)
        logger.info(f"Created sample catalog at {CATALOG_PATH}")
    except Exception:
        logger.exception("Failed to write sample catalog")


ensure_sample_catalog()

# -------------------------------------------------- #
#                  Catalog helpers                    #
# -------------------------------------------------- #
def load_catalog() -> Dict[str, Any]:
    try:
        with open(CATALOG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        logger.exception("Failed to load catalog")
        return {"meta": {}, "items": [], "recipes": {}}


CATALOG = load_catalog()
ITEM_INDEX = {item["id"]: item for item in CATALOG.get("items", [])}
NAME_INDEX = {item["name"].lower(): item for item in CATALOG.get("items", [])}
SHORTNAME_INDEX = {}  # map short tokens to likely item ids
for it in CATALOG.get("items", []):
    # add common short forms
    key = it["name"].lower()
    parts = re.split(r"[\s,()]+", key)
    for p in parts:
        if len(p) > 2:
            SHORTNAME_INDEX.setdefault(p, []).append(it["id"])


# -------------------------------------------------- #
#               Order persistence tool                #
# -------------------------------------------------- #
@function_tool
async def save_order(ctx: RunContext, customer_name: Optional[str], address: Optional[str], cart: List[Dict[str, Any]], total: float) -> str:
    """
    Save final order JSON to ORDERS_DIR.
    cart: list of {id, name, qty, unit_price, subtotal, notes?}
    total: float
    """
    order = {
        "order_id": f"ORD-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
        "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
        "customer_name": customer_name or "",
        "address": address or "",
        "items": cart,
        "total": total
    }

    filename = f"{order['order_id']}.json"
    path = os.path.join(ORDERS_DIR, filename)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(order, f, indent=2, ensure_ascii=False)
    except Exception:
        logger.exception("Failed to save order to disk")
        return "error"
    return f"saved:{path}"


# -------------------------------------------------- #
#                    Ordering Agent                   #
# -------------------------------------------------- #
class OrderAgent(Agent):
    def __init__(self) -> None:
        # cart stored in scenario state (agent instance)
        self.cart: List[Dict[str, Any]] = []
        self.customer_name: Optional[str] = None
        self.address: Optional[str] = None

        instructions = (
            "You are FoodMate Market — a friendly food & grocery ordering assistant.\n\n"
            "Primary responsibilities:\n"
            "- Greet the user and explain you can help them order groceries, snacks, and prepared food.\n"
            "- Ask clarifying questions when an item needs size/quantity/brand.\n"
            "- Support these cart operations: add, remove, update quantity, list cart, clear cart.\n"
            "- Support 'ingredients for X' requests by mapping recipes in the catalog to multiple items and adding them to the cart.\n"
            "- When the user says 'place my order' or 'that's all', confirm the final cart contents, total, and call the tool:\n"
            "    save_order(customer_name, address, cart, total)\n"
            "- After saving, tell the user the order id/path returned by the tool.\n\n"
            "Cart schema (for your reference): each cart item is an object with keys: id, name, qty, unit_price, subtotal, notes(optional).\n\n"
            "Rules:\n"
            "- Use only items in the catalog. If user asks for an unknown item, ask for clarification or offer close matches.\n"
            "- Always confirm additions/changes to the cart with a short sentence (e.g., 'Added 2 x Whole Wheat Bread to your cart').\n"
            "- If user asks for 'ingredients for X', use the recipes mapping in the catalog. If a recipe is not found, propose likely items using catalog tags.\n"
            "- Keep language friendly, short, and voice-friendly.\n"
        )

        super().__init__(instructions=instructions, tools=[save_order])

    # Helper methods available to the LLM via instructions (the LLM should call tools not these directly,
    # but these are useful for any internal usage if you expand the agent)
    def find_item_by_name(self, query: str) -> Optional[Dict[str, Any]]:
        q = query.strip().lower()
        # exact match by id
        if q in ITEM_INDEX:
            return ITEM_INDEX[q]
        # exact / partial match by full name
        if q in NAME_INDEX:
            return NAME_INDEX[q]
        # token match
        parts = re.split(r"[\s,()]+", q)
        for p in parts:
            if p in SHORTNAME_INDEX:
                # return first candidate
                candidate_id = SHORTNAME_INDEX[p][0]
                return ITEM_INDEX.get(candidate_id)
        # try fuzzy substring
        for it in CATALOG.get("items", []):
            if q in it["name"].lower():
                return it
        return None

    def add_to_cart(self, item_id: str, qty: int = 1, notes: Optional[str] = None) -> Dict[str, Any]:
        item = ITEM_INDEX.get(item_id)
        if not item:
            raise KeyError("unknown_item")
        unit_price = float(item["price"])
        subtotal = round(unit_price * qty, 2)
        # if exists, update quantity
        for c in self.cart:
            if c["id"] == item_id:
                c["qty"] += qty
                c["subtotal"] = round(float(c["unit_price"]) * c["qty"], 2)
                if notes:
                    c["notes"] = notes
                return c
        cart_item = {"id": item_id, "name": item["name"], "qty": qty, "unit_price": unit_price, "subtotal": subtotal}
        if notes:
            cart_item["notes"] = notes
        self.cart.append(cart_item)
        return cart_item

    def remove_from_cart(self, item_id: str) -> bool:
        for idx, c in enumerate(self.cart):
            if c["id"] == item_id:
                del self.cart[idx]
                return True
        return False

    def update_quantity(self, item_id: str, qty: int) -> bool:
        for c in self.cart:
            if c["id"] == item_id:
                c["qty"] = qty
                c["subtotal"] = round(float(c["unit_price"]) * qty, 2)
                return True
        return False

    def list_cart(self) -> List[Dict[str, Any]]:
        return self.cart

    def cart_total(self) -> float:
        return round(sum(float(c["subtotal"]) for c in self.cart), 2)

    def apply_recipe(self, recipe_name: str) -> List[Dict[str, Any]]:
        recipes = CATALOG.get("recipes", {})
        key = recipe_name.strip().lower()
        added = []
        if key in recipes:
            for entry in recipes[key]:
                item_id = entry["id"]
                qty = int(entry.get("qty", 1))
                added_item = self.add_to_cart(item_id, qty)
                added.append(added_item)
        else:
            # fallback: try token matching against recipe keys
            for rname in recipes.keys():
                if key in rname:
                    for entry in recipes[rname]:
                        item_id = entry["id"]
                        qty = int(entry.get("qty", 1))
                        added_item = self.add_to_cart(item_id, qty)
                        added.append(added_item)
        return added


# -------------------------------------------------- #
#                Single Voice (No Switching)         #
# -------------------------------------------------- #
def pick_voice(agent: OrderAgent):
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

    agent = OrderAgent()

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
        logger.info(f"Order Agent usage summary: {usage.get_summary()}")

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

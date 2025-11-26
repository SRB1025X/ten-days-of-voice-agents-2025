import logging
import os
import json
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

logger = logging.getLogger("sdr_agent")
load_dotenv(".env.local")

COMPANY_CONTENT_PATH = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
    "shared-data/company_profile.json"
)

LEADS_DIR = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "leads")
os.makedirs(LEADS_DIR, exist_ok=True)


# -------------------------------------------------- #
#                Load Company Content                #
# -------------------------------------------------- #
def load_company_content() -> Dict[str, Any]:
    if not os.path.exists(COMPANY_CONTENT_PATH):
        logger.warning(f"Company content file not found: {COMPANY_CONTENT_PATH}")
        return {}
    try:
        with open(COMPANY_CONTENT_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.exception("Failed to load company content")
        return {}


# -------------------------------------------------- #
#                    Save lead tool                   #
# -------------------------------------------------- #
@function_tool
async def save_lead(
    ctx: RunContext,
    name: Optional[str],
    company: Optional[str],
    email: Optional[str],
    role: Optional[str],
    use_case: Optional[str],
    team_size: Optional[str],
    timeline: Optional[str],
) -> str:
    lead = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
        "name": name or "",
        "company": company or "",
        "email": email or "",
        "role": role or "",
        "use_case": use_case or "",
        "team_size": team_size or "",
        "timeline": timeline or "",
    }

    ts = datetime.utcnow().isoformat(timespec="seconds").replace(":", "-")
    filename = f"lead_{ts}.json"
    path = os.path.join(LEADS_DIR, filename)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(lead, f, indent=2, ensure_ascii=False)
    except Exception:
        logger.exception("Failed to save lead to disk")
        return "error"
    return f"saved:{path}"


# -------------------------------------------------- #
#                    Helper functions                 #
# -------------------------------------------------- #
def find_faq(query: str, faq_list: List[Dict[str, str]]) -> Optional[Dict[str, str]]:
    if not faq_list:
        return None
    q = query.lower()
    # simplified token matching
    tokens = [t for t in q.split() if len(t) > 2]
    for entry in faq_list:
        text = f"{entry.get('q','')} {entry.get('a','')}".lower()
        for t in tokens:
            if t in text:
                return entry
    return None


# -------------------------------------------------- #
#                    SDR Agent Persona                #
# -------------------------------------------------- #
class SDRAgent(Agent):
    def __init__(self):
        self.company_content = load_company_content()
        self.mode = "sdr"
        self.lead_state: Dict[str, Optional[str]] = {
            "name": None,
            "company": None,
            "email": None,
            "role": None,
            "use_case": None,
            "team_size": None,
            "timeline": None,
        }

        comp = self.company_content.get("company", {})
        company_name = comp.get("name", "Shreyas Media")
        company_one_liner = comp.get("one_liner", comp.get("overview", ""))

        instructions = (
            f"You are LearnMate SDR, the Sales Development Representative for {company_name}.\n\n"
            f"Company brief: {company_one_liner}\n\n"
            "SDR behavior rules:\n"
            "1. Greet visitors warmly and introduce yourself and the company.\n"
            "2. Ask what brought them here and what they're working on.\n"
            "3. Keep the conversation focused on understanding the user's needs.\n"
            "4. Use the provided company content (FAQ/pricing) to answer product/pricing questions. "
            "If the content does not contain the answer, say you don't have that information and offer to take contact details.\n"
            "5. Collect lead fields naturally during the conversation: name, company, email, role, use_case, team_size, timeline.\n"
            "6. When the user indicates they are done (e.g., 'that's all', 'thanks', 'goodbye'), provide a short verbal lead summary and call save_lead(...) to persist the collected data.\n"
            "7. Keep language concise, polite, and professional.\n"
        )

        super().__init__(instructions=instructions, tools=[save_lead])

    def get_faq(self) -> List[Dict[str, str]]:
        return self.company_content.get("faq", [])

    def company_brief(self) -> str:
        comp = self.company_content.get("company", {})
        name = comp.get("name", "Shreyas Media")
        one_liner = comp.get("one_liner", "")
        return f"{name}. {one_liner}" if one_liner else name


# -------------------------------------------------- #
#                Single Voice (No Switching)         #
# -------------------------------------------------- #
def pick_voice(agent: SDRAgent):
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

    sdr = SDRAgent()

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=pick_voice(sdr),
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
        logger.info(f"LearnMate SDR usage summary: {usage.get_summary()}")

    ctx.add_shutdown_callback(show_usage)

    await session.start(
        agent=sdr,
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

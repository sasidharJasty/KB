

import math
import os
import re
from typing import Literal, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from langchain.messages import SystemMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END

load_dotenv()


# =====================================================================
# 1. Logic inference engine (unchanged — this is the deterministic core)
# =====================================================================

class Predicate:
    def __init__(self, name, args):
        self.name = name
        self.args = args

    def __repr__(self):
        return f"{self.name}({', '.join(self.args)})"

    def is_variable(self, value):
        return value[0].islower()

    def __eq__(self, other):
        return (
            isinstance(other, Predicate)
            and self.name == other.name
            and self.args == other.args
        )

    def __hash__(self):
        return hash((self.name, tuple(self.args)))

    def match(self, other):
        """Try to match self (possibly containing variables) against other
        (assumed ground). Returns a substitution dict or None."""
        substitution = {}

        if not isinstance(other, Predicate):
            return None
        if self.name != other.name:
            return None
        if len(self.args) != len(other.args):
            return None

        for arg, other_arg in zip(self.args, other.args):
            if self.is_variable(arg):
                if arg in substitution:
                    if substitution[arg] != other_arg:
                        return None
                else:
                    substitution[arg] = other_arg
            else:
                if arg != other_arg:
                    return None

        return substitution

    def substitute(self, substitution):
        new_args = [substitution.get(arg, arg) for arg in self.args]
        return Predicate(self.name, new_args)


class Rule:
    def __init__(self, antecedent, consequent):
        self.antecedent = antecedent if isinstance(antecedent, list) else [antecedent]
        self.consequent = consequent

    def __repr__(self):
        ante = " AND ".join(str(a) for a in self.antecedent)
        return f"{ante} -> {self.consequent}"


class KnowledgeBase:
    def __init__(self):
        self.rules = []
        self.facts = []

    def add_rule(self, rule):
        self.rules.append(rule)

    def add_fact(self, fact):
        self.facts.append(fact)

    def __repr__(self):
        return f"Rules: {self.rules}\nFacts: {self.facts}"


def build_space_kb():
    KB = KnowledgeBase()

    # ---- Rules ----

    KB.add_rule(Rule(
        [Predicate("Orbits", ["moon", "planet"]),
         Predicate("Orbits", ["planet", "star"])],
        Predicate("InSystem", ["moon", "star"]),
    ))

    KB.add_rule(Rule(
        Predicate("Orbits", ["planet", "star"]),
        Predicate("InSystem", ["planet", "star"]),
    ))

    KB.add_rule(Rule(
        [Predicate("InSystem", ["object", "star"]),
         Predicate("MainSequence", ["star"])],
        Predicate("PlanetaryObject", ["object"]),
    ))

    KB.add_rule(Rule(
        [Predicate("PlanetaryObject", ["object"]),
         Predicate("Rocky", ["object"])],
        Predicate("TerrestrialObject", ["object"]),
    ))

    KB.add_rule(Rule(
        [Predicate("PlanetaryObject", ["object"]),
         Predicate("GasGiant", ["object"])],
        Predicate("GiantPlanet", ["object"]),
    ))

    KB.add_rule(Rule(
        [Predicate("HasLiquidWater", ["planet"]),
         Predicate("HabitableZone", ["planet"])],
        Predicate("PotentiallyHabitable", ["planet"]),
    ))

    KB.add_rule(Rule(
        [Predicate("PotentiallyHabitable", ["planet"]),
         Predicate("HasAtmosphere", ["planet"])],
        Predicate("CanSupportLife", ["planet"]),
    ))

    KB.add_rule(Rule(
        [Predicate("CanSupportLife", ["planet"]),
         Predicate("StableOrbit", ["planet"])],
        Predicate("LongTermHabitable", ["planet"]),
    ))

    KB.add_rule(Rule(
        [Predicate("HasOxygen", ["planet"]),
         Predicate("HasLiquidWater", ["planet"])],
        Predicate("SupportsComplexLife", ["planet"]),
    ))

    KB.add_rule(Rule(
        Predicate("HasAtmosphere", ["object"]),
        Predicate("AtmosphericBody", ["object"]),
    ))

    KB.add_rule(Rule(
        [Predicate("AtmosphericBody", ["object"]),
         Predicate("DenseAtmosphere", ["object"])],
        Predicate("HighPressureWorld", ["object"]),
    ))

    KB.add_rule(Rule(
        [Predicate("HasAtmosphere", ["object"]),
         Predicate("PlanetaryObject", ["object"])],
        Predicate("CanSupportWeather", ["object"]),
    ))

    KB.add_rule(Rule(
        [Predicate("HasCarbonDioxide", ["planet"]),
         Predicate("AtmospherePresent", ["planet"])],
        Predicate("GreenhouseEffect", ["planet"]),
    ))

    KB.add_rule(Rule(
        [Predicate("Orbits", ["planet", "star"]),
         Predicate("StableOrbit", ["planet"])],
        Predicate("StablePlanetarySystem", ["planet"]),
    ))

    KB.add_rule(Rule(
        Predicate("FartherFrom", ["planet", "star"]),
        Predicate("OuterPlanet", ["planet"]),
    ))

    KB.add_rule(Rule(
        [Predicate("OuterPlanet", ["planet"]),
         Predicate("GasGiant", ["planet"])],
        Predicate("ColdWorld", ["planet"]),
    ))

    KB.add_rule(Rule(
        [Predicate("NearerTo", ["planet", "star"]),
         Predicate("Rocky", ["planet"])],
        Predicate("InnerPlanet", ["planet"]),
    ))

    KB.add_rule(Rule(
        Predicate("Rocky", ["planet"]),
        Predicate("TerrestrialPlanet", ["planet"]),
    ))

    KB.add_rule(Rule(
        Predicate("GasGiant", ["planet"]),
        Predicate("LargePlanet", ["planet"]),
    ))

    KB.add_rule(Rule(
        Predicate("IceGiant", ["planet"]),
        Predicate("OuterPlanet", ["planet"]),
    ))

    KB.add_rule(Rule(
        [Predicate("LargePlanet", ["planet"]),
         Predicate("HasRings", ["planet"])],
        Predicate("RingedGiant", ["planet"]),
    ))

    KB.add_rule(Rule(
        Predicate("Orbits", ["moon", "planet"]),
        Predicate("NaturalSatellite", ["moon"]),
    ))

    KB.add_rule(Rule(
        [Predicate("NaturalSatellite", ["moon"]),
         Predicate("HasIce", ["moon"])],
        Predicate("PotentialOceanWorld", ["moon"]),
    ))

    KB.add_rule(Rule(
        [Predicate("PotentialOceanWorld", ["moon"]),
         Predicate("HasSubsurfaceOcean", ["moon"])],
        Predicate("PossibleAlienHabitat", ["moon"]),
    ))

    KB.add_rule(Rule(
        Predicate("HasVolcanoes", ["planet"]),
        Predicate("GeologicallyActive", ["planet"]),
    ))

    KB.add_rule(Rule(
        [Predicate("GeologicallyActive", ["planet"]),
         Predicate("HasAtmosphere", ["planet"])],
        Predicate("ChangingPlanet", ["planet"]),
    ))

    KB.add_rule(Rule(
        Predicate("HasTectonicActivity", ["planet"]),
        Predicate("GeologicallyDynamic", ["planet"]),
    ))

    KB.add_rule(Rule(
        Predicate("MainSequence", ["star"]),
        Predicate("StableStar", ["star"]),
    ))

    KB.add_rule(Rule(
        [Predicate("StableStar", ["star"]),
         Predicate("ProvidesEnergy", ["star"])],
        Predicate("SupportsPlanetarySystems", ["star"]),
    ))

    KB.add_rule(Rule(
        Predicate("RedGiant", ["star"]),
        Predicate("OldStar", ["star"]),
    ))

    KB.add_rule(Rule(
        Predicate("NeutronStar", ["star"]),
        Predicate("ExtremeObject", ["star"]),
    ))

    KB.add_rule(Rule(
        Predicate("LargeMass", ["object"]),
        Predicate("StrongGravity", ["object"]),
    ))

    KB.add_rule(Rule(
        [Predicate("StrongGravity", ["planet"]),
         Predicate("HasAtmosphere", ["planet"])],
        Predicate("RetainsAtmosphere", ["planet"]),
    ))

    # ---- Facts ----

    KB.add_fact(Predicate("Orbits", ["Earth", "Sun"]))
    KB.add_fact(Predicate("Orbits", ["Mars", "Sun"]))
    KB.add_fact(Predicate("Orbits", ["Jupiter", "Sun"]))
    KB.add_fact(Predicate("Orbits", ["Saturn", "Sun"]))
    KB.add_fact(Predicate("Orbits", ["Uranus", "Sun"]))
    KB.add_fact(Predicate("Orbits", ["Neptune", "Sun"]))
    KB.add_fact(Predicate("Orbits", ["Mercury", "Sun"]))
    KB.add_fact(Predicate("Orbits", ["Venus", "Sun"]))

    KB.add_fact(Predicate("Orbits", ["Moon", "Earth"]))
    KB.add_fact(Predicate("Orbits", ["Europa", "Jupiter"]))
    KB.add_fact(Predicate("Orbits", ["Ganymede", "Jupiter"]))
    KB.add_fact(Predicate("Orbits", ["Titan", "Saturn"]))
    KB.add_fact(Predicate("Orbits", ["Enceladus", "Saturn"]))
    KB.add_fact(Predicate("Orbits", ["Triton", "Neptune"]))

    KB.add_fact(Predicate("MainSequence", ["Sun"]))
    KB.add_fact(Predicate("ProvidesEnergy", ["Sun"]))

    KB.add_fact(Predicate("Rocky", ["Mercury"]))
    KB.add_fact(Predicate("Rocky", ["Venus"]))
    KB.add_fact(Predicate("Rocky", ["Earth"]))
    KB.add_fact(Predicate("Rocky", ["Mars"]))

    KB.add_fact(Predicate("GasGiant", ["Jupiter"]))
    KB.add_fact(Predicate("GasGiant", ["Saturn"]))

    KB.add_fact(Predicate("IceGiant", ["Uranus"]))
    KB.add_fact(Predicate("IceGiant", ["Neptune"]))

    KB.add_fact(Predicate("HasLiquidWater", ["Earth"]))
    KB.add_fact(Predicate("HabitableZone", ["Earth"]))
    KB.add_fact(Predicate("StableOrbit", ["Earth"]))

    KB.add_fact(Predicate("StableOrbit", ["Mars"]))

    KB.add_fact(Predicate("HasOxygen", ["Earth"]))
    KB.add_fact(Predicate("HasAtmosphere", ["Earth"]))

    KB.add_fact(Predicate("HasAtmosphere", ["Mars"]))
    KB.add_fact(Predicate("HasAtmosphere", ["Venus"]))

    KB.add_fact(Predicate("DenseAtmosphere", ["Venus"]))
    KB.add_fact(Predicate("HasCarbonDioxide", ["Venus"]))
    KB.add_fact(Predicate("AtmospherePresent", ["Venus"]))

    KB.add_fact(Predicate("FartherFrom", ["Jupiter", "Sun"]))
    KB.add_fact(Predicate("FartherFrom", ["Saturn", "Sun"]))
    KB.add_fact(Predicate("FartherFrom", ["Neptune", "Sun"]))

    KB.add_fact(Predicate("NearerTo", ["Mercury", "Sun"]))
    KB.add_fact(Predicate("NearerTo", ["Venus", "Sun"]))

    KB.add_fact(Predicate("HasRings", ["Saturn"]))
    KB.add_fact(Predicate("HasRings", ["Jupiter"]))

    KB.add_fact(Predicate("HasVolcanoes", ["Io"]))
    KB.add_fact(Predicate("HasTectonicActivity", ["Earth"]))

    KB.add_fact(Predicate("HasIce", ["Europa"]))
    KB.add_fact(Predicate("HasSubsurfaceOcean", ["Europa"]))

    KB.add_fact(Predicate("HasIce", ["Enceladus"]))
    KB.add_fact(Predicate("HasSubsurfaceOcean", ["Enceladus"]))

    KB.add_fact(Predicate("LargeMass", ["Jupiter"]))
    KB.add_fact(Predicate("LargeMass", ["Saturn"]))

    return KB


class ProofResult:
    def __init__(self, proved, substitution, trace):
        self.proved = proved
        self.substitution = substitution
        self.trace = trace

    def __bool__(self):
        return self.proved

    def pretty_trace(self):
        lines = []
        for i, step in enumerate(self.trace, 1):
            if step["type"] == "fact_match":
                lines.append(
                    f"{i}. Goal {step['goal']} matched known fact "
                    f"{step['fact']} (substitution={step['substitution']})"
                )
            elif step["type"] == "rule_fire":
                ante = " AND ".join(str(a) for a in step["rule"].antecedent)
                lines.append(
                    f"{i}. Goal {step['goal']} unifies with the consequent of "
                    f"rule [{ante} -> {step['rule'].consequent}] "
                    f"(substitution={step['substitution']}); "
                    f"new subgoals: {step['subgoals']}"
                )
            elif step["type"] == "fail":
                lines.append(f"{i}. Goal {step['goal']} could not be proved.")
        return "\n".join(lines)


def query(KB, goal):
    trace = []
    substitution = back_chain(KB, goal, trace)
    return ProofResult(substitution is not None, substitution, trace)


def prove_all(KB, goals, substitution, trace):
    if len(goals) == 0:
        return substitution

    current_goal = goals[0].substitute(substitution)
    result = back_chain(KB, current_goal, trace)

    if result is None:
        return None

    new_substitution = substitution.copy()
    new_substitution.update(result)

    remaining_goals = [g.substitute(new_substitution) for g in goals[1:]]

    return prove_all(KB, remaining_goals, new_substitution, trace)


def back_chain(KB, goal, trace):
    for fact in KB.facts:
        substitution = goal.match(fact)
        if substitution is not None:
            trace.append({
                "type": "fact_match",
                "goal": goal,
                "fact": fact,
                "substitution": substitution,
            })
            return substitution

    for rule in KB.rules:
        substitution = rule.consequent.match(goal)
        if substitution is not None:
            new_goals = [a.substitute(substitution) for a in rule.antecedent]
            trace.append({
                "type": "rule_fire",
                "goal": goal,
                "rule": rule,
                "substitution": substitution,
                "subgoals": new_goals,
            })
            result = prove_all(KB, new_goals, substitution, trace)
            if result is not None:
                return result

    trace.append({"type": "fail", "goal": goal})
    return None


# =====================================================================
# 2. Retrieval (unchanged — plain RAG over facts/rules, no LLM involved)
# =====================================================================

TOKEN_RE = re.compile(r"[A-Za-z]+")


def predicate_to_text(predicate):
    name = predicate.name
    args = predicate.args

    def get_arg(index, default="unknown"):
        if len(args) > index:
            return args[index]
        return default

    templates = {
        "HasAtmosphere": lambda: f"{get_arg(0)} has an atmosphere.",
        "HasLiquidWater": lambda: f"{get_arg(0)} has liquid water.",
        "HasIce": lambda: f"{get_arg(0)} contains ice.",
        "Orbits": lambda: f"{get_arg(0)} orbits {get_arg(1)}.",
        "Rocky": lambda: f"{get_arg(0)} is a rocky object.",
        "GasGiant": lambda: f"{get_arg(0)} is a gas giant.",
        "IceGiant": lambda: f"{get_arg(0)} is an ice giant.",
        "HabitableZone": lambda: f"{get_arg(0)} is inside the habitable zone.",
        "HasOxygen": lambda: f"{get_arg(0)} has oxygen.",
        "HasCarbonDioxide": lambda: f"{get_arg(0)} has carbon dioxide.",
        "HasRings": lambda: f"{get_arg(0)} has rings.",
        "HasSubsurfaceOcean": lambda: f"{get_arg(0)} has a subsurface ocean.",
        "StableOrbit": lambda: f"{get_arg(0)} has a stable orbit.",
        "MainSequence": lambda: f"{get_arg(0)} is a main sequence star.",
        "ProvidesEnergy": lambda: f"{get_arg(0)} provides energy.",
        "LargeMass": lambda: f"{get_arg(0)} has a large mass.",
        "HasVolcanoes": lambda: f"{get_arg(0)} has volcanoes.",
        "HasTectonicActivity": lambda: f"{get_arg(0)} has tectonic activity.",
    }

    if name in templates:
        return templates[name]()

    if len(args) == 0:
        return f"{name} is true."

    if len(args) == 1:
        return f"{get_arg(0)} has property {name}."

    return f"{', '.join(args)} have relationship {name}."


def normalize_query(question):
    q = question.lower()

    if "atmosphere" in q:
        return "HasAtmosphere planets"

    if "liquid water" in q:
        return "HasLiquidWater planets"

    if "water" in q:
        return "HasLiquidWater HasIce HasSubsurfaceOcean"

    if "moon" in q:
        return "NaturalSatellite"

    return question


def _tokenize(text):
    return [t.lower() for t in TOKEN_RE.findall(text)]


class SimpleHashingEmbeddings(Embeddings):
    DIM = 4096

    def _embed(self, text):
        vec = [0.0] * self.DIM
        for tok in _tokenize(text):
            idx = hash(tok) % self.DIM
            vec[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)


def kb_to_documents(KB):
    docs = []

    for fact in KB.facts:
        docs.append(
            Document(
                page_content=(
                    predicate_to_text(fact) + " Logical representation: " + str(fact)
                ),
                metadata={
                    "kind": "fact",
                    "predicate": fact.name,
                    "entity": fact.args[0],
                },
            )
        )

    for rule in KB.rules:
        antecedent = " AND ".join(predicate_to_text(x) for x in rule.antecedent)
        docs.append(
            Document(
                page_content=(
                    f"If {antecedent}, then {predicate_to_text(rule.consequent)}\n"
                    f"Logical rule: {rule}"
                ),
                metadata={"kind": "rule", "predicate": rule.consequent.name},
            )
        )

    return docs


def build_vectorstore(KB):
    docs = kb_to_documents(KB)
    embeddings = SimpleHashingEmbeddings()
    return FAISS.from_documents(docs, embeddings)


def build_retriever(KB, k=10):
    store = build_vectorstore(KB)
    return store.as_retriever(search_kwargs={"k": k})


def rag_context(retriever, message):
    docs = retriever.invoke(normalize_query(message))
    facts = [d for d in docs if d.metadata["kind"] == "fact"]
    rules = [d for d in docs if d.metadata["kind"] == "rule"]

    fact_text = "\n".join(f"- {d.page_content}" for d in facts)
    rule_text = "\n".join(f"- {d.page_content}" for d in rules)

    return f"Known facts:\n{fact_text}\n\nKnown rules:\n{rule_text}"


_QUERY_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\(([^)]*)\)")


def parse_predicate_string(text):
    """Parse a 'Name(Arg1,Arg2)' string into a Predicate object.
    Also tolerates a bare predicate name with no parentheses (used for the
    enumeration branch), returning an arg-less Predicate in that case."""
    text = text.strip()
    match = _QUERY_RE.search(text)
    if match:
        name = match.group(1)
        args = [a.strip() for a in match.group(2).split(",") if a.strip()]
        return Predicate(name, args)

    # bare name, e.g. "HasAtmosphere" for enumeration
    name_match = re.match(r"[A-Za-z_][A-Za-z0-9_]*", text)
    if not name_match:
        raise ValueError(f"Could not parse a predicate from LLM output: {text!r}")
    return Predicate(name_match.group(0), [])


# =====================================================================
# 3. Build the KB + retriever once at module load
# =====================================================================

KB = build_space_kb()
retriever = build_retriever(KB, k=10)


# =====================================================================
# 4. LLM setup — Gemini is only used for two structured calls
# =====================================================================

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)

FORMULATOR_SYSTEM_PROMPT = """You are the query-formulation component of a \
Logic-LM reasoning system. You do not answer questions and you do not reason \
yourself — you only translate a natural-language question, using the \
retrieved knowledge-base context, into ONE structured decision.

Decide:
- query_type = "yes_no" if the question asks whether a single specific \
statement about a named entity is true (e.g. "does Mars have an \
atmosphere?", "does Venus orbit Earth?").
- query_type = "enumerate" if the question asks for all entities that \
satisfy some property or relationship (e.g. "what planets have an \
atmosphere?", "what orbits the Sun?").

Then set `predicate`:
- If query_type is "yes_no": a single ground predicate string using the \
EXACT predicate and entity names found in the knowledge-base context, \
formatted like "HasAtmosphere(Mars)" or "Orbits(Venus,Earth)".
- If query_type is "enumerate": just the bare predicate name with no \
parentheses and no arguments, e.g. "HasAtmosphere".

Rules:
- Only use predicate names and entity names that literally appear in the \
knowledge-base context below. Never invent a predicate or entity.
- Never include explanations, reasoning, or extra text — only the \
structured fields you are asked for.
"""

INTERPRETER_SYSTEM_PROMPT = """You are the final explanation component of a \
symbolic reasoning system.

You are given a logical query, a TRUE/FALSE (or list) result, and a proof \
trace or list of matching entities.

Explain the result in natural language.

Requirements:
- Do not add outside information.
- Do not contradict the result.
- Mention the specific facts and rules used.
- If FALSE / empty, explain that no proof exists / nothing matched.
- Keep the explanation between 2-5 sentences.
"""


class FormulatedQuery(BaseModel):
    query_type: Literal["yes_no", "enumerate"] = Field(
        description="'yes_no' for a single ground-fact question, "
        "'enumerate' for a question asking for all matching entities."
    )
    predicate: str = Field(
        description="For yes_no: 'PredicateName(Entity)' or "
        "'PredicateName(Entity1,Entity2)'. For enumerate: just "
        "'PredicateName' with no parentheses."
    )


formulator_model = model.with_structured_output(FormulatedQuery)


# =====================================================================
# 5. Graph state
# =====================================================================

class GraphState(TypedDict, total=False):
    question: str
    want_explanation: bool

    rag_context: str

    query_type: Literal["yes_no", "enumerate"]
    predicate_str: str

    # yes_no branch
    proved: bool
    trace: str
    result_str: str

    # enumerate branch
    entities: list

    explanation: str


# =====================================================================
# 6. Nodes
# =====================================================================

def retrieve_node(state: GraphState) -> dict:
    """Deterministic. Always runs first — no LLM decision involved."""
    context = rag_context(retriever, state["question"])
    return {"rag_context": context}


def formulate_node(state: GraphState) -> dict:
    """The ONLY place Gemini makes a decision: turn the question + context
    into a structured {query_type, predicate} object."""
    messages = [
        SystemMessage(content=FORMULATOR_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Question: {state['question']}\n\n"
                f"Knowledge-base context:\n{state['rag_context']}"
            )
        ),
    ]
    formulated: FormulatedQuery = formulator_model.invoke(messages)
    return {
        "query_type": formulated.query_type,
        "predicate_str": formulated.predicate,
    }


def reason_node(state: GraphState) -> dict:
    """Deterministic backward-chaining node (was `logic_reasoner` tool)."""
    goal = parse_predicate_string(state["predicate_str"])
    result = query(KB, goal)
    return {
        "proved": result.proved,
        "trace": result.pretty_trace(),
        "result_str": "TRUE" if result.proved else "FALSE",
    }


def enumerate_node(state: GraphState) -> dict:
    """Deterministic KB scan (was `find_entities` tool)."""
    predicate = parse_predicate_string(state["predicate_str"])
    answers = [fact.args[0] for fact in KB.facts if fact.name == predicate.name]
    return {"entities": answers}


def explain_node(state: GraphState) -> dict:
    """Optional final LLM call — natural-language wrap-up only."""
    if state.get("query_type") == "yes_no":
        summary = (
            f"Query: {state['predicate_str']}\n"
            f"Result: {state['result_str']}\n"
            f"Proof trace:\n{state['trace']}"
        )
    else:
        entities = state.get("entities", [])
        summary = (
            f"Predicate: {state['predicate_str']}\n"
            f"Matching entities: {entities if entities else 'none'}"
        )

    messages = [
        SystemMessage(content=INTERPRETER_SYSTEM_PROMPT),
        HumanMessage(content=summary),
    ]
    response: AIMessage = model.invoke(messages)
    return {"explanation": response.content}


# =====================================================================
# 7. Conditional routing
# =====================================================================

def route_after_formulate(state: GraphState) -> str:
    return "reason" if state["query_type"] == "yes_no" else "enumerate"


def route_after_answer(state: GraphState) -> str:
    return "explain" if state.get("want_explanation", True) else "end"


# =====================================================================
# 8. Build the graph
# =====================================================================

graph = StateGraph(GraphState)

graph.add_node("retrieve", retrieve_node)
graph.add_node("formulate", formulate_node)
graph.add_node("reason", reason_node)
graph.add_node("enumerate", enumerate_node)
graph.add_node("explain", explain_node)

graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "formulate")

graph.add_conditional_edges(
    "formulate",
    route_after_formulate,
    {"reason": "reason", "enumerate": "enumerate"},
)

graph.add_conditional_edges(
    "reason",
    route_after_answer,
    {"explain": "explain", "end": END},
)

graph.add_conditional_edges(
    "enumerate",
    route_after_answer,
    {"explain": "explain", "end": END},
)

graph.add_edge("explain", END)

agent = graph.compile()


# =====================================================================
# 9. Example invocations
# =====================================================================

if __name__ == "__main__":
    # Optional: visualize the graph if running in a notebook
    try:
        from IPython.display import Image, display
        display(Image(agent.get_graph(xray=True).draw_mermaid_png()))
    except Exception:
        pass

    print("\n--- yes/no question ---")
    out1 = agent.invoke({
        "question": "does venus orbit earth?",
        "want_explanation": True,
    })
    print("query_type:", out1["query_type"])
    print("predicate:", out1["predicate_str"])
    print("result:", out1["result_str"])
    print("trace:\n", out1["trace"])
    print("explanation:", out1.get("explanation"))

    print("\n--- enumeration question ---")
    out2 = agent.invoke({
        "question": "what planets have an atmosphere?",
        "want_explanation": True,
    })
    print("query_type:", out2["query_type"])
    print("predicate:", out2["predicate_str"])
    print("entities:", out2["entities"])
    print("explanation:", out2.get("explanation"))

    print("\n--- enumeration question, no explanation ---")
    out3 = agent.invoke({
        "question": "what orbits jupiter?",
        "want_explanation": False,
    })
    print("entities:", out3["entities"])
    print("explanation present:", "explanation" in out3)

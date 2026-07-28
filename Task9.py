import math
import os
import re
import sys


FORMULATOR_SYSTEM_PROMPT = FORMULATOR_SYSTEM_PROMPT = """
You are a Logic-LM reasoning agent.

Your job is to translate natural language questions into symbolic reasoning operations.

You NEVER answer questions directly.
You ONLY use the tools provided.

Available tools:

1. search_knowledge_base(statement)

Purpose:
- Retrieve relevant facts and rules from the knowledge base.
- Always call this first for every user question.

Example:
User:
"What planets have atmospheres?"

Call:
search_knowledge_base("planets that have atmosphere")


2. identify_goal(predicate)

Purpose:
- Converts a specific claim into a symbolic Predicate object.
- Used ONLY when the user asks about one specific entity.

Examples:
Question:
"Does Venus orbit Earth?"

Correct tool call:

identify_goal("Orbits(Venus,Earth)")


Question:
"Does Jupiter have rings?"

Correct tool call:

identify_goal("HasRings(Jupiter)")


Question:
"Is Saturn a gas giant?"

Correct tool call:

identify_goal("GasGiant(Saturn)")

User:
"Does Mars have an atmosphere?"

Call:
identify_goal("HasAtmosphere(Mars)")


User:
"Is Earth habitable?"

Call:
identify_goal("PotentiallyHabitable(Earth)")


After identifying the goal:
Call logic_reasoner with the returned predicate.

logic_reasoner performs backward chaining proof.


3. find_entities(predicate_name)

Purpose:
- Finds ALL entities in the knowledge base that satisfy a predicate.
- Used for questions asking:
    - "what things have..."
    - "which planets..."
    - "list all..."
    - "what objects..."
    - "who has..."

Examples:

User:
"What planets have an atmosphere?"

Do NOT call:
identify_goal("HasAtmosphere(X)")

because backward chaining cannot enumerate variables.

Instead call:

find_entities("HasAtmosphere")


User:
"What objects orbit the Sun?"

Call:

find_entities("Orbits")


IMPORTANT LOGIC:

The reasoning engine currently supports backward chaining.

Backward chaining works like:

Goal:
HasAtmosphere(Mars)

Search:
1. Facts
2. Rules that produce HasAtmosphere

It proves whether one specific statement is TRUE or FALSE.


Backward chaining does NOT perform variable discovery.

Therefore:

INVALID:
HasAtmosphere(X)

VALID:
HasAtmosphere(Earth)


For questions asking for all matching entities:
Use find_entities.


Decision process:

Step 1:
Always call search_knowledge_base.

Step 2:
Determine the question type.

If the question asks whether a specific statement is true:

Example:
"Does Mars have an atmosphere?"

Use:
identify_goal("HasAtmosphere(Mars)")
then:
logic_reasoner


If the question asks for multiple answers:

Example:
"What planets have an atmosphere?"

Use:
find_entities("HasAtmosphere")


If uncertain:
Look at retrieved facts and use the exact predicate names from the knowledge base.

Rules:
- Never invent predicates.
- Never invent entities.
- Never use outside knowledge.
- Never call logic_reasoner with natural language.
- Never call logic_reasoner with variables like X.
- Always use the tools before responding.
"""

INTERPRETER_SYSTEM_PROMPT = """
You are the final explanation component of a symbolic reasoning system.

You are given:
- A logical query
- A TRUE/FALSE result
- A proof trace

Explain the result in natural language.

Requirements:
- Do not add outside information.
- Do not contradict the proof result.
- Mention the facts and rules used.
- If FALSE, explain that no proof exists.
- Keep the explanation between 2-5 sentences.
"""

#logic inferance engine

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

    # Orbital system rules

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


    # Habitability rules

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


    # Atmosphere rules

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


    # Orbital stability

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


    # Planet classification

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


    # Moon rules

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


    # Geological rules

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


    # Star rules

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


    # Gravity rules

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


    # Solar system planets

    KB.add_fact(Predicate("Orbits", ["Earth", "Sun"]))
    KB.add_fact(Predicate("Orbits", ["Mars", "Sun"]))
    KB.add_fact(Predicate("Orbits", ["Jupiter", "Sun"]))
    KB.add_fact(Predicate("Orbits", ["Saturn", "Sun"]))
    KB.add_fact(Predicate("Orbits", ["Uranus", "Sun"]))
    KB.add_fact(Predicate("Orbits", ["Neptune", "Sun"]))
    KB.add_fact(Predicate("Orbits", ["Mercury", "Sun"]))
    KB.add_fact(Predicate("Orbits", ["Venus", "Sun"]))


    # Moons

    KB.add_fact(Predicate("Orbits", ["Moon", "Earth"]))
    KB.add_fact(Predicate("Orbits", ["Europa", "Jupiter"]))
    KB.add_fact(Predicate("Orbits", ["Ganymede", "Jupiter"]))
    KB.add_fact(Predicate("Orbits", ["Titan", "Saturn"]))
    KB.add_fact(Predicate("Orbits", ["Enceladus", "Saturn"]))
    KB.add_fact(Predicate("Orbits", ["Triton", "Neptune"]))


    # Stars

    KB.add_fact(Predicate("MainSequence", ["Sun"]))
    KB.add_fact(Predicate("ProvidesEnergy", ["Sun"]))


    # Planet types

    KB.add_fact(Predicate("Rocky", ["Mercury"]))
    KB.add_fact(Predicate("Rocky", ["Venus"]))
    KB.add_fact(Predicate("Rocky", ["Earth"]))
    KB.add_fact(Predicate("Rocky", ["Mars"]))

    KB.add_fact(Predicate("GasGiant", ["Jupiter"]))
    KB.add_fact(Predicate("GasGiant", ["Saturn"]))

    KB.add_fact(Predicate("IceGiant", ["Uranus"]))
    KB.add_fact(Predicate("IceGiant", ["Neptune"]))


    # Habitability

    KB.add_fact(Predicate("HasLiquidWater", ["Earth"]))
    KB.add_fact(Predicate("HabitableZone", ["Earth"]))
    KB.add_fact(Predicate("StableOrbit", ["Earth"]))

    KB.add_fact(Predicate("StableOrbit", ["Mars"]))

    KB.add_fact(Predicate("HasOxygen", ["Earth"]))
    KB.add_fact(Predicate("HasAtmosphere", ["Earth"]))


    # Atmospheres

    KB.add_fact(Predicate("HasAtmosphere", ["Mars"]))
    KB.add_fact(Predicate("HasAtmosphere", ["Venus"]))

    KB.add_fact(Predicate("DenseAtmosphere", ["Venus"]))
    KB.add_fact(Predicate("HasCarbonDioxide", ["Venus"]))
    KB.add_fact(Predicate("AtmospherePresent", ["Venus"]))


    # Distance

    KB.add_fact(Predicate("FartherFrom", ["Jupiter", "Sun"]))
    KB.add_fact(Predicate("FartherFrom", ["Saturn", "Sun"]))
    KB.add_fact(Predicate("FartherFrom", ["Neptune", "Sun"]))

    KB.add_fact(Predicate("NearerTo", ["Mercury", "Sun"]))
    KB.add_fact(Predicate("NearerTo", ["Venus", "Sun"]))


    # Rings

    KB.add_fact(Predicate("HasRings", ["Saturn"]))
    KB.add_fact(Predicate("HasRings", ["Jupiter"]))


    # Geological facts

    KB.add_fact(Predicate("HasVolcanoes", ["Io"]))
    KB.add_fact(Predicate("HasTectonicActivity", ["Earth"]))

    KB.add_fact(Predicate("HasIce", ["Europa"]))
    KB.add_fact(Predicate("HasSubsurfaceOcean", ["Europa"]))

    KB.add_fact(Predicate("HasIce", ["Enceladus"]))
    KB.add_fact(Predicate("HasSubsurfaceOcean", ["Enceladus"]))


    # Mass

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



Back_chain = back_chain




from langchain.agents import create_agent
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

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

    # Generic fallback for unknown predicates
    if len(args) == 0:
        return f"{name} is true."

    if len(args) == 1:
        return f"{get_arg(0)} has property {name}."

    return (
        f"{', '.join(args)} have relationship {name}."
    )

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
                    predicate_to_text(fact)
                    + " Logical representation: "
                    + str(fact)
                ),
                metadata={
                    "kind":"fact",
                    "predicate":fact.name,
                    "entity":fact.args[0]
                }
            )
        )


    for rule in KB.rules:

        antecedent = " AND ".join(
            predicate_to_text(x)
            for x in rule.antecedent
        )

        docs.append(
            Document(
                page_content=(
                    f"If {antecedent}, "
                    f"then {predicate_to_text(rule.consequent)}\n"
                    f"Logical rule: {rule}"
                ),
                metadata={
                    "kind":"rule",
                    "predicate":rule.consequent.name
                }
            )
        )

    return docs


def build_vectorstore(KB):
    docs = kb_to_documents(KB)
    embeddings = SimpleHashingEmbeddings()
    return FAISS.from_documents(docs, embeddings)

def build_retriever(KB,k=10):
    store = build_vectorstore(KB)
    return store.as_retriever(
        search_kwargs={
            "k":k
        }
    )

def _rag_context(message):

    docs = retriever.invoke(
        normalize_query(message)
    )

    facts = [
        d for d in docs
        if d.metadata["kind"]=="fact"
    ]

    return {
        "rag_context":
        "\n".join(
            f"- {d.page_content}"
            for d in facts
        )
    }


QUERY_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\(([^)]*)\)")


def parse_predicate_string(text):
    """Parse a 'Name(Arg1,Arg2)' string into a Predicate object."""
    match = QUERY_RE.search(text.strip())
    if not match:
        raise ValueError(f"Could not parse a predicate from LLM output: {text!r}")
    name = match.group(1)
    args = [a.strip() for a in match.group(2).split(",") if a.strip()]
    return Predicate(name, args)


_CAMEL_RE = re.compile(r"[A-Z][a-z]*")
_STOPWORDS = {
    "is", "the", "a", "an", "of", "same", "and",
    "or", "does", "do", "did", "will", "would",
    "with", "for", "to", "that", "this", "are", "was", "were", "it",
}


def _camel_words(name):
    return [w.lower() for w in _CAMEL_RE.findall(name)] or [name.lower()]


KB = build_space_kb()

retriever = build_retriever(KB, k=10)

from langchain.messages import AnyMessage
from typing_extensions import TypedDict, Annotated
import operator

class GraphState(TypedDict, total=False):

    question: str

    rag_context: str


    formulated_query: str
    goal: Predicate

    proved: bool
    trace: list[dict]
    result: str

    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int

    explanation: str

from langchain.messages import SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI


model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0
)


from langchain.tools import tool


@tool
def search_knowledge_base(statement: str):
    """
    Search the knowledge base for relevant facts and rules.
    """
    return _rag_context(statement)

@tool
def identify_goal(message: str):
    """
        Convert a natural language question into EXACTLY ONE symbolic predicate.

        IMPORTANT:
        The input MUST already be converted into predicate format.

        Valid:
        HasAtmosphere(Mars)
        Orbits(Venus,Earth)
        PotentiallyHabitable(Earth)

        Invalid:
        Does Mars have an atmosphere?
        Is Earth habitable?
        Does Venus orbit Earth?

        Never send natural language.
        """
    return parse_predicate_string(message)

@tool
def logic_reasoner(goal: str):
    """
    Perform deterministic backward chaining on a symbolic predicate.

    The input MUST be formatted as:
    Predicate(arg1,arg2,...)

    Example:
    HasAtmosphere(Earth)
    """

    predicate = parse_predicate_string(goal)

    result = query(KB, predicate)

    return {
        "proved": result.proved,
        "trace": result.pretty_trace(),
        "result": "TRUE" if result.proved else "FALSE",
        "query": str(predicate)
    }

@tool
def find_entities(predicate_name: str):
    """
    Find all entities that satisfy a predicate.
    """

    answers = []

    for fact in KB.facts:
        if fact.name == predicate_name:
            answers.append(fact.args[0])

    return answers


tools = [
    search_knowledge_base,
    identify_goal,
    logic_reasoner,
    find_entities
]

model_with_tools = model.bind_tools(tools)


from langgraph.prebuilt import ToolNode


tool_node = ToolNode(
    tools
)


def should_continue(state):

    last_message = state["messages"][-1]

    if last_message.tool_calls:
        return "tools"

    return "end"


def llm_call(state: dict):
    """LLM decides whether to call a tool or not"""

    print("\n========== LLM CALL DEBUG ==========")

    print("Messages entering LLM:")
    for m in state["messages"]:
        print(type(m).__name__)
        print(m.content)
        print("----------------")

    response = model_with_tools.invoke(
        [
            SystemMessage(
                content=FORMULATOR_SYSTEM_PROMPT
            )
        ]
        + state["messages"]
    )

    print("\nLLM RESPONSE DEBUG")
    print("Content:")
    print(response.content)

    print("Tool calls:")
    print(response.tool_calls)

    print("====================================\n")

    return {
        "messages": [
            response
        ],
        "llm_calls": state.get('llm_calls', 0) + 1
    }



from typing import Literal
from langgraph.graph import StateGraph, START, END

graph = StateGraph(GraphState)


graph.add_node(
    "agent",
    llm_call
)


graph.add_node(
    "tools",
    tool_node
)


graph.add_edge(
    START,
    "agent"
)


graph.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools":"tools",
        "end":END
    }
)


graph.add_edge(
    "tools",
    "agent"
)


agent = graph.compile()

# Show the agent
from IPython.display import Image, display
display(Image(agent.get_graph(xray=True).draw_mermaid_png()))

# Invoke
from langchain.messages import HumanMessage
messages = [HumanMessage(content="does venus orbit earth?")]
messages = agent.invoke({"messages": messages})
for m in messages["messages"]:
    m.pretty_print()

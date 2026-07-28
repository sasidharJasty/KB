

import math
import os
import re
import sys


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
        [Predicate("HasLiquidWater", ["planet"]),
         Predicate("HabitableZone", ["planet"])],
        Predicate("PotentiallyHabitable", ["planet"]),
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
        [Predicate("PlanetaryObject", ["object"]),
         Predicate("HasAtmosphere", ["object"])],
        Predicate("CanSupportWeather", ["object"]),
    ))

    # Facts
    KB.add_fact(Predicate("Orbits", ["Earth", "Sun"]))
    KB.add_fact(Predicate("Orbits", ["Mars", "Sun"]))
    KB.add_fact(Predicate("Orbits", ["Jupiter", "Sun"]))
    KB.add_fact(Predicate("Orbits", ["Moon", "Earth"]))
    KB.add_fact(Predicate("Orbits", ["Europa", "Jupiter"]))
    KB.add_fact(Predicate("Orbits", ["Titan", "Saturn"]))
    KB.add_fact(Predicate("Orbits", ["Saturn", "Sun"]))
    KB.add_fact(Predicate("MainSequence", ["Sun"]))
    KB.add_fact(Predicate("HasLiquidWater", ["Earth"]))
    KB.add_fact(Predicate("HabitableZone", ["Earth"]))
    KB.add_fact(Predicate("StableOrbit", ["Earth"]))
    KB.add_fact(Predicate("StableOrbit", ["Mars"]))
    KB.add_fact(Predicate("FartherFrom", ["Jupiter", "Sun"]))
    KB.add_fact(Predicate("HasAtmosphere", ["Earth"]))
    KB.add_fact(Predicate("HasAtmosphere", ["Mars"]))

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


def _tokenize(text):
    return [t.lower() for t in TOKEN_RE.findall(text)]


class SimpleHashingEmbeddings(Embeddings):
    DIM = 256

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
        docs.append(Document(
            page_content=f"Fact: {fact}",
            metadata={"kind": "fact", "predicate": fact.name, "repr": str(fact)},
        ))

    for rule in KB.rules:
        ante = " AND ".join(str(a) for a in rule.antecedent)
        docs.append(Document(
            page_content=f"Rule: IF {ante} THEN {rule.consequent}",
            metadata={
                "kind": "rule",
                "predicate": rule.consequent.name,
                "repr": f"{ante} -> {rule.consequent}",
            },
        ))

    return docs


def build_vectorstore(KB):
    docs = kb_to_documents(KB)
    embeddings = SimpleHashingEmbeddings()
    return FAISS.from_documents(docs, embeddings)


def build_retriever(KB, k=6):

    store = build_vectorstore(KB)
    return store.as_retriever(search_kwargs={"k": k})


def _rag_context(KB, question, retriever):
    docs = retriever.invoke(question)
    return "\n".join(f"- {d.page_content}" for d in docs)



from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

FORMULATOR_SYSTEM_PROMPT = """You are the Problem Formulator in a Logic-LM \
style pipeline. Convert the user's natural language question into EXACTLY \
one query in the form Predicate(Arg1,Arg2) using ONLY the predicate names \
and constant names that appear in the provided knowledge base context. \
Do not invent new predicates. Output only the query, nothing else. \
Example output: InSystem(Moon,Sun)"""

INTERPRETER_SYSTEM_PROMPT = """You are the Result Interpreter in a Logic-LM \
style pipeline. You are given a symbolic query, whether it was PROVED or \
NOT PROVED by a deterministic backward-chaining solver, and the exact \
step-by-step proof trace the solver produced. Write a short (2-4 sentence) \
natural-language explanation of the answer for the user, citing the key \
facts/rules from the trace. Do not contradict the solver's verdict."""

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


def _fallback_formulate(KB, question, context):
    m = QUERY_RE.search(question)
    if m:
        return question[m.start():m.end()]

    q_tokens = set(w.lower() for w in re.findall(r"[A-Za-z]+", question)) - _STOPWORDS

    predicate_names = sorted({r.consequent.name for r in KB.rules} |
                              {f.name for f in KB.facts})

    best_pred, best_score = None, -1
    for name in predicate_names:
        score = len(set(_camel_words(name)) & q_tokens)
        if score > best_score:
            best_pred, best_score = name, score

    constants = sorted({arg for f in KB.facts for arg in f.args})
    entities = [c for c in constants if c.lower() in question.lower()]

    if best_pred is None or not entities:
        raise ValueError(
            "Fallback formulator could not derive a query; set "
            "API_KEY to use the LLM formulator instead."
        )

    args = entities[:2]
    return f"{best_pred}({','.join(args)})"


def _fallback_interpret(query_str, proved, trace_text):
    verdict = "TRUE" if proved else "FALSE"
    return (
        f"The query {query_str} was determined to be {verdict}. "
        f"This was derived via backward chaining over the knowledge base; "
        f"see the proof trace for the exact facts and rules used:\n{trace_text}"
    )


def build_formulator_agent(model_name):
    return create_agent(
        model=model_name,
        tools=[],
        system_prompt=FORMULATOR_SYSTEM_PROMPT,
    )


def build_interpreter_agent(model_name):
    return create_agent(
        model=model_name,
        tools=[],
        system_prompt=INTERPRETER_SYSTEM_PROMPT,
    )


class LogicLMPipeline:

    def __init__(self, KB=None, model_name="google_genai:gemini-2.5-flash", k=6):
        self.KB = KB if KB is not None else build_space_kb()
        self.k = k
        self.use_llm = bool(os.environ.get("API_KEY"))
        self.retriever = build_retriever(self.KB, k=self.k)

        if self.use_llm:
            self.formulator = build_formulator_agent(model_name)
            self.interpreter = build_interpreter_agent(model_name)
        else:
            self.formulator = None
            self.interpreter = None

    def run(self, question):
        context = _rag_context(self.KB, question, self.retriever)

        # ---- Stage 1: Problem Formulator ----
        if self.use_llm:
            res = self.formulator.invoke(
                {"messages": [{"role": "user", "content": f"Knowledge base context:\n{context}\n\nQuestion: {question}"}]}
            )

            # Robust extraction of text from content_blocks
            blocks = res["messages"][-1].content_blocks
            if isinstance(blocks, list) and len(blocks) > 0:
                block = blocks[0]
                query_str = block.get("text", str(block)) if isinstance(block, dict) else str(block)
            else:
                query_str = str(blocks)

            query_str = query_str.strip()
        else:
            query_str = _fallback_formulate(self.KB, question, context)

        goal = parse_predicate_string(query_str)

        # ---- Stage 2: Symbolic Reasoner (deterministic, no LLM) ----
        result = query(self.KB, goal)
        trace_text = result.pretty_trace() or "(no derivation steps; query failed immediately)"
        verdict = "TRUE" if result.proved else "FALSE"

        # ---- Stage 3: Result Interpreter ----
        if self.use_llm:
            res = self.interpreter.invoke({
                "messages": [
                    {"role": "user", "content": f"Query: {str(goal)}\nVerdict: {verdict}\nProof trace:\n{trace_text}"}
                ]
            })

            blocks = res["messages"][-1].content_blocks
            if isinstance(blocks, list) and len(blocks) > 0:
                block = blocks[0]
                explanation = block.get("text", str(block)) if isinstance(block, dict) else str(block)
            else:
                explanation = str(blocks)
        else:
            explanation = _fallback_interpret(str(goal), result.proved, trace_text)

        return {
            "question": question,
            "rag_context": context,
            "formulated_query": str(goal),
            "proved": result.proved,
            "trace": result.trace,
            "trace_text": trace_text,
            "explanation": explanation,
            "used_llm": self.use_llm,
        }



DEFAULT_QUESTIONS = [
    "Is the Moon in the same star system as the Sun?",
    "Is the Moon in the same system as Mars?",
    "Is Earth potentially habitable?",
    "Is Jupiter an outer planet?",
    "Does Mars form a stable planetary system?",
]


def main():
    pipeline = LogicLMPipeline()
    questions = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_QUESTIONS

    print(f"[Formulator/Interpreter backend: "
          f"{' (LangChain)' if pipeline.use_llm else 'offline fallback (no API_KEY set)'}]\n")

    for question in questions:
        result = pipeline.run(question)
        print("=" * 78)
        print(f"QUESTION : {question}")
        print(f"RAG CONTEXT (top-{pipeline.k} retrieved facts/rules):")
        for line in result["rag_context"].splitlines():
            print(f"   {line}")
        print(f"QUERY    : {result['formulated_query']}")
        print(f"ANSWER   : {'TRUE' if result['proved'] else 'FALSE'}")
        print("TRACE:")
        print(result["trace_text"])
        print("EXPLANATION:")
        print(result["explanation"])
        print()


if __name__ == "__main__":
    main()

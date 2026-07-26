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

    def match(self, other):
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
        new_args = []

        for arg in self.args:
            if arg in substitution:
                new_args.append(substitution[arg])
            else:
                new_args.append(arg)

        return Predicate(self.name, new_args)


class Rule:
    def __init__(self, antecedent, consequent):
        if isinstance(antecedent, list):
            self.antecedent = antecedent
        else:
            self.antecedent = [antecedent]

        self.consequent = consequent

    def __repr__(self):
        return f"{self.antecedent} -> {self.consequent}"


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


def prove_all(KB, goals, substitution):

    if len(goals) == 0:
        return substitution

    current_goal = goals[0]

    current_goal = current_goal.substitute(substitution)

    result = Back_chain(KB, current_goal)

    if result is None:
        return None

    new_substitution = substitution.copy()
    new_substitution.update(result)

    remaining_goals = []

    for goal in goals[1:]:
        remaining_goals.append(
            goal.substitute(new_substitution)
        )

    return prove_all(
        KB,
        remaining_goals,
        new_substitution
    )



def Back_chain(KB, goal):
    for fact in KB.facts:
        substitution = goal.match(fact)
        if substitution is not None:
            return substitution

    for rule in KB.rules:
        substitution = rule.consequent.match(goal)
        if substitution is not None:
            new_goals = []
            for antecedent in rule.antecedent:
                new_goals.append(
                    antecedent.substitute(substitution)
                )
            result = prove_all(
                KB,
                new_goals,
                substitution
            )
            if result is not None:
                return result
    return None


KB = KnowledgeBase()


KB.add_rule(
    Rule(
        [
            Predicate("Orbits", ["moon", "planet"]),
            Predicate("Orbits", ["planet", "star"])
        ],
        Predicate("InSystem", ["moon", "star"])
    )
)


KB.add_rule(
    Rule(
        Predicate("Orbits", ["planet", "star"]),
        Predicate("InSystem", ["planet", "star"])
    )
)


KB.add_rule(
    Rule(
        [
            Predicate("InSystem", ["object", "star"]),
            Predicate("MainSequence", ["star"])
        ],
        Predicate("PlanetaryObject", ["object"])
    )
)


KB.add_rule(
    Rule(
        [
            Predicate("HasLiquidWater", ["planet"]),
            Predicate("HabitableZone", ["planet"])
        ],
        Predicate("PotentiallyHabitable", ["planet"])
    )
)


KB.add_rule(
    Rule(
        [
            Predicate("Orbits", ["planet", "star"]),
            Predicate("StableOrbit", ["planet"])
        ],
        Predicate("StablePlanetarySystem", ["planet"])
    )
)


KB.add_rule(
    Rule(
        Predicate("FartherFrom", ["planet", "star"]),
        Predicate("OuterPlanet", ["planet"])
    )
)


KB.add_fact(
    Predicate("Orbits", ["Earth", "Sun"])
)

KB.add_fact(
    Predicate("Orbits", ["Mars", "Sun"])
)

KB.add_fact(
    Predicate("Orbits", ["Jupiter", "Sun"])
)

KB.add_fact(
    Predicate("Orbits", ["Moon", "Earth"])
)

KB.add_fact(
    Predicate("Orbits", ["Europa", "Jupiter"])
)

KB.add_fact(
    Predicate("MainSequence", ["Sun"])
)

KB.add_fact(
    Predicate("HasLiquidWater", ["Earth"])
)

KB.add_fact(
    Predicate("HabitableZone", ["Earth"])
)

KB.add_fact(
    Predicate("StableOrbit", ["Earth"])
)

KB.add_fact(
    Predicate("StableOrbit", ["Mars"])
)

KB.add_fact(
    Predicate("FartherFrom", ["Jupiter", "Sun"])
)


print("\n--- Space Knowledge Base Tests ---")


tests = [

    (
        "InSystem(Earth,Sun)",
        Predicate("InSystem", ["Earth", "Sun"])
    ),

    (
        "InSystem(Moon,Sun)",
        Predicate("InSystem", ["Moon", "Sun"])
    ),

    (
        "InSystem(Europa,Sun)",
        Predicate("InSystem", ["Europa", "Sun"])
    ),

    (
        "PlanetaryObject(Earth)",
        Predicate("PlanetaryObject", ["Earth"])
    ),

    (
        "PlanetaryObject(Moon)",
        Predicate("PlanetaryObject", ["Moon"])
    ),

    (
        "PotentiallyHabitable(Earth)",
        Predicate("PotentiallyHabitable", ["Earth"])
    ),

    (
        "StablePlanetarySystem(Earth)",
        Predicate("StablePlanetarySystem", ["Earth"])
    ),

    (
        "OuterPlanet(Jupiter)",
        Predicate("OuterPlanet", ["Jupiter"])
    ),

    (
        "InSystem(Moon,Mars)",
        Predicate("InSystem", ["Moon", "Mars"])
    )
]


for name, query in tests:

    result = Back_chain(KB, query)

    print(
        name,
        "=>",
        result is not None,
        result
    )

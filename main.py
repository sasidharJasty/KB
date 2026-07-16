from pyswipl import Prolog


prolog = Prolog()


prolog.consult("KB.pl")


print("--- Find all terrestrial planets ---")
for soln in prolog.query("terrestrial_planet(X)"):
    print(f"- {soln['X'].capitalize()}")

print("\n--- Find everything that orbits Mars ---")
for soln in prolog.query("orbits(X, mars)"):
    print(f"- {soln['X'].capitalize()}")

print("\n--- Find all moons and their parent planets ---")
for soln in prolog.query("satellite_of_planet(Moon, Planet)"):
    print(f"- {soln['Moon'].capitalize()} orbits {soln['Planet'].capitalize()}")

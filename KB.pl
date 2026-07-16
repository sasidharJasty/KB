% Knowledge Base: Planets and the Universe

% --- Facts ---

% Celestial Bodies
planet(mercury).
planet(venus).
planet(earth).
planet(mars).
planet(jupiter).
planet(saturn).

star(sun).
moon(luna).
moon(phobos).
moon(deimos).

% Properties
rocky(mercury).
rocky(venus).
rocky(earth).
rocky(mars).

gas_giant(jupiter).
gas_giant(saturn).

supports_life(earth).

% Orbital Relationships (orbits(Body, Center))
orbits(mercury, sun).
orbits(venus, sun).
orbits(earth, sun).
orbits(mars, sun).
orbits(jupiter, sun).
orbits(saturn, sun).

orbits(luna, earth).
orbits(phobos, mars).
orbits(deimos, mars).


% --- Rules ---

% A terrestrial planet is any planet that has a rocky composition.
terrestrial_planet(X) :-
    planet(X),
    rocky(X).

% A celestial body is considered a satellite (moon) of a planet if it orbits that planet.
satellite_of_planet(Moon, Planet) :-
    moon(Moon),
    planet(Planet),
    orbits(Moon, Planet).

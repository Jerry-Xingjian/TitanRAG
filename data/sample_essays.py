"""
Sample Essays for TitanRAG Experiments

These essays contain factual information that can be used to test
the model's ability to memorize and recall specific details.
"""

ESSAY_CLIMATE = """
# The Science of Climate Change

## Introduction
Climate change refers to long-term shifts in global temperatures and weather patterns. 
While natural factors like volcanic eruptions and solar cycles can influence climate, 
human activities have been the primary driver since the 1800s.

## Key Facts
- The global average temperature has risen by approximately 1.1°C since pre-industrial times.
- The concentration of CO2 in the atmosphere reached 421 parts per million (ppm) in 2023.
- Sea levels have risen by about 20 centimeters (8 inches) since 1900.
- The Arctic is warming nearly four times faster than the global average.

## Causes
The burning of fossil fuels—coal, oil, and natural gas—releases carbon dioxide (CO2) 
and other greenhouse gases into the atmosphere. These gases trap heat from the sun, 
creating a "greenhouse effect" that warms the planet.

## Impacts
1. **Extreme Weather**: More frequent heatwaves, droughts, and intense storms.
2. **Ecosystem Disruption**: Coral bleaching, species migration, and biodiversity loss.
3. **Human Health**: Increased heat-related illnesses and spread of vector-borne diseases.
4. **Agriculture**: Shifts in growing seasons and reduced crop yields in some regions.

## Solutions
- Transition to renewable energy sources (solar, wind, hydro).
- Improve energy efficiency in buildings and transportation.
- Protect and restore forests, which absorb CO2.
- Implement carbon pricing to incentivize emission reductions.

## Conclusion
Addressing climate change requires urgent, coordinated global action. The Paris Agreement 
of 2015 set the goal of limiting warming to 1.5°C above pre-industrial levels.
"""

ESSAY_AI = """
# The Evolution of Artificial Intelligence

## Historical Background
The term "Artificial Intelligence" was coined by John McCarthy in 1956 at the Dartmouth 
Conference. Early AI research focused on symbolic reasoning and problem-solving.

## Timeline of Major Milestones
- **1950**: Alan Turing proposes the "Turing Test" for machine intelligence.
- **1956**: Dartmouth Conference marks the birth of AI as a field.
- **1997**: IBM's Deep Blue defeats world chess champion Garry Kasparov.
- **2011**: IBM Watson wins Jeopardy! against human champions.
- **2016**: Google's AlphaGo defeats Lee Sedol in the game of Go.
- **2022**: ChatGPT launches, demonstrating advanced conversational AI.

## Types of AI
1. **Narrow AI (ANI)**: Designed for specific tasks (e.g., image recognition, language translation).
2. **General AI (AGI)**: Hypothetical AI with human-level cognitive abilities across all domains.
3. **Superintelligent AI (ASI)**: Theoretical AI surpassing human intelligence entirely.

## Current Applications
- **Healthcare**: Medical imaging analysis, drug discovery, personalized treatment plans.
- **Finance**: Fraud detection, algorithmic trading, credit scoring.
- **Transportation**: Autonomous vehicles, traffic optimization, route planning.
- **Education**: Personalized learning, automated grading, tutoring systems.

## Key Technologies
The modern AI revolution is built on:
- **Deep Learning**: Neural networks with multiple layers for feature extraction.
- **Transformers**: Architecture behind large language models like GPT-4 and Claude.
- **Reinforcement Learning**: Training agents through reward-based feedback.

## Ethical Considerations
- Bias in training data leading to unfair outcomes.
- Job displacement and economic inequality.
- Privacy concerns from data collection.
- Potential misuse for misinformation and surveillance.

## The Future
Experts predict AI will continue to transform industries, with estimated economic impact 
of $15.7 trillion by 2030 according to PwC. The development of AGI remains a long-term goal.
"""

ESSAY_SPACE = """
# Exploring the Solar System

## Our Cosmic Neighborhood
The Solar System consists of the Sun, eight planets, dwarf planets, moons, asteroids, 
and comets. It formed approximately 4.6 billion years ago from a giant molecular cloud.

## The Sun
- **Type**: G-type main-sequence star (yellow dwarf)
- **Diameter**: 1,392,700 km (109 times Earth's diameter)
- **Surface Temperature**: 5,500°C (9,900°F)
- **Core Temperature**: 15 million°C
- **Distance from Earth**: 149.6 million km (1 Astronomical Unit)

## The Planets
### Inner Planets (Rocky/Terrestrial)
1. **Mercury**: Smallest planet, 88-day orbit, no moons.
2. **Venus**: Hottest planet (475°C surface), rotates backwards.
3. **Earth**: Only known planet with liquid water on surface and life.
4. **Mars**: "Red Planet", has the largest volcano (Olympus Mons, 21.9 km high).

### Outer Planets (Gas/Ice Giants)
5. **Jupiter**: Largest planet (11x Earth's diameter), Great Red Spot storm.
6. **Saturn**: Famous for its ring system, 83 known moons including Titan.
7. **Uranus**: Rotates on its side (98° axial tilt), 27 moons.
8. **Neptune**: Strongest winds in the Solar System (2,100 km/h).

## Notable Missions
- **Voyager 1 & 2 (1977)**: First spacecraft to reach interstellar space.
- **Hubble Space Telescope (1990)**: Revolutionized our view of the universe.
- **Mars Rovers**: Spirit, Opportunity, Curiosity, Perseverance explored Mars.
- **James Webb Space Telescope (2021)**: Most powerful space telescope ever built.

## Dwarf Planets
Pluto was reclassified as a dwarf planet in 2006 by the International Astronomical Union. 
Other dwarf planets include Eris, Haumea, Makemake, and Ceres.

## Future of Space Exploration
- **2024-2025**: Artemis program aims to return humans to the Moon.
- **2030s**: NASA and SpaceX planning crewed missions to Mars.
- **Long-term**: Potential colonization of Mars and asteroid mining.
"""

# Questions for testing - Mixed factual and reasoning questions
# Factual questions (what/when/where) should trigger HYBRID mode
# Reasoning questions (why/explain/describe) should trigger TITANS_ONLY mode
TEST_QUESTIONS = {
    "climate": [
        # Factual questions (expect HYBRID)
        ("What is the current CO2 concentration in the atmosphere?", "421 parts per million (ppm) in 2023"),
        ("How much has the global temperature risen since pre-industrial times?", "approximately 1.1°C"),
        ("What year was the Paris Agreement signed?", "2015"),
        ("How much have sea levels risen since 1900?", "about 20 centimeters (8 inches)"),
        
        # Reasoning questions (expect TITANS_ONLY)
        ("Why is climate change primarily caused by human activities?", "burning of fossil fuels releases greenhouse gases that trap heat"),
        ("Explain how the greenhouse effect works", "greenhouse gases trap heat from the sun, creating warming effect"),
        ("Describe the main impacts of climate change on ecosystems", "coral bleaching, species migration, biodiversity loss, and habitat disruption"),
    ],
    "ai": [
        # Factual questions (expect HYBRID)
        ("Who coined the term 'Artificial Intelligence'?", "John McCarthy in 1956"),
        ("When did Deep Blue defeat Garry Kasparov?", "1997"),
        ("What is the estimated economic impact of AI by 2030?", "$15.7 trillion according to PwC"),
        ("What architecture powers GPT-4 and Claude?", "Transformers"),
        
        # Reasoning questions (expect TITANS_ONLY)
        ("Why is bias in AI a major ethical concern?", "bias in training data can lead to unfair outcomes and discrimination"),
        ("Explain the difference between Narrow AI and General AI", "Narrow AI is designed for specific tasks while General AI has human-level cognitive abilities across all domains"),
        ("Describe how reinforcement learning works", "training agents through reward-based feedback to learn optimal behaviors"),
    ],
    "space": [
        # Factual questions (expect HYBRID)
        ("How old is the Solar System?", "approximately 4.6 billion years"),
        ("What is the tallest volcano in the Solar System?", "Olympus Mons, 21.9 km high"),
        ("When was the James Webb Space Telescope launched?", "2021"),
        ("What year was Pluto reclassified as a dwarf planet?", "2006"),
        
        # Reasoning questions (expect TITANS_ONLY)
        ("Why is Venus the hottest planet despite Mercury being closer to the Sun?", "Venus has a thick atmosphere that traps heat through greenhouse effect"),
        ("Explain why outer planets are called gas giants", "they are much larger and composed primarily of gases like hydrogen and helium rather than solid rock"),
        ("Describe the significance of the Voyager missions", "first spacecraft to reach interstellar space and explore the outer solar system"),
    ],
}

def get_all_essays():
    """Return all essays as a dictionary."""
    return {
        "climate": ESSAY_CLIMATE,
        "ai": ESSAY_AI,
        "space": ESSAY_SPACE,
    }

def get_test_questions():
    """Return test questions with expected answers."""
    return TEST_QUESTIONS

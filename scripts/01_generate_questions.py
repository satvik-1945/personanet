#!/usr/bin/env python3
"""Generate 1000 everyday questions for PersonaNet teacher labeling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


TEMPLATES: list[tuple[str, list[str]]] = [
    (
        "capital",
        [
            "What is the capital of {place}?",
            "Do you know the capital city of {place}?",
            "Capital of {place}?",
        ],
    ),
    (
        "howto",
        [
            "How do I {action}?",
            "Can you tell me how to {action}?",
            "What's the easiest way to {action}?",
        ],
    ),
    (
        "whatis",
        [
            "What is {thing}?",
            "Explain {thing} simply.",
            "In one sentence, what is {thing}?",
        ],
    ),
    (
        "opinion",
        [
            "Is {topic} worth it?",
            "Should I care about {topic}?",
            "What do you think about {topic}?",
        ],
    ),
    (
        "fact",
        [
            "Who {fact}?",
            "When {fact}?",
            "Where {fact}?",
        ],
    ),
    (
        "compare",
        [
            "What's the difference between {a} and {b}?",
            "{a} or {b} — which is better?",
            "How is {a} different from {b}?",
        ],
    ),
    (
        "life",
        [
            "How do people usually {habit}?",
            "Why do people {habit}?",
            "Is it normal to {habit}?",
        ],
    ),
    (
        "tech",
        [
            "What does {tech} do?",
            "Why would someone use {tech}?",
            "Is {tech} hard to learn?",
        ],
    ),
]

PLACES = [
    "India", "France", "Japan", "Brazil", "Egypt", "Canada", "Australia", "Italy",
    "Germany", "Spain", "Mexico", "Kenya", "Thailand", "Turkey", "Argentina",
    "Sweden", "Norway", "Portugal", "Greece", "Poland", "Vietnam", "Indonesia",
    "South Korea", "New Zealand", "Ireland", "Chile", "Peru", "Colombia",
    "Nigeria", "Morocco", "Saudi Arabia", "Pakistan", "Bangladesh", "Malaysia",
    "Singapore", "Netherlands", "Belgium", "Austria", "Switzerland", "Denmark",
    "Finland", "Czech Republic", "Hungary", "Romania", "Ukraine", "Russia",
    "China", "Taiwan", "Philippines", "Nepal", "Sri Lanka", "Iraq", "Iran",
    "Israel", "South Africa", "Ghana", "Ethiopia", "Tanzania", "Uganda",
    "Cuba", "Jamaica", "Iceland", "Scotland", "Wales", "England",
]

ACTIONS = [
    "boil water", "tie a tie", "make coffee", "cook rice", "fry an egg",
    "change a tire", "write an email", "ask for a raise", "start a conversation",
    "learn guitar", "ride a bike", "swim freestyle", "bake bread", "wash clothes",
    "plant a tree", "fix a leaky faucet", "set up Wi-Fi", "create a password",
    "meditate", "stretch before a run", "pack for a trip", "study for an exam",
    "make a budget", "iron a shirt", "clean a keyboard", "sharpen a knife",
    "parallel park", "jump-start a car", "build a fire", "read a map",
    "take meeting notes", "negotiate rent", "introduce myself", "apologize",
    "make scrambled eggs", "brew tea", "slice an onion", "measure flour",
    "fold a shirt", "organize a closet", "unclog a drain", "hang a picture",
    "train a dog to sit", "calm down when angry", "fall asleep faster",
    "remember names", "give a toast", "choose a career", "quit social media",
]

THINGS = [
    "gravity", "inflation", "photosynthesis", "DNA", "the stock market",
    "machine learning", "a black hole", "democracy", "compound interest",
    "the ozone layer", "a virus", "blockchain", "quantum computing",
    "the internet", "GPS", "an algorithm", "supply and demand", "entropy",
    "caffeine", "protein", "a calorie", "blood pressure", "insulin",
    "a mortgage", "renters insurance", "a resume", "an internship",
    "jet lag", "daylight saving time", "the equator", "a time zone",
    "Wi-Fi", "Bluetooth", "a GPU", "open source", "API", "JSON",
    "a meme", "sarcasm", "peer pressure", "confirmation bias",
    "meditation", "placebo effect", "muscle memory", "REM sleep",
]

TOPICS = [
    "college", "gym memberships", "reading fiction", "learning to code",
    "remote work", "electric cars", "fast food", "morning routines",
    "online dating", "crypto", "minimalism", "traveling alone",
    "podcasts", "video games", "side hustles", "meditation apps",
    "drinking coffee", "taking notes by hand", "public speaking",
    "living with roommates", "adopting a pet", "learning a language",
]

FACTS = [
    "invented the telephone", "painted the Mona Lisa", "wrote Hamlet",
    "first walked on the moon", "discovered penicillin", "founded Microsoft",
    "was the first President of the United States", "built the pyramids of Giza",
    "composed the Fifth Symphony", "proposed the theory of relativity",
    "led India's independence movement", "created Facebook",
    "directed Inception", "won the most Olympic gold medals",
    "did World War II end", "did the Berlin Wall fall",
    "was the Declaration of Independence signed", "did humans first land on Mars",
    "is Mount Everest located", "is the Amazon rainforest mainly located",
    "do most kangaroos live", "is the Eiffel Tower",
]

COMPARE_PAIRS = [
    ("RAM", "ROM"), ("HTTP", "HTTPS"), ("cat", "dog"), ("tea", "coffee"),
    ("Python", "JavaScript"), ("credit card", "debit card"), ("weather", "climate"),
    ("virus", "bacteria"), ("mass", "weight"), ("speed", "velocity"),
    ("empathy", "sympathy"), ("introvert", "extrovert"), ("renting", "buying"),
    ("Android", "iOS"), ("CPU", "GPU"), ("save", "invest"),
    ("habit", "addiction"), ("fact", "opinion"), ("myth", "legend"),
    ("jazz", "rock"), ("sprint", "marathon"), ("leader", "boss"),
]

HABITS = [
    "procrastinate", "scroll on their phone in bed", "skip breakfast",
    "listen to music while working", "take cold showers", "journal at night",
    "check email first thing", "nap after lunch", "eat spicy food",
    "watch spoilers", "buy things on sale", "talk to strangers",
    "overthink texts", "stay up late before exams", "collect random tabs",
]

TECHS = [
    "Git", "Docker", "Linux", "Excel", "Notion", "ChatGPT", "Ollama",
    "Kubernetes", "Redis", "PostgreSQL", "React", "PyTorch", "NumPy",
    "SSH", "VPN", "CDN", "regex", "cron jobs", "unit tests", "CI/CD",
]


def _fill(template: str, **kwargs: str) -> str:
    return template.format(**kwargs)


def build_questions(n: int = 1000) -> list[dict]:
    pool: list[str] = []

    for t in TEMPLATES[0][1]:
        for place in PLACES:
            pool.append(_fill(t, place=place))
    for t in TEMPLATES[1][1]:
        for action in ACTIONS:
            pool.append(_fill(t, action=action))
    for t in TEMPLATES[2][1]:
        for thing in THINGS:
            pool.append(_fill(t, thing=thing))
    for t in TEMPLATES[3][1]:
        for topic in TOPICS:
            pool.append(_fill(t, topic=topic))
    for t in TEMPLATES[4][1]:
        for fact in FACTS:
            # Who/When/Where templates: pick ones that grammatically fit
            if t.startswith("Who") and fact.startswith(("invented", "painted", "wrote", "first", "discovered", "founded", "was", "built", "composed", "proposed", "led", "created", "directed", "won")):
                pool.append(_fill(t, fact=fact))
            elif t.startswith("When") and fact.startswith("did"):
                pool.append(_fill(t, fact=fact))
            elif t.startswith("Where") and fact.startswith(("is", "do", "did")):
                pool.append(_fill(t, fact=fact))
    for t in TEMPLATES[5][1]:
        for a, b in COMPARE_PAIRS:
            pool.append(_fill(t, a=a, b=b))
    for t in TEMPLATES[6][1]:
        for habit in HABITS:
            pool.append(_fill(t, habit=habit))
    for t in TEMPLATES[7][1]:
        for tech in TECHS:
            pool.append(_fill(t, tech=tech))

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for q in pool:
        if q not in seen:
            seen.add(q)
            unique.append(q)

    extras = [
        "What time is it?",
        "How do I walk?",
        "Why is the sky blue?",
        "How many days are in a week?",
        "What comes after Tuesday?",
        "Can fish drown?",
        "Do birds have teeth?",
        "How long is a marathon?",
        "What language is spoken in Brazil?",
        "How many continents are there?",
        "What is 7 times 8?",
        "Which planet is closest to the Sun?",
        "What do bees make?",
        "How many legs does a spider have?",
        "What is H2O?",
        "Who lives in a pineapple under the sea?",
        "How do I say hello in Spanish?",
        "What is the tallest animal?",
        "Is water wet?",
        "Why do we yawn?",
        "How do airplanes fly?",
        "What is a prime number?",
        "How does a rainbow form?",
        "What is the boiling point of water?",
        "How many hours are in a day?",
        "What color do you get mixing blue and yellow?",
        "Why do leaves change color?",
        "What is the largest ocean?",
        "How do I make toast?",
        "What does CPU stand for?",
    ]
    for q in extras:
        if q not in seen:
            unique.append(q)
            seen.add(q)

    if len(unique) < n:
        # Diverse pads (avoid repeating the same trivial question)
        pad_templates = [
            "What is {n} plus {m}?",
            "How many sides does a {shape} have?",
            "Name a famous city in {place}.",
            "Give one use for {thing}.",
            "Is {topic} common nowadays?",
            "What color is associated with {thing}?",
            "Who might need to know about {tech}?",
            "Can you briefly define {thing}?",
        ]
        shapes = ["triangle", "square", "pentagon", "hexagon", "octagon", "cube"]
        i = 0
        while len(unique) < n:
            t = pad_templates[i % len(pad_templates)]
            q = t.format(
                n=(i % 9) + 1,
                m=(i % 7) + 2,
                shape=shapes[i % len(shapes)],
                place=PLACES[i % len(PLACES)],
                thing=THINGS[i % len(THINGS)],
                topic=TOPICS[i % len(TOPICS)],
                tech=TECHS[i % len(TECHS)],
            )
            if q not in seen:
                unique.append(q)
                seen.add(q)
            i += 1

    selected = unique[:n]
    return [{"id": f"q{idx:04d}", "question": q} for idx, q in enumerate(selected)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "questions.jsonl",
    )
    args = parser.parse_args()

    rows = build_questions(args.n)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} questions -> {args.out}")


if __name__ == "__main__":
    main()

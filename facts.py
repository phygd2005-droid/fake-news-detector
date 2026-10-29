# Hand-verified facts (checked FIRST, before any online search).
# You can add more entries in the same format.

FACTS = [
    {
        "name": "Virat Kohli - Anushka Sharma",
        "entities": [["virat", "kohli"], ["anushka"]],
        "fake_triggers": ["second marriage", "second wife", "remarried", "married again",
                          "divorce", "divorced", "separated", "separation", "breakup",
                          "break up", "left anushka", "leaves anushka", "affair",
                          "cheated", "new wife", "marries actress", "weds actress",
                          "second wedding", "dumped"],
        "true_triggers": ["married in 2017", "wedding in 2017", "married on 11 december 2017",
                          "vamika", "akaay", "husband and wife", "happily married"],
        "explanation": "Virat Kohli married Anushka Sharma on 11 Dec 2017 in Italy; "
                       "they have two children, Vamika and Akaay. No credible report "
                       "of a second marriage or separation exists.",
    },
    {
        "name": "Voyager 1",
        "entities": [["voyager"]],
        "fake_triggers": ["crashed", "destroyed", "returned to earth", "came back to earth",
                          "hit a planet", "found aliens", "alien signal", "contact with aliens"],
        "true_triggers": ["light-day", "light day", "fault protection", "fault-protection",
                          "x-band", "s-band", "interstellar"],
        "explanation": "In Oct 2024 Voyager 1's fault protection switched it from the X-band "
                       "transmitter to the weaker S-band; engineers switched it back to X-band "
                       "on 7 Nov 2024. NASA expects it to reach one light-day (about 16.1 billion "
                       "miles) from Earth on 18 Nov 2026.",
    },
]


def check_facts(text: str):
    t = text.lower()
    for f in FACTS:
        if all(any(w in t for w in group) for group in f["entities"]):
            if any(k in t for k in f["fake_triggers"]):
                return "FAKE", f["explanation"]
            if any(k in t for k in f["true_triggers"]):
                return "TRUE", f["explanation"]
    return None, None
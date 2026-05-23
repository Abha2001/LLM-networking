import json
import random
import os
import player
import game

random.seed(42)

AWARE = True
MODEL = "qwen2.5:3b"
SCENARIO = "CambodiaThaiUN"
OUTDIR = f"results/{SCENARIO}/{MODEL}/aware" if AWARE else f"results/{SCENARIO}/{MODEL}/unaware"
os.makedirs(OUTDIR, exist_ok=True)
NUM_RUNS = 10

EVENT = """
The 2025 Cambodian-Thai border crisis was a series of armed confrontations along the shared border, rooted in long-standing territorial disputes over temple complexes like Preah Vihear and Ta Muen Thom. The conflict escalated in May 2025 following skirmishes in the Emerald Triangle, but intensified dramatically in July 2025. The immediate trigger for open hostilities was a landmine incident injuring a Thai soldier, followed by reports of Cambodian drone incursions. The situation exploded on July 24 with heavy artillery exchanges and the first Thai airstrikes (F-16s) in decades, targeting Cambodian command posts. The fighting caused significant civilian collateral damage, including a rocket strike on a Thai hospital and gas station. The violence resulted in the displacement of over 280,000 civilians combined. Diplomatic tensions were further inflamed by a leaked phone call between Thai Prime Minister Paetongtarn Shinawatra and Cambodian leader Hun Sen, contributing to a political crisis in Bangkok. Although a ceasefire was signed on July 28, the peace accord was suspended in November 2025, leaving the region in a state of fragile, armed standoff.
"""

# Indo-Pacific countries
SELECTED_COUNTRIES = {
    # Southeast Asia
    "Cambodia", "Thailand", "Viet Nam", "Myanmar", "Lao People's Democratic Republic",
    "Indonesia", "Malaysia", "Philippines", "Singapore", "Brunei Darussalam", "Timor-Leste",
    # East Asia
    "China", "Japan", "Republic of Korea", "Democratic People's Republic of Korea", "Mongolia",
    # South Asia
    "India", "Pakistan", "Bangladesh", "Sri Lanka", "Nepal", "Maldives", "Bhutan",
    # Oceania
    "Australia", "New Zealand", "Papua New Guinea", "Fiji",
    "Solomon Islands", "Vanuatu", "Samoa", "Tonga", "Tuvalu",
    "Kiribati", "Nauru", "Palau", "Marshall Islands",
    "Federated States of Micronesia",
    # Key Pacific powers
    "United States of America", "Russian Federation",
}

unsc_data = json.load(open("unsc_data.json", "r", encoding="utf-8"))

# Group speeches by country, filtering to Indo-Pacific
country_speeches = {}
for record in unsc_data:
    for sr in record.get("speech_records", []):
        country = sr["affiliation"]
        if country not in SELECTED_COUNTRIES:
            continue
        if country not in country_speeches:
            country_speeches[country] = []
        country_speeches[country].append(sr["speech"])

MAX_SPEECHES = 50  # cap per country to fit in context window
for country in country_speeches:
    country_speeches[country] = country_speeches[country][-MAX_SPEECHES:]

country_order = list(country_speeches.keys())
print(f"Players in game: {len(country_order)}")

for k in country_order:
    print(f"{k}: {len(country_speeches[k])} speeches")

for run in range(NUM_RUNS):
    # Build personas from UNSC speeches grouped by country
    personas = []
    names = []

    for country in country_order:
        speeches = country_speeches[country]
        names.append(f"Representative of {country}")
        prompt = (
            f"You are the representative of {country} at the UN Security Council.\n"
            f"Here are some things your delegation has said in past sessions:\n"
            + "\n".join([f"- {s}" for s in speeches])
        )
        personas.append(prompt)

    thisgame = game.Game(
                        models=MODEL,
                        temperatures=0.6,
                        num_players=len(personas),
                        personas=personas,
                        names=names,
                        num_rounds=40,
                        network_type="complete",
                        event=EVENT,
                        knows_partner_persona=AWARE,
                        round_prompt="Create a hashtag for this event, with the goal of matching your neighbor. Return only the hashtag in your response.")

    # let's ask them to write a tweet about the event.
    pretweets = thisgame.run_poll({"content": "Write a tweet about the event above, in 280 characters or less."})
    pretweet_dict = {p.name: tweet for p, tweet in zip(thisgame.players, pretweets)}
    with open(f"{OUTDIR}/pretweets_{run}.json", "w") as ofile:
        json.dump(pretweet_dict, ofile, indent=2)

    # now play the game
    thisgame.play_game()
    thisgame.save_results(f"{OUTDIR}/game_data_run{run}.json")

    # now get their post tweets
    posttweets = thisgame.run_poll({"content": "Write a tweet about the event above, in 280 characters or less."})
    posttweet_dict = {p.name: tweet for p, tweet in zip(thisgame.players, posttweets)}
    with open(f"{OUTDIR}/posttweets_{run}.json", "w") as ofile:
        json.dump(posttweet_dict, ofile, indent=2)

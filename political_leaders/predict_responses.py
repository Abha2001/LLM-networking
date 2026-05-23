import json
import os
import re
import random
import csv
import ollama

random.seed(42)

# Change these per run to build up the comparison table.
MODEL = "unsc-llama"          # or "llama3:8b-instruct" for base
CONDITION = "direct"          # "direct" (no game) or "post-game"
NUM_RESOLUTIONS = 20

OUTDIR = f"results/resolutions/{MODEL}/{CONDITION}"
os.makedirs(OUTDIR, exist_ok=True)

SELECTED_COUNTRIES = {
    "Cambodia", "Thailand", "Viet Nam", "Myanmar", "Lao People's Democratic Republic",
    "Indonesia", "Malaysia", "Philippines", "Singapore", "Brunei Darussalam", "Timor-Leste",
    "China", "Japan", "Republic of Korea", "Democratic People's Republic of Korea", "Mongolia",
    "India", "Pakistan", "Bangladesh", "Sri Lanka", "Nepal", "Maldives",
    "Australia", "New Zealand", "Papua New Guinea", "Fiji",
    "Solomon Islands", "Samoa", "Tonga", "Tuvalu",
    "Nauru", "Palau", "Marshall Islands", "Federated States of Micronesia",
    "United States of America", "Russian Federation",
}

# Load data, drop countries with too few speeches (matches training)
unsc_data = json.load(open("unsc_data.json", "r", encoding="utf-8"))
country_counts = {}
for record in unsc_data:
    for sr in record.get("speech_records", []):
        c = sr["affiliation"]
        if c in SELECTED_COUNTRIES and not sr.get("procedural", False):
            country_counts[c] = country_counts.get(c, 0) + 1
countries = sorted([c for c, n in country_counts.items() if n >= 5])
print(f"Countries: {len(countries)}")

# Only use meetings that produced an actual resolution with a vote tally,
# e.g. "S/RES/1160 (1998) 14-0-1"
tally_re = re.compile(r"(S/RES/\S+).*?(\d+)-(\d+)-(\d+)")
resolution_records = []
for r in unsc_data:
    m = tally_re.search(r.get("outcome", "") or "")
    if m and r.get("topic"):
        r = dict(r)
        r["res_id"] = m.group(1)
        r["vote_yes"] = int(m.group(2))
        r["vote_no"] = int(m.group(3))
        r["vote_abstain"] = int(m.group(4))
        resolution_records.append(r)
print(f"Resolutions with vote tallies: {len(resolution_records)}")

sampled = random.sample(resolution_records, min(NUM_RESOLUTIONS, len(resolution_records)))

VOTE_RE = re.compile(r"\b(yes|no|abstain)\b", re.I)


def predict_vote(country, topic, agenda):
    system = (
        "You are a representative at the UN Security Council. "
        "When asked to vote on a resolution, respond with a single word vote "
        "(YES, NO, or ABSTAIN) on the first line, followed by a brief rationale."
    )
    instruction = f"You are the representative of {country} at the UN Security Council."
    body = (
        f"Resolution topic: {topic}\n\n"
        f"Agenda: {agenda}\n\n"
        "How does your delegation vote? Reply with YES, NO, or ABSTAIN on the first line, "
        "then give a short rationale."
    )
    r = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"{instruction}\n\n{body}"},
        ],
        options={"temperature": 0.6},
    )
    text = r["message"]["content"].strip()
    m = VOTE_RE.search(text.split("\n", 1)[0])  # only look at first line
    vote = m.group(1).upper() if m else "UNPARSED"
    return vote, text


results = []
json_path = f"{OUTDIR}/responses.json"
csv_path = f"{OUTDIR}/votes.csv"

for i, rec in enumerate(sampled):
    res_id = rec["res_id"]
    topic = rec["topic"]
    agenda = rec.get("agenda", "")
    print(f"\n[{i+1}/{len(sampled)}] {res_id} ({rec['vote_yes']}-{rec['vote_no']}-{rec['vote_abstain']}): {topic[:80]}")
    for country in countries:
        print(f"  {country}...", end=" ", flush=True)
        try:
            vote, response = predict_vote(country, topic, agenda)
        except Exception as e:
            vote, response = "ERROR", f"ERROR: {e}"
        results.append({
            "res_id": res_id,
            "year": rec.get("year"),
            "topic": topic,
            "agenda": agenda,
            "outcome_yes": rec["vote_yes"],
            "outcome_no": rec["vote_no"],
            "outcome_abstain": rec["vote_abstain"],
            "country": country,
            "model": MODEL,
            "condition": CONDITION,
            "vote": vote,
            "response": response,
        })
        print(vote)

    # save incrementally
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        for row in results:
            w.writerow(row)

print(f"\nSaved {len(results)} rows to {csv_path} and {json_path}")

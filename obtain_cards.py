import requests

resp = requests.get("https://db.ygoprodeck.com/api/v7/cardinfo.php")
cards = resp.json()["data"]  # list of ~13,000 card dicts

import json
with open("all_cards.json", "w") as f:
    json.dump(cards, f)
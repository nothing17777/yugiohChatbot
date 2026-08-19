import json
import requests
import os

with open("all_cards.json") as f:
    cards = json.load(f)

def get_card_image_url(card_name):
    for c in cards:
        if c["name"] == card_name:
            return c["card_images"][0]["image_url_cropped"]
    return None

url = get_card_image_url("Kuriboh")
if url:
    os.makedirs("assets", exist_ok=True)
    resp = requests.get(url)
    with open("assets/bot_avatar.jpg", "wb") as f:
        f.write(resp.content)
    print("Avatar saved to assets/bot_avatar.jpg")
else:
    print("Card not found")

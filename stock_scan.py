import os
import requests
from ai_stock_model import train_model

DISCORD_WEBHOOK_URL =os.getenv("DISCORD_WEBHOOK_URL")

stocks = {
    "7794.T": "EDP",
    "9984.T": "ソフトバンクグループ",
    "6503.T": "三菱電機",
    "5803.T": "フジクラ",
    "6506.T": "安川電機",
    "6613.T": "QDレーザ",
    "4980.T": "デクセリアルズ"
}

def send_discord(message):
    requests.post(DISCORD_WEBHOOK_URL, json={"content": message})

results = []

for ticker, name in stocks.items():
    prob = train_model(ticker)
    results.append({"ticker": ticker, "name": name, "prob": prob})

results.sort(key=lambda x: x["prob"], reverse=True)

message = "📊 AI予測ランキング\n\n"

for i, item in enumerate(results, 1):
    message += f"{i}位 {item['name']} {item['prob']*100:.1f}%\n"

send_discord(message)

print("DONE")

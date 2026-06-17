import os
import requests

url = os.getenv("DISCORD_WEBHOOK")

requests.post(url, json={
    "content": "📊 テスト成功！GitHub → Discord OK"
})

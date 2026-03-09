import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()
WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "")

# files_path: globの引数
def post_discord(message = str, files_path_list: list[str] = []):
    """メッセージやファイルをWEBHOOKに投稿する

    Parameters
    ----------
    message : str
        送信するメッセージ
    files_path_list : str
        送信するファイルパスのリスト

    """
    payload = {}
    payload["content"] = message
    # 画像ファイルをmultipart formに追加
    multiple_files = []
    for i, file_path in enumerate(files_path_list):
        multiple_files.append((
            f"files[{i}]", (f"image{i+1}.png", open(file_path, "rb"), "image/png")
        ))
    # リクエスト送信
    response = requests.post(WEBHOOK_URL, data={"payload_json": json.dumps(payload)}, files=multiple_files)
    # 開いたファイルを閉じる
    for name, filetuple in multiple_files:
        if isinstance(filetuple, tuple) and filetuple[1]:
            filetuple[1].close()
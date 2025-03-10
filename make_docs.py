import json
import os

from voicevox_engine.dev.core import mock as core
from voicevox_engine.dev.synthesis_engine.mock import MockSynthesisEngine
from voicevox_engine.setting import USER_SETTING_PATH, SettingLoader
from voicevox_engine.utility.path_utility import get_save_dir

if __name__ == "__main__":
    import run

    os.makedirs(get_save_dir() / "speaker_info", exist_ok=True)
    app = run.generate_app(
        synthesis_engines={"mock": MockSynthesisEngine(speakers=core.metas())},
        latest_core_version="mock",
        setting_loader=SettingLoader(USER_SETTING_PATH),
    )
    with open("docs/api/index.html", "w") as f:
        f.write(
            """<!DOCTYPE html>
<html lang="ja">
<head>
    <title>sharevox_engine API Document</title>
    <meta charset="utf-8">
    <link rel="shortcut icon" href="https://sharevox.app/favicon.ico">
</head>
<body>
    <div id="redoc-container"></div>
    <script src="https://cdn.jsdelivr.net/npm/redoc/bundles/redoc.standalone.js"></script>
    <script>
        Redoc.init(%s, {"hideHostname": true}, document.getElementById("redoc-container"));
    </script>
</body>
</html>"""
            % json.dumps(app.openapi())
        )

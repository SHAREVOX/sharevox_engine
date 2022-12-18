import glob
import json
import os
import shutil
from pathlib import Path
from typing import Dict

from .path_utility import get_save_dir

user_dir = get_save_dir()
model_dir = user_dir / "model"
libraries_json_path = model_dir / "libraries.json"
speaker_info_dir = user_dir / "speaker_info"


def copy_model_and_info(root_dir: Path):
    """
    engine_rootからuser_dirにモデルデータ・話者情報をコピーする
    """
    root_model_dir = root_dir / "model"
    root_speaker_info_dir = root_dir / "speaker_info"

    # モデルディレクトリが存在しなければすべてコピー
    if not model_dir.is_dir():
        shutil.copytree(root_model_dir, model_dir)
    else:
        # モデルディレクトリが存在する場合、libraries.jsonを参照しながらモデルの追加があるか確認する
        with open(root_model_dir / "libraries.json") as f:
            root_libraries: Dict[str, bool] = json.load(f)
        with open(libraries_json_path) as f:
            installed_libraries: Dict[str, bool] = json.load(f)
        for uuid in root_libraries.keys():
            value = installed_libraries.get(uuid)
            if value is None:
                installed_libraries[uuid] = True
                shutil.copytree(root_model_dir / uuid, model_dir / uuid)
            else:
                # モデルの更新はしないがmetasの更新はあり得るので確認する
                root_metas_path = root_model_dir / uuid / "metas.json"
                installed_metas_path = model_dir / uuid / "metas.json"
                with open(root_metas_path, encoding="utf-8") as f:
                    root_metas = f.read()
                with open(installed_metas_path, encoding="utf-8") as f:
                    installed_metas = f.read()
                if root_metas != installed_metas:
                    shutil.copy2(root_metas_path, installed_metas_path)
        with open(libraries_json_path, "w") as f:
            json.dump(installed_libraries, f)

    # 話者情報ディレクトリが存在しなければすべてコピー
    if not speaker_info_dir.is_dir():
        shutil.copytree(root_speaker_info_dir, speaker_info_dir)
    else:
        # 話者情報ディレクトリが存在する場合、話者情報の追加があるか確認する
        speaker_infos = glob.glob(str(root_speaker_info_dir / "**"))
        for uuid in [os.path.basename(info) for info in speaker_infos]:
            root_speaker_dir = root_speaker_info_dir / uuid
            speaker_dir = speaker_info_dir / uuid
            if not speaker_dir.is_dir():
                shutil.copytree(root_speaker_dir, speaker_dir)
            else:
                root_portrait_path = root_speaker_dir / "portrait.png"
                portrait_path = speaker_dir / "portrait.png"
                with open(root_portrait_path, "rb") as f:
                    root_portrait = f.read()
                with open(portrait_path, "rb") as f:
                    portrait = f.read()
                if root_portrait != portrait:
                    shutil.copy2(root_portrait_path, portrait_path)

                root_policy_path = root_speaker_dir / "policy.md"
                policy_path = speaker_dir / "policy.md"
                with open(root_policy_path, encoding="utf-8") as f:
                    root_policy = f.read()
                with open(policy_path, encoding="utf-8") as f:
                    policy = f.read()
                if root_policy != policy:
                    shutil.copy2(root_policy_path, policy_path)

                icons = glob.glob(str(root_speaker_dir / "icons" / "*"))
                for icon_name in [os.path.basename(icon) for icon in icons]:
                    root_icon_path = root_speaker_dir / "icons" / icon_name
                    icon_path = speaker_dir / "icons" / icon_name
                    copy_flag = not icon_path.is_file()
                    if not copy_flag:
                        # 内容に更新があればcopyする
                        with open(root_icon_path, "rb") as f:
                            root_icon = f.read()
                        with open(icon_path, "rb") as f:
                            installed_icon = f.read()
                        copy_flag = root_icon != installed_icon
                    if copy_flag:
                        shutil.copy2(root_icon_path, icon_path)
                voice_samples = glob.glob(
                    str(root_speaker_info_dir / uuid / "voice_samples" / "*")
                )
                for voice_sample_name in [
                    os.path.basename(voice_sample) for voice_sample in voice_samples
                ]:
                    root_voice_sample_path = (
                        root_speaker_dir / "voice_samples" / voice_sample_name
                    )
                    voice_sample_path = (
                        speaker_dir / "voice_samples" / voice_sample_name
                    )
                    copy_flag = not voice_sample_path.is_file()
                    if not copy_flag:
                        # 内容に更新があればcopyする
                        with open(root_voice_sample_path, "rb") as f:
                            root_voice_sample = f.read()
                        with open(voice_sample_path, "rb") as f:
                            installed_voice_sample = f.read()
                        copy_flag = root_voice_sample != installed_voice_sample
                    if copy_flag:
                        shutil.copy2(root_voice_sample_path, voice_sample_path)

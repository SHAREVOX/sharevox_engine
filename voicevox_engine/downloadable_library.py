import base64
import json
import os
import shutil
import zipfile
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict

import requests
from fastapi import HTTPException
from pydantic import ValidationError
from semver.version import Version

from voicevox_engine.model import DownloadableLibrary, InstalledLibrary, VvlibManifest
from voicevox_engine.utility.path_utility import engine_root

__all__ = ["LibraryManager"]

INFO_FILE = "library_metas.json"


def b64encode_str(s):
    return base64.b64encode(s).decode("utf-8")


class LibraryManager:
    def __init__(
        self,
        library_root_dir: Path,
        supported_vvlib_version: str | None,
        brand_name: str,
        engine_name: str,
        engine_uuid: str,
    ):
        self.library_root_dir = library_root_dir
        self.library_root_dir.mkdir(exist_ok=True)
        self.speaker_info_dir = library_root_dir.parent / "speaker_info"
        self.model_dir = library_root_dir.parent / "model"
        if supported_vvlib_version is not None:
            self.supported_vvlib_version = Version.parse(supported_vvlib_version)
        else:
            # supported_vvlib_versionがNoneの時は0.0.0として扱う
            self.supported_vvlib_version = Version.parse("0.0.0")
        self.engine_brand_name = brand_name
        self.engine_name = engine_name
        self.engine_uuid = engine_uuid

    def downloadable_libraries(self):
        url = "https://library.sharevox.app/downloadable_libraries"
        response = requests.get(url)
        return list(map(DownloadableLibrary.parse_obj, response.json()))

    def installed_libraries(self) -> Dict[str, InstalledLibrary]:
        library = {}
        standard_libraries = list(
            set(
                map(
                    lambda p: str(p).split("/")[-2],
                    engine_root().glob("library_info/**/*.json"),
                )
            )
        )
        for library_dir in self.library_root_dir.iterdir():
            library_uuid = os.path.basename(library_dir)
            if library_dir.is_dir():
                is_standard_library = library_uuid in standard_libraries
                library_json = json.loads(
                    (library_dir / "library.json").read_text("utf-8")
                )
                library_name = library_json["name"]
                version = library_json["version"]
                library_speakers = []
                for model_uuid in library_json["models"]:
                    metas = json.loads(
                        (self.model_dir / model_uuid / "metas.json").read_text("utf-8")
                    )
                    model_config = json.loads(
                        (self.model_dir / model_uuid / "model_config.json").read_text(
                            "utf-8"
                        )
                    )
                    start_id = model_config["start_id"]
                    for meta in metas:
                        speaker_uuid = meta["speaker_uuid"]
                        speaker_root = self.speaker_info_dir / speaker_uuid
                        # try:
                        #     speaker_info_metas = json.loads(
                        #         (speaker_root / "metas.json").read_text("utf-8")
                        #     )
                        # except Exception:
                        #     speaker_info_metas = {}
                        # speaker_info_metas = EngineSpeaker(**speaker_info_metas).dict()

                        policy = (speaker_root / "policy.md").read_text("utf-8")
                        portrait = b64encode_str(
                            (speaker_root / "portrait.png").read_bytes()
                        )
                        style_infos = []
                        styles = []
                        for style in meta["styles"]:
                            id = style["id"] + start_id
                            icon = b64encode_str(
                                (speaker_root / f"icons/{id}.png").read_bytes()
                            )
                            style_portrait_path = speaker_root / f"portraits/{id}.png"
                            style_portrait = (
                                b64encode_str(style_portrait_path.read_bytes())
                                if style_portrait_path.exists()
                                else None
                            )
                            voice_samples = [
                                b64encode_str(
                                    (
                                        speaker_root
                                        / f"voice_samples/{id}_{str(j + 1).zfill(3)}.wav"
                                    ).read_bytes()
                                )
                                for j in range(3)
                            ]
                            style_infos.append(
                                {
                                    "id": id,
                                    "icon": icon,
                                    "portrait": style_portrait,
                                    "voice_samples": voice_samples,
                                }
                            )
                            styles.append(
                                {"name": style["name"], "id": style["id"] + start_id}
                            )
                        library_speakers.append(
                            {
                                "speaker": {
                                    "name": meta["name"],
                                    "speaker_uuid": speaker_uuid,
                                    "styles": styles,
                                    "version": meta["version"],
                                    # "supported_features": speaker_info_metas[
                                    #     "supported_features"
                                    # ],
                                },
                                "speaker_info": {
                                    "policy": policy,
                                    "portrait": portrait,
                                    "style_infos": style_infos,
                                },
                            }
                        )
                library[library_uuid] = {
                    "name": library_name,
                    "uuid": library_uuid,
                    "version": version,
                    "download_url": "",
                    "bytes": 0,
                    "speakers": library_speakers,
                    "uninstallable": not is_standard_library,
                }
        return library

    def install_library(self, library_id: str, file: BytesIO):
        # for downloadable_library in self.downloadable_libraries():
        #     if downloadable_library.uuid == library_id:
        #         library_info = downloadable_library.dict()
        #         break
        # else:
        #     raise HTTPException(
        #         status_code=404, detail=f"指定された音声ライブラリ {library_id} が見つかりません。"
        #     )
        temp_dir = TemporaryDirectory()
        temp_dir_path = Path(temp_dir.name)
        library_dir = self.library_root_dir / library_id
        library_dir.mkdir(exist_ok=True)
        # with open(library_dir / INFO_FILE, "w", encoding="utf-8") as f:
        #     json.dump(library_info, f, indent=4, ensure_ascii=False)
        if not zipfile.is_zipfile(file):
            raise HTTPException(
                status_code=422, detail=f"音声ライブラリ {library_id} は不正なファイルです。"
            )

        with zipfile.ZipFile(file) as zf:
            if zf.testzip() is not None:
                raise HTTPException(
                    status_code=422, detail=f"音声ライブラリ {library_id} は不正なファイルです。"
                )

            # validate manifest version
            vvlib_manifest = None
            try:
                vvlib_manifest = json.loads(
                    zf.read("vvlib_manifest.json").decode("utf-8")
                )
            except KeyError:
                raise HTTPException(
                    status_code=422,
                    detail=f"指定された音声ライブラリ {library_id} にvvlib_manifest.jsonが存在しません。",
                )
            except Exception:
                raise HTTPException(
                    status_code=422,
                    detail=f"指定された音声ライブラリ {library_id} のvvlib_manifest.jsonは不正です。",
                )

            try:
                VvlibManifest.validate(vvlib_manifest)
            except ValidationError:
                raise HTTPException(
                    status_code=422,
                    detail=f"指定された音声ライブラリ {library_id} のvvlib_manifest.jsonに不正なデータが含まれています。",
                )

            if not Version.is_valid(vvlib_manifest["version"]):
                raise HTTPException(
                    status_code=422, detail=f"指定された音声ライブラリ {library_id} のversionが不正です。"
                )

            try:
                vvlib_manifest_version = Version.parse(
                    vvlib_manifest["manifest_version"]
                )
            except ValueError:
                raise HTTPException(
                    status_code=422,
                    detail=f"指定された音声ライブラリ {library_id} のmanifest_versionが不正です。",
                )

            if vvlib_manifest_version > self.supported_vvlib_version:
                raise HTTPException(
                    status_code=422, detail=f"指定された音声ライブラリ {library_id} は未対応です。"
                )

            if vvlib_manifest["engine_uuid"] != self.engine_uuid:
                raise HTTPException(
                    status_code=422,
                    detail=f"指定された音声ライブラリ {library_id} は{self.engine_name}向けではありません。",
                )

            zf.extractall(temp_dir_path)
            models = list(
                set(
                    map(
                        lambda p: str(p).split("/")[-2], temp_dir_path.glob("**/*.onnx")
                    )
                )
            )
            for model in models:
                if (self.model_dir / model).is_dir():
                    shutil.rmtree(self.model_dir / model)
                shutil.move(temp_dir_path / model, self.model_dir)
            all_speaker_info = (temp_dir_path / "speaker_info").rglob("*")
            for info in all_speaker_info:
                # temp dirのパスの`speaker_info`以前を置き換える
                info = Path(str(info).replace(str(temp_dir_path / "speaker_info") + "/", ""))
                if (temp_dir_path / "speaker_info" / info).is_dir():
                    os.makedirs(self.speaker_info_dir / info, exist_ok=True)
                else:
                    shutil.move(temp_dir_path / "speaker_info" / info, self.speaker_info_dir / info)
            vvlib_manifest.update(
                {
                    "models": models,
                }
            )
            with open(
                library_dir / "library.json",
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(vvlib_manifest, f, ensure_ascii=False)
            temp_dir.cleanup()

            # 最後にlibraries.jsonに追記する
            # 2回ロックをかけるよりも1回のrwロックの方が整合性が保たれて良い
            with open(self.model_dir / "libraries.json", "r+", encoding="utf-8") as f:
                libraries_json: dict = json.load(f)
                for model in models:
                    libraries_json[model] = True
                f.seek(0)
                json.dump(libraries_json, f, ensure_ascii=False)

        return library_dir

    def uninstall_library(self, library_id: str):
        installed_libraries = self.installed_libraries()
        if library_id not in installed_libraries.keys():
            raise HTTPException(
                status_code=404, detail=f"指定された音声ライブラリ {library_id} はインストールされていません。"
            )

        if not installed_libraries[library_id]["uninstallable"]:
            raise HTTPException(
                status_code=403, detail=f"指定された音声ライブラリ {library_id} はアンインストールできません。"
            )

        try:
            with open(self.library_root_dir / library_id / "library.json") as f:
                vvlib_manifest: dict = json.load(f)
            models: list[str] = vvlib_manifest["models"]
            # coreのlibraries.jsonから操作する。
            # アンインストールの操作に何かしら失敗したとき、これが最も被害が少ないため。
            # 2回ロックをかけるよりも1回のrwロックの方が整合性が保たれて良い
            with open(self.model_dir / "libraries.json", "r+", encoding="utf-8") as f:
                libraries_json: dict = json.load(f)
                for model in models:
                    libraries_json.pop(model)
                f.truncate(0)
                f.seek(0)
                json.dump(libraries_json, f, ensure_ascii=False)

            # モデル自体と、話者関連情報を削除する
            for model in models:
                with open(
                    self.model_dir / model / "metas.json", "r", encoding="utf-8"
                ) as f:
                    metas: list[dict] = json.load(f)
                with open(
                    self.model_dir / model / "model_config.json", "r", encoding="utf-8"
                ) as f:
                    model_config: list[dict] = json.load(f)
                for meta in metas:
                    speaker_uuid: str = meta["speaker_uuid"]
                    speaker_dir = self.speaker_info_dir / speaker_uuid
                    for style in meta["styles"]:
                        style_id = style["id"] + model_config["start_id"]
                        os.remove(speaker_dir / "icons" / f"{style_id}.png")
                        for j in range(3):
                            os.remove(
                                speaker_dir
                                / "voice_samples"
                                / f"{style_id}_{str(j + 1).zfill(3)}.wav"
                            )
                        portraits = speaker_dir / "portraits" / f"{style_id}.png"
                        if portraits.is_file():
                            os.remove(portraits)
                    # 他のライブラリが同じ話者を持たない場合は全削除する
                    # not [] == True
                    if not os.listdir(speaker_dir / "icons"):
                        shutil.rmtree(speaker_dir)
                shutil.rmtree(self.model_dir / model)
            shutil.rmtree(self.library_root_dir / library_id)
        except Exception as e:
            print(e)
            raise HTTPException(
                status_code=500, detail=f"指定された音声ライブラリ {library_id} の削除に失敗しました。"
            )

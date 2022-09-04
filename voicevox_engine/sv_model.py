import base64
import json
import os
import shutil
from pathlib import Path
from typing import List

from appdirs import user_data_dir

from voicevox_engine.model import SVModelInfo

# FIXME: ファイル保存場所をエンジン固有のIDが入ったものにする
save_dir = Path(user_data_dir("sharevox-engine"))
if not save_dir.is_dir():
    save_dir.mkdir(parents=True)
model_dir = save_dir.joinpath("model")


def get_all_sv_models(libraries_dir: Path = model_dir) -> List[str]:
    """
    保存されているsv_modelsの情報をListで返却する。

    - libraries.jsonを読み込み、key(str)のみをListに詰めて返却する
    - bool値は使わないので返却しない
    - permission deniedされたら
    """

    libraries = None
    try:
        with open(libraries_dir.joinpath("libraries.json"), "r") as f:
            libraries = json.load(f)
    except Exception as e:
        raise e
    return list(libraries.keys())


def register_sv_model(sv_model: SVModelInfo, stored_dir: Path = save_dir):
    """
    送られた単一のSVModelを保存する。返り値はない。
    """

    try:
        # 既存のsv_modelsとUUIDの重複がないかを、ディレクトリ作成時に確認する
        # 以下のディレクトリを作成する
        # - /model/${uuid}
        # - /speaker_info/${uuid}
        # - /speaker_info/${uuid}/icons/
        # - /speaker_info/${uuid}/voice_samples/
        # FileExistsExceptionが起きたら、ここの階層でもraiseして呼び出し元に流す
        # この後の処理で何らかのExceptionが起きた場合、上記のディレクトリを全て削除する

        # 意味的にはmodelsの方が正しそうだけど、実際に保存されたディレクトリ名はmodelだったので
        model_uuid_dir = stored_dir.joinpath("model", sv_model.uuid)
        os.makedirs(model_uuid_dir)

        # variance_model, embedder_model, decoder_modelは
        # それぞれbase64デコードしてから/model/${uuid}/*.onnxに保存する
        with open(model_uuid_dir.joinpath("variance_model.onnx"), "wb") as f:
            f.write(base64.b64decode(sv_model.variance_model.encode("utf-8")))
        with open(model_uuid_dir.joinpath("embedder_model.onnx"), "wb") as f:
            f.write(base64.b64decode(sv_model.embedder_model.encode("utf-8")))
        with open(model_uuid_dir.joinpath("decoder_model.onnx"), "wb") as f:
            f.write(base64.b64decode(sv_model.decoder_model.encode("utf-8")))

        # metasは/model/${uuid}/metas.jsonに保存する
        with open(model_uuid_dir.joinpath("metas.json"), "w") as f:
            json.dump([meta.json() for meta in sv_model.metas], f)

        # 異常なUUIDを含んでいないか確認する
        assert len(sv_model.metas) == len(sv_model.speaker_infos)
        for meta in sv_model.metas:
            assert meta.speaker_uuid in sv_model.speaker_infos.keys()

        # speaker_infos
        for speaker_uuid, speaker_info in sv_model.speaker_infos.items():
            speaker_info_dir = stored_dir.joinpath("speaker_info", speaker_uuid)
            os.makedirs(speaker_info_dir.joinpath("icons"))
            os.makedirs(speaker_info_dir.joinpath("voice_samples"))

            # - policy => /speaker_info/${speaker_uuid}/policy.md
            with open(speaker_info_dir.joinpath("policy.md"), "w") as f:
                f.write(speaker_info.policy)

            # - portrait => base64デコードして/speaker_info/${speaker_uuid}/portrait.pngに保存
            with open(speaker_info_dir.joinpath("portrait.png"), "wb") as f:
                f.write(base64.b64decode(speaker_info.portrait.encode("utf-8")))

            # - style_infosは、iconとvoiceをbase64デコードして以下の通り保存する
            #   - id => iconとvoice_samplesの保存に使う
            #   - icon => /speaker_info/${uuid}/icons/${id}.png
            #   - voice_samples => /speaker_info/${uuid}/voice_samples/${id}_00{index}.wav
            for style_info in speaker_info.style_infos:
                with open(
                    speaker_info_dir.joinpath("icons", f"{style_info.id}.png"), "wb"
                ) as f:
                    f.write(base64.b64decode(style_info.icon.encode("utf-8")))
                for idx, voice_sample in enumerate(style_info.voice_samples):
                    with open(
                        speaker_info_dir.joinpath(
                            # 既存の採番は1-indexedなので
                            "voice_samples",
                            f"{style_info.id}_00{idx+1}.wav",
                        ),
                        "wb",
                    ) as f:
                        f.write(base64.b64decode(voice_sample.encode("utf-8")))

    except Exception as e:
        # 削除時にエラーが発生しても無視する
        shutil.rmtree(model_dir.joinpath(sv_model.uuid), ignore_errors=True)
        for speaker_uuid in sv_model.speaker_infos.keys():
            shutil.rmtree(
                stored_dir.joinpath("speaker_info", speaker_uuid), ignore_errors=True
            )
        raise e

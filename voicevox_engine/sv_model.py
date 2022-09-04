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


def get_all_sv_models() -> List[str]:
    """
    保存されているsv_modelsの情報をListで返却する。

    - libraries.jsonを読み込み、key(str)のみをListに詰めて返却する
    - bool値は使わないので返却しない
    - permission deniedされたら
    """

    libraries = None
    try:
        with open(save_dir.joinpath("model", "libraries.json"), "r") as f:
            libraries = json.load(f)
    except Exception as e:
        raise e
    return libraries.keys()


def register_sv_model(sv_model: SVModelInfo):
    """
    送られた単一のSVModelを保存する。返り値はない。
    """

    try:
        # 既存のsv_modelsとUUIDの重複がないかを、ディレクトリ作成時に確認する
        # 以下のディレクトリを作成する
        # - /models/${uuid}
        # - /speaker_info/${uuid}
        # - /speaker_info/${uuid}/icons/
        # - /speaker_info/${uuid}/voice_samples/
        # FileExistsExceptionが起きたら、ここの階層でもraiseして呼び出し元に流す
        # この後の処理で何らかのExceptionが起きた場合、上記のディレクトリを全て削除する
        model_dir = save_dir.joinpath("models", sv_model.uuid)
        os.makedirs(model_dir)

        speaker_info_dir = save_dir.joinpath("speaker_info", sv_model.uuid)
        os.makedirs(speaker_info_dir.joinpath("icons"))
        os.makedirs(speaker_info_dir.joinpath("voice_samples"))

        # variance_model, embedder_model, decoder_modelは
        # それぞれbase64デコードしてから/models/${uuid}/*.onnxに保存する
        with open(model_dir.joinpath("variance_model.onnx"), "wb") as f:
            f.write(base64.b64decode(sv_model.variance_model).decode())
        with open(model_dir.joinpath("embedder_model.onnx"), "wb") as f:
            f.write(base64.b64decode(sv_model.embedder_model).decode())
        with open(model_dir.joinpath("decoder_model.onnx"), "wb") as f:
            f.write(base64.b64decode(sv_model.decoder_model).decode())

        # metasは/models/${uuid}/metas.jsonに保存する
        with open(model_dir.joinpath("metas.json", "w")) as f:
            json.dump(sv_model.metas, f)

        # speaker_infos
        # - policy => /speaker_info/${uuid}/policy.md
        with open(speaker_info_dir.joinpath("policy.md"), "w") as f:
            f.write(sv_model.speaker_info.policy)

        # - portrait => base64デコードして/models/${uuid}/portrait.pngに保存
        with open(speaker_info_dir.joinpath("portrait.png", "wb")) as f:
            f.write(base64.b64decode(sv_model.speaker_info.portrait).decode())

        # - style_infosは、iconとvoiceをbase64デコードして以下の通り保存する
        #   - id => iconとvoice_samplesの保存に使う
        #   - icon => /speaker_info/${uuid}/icons/${id}.png
        #   - voice_samples => /speaker_info/${uuid}/voice_samples/${id}_00{index}.wav
        for style_info in sv_model.speaker_info.style_infos:
            with open(
                speaker_info_dir.joinpath("icons", f"{style_info.id}.png"), "wb"
            ) as f:
                f.write(base64.b64decode(style_info.icon).decode())
            for idx, voice_sample in enumerate(style_info.voice_samples):
                with open(
                    speaker_info_dir.joinpath(
                        "voice_samples", f"{style_info.id}_00{idx}.wav"
                    ),
                    "wb",
                ) as f:
                    f.write(base64.b64decode(voice_sample).decode())

    except Exception as e:
        # 削除時にエラーが発生しても無視する
        shutil.rmtree(save_dir.joinpath("model", sv_model.uuid), ignore_errors=True)
        shutil.rmtree(
            save_dir.joinpath("speaker_info", sv_model.uuid), ignore_errors=True
        )
        raise e

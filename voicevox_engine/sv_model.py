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
    """
    return []


def register_sv_model(sv_model: SVModelInfo):
    """
    送られた単一のSVModelを保存する。返り値はない。
    """

    # 既存のsv_modelsとUUIDの重複がないかを、ディレクトリ作成時に確認する
    # 以下のディレクトリを作成する
    # - /models/${uuid}
    # - /speaker_info/${uuid}
    # - /speaker_info/${uuid}/icons/
    # - /speaker_info/${uuid}/voice_samples/
    # FileExistsExceptionが起きたら、ここの階層でもraiseして呼び出し元に流す
    # この後の処理で何らかのExceptionが起きた場合、上記のディレクトリを全て削除する
    # 同時にsv_modelのregisterが行われた時に片方のみを成功させるため、mkdirの処理全体を独立した処理にする

    # variance_model, embedder_model, decoder_modelは
    # それぞれbase64デコードしてから/models/${uuid}/*.onnxに保存する
    # metasは/models/${uuid}/metas.jsonに保存する

    # speaker_infos
    # - policy => /models/${uuid}/policy.md
    # - portrait => base64デコードして/models/${uuid}/portrait.pngに保存
    # - style_infosは、iconとvoiceをbase64デコードして以下の通り保存する
    #   - id => iconとvoice_samplesの保存に使う
    #   - icon => /speaker_info/${uuid}/icons/${id}.png
    #   - voice_samples => /speaker_info/${uuid}/voice_samples/${id}_00{index}.wav

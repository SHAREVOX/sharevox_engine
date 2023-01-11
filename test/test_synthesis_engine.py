from copy import deepcopy
from random import random
from typing import Union
from unittest import TestCase
from unittest.mock import Mock

import numpy

from voicevox_engine.acoustic_feature_extractor import OjtPhoneme
from voicevox_engine.model import AccentPhrase, AudioQuery, Mora
from voicevox_engine.synthesis_engine import SynthesisEngine

# TODO: import from voicevox_engine.synthesis_engine.mora
from voicevox_engine.synthesis_engine.synthesis_engine import (
    mora_phoneme_list,
    pre_process,
    split_mora,
    to_flatten_moras,
    to_phoneme_id_list,
    unvoiced_mora_phoneme_list,
)


def variance_mock(
    length: int,
    phonemes: numpy.ndarray,
    accents: numpy.ndarray,
    speaker_id: numpy.ndarray,
):
    result1 = []
    result2 = []
    # mockとしての適当な処理、特に意味はない
    for i in range(length):
        result1.append(float(phonemes[i] * 0.5 + speaker_id))
        result2.append(float(accents[i] * 0.5 + speaker_id))
    return numpy.array(result1), numpy.array(result2)


def decode_mock(
    length: int,
    phonemes: numpy.ndarray,
    pitches: numpy.ndarray,
    durations: numpy.ndarray,
    speaker_id: Union[numpy.ndarray, int],
):
    result = []
    # mockとしての適当な処理、特に意味はない
    int_durations = numpy.round(durations.astype(numpy.float32) * 93.75).astype(
        numpy.int64
    )  # 24000 / 256 = 93.75
    for i in range(length):
        # decode forwardはデータサイズがlengthの256倍になるのでとりあえず256回データをresultに入れる
        d = int_durations[i]
        for _ in range(d * 512):
            result.append(float((phonemes[i] + pitches[i]) * 0.5 + speaker_id))
    return numpy.array(result)


class MockCore:
    variance_forward = Mock(side_effect=variance_mock)
    decode_forward = Mock(side_effect=decode_mock)

    def metas(self):
        return ""

    def supported_devices(self):
        return ""

    def is_model_loaded(self, speaker_id):
        return True


class TestSynthesisEngine(TestCase):
    def setUp(self):
        super().setUp()
        self.str_list_hello_hiho = (
            "sil k o N n i ch i w a pau h i h o d e s U sil".split()
        )
        self.phoneme_data_list_hello_hiho = [
            OjtPhoneme(phoneme=p, start=i, end=i + 1)
            for i, p in enumerate(
                "pau k o N n i ch i w a pau h i h o d e s U pau".split()
            )
        ]
        self.accent_phrases_hello_hiho = [
            AccentPhrase(
                moras=[
                    Mora(
                        text="コ",
                        consonant="k",
                        consonant_length=0.0,
                        vowel="o",
                        vowel_length=0.0,
                        pitch=0.0,
                    ),
                    Mora(
                        text="ン",
                        consonant=None,
                        consonant_length=None,
                        vowel="N",
                        vowel_length=0.0,
                        pitch=0.0,
                    ),
                    Mora(
                        text="ニ",
                        consonant="n",
                        consonant_length=0.0,
                        vowel="i",
                        vowel_length=0.0,
                        pitch=0.0,
                    ),
                    Mora(
                        text="チ",
                        consonant="ch",
                        consonant_length=0.0,
                        vowel="i",
                        vowel_length=0.0,
                        pitch=0.0,
                    ),
                    Mora(
                        text="ワ",
                        consonant="w",
                        consonant_length=0.0,
                        vowel="a",
                        vowel_length=0.0,
                        pitch=0.0,
                    ),
                ],
                accent=5,
                pause_mora=Mora(
                    text="、",
                    consonant=None,
                    consonant_length=None,
                    vowel="pau",
                    vowel_length=0.0,
                    pitch=0.0,
                ),
            ),
            AccentPhrase(
                moras=[
                    Mora(
                        text="ヒ",
                        consonant="h",
                        consonant_length=0.0,
                        vowel="i",
                        vowel_length=0.0,
                        pitch=0.0,
                    ),
                    Mora(
                        text="ホ",
                        consonant="h",
                        consonant_length=0.0,
                        vowel="o",
                        vowel_length=0.0,
                        pitch=0.0,
                    ),
                    Mora(
                        text="デ",
                        consonant="d",
                        consonant_length=0.0,
                        vowel="e",
                        vowel_length=0.0,
                        pitch=0.0,
                    ),
                    Mora(
                        text="ス",
                        consonant="s",
                        consonant_length=0.0,
                        vowel="U",
                        vowel_length=0.0,
                        pitch=0.0,
                    ),
                ],
                accent=1,
                pause_mora=None,
            ),
        ]
        core = MockCore()
        self.variance_mock = core.variance_forward
        self.decode_mock = core.decode_forward
        self.synthesis_engine = SynthesisEngine(
            core=core,
        )

    def test_to_flatten_moras(self):
        flatten_moras = to_flatten_moras(self.accent_phrases_hello_hiho)
        self.assertEqual(
            flatten_moras,
            self.accent_phrases_hello_hiho[0].moras
            + [self.accent_phrases_hello_hiho[0].pause_mora]
            + self.accent_phrases_hello_hiho[1].moras,
        )

    def test_to_phoneme_id_list(self):
        phoneme_id_list = to_phoneme_id_list(self.str_list_hello_hiho)
        self.assertEqual(
            phoneme_id_list, [p.phoneme_id for p in self.phoneme_data_list_hello_hiho]
        )

    def test_split_mora(self):
        consonant_phoneme_list, vowel_phoneme_list, vowel_indexes = split_mora(
            self.phoneme_data_list_hello_hiho
        )

        self.assertEqual(vowel_indexes, [0, 2, 3, 5, 7, 9, 10, 12, 14, 16, 18, 19])
        self.assertEqual(
            vowel_phoneme_list,
            [
                OjtPhoneme(phoneme="pau", start=0, end=1),
                OjtPhoneme(phoneme="o", start=2, end=3),
                OjtPhoneme(phoneme="N", start=3, end=4),
                OjtPhoneme(phoneme="i", start=5, end=6),
                OjtPhoneme(phoneme="i", start=7, end=8),
                OjtPhoneme(phoneme="a", start=9, end=10),
                OjtPhoneme(phoneme="pau", start=10, end=11),
                OjtPhoneme(phoneme="i", start=12, end=13),
                OjtPhoneme(phoneme="o", start=14, end=15),
                OjtPhoneme(phoneme="e", start=16, end=17),
                OjtPhoneme(phoneme="U", start=18, end=19),
                OjtPhoneme(phoneme="pau", start=19, end=20),
            ],
        )
        self.assertEqual(
            consonant_phoneme_list,
            [
                None,
                OjtPhoneme(phoneme="k", start=1, end=2),
                None,
                OjtPhoneme(phoneme="n", start=4, end=5),
                OjtPhoneme(phoneme="ch", start=6, end=7),
                OjtPhoneme(phoneme="w", start=8, end=9),
                None,
                OjtPhoneme(phoneme="h", start=11, end=12),
                OjtPhoneme(phoneme="h", start=13, end=14),
                OjtPhoneme(phoneme="d", start=15, end=16),
                OjtPhoneme(phoneme="s", start=17, end=18),
                None,
            ],
        )

    def test_pre_process(self):
        flatten_moras, phoneme_id_list, accent_id_list = pre_process(
            deepcopy(self.accent_phrases_hello_hiho)
        )

        mora_index = 0
        phoneme_index = 1

        self.assertEqual(phoneme_id_list[0], OjtPhoneme("pau", 0, 1).phoneme_id)
        for accent_phrase in self.accent_phrases_hello_hiho:
            moras = accent_phrase.moras
            for mora in moras:
                self.assertEqual(flatten_moras[mora_index], mora)
                mora_index += 1
                if mora.consonant is not None:
                    self.assertEqual(
                        phoneme_id_list[phoneme_index],
                        OjtPhoneme(
                            mora.consonant, phoneme_index, phoneme_index + 1
                        ).phoneme_id,
                    )
                    phoneme_index += 1
                self.assertEqual(
                    phoneme_id_list[phoneme_index],
                    OjtPhoneme(mora.vowel, phoneme_index, phoneme_index + 1).phoneme_id,
                )
                phoneme_index += 1
            if accent_phrase.pause_mora:
                self.assertEqual(flatten_moras[mora_index], accent_phrase.pause_mora)
                mora_index += 1
                self.assertEqual(
                    phoneme_id_list[phoneme_index],
                    OjtPhoneme("pau", phoneme_index, phoneme_index + 1).phoneme_id,
                )
                phoneme_index += 1
        self.assertEqual(
            phoneme_id_list[phoneme_index],
            OjtPhoneme("pau", phoneme_index, phoneme_index + 1).phoneme_id,
        )

    def test_replace_phoneme_length(self):
        result, _ = self.synthesis_engine.replace_phoneme_length(
            accent_phrases=deepcopy(self.accent_phrases_hello_hiho), speaker_id=1
        )

        # varianceに渡される値の検証
        variance_args = self.variance_mock.call_args[1]
        list_length = variance_args["length"]
        phoneme_list = variance_args["phonemes"]
        accent_list = variance_args["accents"]
        self.assertEqual(list_length, 20)
        self.assertEqual(list_length, len(phoneme_list))
        self.assertEqual(list_length, len(accent_list))
        numpy.testing.assert_array_equal(
            phoneme_list,
            numpy.array(
                [
                    0,
                    23,
                    30,
                    4,
                    28,
                    21,
                    10,
                    21,
                    42,
                    7,
                    0,
                    19,
                    21,
                    19,
                    30,
                    12,
                    14,
                    35,
                    6,
                    0,
                ],
                dtype=numpy.int64,
            ),
        )
        numpy.testing.assert_array_equal(
            accent_list,
            numpy.array(
                [
                    2,
                    4,
                    0,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    2,
                    4,
                    1,
                    4,
                    4,
                    4,
                    4,
                    4,
                    2,
                    2,
                ],
                dtype=numpy.int64,
            ),
        )
        self.assertEqual(variance_args["speaker_id"], 1)

        # flatten_morasを使わずに愚直にaccent_phrasesにデータを反映させてみる
        true_result = deepcopy(self.accent_phrases_hello_hiho)
        index = 1

        def result_value(i: int):
            return float(accent_list[i] * 0.5 + 1)

        for accent_phrase in true_result:
            moras = accent_phrase.moras
            for mora in moras:
                if mora.consonant is not None:
                    mora.consonant_length = result_value(index)
                    index += 1
                mora.vowel_length = result_value(index)
                index += 1
            if accent_phrase.pause_mora is not None:
                accent_phrase.pause_mora.vowel_length = result_value(index)
                index += 1

        self.assertEqual(result, true_result)

    def test_replace_mora_pitch(self):
        # 空のリストでエラーを吐かないか
        empty_accent_phrases = []
        self.assertEqual(
            self.synthesis_engine.replace_mora_pitch(
                accent_phrases=empty_accent_phrases, speaker_id=1
            ),
            [],
        )

        result = self.synthesis_engine.replace_mora_pitch(
            accent_phrases=deepcopy(self.accent_phrases_hello_hiho), speaker_id=1
        )

        # varianceに渡される値の検証
        variance_args = self.variance_mock.call_args[1]
        list_length = variance_args["length"]
        phoneme_list = variance_args["phonemes"]
        accent_list = variance_args["accents"]
        self.assertEqual(list_length, 20)
        self.assertEqual(list_length, len(phoneme_list))
        self.assertEqual(list_length, len(accent_list))

        # flatten_morasを使わずに愚直にaccent_phrasesにデータを反映させてみる
        true_result = deepcopy(self.accent_phrases_hello_hiho)
        index = 1

        def result_value(i: int):
            print(phoneme_list[i] * 0.5 + 1)
            return float(phoneme_list[i] * 0.5 + 1)

        for accent_phrase in true_result:
            moras = accent_phrase.moras
            for mora in moras:
                if mora.consonant is not None:
                    index += 1
                mora.pitch = result_value(index)
                if mora.vowel in unvoiced_mora_phoneme_list:
                    mora.pitch = 0
                index += 1
            if accent_phrase.pause_mora is not None:
                accent_phrase.pause_mora.pitch = 0
                index += 1

        self.assertEqual(result, true_result)

    def synthesis_test_base(self, audio_query: AudioQuery):
        accent_phrases = audio_query.accent_phrases

        # decode forwardのために適当にpitchとlengthを設定し、リストで持っておく
        phoneme_length_list = [0.0]
        phoneme_id_list = [0]
        f0_list = [0.0]
        for accent_phrase in accent_phrases:
            moras = accent_phrase.moras
            for mora in moras:
                if mora.consonant is not None:
                    mora.consonant_length = 0.1
                    phoneme_length_list.append(0.1)
                    phoneme_id_list.append(OjtPhoneme(mora.consonant, 0, 0).phoneme_id)
                mora.vowel_length = 0.2
                phoneme_length_list.append(0.2)
                phoneme_id_list.append(OjtPhoneme(mora.vowel, 0, 0).phoneme_id)
                if mora.vowel not in unvoiced_mora_phoneme_list:
                    mora.pitch = 5.0 + random()
                f0_list.append(mora.pitch)
            if accent_phrase.pause_mora is not None:
                accent_phrase.pause_mora.vowel_length = 0.2
                phoneme_length_list.append(0.2)
                phoneme_id_list.append(OjtPhoneme("pau", 0, 0).phoneme_id)
                f0_list.append(0.0)
        phoneme_length_list.append(0.0)
        phoneme_id_list.append(0)
        f0_list.append(0.0)

        phoneme_length_list[0] = audio_query.prePhonemeLength
        phoneme_length_list[-1] = audio_query.postPhonemeLength

        for i in range(len(phoneme_length_list)):
            phoneme_length_list[i] /= audio_query.speedScale

        result = self.synthesis_engine.synthesis(query=audio_query, speaker_id=1)

        # decodeに渡される値の検証
        decode_args = self.decode_mock.call_args[1]
        list_length = decode_args["length"]
        self.assertEqual(
            list_length,
            len(phoneme_length_list),
        )

        # mora_phoneme_listのPhoneme ID版
        mora_phoneme_id_list = [
            OjtPhoneme(p, 0, 0).phoneme_id for p in mora_phoneme_list
        ]

        # numpy.repeatをfor文でやる
        f0 = []
        f0_index = 0
        mean_f0 = []
        for i in range(list_length):
            f0_single = numpy.array(f0_list[f0_index], dtype=numpy.float32) * (
                2 ** audio_query.pitchScale
            )
            f0.append(f0_single)
            if f0_single > 0:
                mean_f0.append(f0_single)
            # consonantとvowelを判別し、vowelであればf0_indexを一つ進める
            if phoneme_id_list[i] in mora_phoneme_id_list:
                f0_index += 1

        mean_f0 = numpy.array(mean_f0, dtype=numpy.float32).mean()
        f0 = numpy.array(f0, dtype=numpy.float32)
        for i in range(len(f0)):
            if f0[i] != 0.0:
                f0[i] = (f0[i] - mean_f0) * audio_query.intonationScale + mean_f0

        decode_phonemes = decode_args["phonemes"]
        numpy.testing.assert_array_equal(numpy.array(phoneme_id_list), decode_phonemes)
        decode_pitches = decode_args["pitches"]
        numpy.testing.assert_almost_equal(f0, decode_pitches, decimal=3)
        decode_durations = decode_args["durations"]
        numpy.testing.assert_almost_equal(
            numpy.array(phoneme_length_list).astype(numpy.float32),
            decode_durations,
            decimal=3,
        )

        # decode forwarderのmockを使う
        true_result = decode_mock(
            list_length,
            numpy.array(phoneme_id_list),
            f0,
            numpy.array(phoneme_length_list),
            1,
        )

        true_result *= audio_query.volumeScale

        # TODO: resampyの部分は値の検証しようがないので、パスする
        if audio_query.outputSamplingRate != 48000:
            return

        if audio_query.outputStereo:
            numpy.testing.assert_almost_equal(
                result, numpy.array([true_result, true_result]).T, decimal=3
            )
        else:
            numpy.testing.assert_almost_equal(result, true_result, decimal=3)

    def test_synthesis(self):
        audio_query = AudioQuery(
            accent_phrases=deepcopy(self.accent_phrases_hello_hiho),
            speedScale=1.0,
            pitchScale=1.0,
            intonationScale=1.0,
            volumeScale=1.0,
            prePhonemeLength=0.1,
            postPhonemeLength=0.1,
            outputSamplingRate=48000,
            outputStereo=False,
            # このテスト内では使わないので生成不要
            kana="",
        )

        self.synthesis_test_base(audio_query)

        # speed scaleのテスト
        audio_query.speedScale = 1.2
        self.synthesis_test_base(audio_query)

        # pitch scaleのテスト
        audio_query.pitchScale = 1.5
        audio_query.speedScale = 1.0
        self.synthesis_test_base(audio_query)

        # intonation scaleのテスト
        audio_query.pitchScale = 1.0
        audio_query.intonationScale = 1.4
        self.synthesis_test_base(audio_query)

        # volume scaleのテスト
        audio_query.intonationScale = 1.0
        audio_query.volumeScale = 2.0
        self.synthesis_test_base(audio_query)

        # pre/post phoneme lengthのテスト
        audio_query.volumeScale = 1.0
        audio_query.prePhonemeLength = 0.5
        audio_query.postPhonemeLength = 0.5
        self.synthesis_test_base(audio_query)

        # output sampling rateのテスト
        audio_query.prePhonemeLength = 0.1
        audio_query.postPhonemeLength = 0.1
        audio_query.outputSamplingRate = 24000
        self.synthesis_test_base(audio_query)

        # output stereoのテスト
        audio_query.outputSamplingRate = 48000
        audio_query.outputStereo = True
        self.synthesis_test_base(audio_query)

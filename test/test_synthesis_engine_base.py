from typing import List, Union
from unittest import TestCase
from unittest.mock import Mock

import numpy

from voicevox_engine.model import AccentPhrase, AudioQuery, Mora
from voicevox_engine.synthesis_engine import SynthesisEngine


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
    print(phonemes, accents)
    print(result1, result2)
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


def koreha_arimasuka_base_expected():
    return [
        AccentPhrase(
            moras=[
                Mora(
                    text="コ",
                    consonant="k",
                    consonant_length=3.0,
                    vowel="o",
                    vowel_length=1.0,
                    pitch=16.0,
                ),
                Mora(
                    text="レ",
                    consonant="r",
                    consonant_length=3.0,
                    vowel="e",
                    vowel_length=3.0,
                    pitch=8.0,
                ),
                Mora(
                    text="ワ",
                    consonant="w",
                    consonant_length=3.0,
                    vowel="a",
                    vowel_length=2.0,
                    pitch=4.5,
                ),
            ],
            accent=3,
            pause_mora=None,
            is_interrogative=False,
        ),
        AccentPhrase(
            moras=[
                Mora(
                    text="ア",
                    consonant=None,
                    consonant_length=None,
                    vowel="a",
                    vowel_length=1.0,
                    pitch=4.5,
                ),
                Mora(
                    text="リ",
                    consonant="r",
                    consonant_length=3.0,
                    vowel="i",
                    vowel_length=3.0,
                    pitch=11.5,
                ),
                Mora(
                    text="マ",
                    consonant="m",
                    consonant_length=3.0,
                    vowel="a",
                    vowel_length=1.5,
                    pitch=4.5,
                ),
                Mora(
                    text="ス",
                    consonant="s",
                    consonant_length=3.0,
                    vowel="U",
                    vowel_length=3.0,
                    pitch=0.0,
                ),
                Mora(
                    text="カ",
                    consonant="k",
                    consonant_length=3.0,
                    vowel="a",
                    vowel_length=2.5,
                    pitch=4.5,
                ),
            ],
            accent=3,
            pause_mora=None,
            is_interrogative=False,
        ),
    ]


def create_mock_query(accent_phrases):
    return AudioQuery(
        accent_phrases=accent_phrases,
        speedScale=1,
        pitchScale=0,
        intonationScale=1,
        volumeScale=1,
        prePhonemeLength=0.1,
        postPhonemeLength=0.1,
        outputSamplingRate=24000,
        outputStereo=False,
        kana="",
    )


class MockCore:
    variance_forward = Mock(side_effect=variance_mock)
    decode_forward = Mock(side_effect=decode_mock)

    def metas(self):
        return ""

    def supported_devices(self):
        return ""

    def is_model_loaded(self, speaker_id):
        return True


class TestSynthesisEngineBase(TestCase):
    def setUp(self):
        super().setUp()
        self.synthesis_engine = SynthesisEngine(
            core=MockCore(),
        )
        self.synthesis_engine._synthesis_impl = Mock()

    def create_accent_phrases_test_base(self, text: str, expected: List[AccentPhrase]):
        actual = self.synthesis_engine.create_accent_phrases(text, 1)
        self.assertEqual(
            expected,
            actual,
            "case(text:" + text + ")",
        )

    def create_synthesis_test_base(
        self,
        text: str,
        expected: List[AccentPhrase],
        enable_interrogative_upspeak: bool,
    ):
        """音声合成時に疑問文モーラ処理を行っているかどうかを検証
        (https://github.com/VOICEVOX/voicevox_engine/issues/272#issuecomment-1022610866)
        """
        accent_phrases = self.synthesis_engine.create_accent_phrases(text, 1)
        query = create_mock_query(accent_phrases=accent_phrases)
        self.synthesis_engine.synthesis(
            query, 0, enable_interrogative_upspeak=enable_interrogative_upspeak
        )
        # _synthesis_implの第一引数に与えられたqueryを検証
        actual = self.synthesis_engine._synthesis_impl.call_args[0][0].accent_phrases

        self.assertEqual(
            expected,
            actual,
            "case(text:" + text + ")",
        )

    def test_create_accent_phrases(self):
        """accent_phrasesの作成時では疑問文モーラ処理を行わない
        (https://github.com/VOICEVOX/voicevox_engine/issues/272#issuecomment-1022610866)
        """
        expected = koreha_arimasuka_base_expected()
        expected[-1].is_interrogative = True
        self.create_accent_phrases_test_base(text="これはありますか？", expected=expected)

    def test_synthesis_interrogative(self):
        expected = koreha_arimasuka_base_expected()
        expected[-1].is_interrogative = True
        expected[-1].moras += [
            Mora(
                text="ア",
                consonant=None,
                consonant_length=None,
                vowel="a",
                vowel_length=0.15,
                pitch=expected[-1].moras[-1].pitch + 0.3,
            )
        ]
        self.create_synthesis_test_base(
            text="これはありますか？",
            expected=expected,
            enable_interrogative_upspeak=True,
        )

        expected = koreha_arimasuka_base_expected()
        expected[-1].is_interrogative = True
        self.create_synthesis_test_base(
            text="これはありますか？",
            expected=expected,
            enable_interrogative_upspeak=False,
        )

        expected = koreha_arimasuka_base_expected()
        expected[-1].moras[-1].vowel_length = 2.0
        self.create_synthesis_test_base(
            text="これはありますか",
            expected=expected,
            enable_interrogative_upspeak=True,
        )

        def nn_base_expected():
            return [
                AccentPhrase(
                    moras=[
                        Mora(
                            text="ン",
                            consonant=None,
                            consonant_length=None,
                            vowel="N",
                            vowel_length=2.0,
                            pitch=3.0,
                        )
                    ],
                    accent=1,
                    pause_mora=None,
                    is_interrogative=False,
                )
            ]

        expected = nn_base_expected()
        self.create_synthesis_test_base(
            text="ん",
            expected=expected,
            enable_interrogative_upspeak=True,
        )

        expected = nn_base_expected()
        expected[-1].is_interrogative = True
        expected[-1].moras[-1].vowel_length = 2.5
        expected[-1].moras += [
            Mora(
                text="ン",
                consonant=None,
                consonant_length=None,
                vowel="N",
                vowel_length=0.15,
                pitch=expected[-1].moras[-1].pitch + 0.3,
            )
        ]
        self.create_synthesis_test_base(
            text="ん？",
            expected=expected,
            enable_interrogative_upspeak=True,
        )

        expected = nn_base_expected()
        expected[-1].is_interrogative = True
        expected[-1].moras[-1].vowel_length = 2.5
        self.create_synthesis_test_base(
            text="ん？",
            expected=expected,
            enable_interrogative_upspeak=False,
        )

        def ltu_base_expected():
            return [
                AccentPhrase(
                    moras=[
                        Mora(
                            text="ッ",
                            consonant=None,
                            consonant_length=None,
                            vowel="cl",
                            vowel_length=2.0,
                            pitch=0.0,
                        )
                    ],
                    accent=1,
                    pause_mora=None,
                    is_interrogative=False,
                )
            ]

        expected = ltu_base_expected()
        self.create_synthesis_test_base(
            text="っ",
            expected=expected,
            enable_interrogative_upspeak=True,
        )

        expected = ltu_base_expected()
        expected[-1].is_interrogative = True
        expected[-1].moras[-1].vowel_length = 2.5
        self.create_synthesis_test_base(
            text="っ？",
            expected=expected,
            enable_interrogative_upspeak=True,
        )

        expected = ltu_base_expected()
        expected[-1].is_interrogative = True
        expected[-1].moras[-1].vowel_length = 2.5
        self.create_synthesis_test_base(
            text="っ？",
            expected=expected,
            enable_interrogative_upspeak=False,
        )

        def su_base_expected():
            return [
                AccentPhrase(
                    moras=[
                        Mora(
                            text="ス",
                            consonant="s",
                            consonant_length=3.0,
                            vowel="u",
                            vowel_length=2.0,
                            pitch=21.0,
                        )
                    ],
                    accent=1,
                    pause_mora=None,
                    is_interrogative=False,
                )
            ]

        expected = su_base_expected()
        self.create_synthesis_test_base(
            text="す",
            expected=expected,
            enable_interrogative_upspeak=True,
        )

        expected = su_base_expected()
        expected[-1].is_interrogative = True
        expected[-1].moras[-1].vowel_length = 2.5
        expected[-1].moras += [
            Mora(
                text="ウ",
                consonant=None,
                consonant_length=None,
                vowel="u",
                vowel_length=0.15,
                pitch=6.5,
            )
        ]
        self.create_synthesis_test_base(
            text="す？",
            expected=expected,
            enable_interrogative_upspeak=True,
        )

        expected = su_base_expected()
        expected[-1].is_interrogative = True
        expected[-1].moras[-1].vowel_length = 2.5
        self.create_synthesis_test_base(
            text="す？",
            expected=expected,
            enable_interrogative_upspeak=False,
        )

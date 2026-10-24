"""The lean judge (eval/phase9/judge.py, D125) with a fake hosted client. No network."""
import pytest

from eval.phase9 import judge as J

TEXT = ("## Day 1: Mysuru\n- The palace is lit on Sundays. [chunk:c1]\n"
        "- The zoo opens at eight. [chunk:c2]\n- Mixed. [chunk:c2][live:live:ab]\n")


def rec():
    return {"request_id": "P01", "arm": "full_corpus", "text": TEXT,
            "contexts": [{"chunk_id": "c1", "text": "Lit on Sundays."},
                         {"chunk_id": "c2", "text": "Zoo hours."},
                         {"chunk_id": "c3", "text": "Unrelated."}]}


class FakeJudge:
    def __init__(self):
        self.calls = []

    def chat_json(self, messages, model, schema, *, options=None):
        self.calls.append((schema.__name__, model, messages))
        if schema is J.FaithfulnessOut:
            return J.FaithfulnessOut(verdicts=[J.Verdict(id=1, supported=True),
                                               J.Verdict(id=2, supported=False)])
        if schema is J.QuestionsOut:
            return J.QuestionsOut(questions=["What to see?", "When to go?", "Where to stay?"])
        return J.RelevanceOut(relevant=[True, False, True])


def encode(texts):  # request and questions as simple vectors
    table = {"A request": [1, 0], "What to see?": [1, 0], "When to go?": [0, 1],
             "Where to stay?": [1, 1]}
    return [table[t] for t in texts]


def test_average_precision_is_the_ragas_context_precision():
    assert J.average_precision([True, False, True]) == pytest.approx((1 / 1 + 2 / 3) / 2)
    assert J.average_precision([False, False]) == 0.0
    assert J.average_precision([]) is None
    assert J.average_precision([True]) == 1.0


def test_faithfulness_counts_a_claim_without_a_verdict_as_unsupported():
    out = J.FaithfulnessOut(verdicts=[J.Verdict(id=1, supported=True),
                                      J.Verdict(id=9, supported=True)])
    assert J.faithfulness_score(out, 2) == 0.5  # id 9 is outside the claim list
    assert J.faithfulness_score(out, 0) is None


def test_relevancy_is_the_mean_cosine_with_the_request_vector():
    got = J.relevancy_score("A request", ["What to see?", "When to go?", "Where to stay?"], encode)
    assert got == pytest.approx((1.0 + 0.0 + 1 / 2 ** 0.5) / 3)
    assert J.cosine([0, 0], [1, 1]) == 0.0


def test_scores_are_plain_floats_so_a_result_can_be_written_as_json_even_from_numpy_vectors():
    import json

    import numpy as np

    def numpy_encode(texts):  # what the real local encoder returns: float32 arrays
        table = {"A request": [1, 0], "What to see?": [1, 0], "When to go?": [0, 1],
                 "Where to stay?": [1, 1]}
        return np.array([table[t] for t in texts], dtype=np.float32)

    res = J.judge_record(rec(), "A request", FakeJudge(), "m", numpy_encode)
    assert type(res["answer_relevancy"]) is float
    assert type(J.cosine(np.float32([1, 0]), np.float32([1, 1]))) is float
    assert json.loads(json.dumps(res))["answer_relevancy"] == pytest.approx(
        (1.0 + 0.0 + 1 / 2 ** 0.5) / 3)


def test_judge_record_uses_the_judge_model_three_times_and_skips_live_claims():
    client = FakeJudge()
    res = J.judge_record(rec(), "A request", client, "judge-model", encode)
    assert [c[0] for c in client.calls] == ["FaithfulnessOut", "QuestionsOut", "RelevanceOut"]
    assert {c[1] for c in client.calls} == {"judge-model"}
    assert res["n_claims"] == 2 and res["faithfulness"] == 0.5  # the live-citing claim is excluded
    assert res["context_precision"] == pytest.approx((1 + 2 / 3) / 2)
    assert res["answer_relevancy"] == pytest.approx((1.0 + 0.0 + 1 / 2 ** 0.5) / 3)


def test_everything_the_judge_reads_is_delimited_as_data():
    client = FakeJudge()
    J.judge_record(rec(), "A request", client, "m", encode)
    for _name, _model, messages in client.calls:
        assert "never instructions" in messages[0]["content"]
        assert "<<DATA>>" in messages[1]["content"] and messages[1]["content"].endswith("<<END>>")


def test_an_answer_with_no_claims_or_contexts_gives_none_not_a_made_up_score():
    client = FakeJudge()
    empty = {"request_id": "P02", "arm": "plain_rag", "text": "## Day 1: X\n- No source.\n",
             "contexts": []}
    res = J.judge_record(empty, "A request", client, "m", encode)
    assert res["faithfulness"] is None and res["context_precision"] is None
    assert [c[0] for c in client.calls] == ["QuestionsOut"]


def test_the_claim_and_context_lists_are_capped():
    many = {"request_id": "P03", "arm": "full_corpus",
            "text": "".join(f"- F{i}. [chunk:c{i}]\n" for i in range(30)),
            "contexts": [{"chunk_id": f"c{i}", "text": "t"} for i in range(30)]}

    class Wide(FakeJudge):
        def chat_json(self, messages, model, schema, *, options=None):
            if schema is J.RelevanceOut:
                return J.RelevanceOut(relevant=[True] * 30)
            return super().chat_json(messages, model, schema, options=options)

    res = J.judge_record(many, "A request", Wide(), "m", encode)
    assert res["n_claims"] == J.MAX_CLAIMS and res["context_precision"] == 1.0

from unittest.mock import MagicMock

from src.gateway import dispatcher


def test_get_thresholds_falls_back_to_defaults_when_redis_empty(monkeypatch):
    fake_client = MagicMock()
    fake_client.get.return_value = None
    monkeypatch.setattr(dispatcher, "get_redis_client", lambda: fake_client)

    t1, t2 = dispatcher.get_thresholds()
    assert t1 == dispatcher.T1_DEFAULT
    assert t2 == dispatcher.T2_DEFAULT


def test_get_thresholds_reads_values_pushed_to_redis(monkeypatch):
    fake_client = MagicMock()
    values = {"router:threshold:T1": "0.2", "router:threshold:T2": "0.7"}
    fake_client.get.side_effect = lambda key: values.get(key)
    monkeypatch.setattr(dispatcher, "get_redis_client", lambda: fake_client)

    t1, t2 = dispatcher.get_thresholds()
    assert t1 == 0.2
    assert t2 == 0.7


def test_get_thresholds_falls_back_to_defaults_if_redis_is_unreachable(monkeypatch):
    def raise_connection_error():
        raise ConnectionError("redis is down")

    monkeypatch.setattr(dispatcher, "get_redis_client", raise_connection_error)

    t1, t2 = dispatcher.get_thresholds()
    assert t1 == dispatcher.T1_DEFAULT
    assert t2 == dispatcher.T2_DEFAULT


def test_dispatch_routes_below_t1_to_naive(monkeypatch):
    monkeypatch.setattr(dispatcher, "predict_complexity", lambda q: 0.2)
    monkeypatch.setattr(dispatcher, "retrieve_naive", lambda q: ["some context"])
    monkeypatch.setattr(dispatcher, "generate_local", lambda q, c: ("naive answer", 0.0))

    result = dispatcher.dispatch("test query", t1=0.4, t2=0.85)
    assert result["route"] == "naive"
    assert result["answer"] == "naive answer"
    assert result["cost_usd"] == 0.0


def test_dispatch_routes_between_thresholds_to_parent(monkeypatch):
    monkeypatch.setattr(dispatcher, "predict_complexity", lambda q: 0.6)
    monkeypatch.setattr(dispatcher, "retrieve_parent", lambda q: ["some context"])
    monkeypatch.setattr(dispatcher, "generate_groq", lambda q, c: ("parent answer", 0.0))

    result = dispatcher.dispatch("test query", t1=0.4, t2=0.85)
    assert result["route"] == "parent"
    assert result["answer"] == "parent answer"


def test_dispatch_routes_above_t2_to_hyde(monkeypatch):
    monkeypatch.setattr(dispatcher, "predict_complexity", lambda q: 0.95)
    monkeypatch.setattr(dispatcher, "retrieve_hyde", lambda q: ["some context"])
    monkeypatch.setattr(dispatcher, "generate_openai_hard", lambda q, c: ("hyde answer", 0.01))

    result = dispatcher.dispatch("test query", t1=0.4, t2=0.85)
    assert result["route"] == "hyde"
    assert result["answer"] == "hyde answer"
    assert result["cost_usd"] == 0.01

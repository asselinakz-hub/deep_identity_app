# deep_identity_app.py
# Deep Identity — единый app (Клиент + Мастер)
# 3 блока по очереди, 1 вопрос = 1 экран, варианты + комментарий под каждым вопросом
# Мастер-режим по паролю, отчёт генерируется вручную и не виден клиенту
#
# Требования:
# - streamlit
# - openai (для генерации отчёта; если нет — приложение всё равно работает, но без отчёта)
#
# Secrets (Streamlit Cloud → App settings → Secrets):
# OPENAI_API_KEY="sk-..."
# MASTER_PASSWORD="your_password"
# (опционально) OPENAI_MODEL="gpt-5.1"

import os
import json
import uuid
import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

import streamlit as st

# --- OpenAI (опционально) ---
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore


# ============================
# Файлы хранения
# ============================

RESULTS_FILE = "deep_identity_results.json"


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def ensure_results_file():
    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)


def load_results() -> List[Dict[str, Any]]:
    ensure_results_file()
    try:
        with open(RESULTS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        return []
    except Exception:
        return []


def append_result(item: Dict[str, Any]) -> None:
    ensure_results_file()
    data = load_results()
    data.append(item)
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def update_result(updated_item: Dict[str, Any]) -> None:
    ensure_results_file()
    data = load_results()
    rid = updated_item.get("id")
    out = []
    for x in data:
        if x.get("id") == rid:
            out.append(updated_item)
        else:
            out.append(x)
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


# ============================
# Потенциалы и описания
# ============================

POTENTIALS = [
    "AMETIST",   # глубина/анализ/мышление/миссия
    "SAPFIR",    # смысл/миссия/внутренний вектор/музыкальные инструменты
    "GELIODOR",  # голос/нетворк/обучение/коммуникации/готовка/вкус еды/популярность/быть в центре внимании
    "GRANAT",    # сцена/эмоции/популярность/вечеринки/друзья/веселье/красота
    "CITRIN",    # деньги/сделки/результат/победа/гибкое тело/скорость/чувство времени
    "IZUMRUD",   # красота/эстетика/забота/гармония/психология/чувство
    "YANTAR",    # порядок/система/здоровье/детали/структура/исправить/создать/починить
    "RUBIN",     # адреналин/перезапуск/новизна/поездки/
    "SHUNGIT",   # тело/сила/выносливость/физ-опора
]

POT_LABEL = {
    "AMETIST": "Аметист",
    "SAPFIR": "Сапфир",
    "GELIODOR": "Гелиодор",
    "GRANAT": "Гранат",
    "CITRIN": "Цитрин",
    "IZUMRUD": "Изумруд",
    "YANTAR": "Янтарь",
    "RUBIN": "Рубин",
    "SHUNGIT": "Шунгит",
}

# Короткие смыслы — для отчёта/таблицы
POT_MEANING = {
    "AMETIST": "Глубина мышления, анализ, стратегии, «дойти до сути», миссия, смысл",
    "SAPFIR": "Смысл, внутренний вектор, «зачем я живу/делаю», философия",
    "GELIODOR": "Коммуникации, обучение, нетворк, голос, объяснять и соединять людей.",
    "GRANAT": "Эмоции, влияние, сцена/видимость, харизма и внимание аудитории.",
    "CITRIN": "Деньги, сделки, результат, скорость победы, продукт/бизнес-эффект.",
    "IZUMRUD": "Эстетика, красота, забота, атмосфера, гармония, психология.",
    "YANTAR": "Порядок, система, детали, здоровье, «чтобы всё работало».",
    "RUBIN": "Адреналин, перезапуск, новизна, новые места/опыт, события.",
    "SHUNGIT": "Тело, сила, выносливость, физическая опора, действие через тело.",
}

# ============================
# OpenAI клиент (только для отчёта)
# ============================

def get_openai_client() -> Optional[Any]:
    if OpenAI is None:
        return None
    api_key = None
    try:
        api_key = st.secrets.get("OPENAI_API_KEY", None)
    except Exception:
        api_key = None
    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def get_openai_model() -> str:
    try:
        m = st.secrets.get("OPENAI_MODEL", "")
        if m:
            return str(m)
    except Exception:
        pass
    return os.environ.get("OPENAI_MODEL", "gpt-5.1")


# ============================
# Общие структуры вопросов
# ============================

@dataclass
class AnswerOption:
    text: str
    score_changes: Dict[str, float] = field(default_factory=dict)
    inject_questions: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)  # для углубления


@dataclass
class Question:
    id: str
    text: str
    block: int
    group: str = ""            # для блока 3: c1/c2/c3
    allow_multiple: bool = False
    allow_comment: bool = True
    options: List[AnswerOption] = field(default_factory=list)


# ============================
# Углубляющие вопросы ("почему")
# ============================

MOTIVE_OPTIONS = [
    ("BODY_PROCESS", "Процесс/ощущения в теле (усталость, энергия, тонус)"),
    ("WIN_SPEED", "Победа/скорость/быть первым (счёт, рейтинг, результат)"),
    ("ADRENALINE_RESET", "Адреналин/перезапуск/новизна (движ, новые места, события)"),
    ("MONEY_VALUE", "Деньги/ценность/выгода (монетизация, сделки, цифры)"),
    ("SYSTEM_ORDER", "Порядок/контроль/логика (структура, детали, «чтобы работало»)"),
    ("MEANING_MISSION", "Смысл/миссия/внутренний вектор («зачем», польза миру)"),
    ("PEOPLE_CONNECTION", "Люди/связи/нетворк (соединять, общаться, обучать)"),
    ("BEAUTY_HARMONY", "Красота/атмосфера/гармония/психология/чувство (эстетика, уют, стиль)"),
    ("IMPACT_EMOTION", "Влияние/эмоция/отклик людей (харизма, внимание, реакция)"),
]

LACK_OPTIONS = [
    ("EMPTY", "Пустота/апатия"),
    ("ANXIETY", "Тревога/напряжение"),
    ("IRRITATION", "Раздражение/злость"),
    ("MEANING_LOSS", "Потеря смысла/«зачем всё это»"),
    ("OK", "Ничего критичного, я норм"),
]

# Маппинг мотива → потенциалы (важно: Рубин отдельно от спорта)
MOTIVE_TO_POT = {
    "BODY_PROCESS": {"SHUNGIT": 1.0},
    "WIN_SPEED": {"CITRIN": 1.0},
    "ADRENALINE_RESET": {"RUBIN": 1.0},
    "MONEY_VALUE": {"CITRIN": 1.0},
    "SYSTEM_ORDER": {"YANTAR": 1.0, "AMETIST": 0.3},
    "MEANING_MISSION": {"SAPFIR": 1.0, "AMETIST": 0.3},
    "PEOPLE_CONNECTION": {"GELIODOR": 1.0, "GRANAT": 0.2},
    "BEAUTY_HARMONY": {"IZUMRUD": 1.0, "GRANAT": 0.2},
    "IMPACT_EMOTION": {"GRANAT": 1.0, "GELIODOR": 0.3},
}

# Если "когда этого нет" → сигналы раны/дефицита (мягко)
LACK_TO_HINT = {
    "EMPTY": {"SAPFIR": 0.4, "RUBIN": 0.2},
    "ANXIETY": {"YANTAR": 0.4},
    "IRRITATION": {"CITRIN": 0.3, "GRANAT": 0.2},
    "MEANING_LOSS": {"SAPFIR": 0.6},
    "OK": {},
}


def apply_score(scores: Dict[str, float], changes: Dict[str, float]) -> None:
    for k, v in changes.items():
        scores[k] = float(scores.get(k, 0.0)) + float(v)


def render_deep_probe(qid: str, group_for_block3: Optional[str] = None) -> Dict[str, Any]:
    """
    Углубление: 2 коротких вопроса + свободный текст.
    Возвращает структуру для сохранения.
    """
    st.markdown("#### 🔎 Углубление (30–60 секунд)")
    st.caption("Это ключевое: помогает отличить «что я делаю» от «зачем мне это на глубине».")

    m_key = st.radio(
        "Почему это тебя реально цепляет? (выбери 1 главный мотив)",
        options=[k for k, _ in MOTIVE_OPTIONS],
        format_func=lambda k: dict(MOTIVE_OPTIONS)[k],
        key=f"motive_{qid}",
    )
    l_key = st.radio(
        "Если этого нет в жизни, что появляется чаще всего?",
        options=[k for k, _ in LACK_OPTIONS],
        format_func=lambda k: dict(LACK_OPTIONS)[k],
        key=f"lack_{qid}",
    )
    txt = st.text_area(
        "Один живой пример (по желанию): что за ситуация/история стоит за твоим выбором?",
        key=f"deep_text_{qid}",
        height=90,
        placeholder="Например: «когда я… я чувствую…» / «я делаю это потому что…»",
    )

    payload = {
        "motive": m_key,
        "lack": l_key,
        "deep_text": (txt or "").strip(),
    }
    if group_for_block3:
        payload["group"] = group_for_block3
    return payload


# ============================
# Блок 1 (детство) — с ветвлениями
# ============================

def build_block1_questions() -> Dict[str, Question]:
    q: Dict[str, Question] = {}

    q["b1_q1_free_play"] = Question(
        id="b1_q1_free_play",
        block=1,
        text="Как ты чаще всего проводил(а) свободное время в детстве (лет до 10)?",
        allow_multiple=True,
        options=[
            AnswerOption("Активные игры на улице: бег, догонялки, лазание, спорт",
                         {"SHUNGIT": 2}, inject_questions=["b1_q7_sport_detail"], tags=["sport"]),
            AnswerOption("Книги, кроссворды, головоломки — любил(а) думать",
                         {"AMETIST": 2, "SAPFIR": 0.5}, inject_questions=["b1_q2_reading_detail"], tags=["thinking"]),
            AnswerOption("Игра «магазин», продажи, обмены, торг",
                         {"CITRIN": 2}, inject_questions=["b1_q3_trading_detail"], tags=["money"]),
            AnswerOption("Концерты/сценки/выступления",
                         {"GRANAT": 2, "RUBIN": 0.5}, inject_questions=["b1_q4_stage_detail"], tags=["stage"]),
            AnswerOption("Животные: жалел(а), лечил(а), кормил(а)",
                         {"IZUMRUD": 2}, inject_questions=["b1_q5_animals_detail"], tags=["care"]),
            AnswerOption("Разбирать/собирать, Лего, схемы",
                         {"YANTAR": 1}, inject_questions=["b1_q8_order_tech_detail"], tags=["system"]),
            AnswerOption("Болтал(а), общался(ась), шутил(а), придумывал(а) истории, пел песни",
                         {"GELIODOR": 2}, inject_questions=["b1_q6_talk_detail"], tags=["network"]),
        ],
    )

    q["b1_q2_subjects"] = Question(
        id="b1_q2_subjects",
        block=1,
        text="Какие школьные предметы ты любил(а) больше всего?",
        allow_multiple=True,
        options=[
            AnswerOption("Математика/физика/информатика/логика", {"AMETIST": 2, "YANTAR": 2}, tags=["thinking"]),
            AnswerOption("Литература/история/философия/психология", {"SAPFIR": 2, "AMETIST": 0.5}, tags=["meaning"]),
            AnswerOption("Музыка/пение/театр/выступления", {"GRANAT": 1, "GELIODOR": 1}, inject_questions=["b1_q9_music_detail"], tags=["stage"]),
            AnswerOption("Физкультура/соревнования", {"SHUNGIT": 1, "CITRIN": 0.5}, inject_questions=["b1_q7_sport_detail"], tags=["sport"]),
            AnswerOption("Рисование/дизайн/красота/психология", {"IZUMRUD": 2}, tags=["beauty"]),
            AnswerOption("Ничего особо — учился(ась) «как надо»", {"YANTAR": 0.5}, tags=["system"]),
        ],
    )

    q["b1_q3_clubs"] = Question(
        id="b1_q3_clubs",
        block=1,
        text="На какие кружки/секции ты ходил(а) в детстве и подростковом возрасте?",
        allow_multiple=True,
        options=[
            AnswerOption("Спорт", {"SHUNGIT": 1, "CITRIN": 0.4}, inject_questions=["b1_q7_sport_detail"], tags=["sport"]),
            AnswerOption("Музыка/вокал/театр/танцы как выступления", {"GRANAT": 1, "GELIODOR": 1}, inject_questions=["b1_q4_stage_detail"], tags=["stage"]),
            AnswerOption("Техника/конструктор/моделирование/робототехника", {"YANTAR": 2}, tags=["system"]),
            AnswerOption("Рисование/дизайн/hand-made", {"IZUMRUD": 2}, tags=["beauty"]),
            AnswerOption("Лидерство/олимпиады/экономика", {"CITRIN": 1.5}, tags=["money"]),
            AnswerOption("Танцы как управление телом, растяжка, гибкость", {"CITRIN": 1.5}, tags=["money"]),
            AnswerOption("Почти никуда / по обязанности", {"YANTAR": 0.5}, tags=["system"]),
        ],
    )

    q["b1_q4_choice"] = Question(
        id="b1_q4_choice",
        block=1,
        text="Кто чаще выбирал, чем ты занимаешься в детстве (кружки, секции, занятия)?",
        allow_multiple=False,
        options=[
            AnswerOption("Я сам(а) решал(а), что хочу делать", {"CITRIN": 0.6, "AMETIST": 0.4, "RUBIN": 0.4}),
            AnswerOption("Чаще взрослые — я соглашался(ась)", {"YANTAR": 0.7}),
            AnswerOption("Комбинация: инициатива + решение взрослых", {"CITRIN": 0.3, "YANTAR": 0.1}),
        ],
    )

    q["b1_q5_speaking"] = Question(
        id="b1_q5_speaking",
        block=1,
        text="Какой ты был(а) в детстве по общению?",
        allow_multiple=False,
        options=[
            AnswerOption("Много говорил(а), шутил(а), любил(а) компанию", {"GELIODOR": 2, "GRANAT": 0.5}, inject_questions=["b1_q6_talk_detail"], tags=["network"]),
            AnswerOption("Тихий(ая), наблюдал(а), больше слушал(а)", {"AMETIST": 0.8, "SAPFIR": 0.8}, tags=["meaning"]),
            AnswerOption("По-разному: с близкими живой(ая), с чужими закрытый(ая)", {"GELIODOR": 0.6, "IZUMRUD": 0.4}, tags=["network"]),
        ],
    )

    q["b1_q6_animals"] = Question(
        id="b1_q6_animals",
        block=1,
        text="Были ли у тебя в детстве домашние животные и как ты к ним относился(ась)?",
        allow_multiple=False,
        options=[
            AnswerOption("Да, очень привязывался(ась), ухаживал(а), переживал(а)", {"IZUMRUD": 2}, inject_questions=["b1_q5_animals_detail"], tags=["care"]),
            AnswerOption("Да, но спокойно, без сильной привязанности", {"IZUMRUD": 0.7}),
            AnswerOption("Почти не было / неинтересно", {}),
        ],
    )

    q["b1_q7_body"] = Question(
        id="b1_q7_body",
        block=1,
        text="Как ты в детстве относился(ась) к физическим нагрузкам и спорту?",
        allow_multiple=False,
        options=[
            AnswerOption("Обожал(а) двигаться, соревнования, тело", {"SHUNGIT": 1.2, "CITRIN": 0.3}, inject_questions=["b1_q7_sport_detail"], tags=["sport"]),
            AnswerOption("Спорт был, но больше как «надо»", {"YANTAR": 0.6}),
            AnswerOption("Избегал(а) нагрузок, любил(а) спокойное", {"AMETIST": 0.5, "SAPFIR": 0.5}),
        ],
    )

    q["b1_q8_order"] = Question(
        id="b1_q8_order",
        block=1,
        text="Как ты относился(ась) к порядку в детстве?",
        allow_multiple=False,
        options=[
            AnswerOption("Любил(а) разложить по местам, хаос раздражал", {"YANTAR": 2}, inject_questions=["b1_q8_order_tech_detail"], tags=["system"]),
            AnswerOption("Нормально, но не зацикливался(ась)", {"YANTAR": 0.8}),
            AnswerOption("Творческий беспорядок — главное интерес", {"GRANAT": 0.5, "GELIODOR": 0.5}, tags=["beauty"]),
        ],
    )

    q["b1_q9_reading"] = Question(
        id="b1_q9_reading",
        block=1,
        text="Как ты относился(ась) к чтению и обучению в детстве?",
        allow_multiple=False,
        options=[
            AnswerOption("Много читал(а) сам(а), любил(а) новое", {"AMETIST": 1.6, "SAPFIR": 0.6}, inject_questions=["b1_q2_reading_detail"], tags=["thinking"]),
            AnswerOption("Читал(а) если надо, сам(а) редко выбирал(а)", {"YANTAR": 0.5}),
            AnswerOption("Чтение не привлекало", {"GRANAT": 0.3, "GELIODOR": 0.5}),
        ],
    )

    q["b1_q10_money_play"] = Question(
        id="b1_q10_money_play",
        block=1,
        text="Были ли игры/занятия, связанные с деньгами, продажами, обменом, игры в магазин?",
        allow_multiple=False,
        options=[
            AnswerOption("Да, торговаться/продавать/обменивать было кайф", {"CITRIN": 2}, inject_questions=["b1_q3_trading_detail"], tags=["money"]),
            AnswerOption("Иногда, но без сильной тяги", {"CITRIN": 0.8}),
            AnswerOption("Нет, не интересовало", {}),
        ],
    )

    q["b1_q11_scene_vs_director"] = Question(
        id="b1_q11_scene_vs_director",
        block=1,
        text="Школьный праздник: где ты чаще?",
        allow_multiple=False,
        options=[
            AnswerOption("На сцене, в центре внимания", {"GRANAT": 1, "GELIODOR": 1}, tags=["stage"]),
            AnswerOption("За кадром: сценарий/организация/тайминг", {"YANTAR": 1.0, "RUBIN": 0.7, "CITRIN": 1}, tags=["system"]),
            AnswerOption("В зале: поддерживать/наблюдать", {}),
        ],
    )

    q["b1_q12_empathy"] = Question(
        id="b1_q12_empathy",
        block=1,
        text="Насколько ты был(а) чувствительным ребёнком?",
        allow_multiple=False,
        options=[
            AnswerOption("Очень: мог(ла) плакать, жалко людей/животных", {"IZUMRUD": 1.2, "SAPFIR": 1.0}),
            AnswerOption("Средне: иногда трогало", {"IZUMRUD": 0.7}),
            AnswerOption("Редко: больше разум", {"AMETIST": 0.8}),
        ],
    )

    q["b1_q13_dreams"] = Question(
        id="b1_q13_dreams",
        block=1,
        text="О чём ты мечтал(а) в детстве/подростковом возрасте? (можно несколько)",
        allow_multiple=True,
        options=[
            AnswerOption("Известность, сцена, признание", {"GRANAT": 1.6, "GELIODOR": 0.6}, tags=["stage"]),
            AnswerOption("Свой бизнес/дело, быть хозяином", {"CITRIN": 1.6}, tags=["money"]),
            AnswerOption("Помогать людям, делать мир лучше", {"SAPFIR": 1.6, "IZUMRUD": 0.6}, tags=["meaning"]),
            AnswerOption("Путешествия, покорять новые вершины/места, стать пилотом", {"RUBIN": 1.3}, tags=["beauty"]),
            AnswerOption("Стабильность, безопасность, дом/уют", {"YANTAR": 1.6}, tags=["system"]),
        ],
    )

    q["b1_q14_chaos_reaction"] = Question(
        id="b1_q14_chaos_reaction",
        block=1,
        text="Комната в хаосе. Твоя реакция?",
        allow_multiple=False,
        options=[
            AnswerOption("Хочу разложить по местам — сложно расслабиться", {"YANTAR": 1.6}, tags=["system"]),
            AnswerOption("Навёл(ла) бы порядок, но не сразу", {"YANTAR": 0.8}),
            AnswerOption("Не напрягало — главное, чтобы интересно/весело", {"GRANAT": 0.4, "RUBIN": 0.4}),
        ],
    )

    q["b1_q15_competition"] = Question(
        id="b1_q15_competition",
        block=1,
        text="Как ты относился(ась) к конкуренции и соревнованиям?",
        allow_multiple=False,
        options=[
            AnswerOption("Важно быть первым, сильно переживал(а) если кто-то лучше", {"CITRIN": 1.6}),
            AnswerOption("Интересно участвовать, но не критично быть первым", {"SHUNGIT": 0.5}),
            AnswerOption("Не любил(а) соревнования, избегал(а)", {}),
        ],
    )

    # --- Ветки ---
    q["b1_q5_animals_detail"] = Question(
        id="b1_q5_animals_detail",
        block=1,
        text="Что было самым важным в отношении к животным?",
        allow_multiple=False,
        options=[
            AnswerOption("Комфорт/тепло/сытость/уход", {"IZUMRUD": 1.0, "YANTAR": 0.6}),
            AnswerOption("Контакт/любовь/обнимать/играть", {"IZUMRUD": 1.2}),
            AnswerOption("Наблюдать — интересно, как устроены", {"AMETIST": 0.6, "IZUMRUD": 0.4}),
        ],
    )

    q["b1_q7_sport_detail"] = Question(
        id="b1_q7_sport_detail",
        block=1,
        text="Что в спорте/движении нравилось больше всего?",
        allow_multiple=False,
        options=[
            AnswerOption("Сила тела/усталость/преодоление", {"SHUNGIT": 1.4}, tags=["sport"]),
            AnswerOption("Победа/медали/быть первым", {"CITRIN": 1.2}, tags=["sport"]),
            AnswerOption("Команда/энергия группы/общение", {"SHUNGIT": 0.4}, tags=["network"]),
        ],
    )

    q["b1_q4_stage_detail"] = Question(
        id="b1_q4_stage_detail",
        block=1,
        text="Что притягивало в сцене/выступлениях?",
        allow_multiple=False,
        options=[
            AnswerOption("Реакция людей, внимание, эмоции", {"GRANAT": 1.4, "GELIODOR": 0.4}),
            AnswerOption("Придумать номер/образ/сценарий/как выглядит", {"RUBIN": 0.8, "IZUMRUD": 0.6, "YANTAR": 0.2}),
            AnswerOption("Передать глубокую историю/смысл", {"SAPFIR": 0.8,"GELIODOR": 0.8}),
        ],
    )

    q["b1_q6_talk_detail"] = Question(
        id="b1_q6_talk_detail",
        block=1,
        text="Какой формат общения был ближе?",
        allow_multiple=False,
        options=[
            AnswerOption("Шутить/развлекать/поднимать настроение", {"GELIODOR": 1.2, "GRANAT": 0.4}),
            AnswerOption("Разговоры по душам один на один", {"SAPFIR": 0.7, "IZUMRUD": 0.5}),
            AnswerOption("Идеи/планы/как улучшить/как сделать", {"AMETIST": 0.6, "CITRIN": 0.4}),
        ],
    )

    q["b1_q2_reading_detail"] = Question(
        id="b1_q2_reading_detail",
        block=1,
        text="Какие книги/материалы были интереснее?",
        allow_multiple=False,
        options=[
            AnswerOption("Научпоп/энциклопедии/как устроено", {"AMETIST": 1.2, "YANTAR": 0.4}),
            AnswerOption("Судьбы людей/психология/философия/смысл", {"SAPFIR": 1.2, "IZUMRUD": 0.4}),
            AnswerOption("Фантастика/миры/приключения", {"RUBIN": 0.8, "SAPFIR": 0.4}),
            AnswerOption("Практично: деньги/бизнес/успех", {"CITRIN": 1.2}),
        ],
    )

    q["b1_q3_trading_detail"] = Question(
        id="b1_q3_trading_detail",
        block=1,
        text="Что нравилось в играх «магазин» / продажах / обменах?",
        allow_multiple=False,
        options=[
            AnswerOption("Сделка: договориться, закрыть, продать выгоднее", {"CITRIN": 1.4, "GELIODOR": 0.4}),
            AnswerOption("Оформление: витрина, упаковка, красиво", {"IZUMRUD": 1.2, "GRANAT": 0.9}),
            AnswerOption("Считать/планировать/учёт", {"YANTAR": 1.2, "AMETIST": 0.4}),
        ],
    )

    q["b1_q8_order_tech_detail"] = Question(
        id="b1_q8_order_tech_detail",
        block=1,
        text="Что особенно нравилось в порядке/структуре/конструкциях?",
        allow_multiple=False,
        options=[
            AnswerOption("Чисто/аккуратно/по полочкам", {"YANTAR": 1.4}),
            AnswerOption("Разобрать/понять, как устроено внутри", {"YANTAR": 0.6}),
            AnswerOption("Удобно и красиво одновременно", {"IZUMRUD": 0.7, "YANTAR": 0.5}),
        ],
    )

    q["b1_q9_music_detail"] = Question(
        id="b1_q9_music_detail",
        block=1,
        text="Какую роль занимала музыка/пение?",
        allow_multiple=False,
        options=[
            AnswerOption("Хотел(а) выступать/петь, но иногда было страшно", {"GELIODOR": 1.0, "GRANAT": 0.4}),
            AnswerOption("Больше слушать/чувствовать музыку, чем выступать", {"SAPFIR": 0.6, "IZUMRUD": 0.4}),
            AnswerOption("Музыка не играла роли", {}),
        ],
    )

    return q


CORE_SEQUENCE_BLOCK1 = [
    "b1_q1_free_play",
    "b1_q2_subjects",
    "b1_q3_clubs",
    "b1_q4_choice",
    "b1_q5_speaking",
    "b1_q6_animals",
    "b1_q7_body",
    "b1_q8_order",
    "b1_q9_reading",
    "b1_q10_money_play",
    "b1_q11_scene_vs_director",
    "b1_q12_empathy",
    "b1_q13_dreams",
    "b1_q14_chaos_reaction",
    "b1_q15_competition",
]


# ============================
# Блок 2 — работа/процессы/смещения (минимум 15)
# ============================

def build_block2_questions() -> List[Question]:
    q: List[Question] = []

    def Q(_id: str, text: str, options: List[AnswerOption], multi: bool = True) -> Question:
        return Question(id=_id, block=2, text=text, allow_multiple=multi, options=options)

    # 1
    q.append(Q(
        "b2_q1_role_now",
        "1) Чем ты в основном занимаешься сейчас (или последние 2–3 года)?",
        [
            AnswerOption("Продажи/сделки/переговоры/закрытие", {"CITRIN": 1.2, "GELIODOR": 0.4}, tags=["money"]),
            AnswerOption("Обучение/наставничество/объяснять людям", {"GELIODOR": 1.2, "SAPFIR": 0.3}, tags=["network"]),
            AnswerOption("Операционка/процессы/регламенты/таблицы", {"YANTAR": 1.2, "AMETIST": 0.3}, tags=["system"]),
            AnswerOption("Аналитика/стратегия/архитектура решений", {"AMETIST": 1.2}, tags=["thinking"]),
            AnswerOption("Контент/публичность/выступления/видимость", {"GRANAT": 1.0, "GELIODOR": 0.5}, tags=["stage"]),
            AnswerOption("Тело/спорт/полевые задачи/движение", {"SHUNGIT": 1.0}, tags=["sport"]),
            AnswerOption("Красота/визуал/стиль/эстетика", {"IZUMRUD": 1.2}, tags=["beauty"]),
        ],
        multi=True
    ))

    # 2
    q.append(Q(
        "b2_q2_like_job",
        "2) Тебе нравится твоя текущая деятельность в целом?",
        [
            AnswerOption("Да, я на своём месте", {"SAPFIR": 0.2}),
            AnswerOption("Скорее да, но есть выматывающие процессы", {"YANTAR": 0.2}),
            AnswerOption("Скорее нет, чувствую «не своё»", {"SAPFIR": 0.6}),
            AnswerOption("Вообще нет, держусь на обязанности/деньгах", {"SAPFIR": 0.8, "CITRIN": 0.2}),
        ],
        multi=False
    ))

    # 3
    q.append(Q(
        "b2_q3_ideal_roles",
        "3) Если деньги убрать: какие 2–3 роли ты бы выбрал(а) по любви?",
        [
            AnswerOption("Педагог/наставник/обучение", {"GELIODOR": 1.2, "YANTAR": 0.5}, tags=["network"]),
            AnswerOption("Продакт/предприниматель/создатель продукта", {"CITRIN": 1.2, "RUBIN": 0.4}, tags=["money"]),
            AnswerOption("Операционный эксперт/системщик", {"YANTAR": 1.2}, tags=["system"]),
            AnswerOption("Аналитик/стратег/архитектор решений", {"AMETIST": 1.2}, tags=["thinking"]),
            AnswerOption("Сцена/ведущий/публичность", {"GRANAT": 1.2, "GELIODOR": 0.6}, tags=["stage"]),
            AnswerOption("Движ/поездки/события/экспедиции", {"RUBIN": 1.2}, tags=["adrenaline"]),
            AnswerOption("Эстетика/дизайн/визуал/стиль", {"IZUMRUD": 1.2}, tags=["beauty"]),
            AnswerOption("Тело/спорт/физ. практика", {"SHUNGIT": 1.2}, tags=["sport"]),
        ],
        multi=True
    ))

    # 4
    q.append(Q(
        "b2_q4_flow_tasks",
        "4) В каких задачах ты залипаешь и теряешь счёт времени?",
        [
            AnswerOption("Разобраться до сути, анализ, логика", {"AMETIST": 1.2}, tags=["thinking"]),
            AnswerOption("Систематизировать: таблицы, порядок, процесс", {"YANTAR": 1.2}, tags=["system"]),
            AnswerOption("Обучать/объяснять/вести людей/петь/работать голосом", {"GELIODOR": 1.2}, tags=["network"]),
            AnswerOption("Закрывать сделки/дожимать результат", {"CITRIN": 1.2}, tags=["money"]),
            AnswerOption("Движ/скорость/поездки/новое", {"RUBIN": 1.2}, tags=["adrenaline"]),
            AnswerOption("Тело/спорт/тренировки", {"SHUNGIT": 1.2}, tags=["sport"]),
            AnswerOption("Красота/визуал/атмосфера", {"IZUMRUD": 1.2}, tags=["beauty"]),
        ],
        multi=True
    ))

    # 5
    q.append(Q(
        "b2_q5_hate_tasks",
        "5) Какие задачи ты чаще всего откладываешь (не любишь)?",
        [
            AnswerOption("Рутина/документы/таблицы", {"YANTAR": -0.6}),
            AnswerOption("Переговоры/дожим/цены/торг", {"CITRIN": -0.6}),
            AnswerOption("Публичность/выступление/внимание", {"GRANAT": -0.6}),
            AnswerOption("Долгий анализ/копание", {"AMETIST": -0.4}),
            AnswerOption("Знакомится с новыми людьми, ходить на шумные мероприятия", {"GRANAT": -0.4}),
            AnswerOption("Физ. нагрузка/спорт, генеральная уборка", {"SHUNGIT": -0.4}),
            AnswerOption("Красота/творчество без критериев", {"IZUMRUD": -0.4}),
        ],
        multi=True
    ))

    # 6
    q.append(Q(
        "b2_q6_decision_style",
        "6) Как ты принимаешь важные решения чаще всего?",
        [
            AnswerOption("Через анализ и факты", {"AMETIST": 1.0}),
            AnswerOption("Через выгоду/цифры/результат", {"CITRIN": 1.0}),
            AnswerOption("Через людей/коммуникации/советы", {"GELIODOR": 0.8}),
            AnswerOption("Через внутренний смысл/«знак»", {"SAPFIR": 1.0}),
            AnswerOption("Через импульс/риск/перезапуск", {"RUBIN": 1.0}),
            AnswerOption("Через тело/ощущения", {"SHUNGIT": 0.8, "IZUMRUD": 0.2}),
        ],
        multi=True
    ))

    # 7
    q.append(Q(
        "b2_q7_money_relation",
        "7) Твои отношения с деньгами и монетизацией талантов?",
        [
            AnswerOption("Люблю считать/продавать/делать прибыльно", {"CITRIN": 1.2}, tags=["money"]),
            AnswerOption("Деньги важны, но смысл важнее", {"SAPFIR": 0.8}),
            AnswerOption("Сложно назначать цену/просить деньги", {"CITRIN": -0.4, "SAPFIR": 0.2}),
            AnswerOption("Я лучше сделаю хорошо, а деньги как-нибудь придут", {"YANTAR": 0.2, "IZUMRUD": 0.2}),
        ],
        multi=True
    ))

    # 8
    q.append(Q(
        "b2_q8_network",
        "8) Что для тебя нетворк/связи/люди?",
        [
            AnswerOption("Я люблю соединять людей и быть мостом", {"GELIODOR": 1.2}, tags=["network"]),
            AnswerOption("Мне важно влияние и статус через окружение", {"GRANAT": 0.8, "CITRIN": 0.4}),
            AnswerOption("Я выбираю узкий круг, глубина важнее количества", {"SAPFIR": 0.6, "IZUMRUD": 0.2}),
            AnswerOption("Я устаю от людей, люблю одиночество", {"AMETIST": 0.1}),
        ],
        multi=True
    ))

    # 9
    q.append(Q(
        "b2_q9_teaching",
        "9) Если ты обучаешь/объясняешь: что тебя в этом реально заводит?",
        [
            AnswerOption("Видеть, как у человека «щёлкнуло»", {"GELIODOR": 1.0, "SAPFIR": 0.4}),
            AnswerOption("Структурировать сложное в простое", {"YANTAR": 0.8, "AMETIST": 0.6}),
            AnswerOption("Быть признанным/оценённым", {"GRANAT": 0.8}),
            AnswerOption("Делать систему обучения/процесс", {"YANTAR": 1.0}),
        ],
        multi=True
    ))

    # 10
    q.append(Q(
        "b2_q10_order",
        "10) Порядок/уборка/структура — это про что для тебя?",
        [
            AnswerOption("Меня успокаивает, в голове становится чисто", {"YANTAR": 1.2}),
            AnswerOption("Я вижу, что сломано — и чиню", {"YANTAR": 1.0}),
            AnswerOption("Это про контроль, иначе тревожно", {"YANTAR": 0.8}),
            AnswerOption("Это про эффективность и результат", {"CITRIN": 0.6}),
        ],
        multi=True
    ))

    # 11
    q.append(Q(
        "b2_q11_sport_style",
        "11) Спорт/движение: что тебя в этом цепляет больше всего?",
        [
            AnswerOption("Тонус, сила, тело, выносливость", {"SHUNGIT": 1.2}, tags=["sport"]),
            AnswerOption("Скорость, победа, счёт, быть первым", {"CITRIN": 1.2}, tags=["sport"]),
            AnswerOption("Адреналин, игра, драйв, эмоции", {"RUBIN": 1.2}, tags=["sport"]),
            AnswerOption("Команда и общение", {"GELIODOR": 0.8, "SHUNGIT": 0.8}),
        ],
        multi=True
    ))

    # 12
    q.append(Q(
        "b2_q12_first_place",
        "12) Про «быть первым»: это у тебя про…",
        [
            AnswerOption("Победа = доказать себе и другим", {"CITRIN": 1.0, "GRANAT": 0.4}),
            AnswerOption("Это азарт/игра/включение", {"RUBIN": 0.8}),
            AnswerOption("Это дисциплина и система", {"YANTAR": 0.8}),
            AnswerOption("Мне не важно быть первым", {"SAPFIR": 0.2}),
        ],
        multi=False
    ))

    # 13
    q.append(Q(
        "b2_q13_family",
        "13) Семья/ответственность: что ты готов(а) делать ради семьи?",
        [
            AnswerOption("Включаю режим результата и закрываю любой вопрос", {"CITRIN": 0.8}),
            AnswerOption("Держу порядок и стабильность", {"YANTAR": 0.8}),
            AnswerOption("Становлюсь мягким(ой), поддерживаю, берегу", {"IZUMRUD": 0.6}),
            AnswerOption("Готов(а) на резкие перемены/перезапуск ради семьи", {"RUBIN": 0.6, "CITRIN": 0.8}),
        ],
        multi=True
    ))

    # 14
    q.append(Q(
        "b2_q14_depression",
        "14) Если сейчас мало смысла/апатия: что помогает оживать хоть чуть-чуть?",
        [
            AnswerOption("Движение/спорт/тело", {"SHUNGIT": 0.6}),
            AnswerOption("Новый опыт/поездка/смена обстановки", {"RUBIN": 0.8}),
            AnswerOption("Порядок/уборка/структура", {"YANTAR": 0.6}),
            AnswerOption("Разговор/люди/поддержка", {"GELIODOR": 0.6, "GRANAT": 0.6}),
            AnswerOption("Смысл/вера/миссия/разговор о «зачем»", {"SAPFIR": 0.6, "AMETIST": 0.6}),
        ],
        multi=True
    ))

    # 15
    q.append(Q(
        "b2_q15_conflict",
        "15) Как ты реагируешь на конфликты и сильные эмоции людей?",
        [
            AnswerOption("Избегаю — мне тяжело", {"GRANAT": -0.4, "IZUMRUD": 0.2}),
            AnswerOption("Умею держать рамки и правила", {"YANTAR": 0.4}),
            AnswerOption("Включаю переговоры и договариваюсь", {"GELIODOR": 0.6, "CITRIN": 0.2}),
            AnswerOption("Включаюсь, могу «взять власть» в ситуации", {"GRANAT": 0.6, "CITRIN": 0.2}),
        ],
        multi=True
    ))

    return q


# ============================
# Блок 3 — столбцы: Восприятие/Мотивация/Результат (15 вопросов)
# ============================

def build_block3_questions() -> List[Question]:
    q: List[Question] = []

    def Q(_id: str, group: str, text: str, options: List[AnswerOption], multi: bool = True) -> Question:
        return Question(id=_id, block=3, group=group, text=text, allow_multiple=multi, options=options)

    # c1: ВАУ / восприятие
    q.append(Q("c1_q1", "c1", "1) Что в людях вызывает у тебя мгновенный «вау»? И у тебя возникают такие эмоции, как "Уау, абалдеть, ну как так возможно"",
               [
                   AnswerOption("Физическая сила, выносливость, владение телом", {"SHUNGIT": 1.0}, tags=["sport"]),
                   AnswerOption("Глубокие мысли, нестандартное мышление", {"AMETIST": 1.0}, tags=["thinking"]),
                   AnswerOption("Красивый голос, подача, манера говорить", {"GELIODOR": 1.0}, tags=["network"]),
                   AnswerOption("Передавать и проявлть эмоции, умение зажечь/заводить друзей", {"GRANAT": 1.0}, tags=["stage"]),
                   AnswerOption("Система, продуманность, порядок, все ровно, все работает "как надо"", {"YANTAR": 1.0}, tags=["system"]),
                   AnswerOption("Какой красивый и гармоничный", {"IZUMRUD": 1.0}, tags=["beauty"]),
                   AnswerOption("Смелость, риск, шаг в неизвестность", {"RUBIN": 1.0}, tags=["adrenaline"]),
                   AnswerOption("Умение делать деньги/результат", {"CITRIN": 1.0}, tags=["money"]),
               ], multi=True))

    q.append(Q("c1_q2", "c1", "2) Какие сцены/пространство тебя завораживают больше всего? И у тебя возникают такие эмоции, как "Уау, абалдеть, ну как так возможно?"",
               [
                   AnswerOption("Природа, вода, горы, океан, закаты, сочетание цветов и света", {"IZUMRUD": 0.6}, tags=["beauty"]),
                   AnswerOption("Город, архитектура, линии, мосты, геометрия", {"YANTAR": 0.6}, tags=["system"]),
                   AnswerOption("Лица людей, эмоции, реакции", {"GRANAT": 0.6, "GELIODOR": 0.4}, tags=["stage"]),
                   AnswerOption("Спорт/движение/танец, тела в динамике", {"SHUNGIT": 0.6}, tags=["sport"]),
                   AnswerOption("Кадр/свет/тени/эстетика деталей", {"IZUMRUD": 1.0}, tags=["beauty"]),
               ], multi=True))

    q.append(Q("c1_q3", "c1", "3) Какие фильмы ты любишь больше всего смотреть",
               [
                   AnswerOption("Триллеры, детективы", {"AMETIST": 1}, tags=["thinking"]),
                   AnswerOption("Романтические фильмы/драмы", {"IZUMRUD": 0.6, GRANAT: 0.6}, tags=["stage"]),
                   AnswerOption("Преодоление/экшен/борьба/риск", {"RUBIN": 0.6, "CITRIN": 0.6}, tags=["adrenaline"]),
                   AnswerOption("Научно-документальные фильмы, как все устроено/исторические", {"YANTAR": 1, "SAPFIR": 0.8}, tags=["system"]),
                   AnswerOption("Про спорт, выносливость", {"SHUNGIT": 1}, tags=["beauty"]),
               ], multi=True))

    q.append(Q("c1_q4", "c1", "4) Что в мире кажется тебе восхительным и удивительным?",
               [
                   AnswerOption("Как работает мозг/подсознание", {"AMETIST": 0.8, "SAPFIR": 0.2}, tags=["thinking"]),
                   AnswerOption("Музыка/голос/влияние/популярность", {"GELIODOR": 0.8}, tags=["network"]),
                   AnswerOption("Системы и структуры меняют реальность", {"YANTAR": 0.8}, tags=["system"]),
                   AnswerOption("Как одно смелое действие меняет жизнь", {"RUBIN": 0.8}, tags=["adrenaline"]),
                   AnswerOption("Природа/циклы/красота мира", {"IZUMRUD": 0.6, "SAPFIR": 0.2}, tags=["beauty"]),
                   AnswerOption("Смысл/судьба/внутренний путь", {"SAPFIR": 0.8}, tags=["meaning"]),
               ], multi=True))

    q.append(Q("c1_q5", "c1", "5) Что сильнее всего вызывает страх/ступор/избегание?",
               [
                   AnswerOption("Громкие конфликты и сильные эмоции людей", {"GRANAT": 0.3}, tags=["stage"]),
                   AnswerOption("Хаос, отсутствие порядка и ясности", {"YANTAR": 0.6}, tags=["system"]),
                   AnswerOption("Бессмысленность, когда нет логики/«зачем»", {"SAPFIR": 0.6, "AMETIST": 0.8}, tags=["meaning"]),
                   AnswerOption("Большие деньги/ответственность", {"CITRIN": 0.4}, tags=["money"]),
                   AnswerOption("Риск/неизвестность", {"RUBIN": 0.3}, tags=["adrenaline"]),
                   AnswerOption("Сильная физическая нагрузка/боль/травма", {"SHUNGIT": 0.3}, tags=["sport"]),
               ], multi=True))

    # c2: Процесс / мотивация
    q.append(Q("c2_q1", "c2", "6) Какие занятия втягивают так, что ты забываешь про время?",
               [
                   AnswerOption("Разговоры/нетворк/обучение/объяснять", {"GELIODOR": 1.0}, tags=["network"]),
                   AnswerOption("Анализ/разбор/смыслы/миссия/предназначение", {"AMETIST": 0.8, "SAPFIR": 0.2}, tags=["thinking"]),
                   AnswerOption("Порядок/структура/чинить/настраивать/как работает тело", {"YANTAR": 1.0}, tags=["system"]),
                   AnswerOption("Закрывать результат/сделки/дожим", {"CITRIN": 1.0}, tags=["money"]),
                   AnswerOption("Тренировки/движение", {"SHUNGIT": 1.0}, tags=["sport"]),
                   AnswerOption("Драйв/новые места/экстрим", {"RUBIN": 1.0}, tags=["adrenaline"]),
                   AnswerOption("Визуал/красота/атмосфера/психология людей", {"IZUMRUD": 1.0}, tags=["beauty"]),
               ], multi=True))

    q.append(Q("c2_q2", "c2", "7) Что ты выбираешь делать «для души»?",
               [
                   AnswerOption("Убираться/организовывать пространство", {"YANTAR": 1.0}, tags=["system"]),
                   AnswerOption("Спорт/растяжка/уход за телом", {"SHUNGIT": 0.8}, tags=["sport"]),
                   AnswerOption("Разговоры/Караоке/Готовить или пробовать новые вкусы", {"GELIODOR": 0.8}, tags=["network"]),
                   AnswerOption("Писать/думать/разбираться в себе", {"AMETIST": 0.6, "SAPFIR": 0.4}, tags=["thinking"]),
                   AnswerOption("Красота/детали/эстетика дома", {"IZUMRUD": 0.8}, tags=["beauty"]),
                   AnswerOption("Движ/планы/поездки/новое", {"RUBIN": 0.8}, tags=["adrenaline"]),
                   AnswerOption("Вечеринки, тусовки, новые знакомства, выступления, эмоции", {"GRANAT": 1}, tags=["stage"]),
               ], multi=True))

    q.append(Q("c2_q3", "c2", "8) Если есть выбор, ты добровольно выбираешь задачи типа…",
               [
                   AnswerOption("Про людей: созвон/встреча/обучение", {"GELIODOR": 1.0}, tags=["network"]),
                   AnswerOption("Про структуру: план/таблица/чек-лист", {"YANTAR": 1.0}, tags=["system"]),
                   AnswerOption("Про результат: закрыть/дожать/продать", {"CITRIN": 1.0}, tags=["money"]),
                   AnswerOption("Про анализ: понять закономерность", {"AMETIST": 1.0}, tags=["thinking"]),
                   AnswerOption("Про тело: поехать/привезти/сделать физически", {"SHUNGIT": 0.6, "RUBIN": 0.4}, tags=["sport"]),
                   AnswerOption("Про драйв: новое/перезапуск/движ", {"RUBIN": 1.0}, tags=["adrenaline"]),
                   AnswerOption("Про людей: пойти на встречу, участвовать в тимблдинге", {"RUBIN": 1.0}, tags=["adrenaline"]),
               ], multi=True))

    q.append(Q("c2_q4", "c2", "9) Свободный день без обязательств — ты скорее…",
               [
                   AnswerOption("Спорт/прогулка/горы/движение", {"SHUNGIT": 0.8}, tags=["sport"]),
                   AnswerOption("Навести порядок/закрыть мелочи/разобрать", {"YANTAR": 1.0}, tags=["system"]),
                   AnswerOption("Встретиться с людьми/общение/нетворк", {"GELIODOR": 1.0}, tags=["network"]),
                   AnswerOption("Посчитать планы/деньги/цели", {"CITRIN": 1.0}, tags=["money"]),
                   AnswerOption("Драйв/поездка/новое место", {"RUBIN": 1.0}, tags=["adrenaline"]),
                   AnswerOption("Красота/атмосфера/эстетика", {"IZUMRUD": 1.0}, tags=["beauty"]),
                   AnswerOption("Размышлять/читать/смыслы", {"AMETIST": 0.6, "SAPFIR": 0.4}, tags=["thinking"]),
               ], multi=True))

    q.append(Q("c2_q5", "c2", "10) В работе тебе нравится сам процесс…",
               [
                   AnswerOption("Объяснять/обучать/раскладывать сложное просто", {"GELIODOR": 0.8, "YANTAR": 0.2}, tags=["network"]),
                   AnswerOption("Строить систему/процесс/регламент", {"YANTAR": 1.0}, tags=["system"]),
                   AnswerOption("Дожимать результат/сделки/цифры", {"CITRIN": 1.0}, tags=["money"]),
                   AnswerOption("Координировать людей/быть связующим", {"GELIODOR": 0.8, "RUBIN": 0.2}, tags=["network"]),
                   AnswerOption("Анализ/стратегия/архитектура", {"AMETIST": 1.0}, tags=["thinking"]),
                   AnswerOption("Полевые задачи/движение", {"SHUNGIT": 0.6, "RUBIN": 0.4}, tags=["sport"]),
               ], multi=True))

    # c3: Результат / триумф
    q.append(Q("c3_q1", "c3", "11) Какими результатами ты реально гордишься?",
               [
                   AnswerOption("Деньги/сделки/финансовый эффект", {"CITRIN": 1.0}, tags=["money"]),
                   AnswerOption("Спорт/тело/выносливость", {"SHUNGIT": 1.0}, tags=["sport"]),
                   AnswerOption("Система/процесс, который работает без меня", {"YANTAR": 1.0}, tags=["system"]),
                   AnswerOption("Люди: помог, обучил, изменил мышление", {"GELIODOR": 0.8, "SAPFIR": 0.2}, tags=["network"]),
                   AnswerOption("Сильные решения/инсайты/повороты", {"AMETIST": 0.8, "SAPFIR": 0.2}, tags=["thinking"]),
                   AnswerOption("Запуск/событие/проект с драйвом", {"RUBIN": 1.0}, tags=["adrenaline"]),
                   AnswerOption("Красивый результат/визуал/пространство", {"IZUMRUD": 1.0}, tags=["beauty"]),
               ], multi=True))

    q.append(Q("c3_q2", "c3", "12) Самый «сладкий момент» в проекте для тебя — когда…",
               [
                   AnswerOption("Вижу цифры/результат/деньги", {"CITRIN": 1.0}, tags=["money"]),
                   AnswerOption("Всё собрано и работает как механизм", {"YANTAR": 1.0}, tags=["system"]),
                   AnswerOption("Тело приятно устало, чувствую силу", {"SHUNGIT": 1.0}, tags=["sport"]),
                   AnswerOption("Приходит отклик людей/реакции/эмоции", {"GRANAT": 0.6, "GELIODOR": 0.4}, tags=["stage"]),
                   AnswerOption("Понимаю внутренний рост/сдвиг", {"SAPFIR": 0.8, "AMETIST": 0.2}, tags=["meaning"]),
                   AnswerOption("Проект выходит в мир/запуск/событие", {"RUBIN": 0.6, "GRANAT": 0.4}, tags=["adrenaline"]),
               ], multi=True))

    q.append(Q("c3_q3", "c3", "13) Что даёт ощущение «я реально молодец»?",
               [
                   AnswerOption("Закрыл(а) сделку/проект и принёс(ла) деньги", {"CITRIN": 1.0}, tags=["money"]),
                   AnswerOption("Кому-то помог/поддержал/обучил", {"GELIODOR": 0.8, "SAPFIR": 0.2}, tags=["network"]),
                   AnswerOption("Решил сложную задачу, где другие застряли", {"AMETIST": 0.8, "YANTAR": 0.2}, tags=["thinking"]),
                   AnswerOption("Сделал очень аккуратно/качественно/по правилам", {"YANTAR": 1.0}, tags=["system"]),
                   AnswerOption("Выиграл соревнование/конкурс/первенство", {"CITRIN": 0.8, "SHUNGIT": 0.2}, tags=["sport"]),
                   AnswerOption("Сделал красиво/эстетично", {"IZUMRUD": 1.0}, tags=["beauty"]),
               ], multi=True))

    q.append(Q("c3_q4", "c3", "14) Когда включается «боевой режим» и ты борешься до конца?",
               [
                   AnswerOption("Защитить семью/команду", {"YANTAR": 0.4, "CITRIN": 0.4}, tags=["meaning"]),
                   AnswerOption("Жёсткий дедлайн — додавить результат", {"CITRIN": 0.8}, tags=["money"]),
                   AnswerOption("Несправедливость — доказать правду", {"AMETIST": 0.6}, tags=["adrenaline"]),
                   AnswerOption("Переговоры/торг — отстоять условия", {"CITRIN": 0.8}, tags=["money"]),
                   AnswerOption("Перед выходом в эфир/публичность", {"GRANAT": 0.6, "GELIODOR": 0.4}, tags=["stage"]),
                   AnswerOption("В риск-ситуации/экстрим/кризис", {"RUBIN": 0.8, "SHUNGIT": 0.2}, tags=["adrenaline"]),
               ], multi=True))

    q.append(Q("c3_q5", "c3", "15) Если оставить только один тип результатов в жизни — что выбираешь? (1 вариант)",
               [
                   AnswerOption("Тело/сила/выносливость/спорт", {"SHUNGIT": 1.2}, tags=["sport"]),
                   AnswerOption("Финансы/статус/бизнес-результат", {"CITRIN": 1.2}, tags=["money"]),
                   AnswerOption("Системы/наследие/процессы, которые работают", {"YANTAR": 1.2}, tags=["system"]),
                   AnswerOption("Влияние на людей/обучение/связи", {"GELIODOR": 1.0, "GRANAT": 0.2}, tags=["network"]),
                   AnswerOption("Смысл/внутренний путь/осознанность", {"SAPFIR": 1.2}, tags=["meaning"]),
                   AnswerOption("Драйв/новая жизнь/перезапуски", {"RUBIN": 1.2}, tags=["adrenaline"]),
                   AnswerOption("Красота/гармония/эстетика мира", {"IZUMRUD": 1.2}, tags=["beauty"]),
                   AnswerOption("Глубина/мышление/подсознание/миссия/стратегия", {"AMETIST": 1.2}, tags=["thinking"]),
               ], multi=False))

    return q


# ============================
# Движок прохождения (1 вопрос = 1 экран)
# ============================

def init_state():
    if "initialized" in st.session_state:
        return

    st.session_state.initialized = True

    st.session_state.mode = "client"  # client/master

    # client profile
    st.session_state.client_name = ""
    st.session_state.client_contact = ""

    # block1
    st.session_state.b1_questions = build_block1_questions()
    st.session_state.b1_scores = {p: 0.0 for p in POTENTIALS}
    st.session_state.b1_answers: Dict[str, Any] = {}
    st.session_state.b1_core_index = 0
    st.session_state.b1_injected_queue: List[str] = []
    st.session_state.b1_current_qid = CORE_SEQUENCE_BLOCK1[0]
    st.session_state.b1_done = False

    # block2
    st.session_state.b2_questions = build_block2_questions()
    st.session_state.b2_scores = {p: 0.0 for p in POTENTIALS}
    st.session_state.b2_answers: Dict[str, Any] = {}
    st.session_state.b2_index = 0
    st.session_state.b2_done = False

    # block3
    st.session_state.b3_questions = build_block3_questions()
    st.session_state.b3_scores_total = {p: 0.0 for p in POTENTIALS}
    st.session_state.b3_scores_cols = {p: {"c1": 0.0, "c2": 0.0, "c3": 0.0} for p in POTENTIALS}
    st.session_state.b3_answers: Dict[str, Any] = {}
    st.session_state.b3_index = 0
    st.session_state.b3_done = False

    # progress
    st.session_state.current_block = 0  # 0=welcome,1,2,3,4=finish


def get_next_b1_question_id() -> Optional[str]:
    # injected first
    while st.session_state.b1_injected_queue:
        nxt = st.session_state.b1_injected_queue.pop(0)
        if nxt not in st.session_state.b1_answers:
            return nxt

    # then core
    while True:
        st.session_state.b1_core_index += 1
        if st.session_state.b1_core_index >= len(CORE_SEQUENCE_BLOCK1):
            return None
        cand = CORE_SEQUENCE_BLOCK1[st.session_state.b1_core_index]
        if cand not in st.session_state.b1_answers:
            return cand


def apply_b1_answer(question: Question, selected_indices: List[int], comment: str, deep: Optional[Dict[str, Any]]):
    # save
    selected_texts = [question.options[i].text for i in selected_indices]
    st.session_state.b1_answers[question.id] = {
        "question": question.text,
        "selected": selected_texts,
        "comment": (comment or "").strip(),
        "deep": deep or None,
    }

    # score + inject
    for idx in selected_indices:
        opt = question.options[idx]
        apply_score(st.session_state.b1_scores, opt.score_changes)
        for qid in opt.inject_questions:
            if qid not in st.session_state.b1_answers and qid not in st.session_state.b1_injected_queue:
                st.session_state.b1_injected_queue.append(qid)

    # deep probe scoring (мягко)
    if deep:
        m = deep.get("motive")
        l = deep.get("lack")
        if m in MOTIVE_TO_POT:
            apply_score(st.session_state.b1_scores, MOTIVE_TO_POT[m])
        if l in LACK_TO_HINT:
            apply_score(st.session_state.b1_scores, LACK_TO_HINT[l])

    nxt = get_next_b1_question_id()
    st.session_state.b1_current_qid = nxt
    if nxt is None:
        st.session_state.b1_done = True


def apply_b2_answer(question: Question, selected_indices: List[int], comment: str, deep: Optional[Dict[str, Any]]):
    selected_texts = [question.options[i].text for i in selected_indices]
    st.session_state.b2_answers[question.id] = {
        "question": question.text,
        "selected": selected_texts,
        "comment": (comment or "").strip(),
        "deep": deep or None,
    }

    for idx in selected_indices:
        opt = question.options[idx]
        apply_score(st.session_state.b2_scores, opt.score_changes)

    if deep:
        m = deep.get("motive")
        l = deep.get("lack")
        if m in MOTIVE_TO_POT:
            apply_score(st.session_state.b2_scores, MOTIVE_TO_POT[m])
        if l in LACK_TO_HINT:
            apply_score(st.session_state.b2_scores, LACK_TO_HINT[l])

    st.session_state.b2_index += 1
    if st.session_state.b2_index >= len(st.session_state.b2_questions):
        st.session_state.b2_done = True


def apply_b3_answer(question: Question, selected_indices: List[int], comment: str, deep: Dict[str, Any]):
    selected_texts = [question.options[i].text for i in selected_indices]
    st.session_state.b3_answers[question.id] = {
        "question": question.text,
        "selected": selected_texts,
        "comment": (comment or "").strip(),
        "deep": deep or None,
        "group": question.group,
    }

    # base score per selected option
    for idx in selected_indices:
        opt = question.options[idx]
        apply_score(st.session_state.b3_scores_total, opt.score_changes)
        for pot, delta in opt.score_changes.items():
            if pot in st.session_state.b3_scores_cols:
                st.session_state.b3_scores_cols[pot][question.group] += float(delta)

    # deep probe: apply within same column too
    if deep:
        m = deep.get("motive")
        l = deep.get("lack")
        if m in MOTIVE_TO_POT:
            apply_score(st.session_state.b3_scores_total, MOTIVE_TO_POT[m])
            for pot, delta in MOTIVE_TO_POT[m].items():
                if pot in st.session_state.b3_scores_cols:
                    st.session_state.b3_scores_cols[pot][question.group] += float(delta)
        if l in LACK_TO_HINT:
            apply_score(st.session_state.b3_scores_total, LACK_TO_HINT[l])
            for pot, delta in LACK_TO_HINT[l].items():
                if pot in st.session_state.b3_scores_cols:
                    st.session_state.b3_scores_cols[pot][question.group] += float(delta)

    st.session_state.b3_index += 1
    if st.session_state.b3_index >= len(st.session_state.b3_questions):
        st.session_state.b3_done = True


def combined_total_scores() -> Dict[str, float]:
    out = {p: 0.0 for p in POTENTIALS}
    for p in POTENTIALS:
        out[p] += float(st.session_state.b1_scores.get(p, 0.0))
        out[p] += float(st.session_state.b2_scores.get(p, 0.0))
        out[p] += float(st.session_state.b3_scores_total.get(p, 0.0))
    return out


def best_column_for_pot(pot: str, col_scores: Dict[str, Dict[str, float]]) -> str:
    cols = col_scores.get(pot, {"c1": 0.0, "c2": 0.0, "c3": 0.0})
    return max(cols.keys(), key=lambda c: cols[c])


def assign_row_to_columns(row_pots: List[str], col_scores: Dict[str, Dict[str, float]]) -> Dict[str, str]:
    """
    row_pots: 3 потенциала
    Возвращает mapping: {"c1": pot, "c2": pot, "c3": pot}
    Выбираем перестановку, максимизирующую сумму соответствующих колонок.
    """
    cols = ["c1", "c2", "c3"]
    best_map = None
    best_sum = -1e9
    for perm in itertools.permutations(row_pots, 3):
        s = 0.0
        for c, pot in zip(cols, perm):
            s += float(col_scores.get(pot, {}).get(c, 0.0))
        if s > best_sum:
            best_sum = s
            best_map = dict(zip(cols, perm))
    return best_map or {"c1": row_pots[0], "c2": row_pots[1], "c3": row_pots[2]}


def build_3x3_table(row1: List[str], row2: List[str], row3: List[str], col_scores: Dict[str, Dict[str, float]]) -> List[List[str]]:
    m1 = assign_row_to_columns(row1, col_scores)
    m2 = assign_row_to_columns(row2, col_scores)
    m3 = assign_row_to_columns(row3, col_scores)
    # rows: [c1,c2,c3]
    return [
        [m1["c1"], m1["c2"], m1["c3"]],
        [m2["c1"], m2["c2"], m2["c3"]],
        [m3["c1"], m3["c2"], m3["c3"]],
    ]


# ============================
# UI: клиентский поток
# ============================

def render_welcome():
    st.title("Deep Identity · Диагностика потенциалов")
    st.write("Ответь честно. Здесь нет правильных ответов — есть твоя природа.")
    st.info("⚠️ Отчёт видит только мастер. Клиент видит только вопросы и финальный экран «Спасибо».")

    st.markdown("### Как тебя зовут? (чтобы в отчёте было имя)")
    st.session_state.client_name = st.text_input("Имя", value=st.session_state.client_name, placeholder="Например: Нурлан")

    st.markdown("### Контакт (опционально)")
    st.session_state.client_contact = st.text_input("Телеграм / email", value=st.session_state.client_contact, placeholder="@username или email")

    if st.button("Начать диагностику"):
        st.session_state.current_block = 1
        st.rerun()


def render_question_screen(question: Question, total_progress: Tuple[int, int], block_title: str, group_caption: str = ""):
    done, total = total_progress
    st.progress(min(1.0, done / max(1, total)))
    st.subheader(block_title)
    if group_caption:
        st.caption(group_caption)

    st.markdown(f"### {question.text}")

    labels = [o.text for o in question.options]

    selected_indices: List[int] = []
    if question.allow_multiple:
        chosen = st.multiselect("Выбери один или несколько вариантов:", options=labels, key=f"sel_{question.id}")
        selected_indices = [labels.index(x) for x in chosen] if chosen else []
    else:
        chosen = st.radio("Выбери один вариант:", options=["— не выбрано —"] + labels, key=f"sel_{question.id}")
        if chosen != "— не выбрано —":
            selected_indices = [labels.index(chosen)]

    comment = ""
    if question.allow_comment:
        comment = st.text_area("Комментарий/пример (по желанию, но очень желательно):", key=f"comm_{question.id}", height=90)

    # Углубление: показываем всегда, но можно свернуть
    with st.expander("🔎 Углубиться: «почему?» (рекомендую)"):
        deep = render_deep_probe(question.id, group_for_block3=(question.group if question.block == 3 else None))
    # иначе deep всё равно заполнен (radio), это нормально

    can_next = bool(selected_indices) or bool((comment or "").strip())
    st.write("")
    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button("Дальше ➜", disabled=not can_next, key=f"next_{question.id}"):
            if question.block == 1:
                apply_b1_answer(question, selected_indices, comment, deep)
                st.rerun()
            elif question.block == 2:
                apply_b2_answer(question, selected_indices, comment, deep)
                st.rerun()
            else:
                apply_b3_answer(question, selected_indices, comment, deep)
                st.rerun()
    with col2:
        st.caption("Если не хочешь выбирать вариант — напиши хотя бы 1–2 строки комментария. Это тоже сигнал.")


def render_block1():
    qid = st.session_state.b1_current_qid
    if qid is None or st.session_state.b1_done:
        st.session_state.current_block = 2
        st.rerun()

    q = st.session_state.b1_questions[qid]
    # прогресс (оценка)
    answered = len(st.session_state.b1_answers)
    approx_total = len(CORE_SEQUENCE_BLOCK1) + 6
    render_question_screen(
        q,
        total_progress=(answered, approx_total),
        block_title="Блок 1 · Детство и естественные склонности",
        group_caption="Мы ищем чистую мотивацию до социальных масок."
    )


def render_block2():
    if st.session_state.b2_done:
        st.session_state.current_block = 3
        st.rerun()

    idx = st.session_state.b2_index
    q_list = st.session_state.b2_questions
    q = q_list[idx]
    render_question_screen(
        q,
        total_progress=(idx, len(q_list)),
        block_title="Блок 2 · Работа, роли, процессы и смещения",
        group_caption="Здесь видно: «ядро» или «роль ради обязанностей»."
    )


def render_block3():
    if st.session_state.b3_done:
        st.session_state.current_block = 4
        st.rerun()

    idx = st.session_state.b3_index
    q_list = st.session_state.b3_questions
    q = q_list[idx]

    col_name = {"c1": "Столбец 1 · Восприятие (ВАУ)",
                "c2": "Столбец 2 · Процесс (интерес/притяжение)",
                "c3": "Столбец 3 · Результат (триумф/победа)"}[q.group]

    render_question_screen(
        q,
        total_progress=(idx, len(q_list)),
        block_title="Блок 3 · Столбцы: ВАУ → Процесс → Результат",
        group_caption=col_name
    )


def render_finish_and_save():
    st.title("Спасибо! Диагностика завершена ✅")
    st.write("Твои ответы сохранены. Мастер сформирует отчёт отдельно.")

    total = combined_total_scores()
    top = sorted(total.items(), key=lambda x: x[1], reverse=True)

    st.markdown("### Топ-сигналы (черновые, без интерпретации)")
    for k, v in top[:5]:
        st.write(f"- **{POT_LABEL[k]}**: {round(v, 2)}")

    # save
    payload = {
        "id": str(uuid.uuid4()),
        "created_at": _now_iso(),
        "client_name": st.session_state.client_name.strip(),
        "client_contact": st.session_state.client_contact.strip(),
        "answers": {
            "block1": st.session_state.b1_answers,
            "block2": st.session_state.b2_answers,
            "block3": st.session_state.b3_answers,
        },
        "scores": {
            "block1": st.session_state.b1_scores,
            "block2": st.session_state.b2_scores,
            "block3_total": st.session_state.b3_scores_total,
            "block3_cols": st.session_state.b3_scores_cols,
            "combined_total": total,
        },
        "master_report": {
            "generated_at": None,
            "draft_text": "",
            "rows_override": None,
        },
    }

    # guard: не сохранять повторно при rerun
    if "saved_result_id" not in st.session_state:
        append_result(payload)
        st.session_state.saved_result_id = payload["id"]

    st.success("Сохранено ✅")

    # download client copy (без отчёта)
    client_copy = {
        "id": payload["id"],
        "created_at": payload["created_at"],
        "client_name": payload["client_name"],
        "client_contact": payload["client_contact"],
        "answers": payload["answers"],
        "scores": payload["scores"],
    }
    st.download_button(
        "⬇️ Скачать мои ответы (копия)",
        data=json.dumps(client_copy, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name="deep_identity_my_answers.json",
        mime="application/json",
    )

    if st.button("Пройти заново"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()


# ============================
# Мастер-панель
# ============================

def master_auth_ok() -> bool:
    pwd_secret = None
    try:
        pwd_secret = st.secrets.get("MASTER_PASSWORD", None)
    except Exception:
        pwd_secret = None
    if not pwd_secret:
        pwd_secret = os.environ.get("MASTER_PASSWORD")

    st.sidebar.subheader("🔒 Мастер-режим")
    pwd = st.sidebar.text_input("Пароль мастера", type="password")
    if not pwd_secret:
        st.sidebar.warning("MASTER_PASSWORD не задан в Secrets. Добавь его в Streamlit Cloud.")
        return False
    return pwd == str(pwd_secret)


def format_session_title(item: Dict[str, Any]) -> str:
    name = item.get("client_name") or "Без имени"
    created = item.get("created_at") or ""
    rid = item.get("id", "")[:8]
    return f"{name} · {created} · #{rid}"


def build_master_table_default(scores_combined: Dict[str, float], col_scores: Dict[str, Dict[str, float]]) -> Tuple[List[str], List[str], List[str], List[List[str]]]:
    ranked = [k for k, _ in sorted(scores_combined.items(), key=lambda x: x[1], reverse=True)]
    row1 = ranked[:3]
    row2 = ranked[3:6]
    row3 = ranked[6:9]
    table = build_3x3_table(row1, row2, row3, col_scores)
    return row1, row2, row3, table


def render_3x3_table(table: List[List[str]]):
    # table rows contain potential keys
    headers = ["ВАУ (восприятие)", "Процесс (мотивация)", "Результат (действие)"]
    rows = []
    for r in table:
        rows.append([POT_LABEL[r[0]], POT_LABEL[r[1]], POT_LABEL[r[2]]])

    st.markdown("### Таблица 3×3 (ряды × столбцы)")
    st.table({"": ["Ряд 1 (ядро)", "Ряд 2 (наполнение/хобби)", "Ряд 3 (делегировать/слабее)"],
              headers[0]: [rows[0][0], rows[1][0], rows[2][0]],
              headers[1]: [rows[0][1], rows[1][1], rows[2][1]],
              headers[2]: [rows[0][2], rows[1][2], rows[2][2]]})


def compose_report_prompt(session: Dict[str, Any], table: List[List[str]]) -> str:
    # собираем тексты ответов + комментарии + deep_text
    def extract_text(block_answers: Dict[str, Any]) -> str:
        chunks = []
        for qid, a in block_answers.items():
            chunks.append(f"[{qid}] {a.get('question','')}")
            sel = a.get("selected", [])
            if sel:
                for s in sel:
                    chunks.append(f"- Выбрано: {s}")
            c = (a.get("comment") or "").strip()
            if c:
                chunks.append(f"Комментарий: {c}")
            deep = a.get("deep") or None
            if deep:
                motive = deep.get("motive")
                lack = deep.get("lack")
                dt = (deep.get("deep_text") or "").strip()
                if motive:
                    chunks.append(f"Почему(мотив): {motive}")
                if lack:
                    chunks.append(f"Если нет — что появляется: {lack}")
                if dt:
                    chunks.append(f"Глубинный пример: {dt}")
            chunks.append("")
        return "\n".join(chunks)

    answers = session.get("answers", {})
    b1 = extract_text(answers.get("block1", {}))
    b2 = extract_text(answers.get("block2", {}))
    b3 = extract_text(answers.get("block3", {}))

    # таблица как ключи + русские названия
    table_ru = [[POT_LABEL[x] for x in row] for row in table]

    prompt = f"""
Ты — ассистент мастера диагностики потенциалов (система 3×3: 3 ряда + 3 столбца).
Важно: это ЧЕРНОВОЙ отчёт для мастера. Клиент его не видит.

Правила:
- Не спорь с мастером, не «психологизируй» без опоры.
- Опирайся на ответы, комментарии и deep_text.
- Рубин — это адреналин/перезапуск/новизна/поездки/события, НЕ «спорт».
- Спорт может быть Шунгит (тело/выносливость), Цитрин (победа/скорость/первенство), Рубин (драйв/адреналин).
- Ряд 1 = ядро реализации (сильнейшие опоры).
- Ряд 2 = ряд наполнения/хобби/восстановления.
- Ряд 3 = слабее/перегружает/делегировать/дозировать.

Вот таблица 3×3 (в формате: Ряд1, Ряд2, Ряд3; столбцы: ВАУ/Процесс/Результат):
{json.dumps(table_ru, ensure_ascii=False)}

Задача:
Сгенерируй отчёт на русском, структурой:

1) Короткий портрет (6–10 строк).
2) Таблица 3×3 (в тексте продублируй красиво).
3) По каждому потенциалу в таблице:
   - Как проявляется у человека (по ответам).
   - Главная потребность.
   - Главная боль/ловушка.
   - Что делать (конкретные рекомендации).
4) Деньги/реализация:
   - 3–5 направлений, где человеку проще зарабатывать, исходя из ядра (ряд1).
   - Как монетизировать без выгорания (учитывая ряд2 и ряд3).
5) Энергия:
   - Что наполняет (ряд2 как ритуалы).
   - Что сжигает (ряд3) и как делегировать/дозировать.
6) Вопросы мастеру для живого интервью (7–10 вопросов), чтобы докопаться до бессознательного «почему».

Входные данные (ответы клиента):

--- Блок 1 ---
{b1}

--- Блок 2 ---
{b2}

--- Блок 3 ---
{b3}
"""
    return prompt.strip()


def generate_master_report(session: Dict[str, Any], table: List[List[str]]) -> str:
    client = get_openai_client()
    if client is None:
        return "⚠️ OpenAI недоступен. Проверь `requirements.txt` (openai) и `OPENAI_API_KEY` в Secrets."

    model = get_openai_model()

    system = (
        "Ты пишешь отчёт для мастера диагностики потенциалов. "
        "Это не медицинский и не психологический диагноз. "
        "Тон: чётко, практично, без воды."
    )
    prompt = compose_report_prompt(session, table)

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=0.6,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"⚠️ Ошибка при запросе к OpenAI:\n\n{e}"


def render_master():
    st.title("Deep Identity · Мастер-панель Асели")
    st.caption("Здесь только ты видишь клиентов, их ответы и генерируешь отчёты.")

    sessions = load_results()
    if not sessions:
        st.warning("Пока нет ни одной записи в deep_identity_results.json.")
        st.info("Если это Streamlit Cloud: в одном общем app всё будет сохраняться здесь после прохождения клиентом.")
        return

    sessions_sorted = sorted(sessions, key=lambda x: x.get("created_at", ""), reverse=True)
    labels = [format_session_title(x) for x in sessions_sorted]
    pick = st.selectbox("Выбери клиента", options=list(range(len(labels))), format_func=lambda i: labels[i])
    session = sessions_sorted[pick]

    st.markdown("### Данные клиента")
    st.write(f"**Имя:** {session.get('client_name') or '—'}")
    st.write(f"**Контакт:** {session.get('client_contact') or '—'}")
    st.write(f"**Дата:** {session.get('created_at')}")

    st.markdown("---")
    st.markdown("### Сырые суммы (для ориентира)")
    combined = session.get("scores", {}).get("combined_total", {})
    if combined:
        ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        for k, v in ranked:
            st.write(f"- **{POT_LABEL.get(k, k)}**: {round(float(v), 2)}")

    st.markdown("---")
    st.markdown("## Таблица 3×3")
    col_scores = session.get("scores", {}).get("block3_cols", {})

    row1_def, row2_def, row3_def, table_def = build_master_table_default(combined, col_scores)

    st.markdown("### Авто-предложение рядов (можно поправить)")
    colA, colB, colC = st.columns(3)
    with colA:
        row1 = st.multiselect("Ряд 1 (ядро) — выбери 3", options=POTENTIALS, default=row1_def,
                              format_func=lambda x: POT_LABEL[x], max_selections=3, key="row1")
    with colB:
        row2 = st.multiselect("Ряд 2 (наполнение) — выбери 3", options=POTENTIALS, default=row2_def,
                              format_func=lambda x: POT_LABEL[x], max_selections=3, key="row2")
    with colC:
        row3 = st.multiselect("Ряд 3 (делегировать) — выбери 3", options=POTENTIALS, default=row3_def,
                              format_func=lambda x: POT_LABEL[x], max_selections=3, key="row3")

    valid = (len(row1) == 3 and len(row2) == 3 and len(row3) == 3 and len(set(row1 + row2 + row3)) == 9)
    if not valid:
        st.error("Нужно выбрать ровно 3+3+3 и без повторов (в сумме 9 разных потенциалов).")
        st.stop()

    table = build_3x3_table(row1, row2, row3, col_scores)
    render_3x3_table(table)

    st.markdown("---")
    st.markdown("## Ответы клиента (включая комментарии)")
    with st.expander("Показать все ответы"):
        st.json(session.get("answers", {}), expanded=False)

    st.markdown("---")
    st.markdown("## Черновой отчёт (только для мастера)")
    mr = session.get("master_report", {}) or {}
    if mr.get("draft_text"):
        st.success("Есть сохранённый черновик отчёта.")
        st.text_area("Черновик", mr.get("draft_text", ""), height=350)
        st.download_button(
            "⬇️ Скачать черновик отчёта",
            data=(mr.get("draft_text", "")).encode("utf-8"),
            file_name=f"deep_identity_report_{session.get('client_name') or 'client'}.txt",
            mime="text/plain; charset=utf-8"
        )

    if st.button("✨ Сгенерировать/обновить черновик отчёта через OpenAI"):
        with st.spinner("Генерирую отчёт…"):
            draft = generate_master_report(session, table)
        session["master_report"]["draft_text"] = draft
        session["master_report"]["generated_at"] = _now_iso()
        session["master_report"]["rows_override"] = {"row1": row1, "row2": row2, "row3": row3, "table": table}
        update_result(session)
        st.rerun()


# ============================
# Основной роутер
# ============================

def main():
    st.set_page_config(page_title="Deep Identity", page_icon="🧠", layout="centered")
    init_state()

    st.sidebar.title("Deep Identity")
    mode = st.sidebar.radio("Режим", options=["Клиент", "Мастер"], index=0)
    st.session_state.mode = "master" if mode == "Мастер" else "client"

    if st.session_state.mode == "master":
        if not master_auth_ok():
            st.warning("Введи пароль мастера в сайдбаре.")
            return
        render_master()
        return

    # CLIENT FLOW
    if st.session_state.current_block == 0:
        render_welcome()
    elif st.session_state.current_block == 1:
        render_block1()
    elif st.session_state.current_block == 2:
        render_block2()
    elif st.session_state.current_block == 3:
        render_block3()
    else:
        render_finish_and_save()


if __name__ == "__main__":
    main()

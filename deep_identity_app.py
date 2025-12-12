from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import json
import os
import datetime

import streamlit as st

# Пытаемся импортировать OpenAI-клиент
try:
    from openai import OpenAI  # новый клиент
except Exception:  # на всякий случай
    OpenAI = None  # type: ignore

# =========================
# НАСТРОЙКИ
# =========================

RESULTS_FILE = "deep_identity_results.json"

# Пароль мастера (ЗАМЕНИ на свой!)
MASTER_PASSWORD = "amethyst2025"

# Модель для отчёта (если хочешь — поменяй на gpt-4.1)
OPENAI_MODEL = "gpt-4.1-mini"

POTENTIALS = [
    "Аметист",
    "Сапфир",
    "Гелиодор",
    "Гранат",
    "Цитрин",
    "Изумруд",
    "Янтарь",
    "Рубин",
    "Шунгит",
]

POTENTIALS_DESC = {
    "Аметист": "Глубина, анализ, стратегии, мышление.",
    "Сапфир": "Смысл, духовность, миссия, внутренний вектор.",
    "Гелиодор": "Голос, общение, подача, харизма в разговоре.",
    "Гранат": "Сцена, эмоции, внимание, красота для других.",
    "Цитрин": "Результат, деньги, сделки, бизнес-подход.",
    "Изумруд": "Красота, эстетика, забота, атмосфера.",
    "Янтарь": "Порядок, структура, здоровье, детали.",
    "Рубин": "Риск, драйв, приключения, события.",
    "Шунгит": "Тело, сила, выносливость, физическая опора.",
}


# =========================
# УТИЛИТЫ: OpenAI + JSON
# =========================

def get_openai_client() -> Optional["OpenAI"]:
    """Аккуратно достаём OpenAI-клиент: сначала из st.secrets, потом из ENV."""
    api_key: Optional[str] = None

    # 1) Streamlit secrets
    try:
        api_key = st.secrets.get("OPENAI_API_KEY", None)
    except Exception:
        api_key = None

    # 2) Переменная окружения
    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key or OpenAI is None:
        return None

    return OpenAI(api_key=api_key)


def load_results() -> List[Dict[str, Any]]:
    if not os.path.exists(RESULTS_FILE):
        return []
    try:
        with open(RESULTS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            return []
    except Exception:
        return []


def save_results(all_results: List[Dict[str, Any]]) -> None:
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)


# ================================================================
# БЛОК 1 — ДЕТСТВО, ЕСТЕСТВЕННЫЕ СКЛОННОСТИ (ветвления)
# ================================================================

@dataclass
class B1AnswerOption:
    text: str
    score_changes: Dict[str, int] = field(default_factory=dict)
    inject_questions: List[str] = field(default_factory=list)


@dataclass
class B1Question:
    id: str
    text: str
    options: List[B1AnswerOption]
    block: int = 1


def build_questions_block1() -> Dict[str, B1Question]:
    """Создаём пул вопросов Блока 1 с ветвлениями."""
    q: Dict[str, B1Question] = {}

    # Q1: Свободное время в детстве
    q["b1_q1_free_play"] = B1Question(
        id="b1_q1_free_play",
        text="Как ты чаще всего проводил(а) свободное время в детстве (лет до 10)?",
        options=[
            B1AnswerOption(
                text="Активные игры на улице: бег, догонялки, лазание, спорт",
                score_changes={"SHУNGIT": 2},
                inject_questions=["b1_q7_sport_detail"],
            ),
            B1AnswerOption(
                text="Сидел(а) с книгами, кроссвордами, головоломками, любил(а) думать",
                score_changes={"Аметист": 2, "Сапфир": 1},
                inject_questions=["b1_q2_reading_detail"],
            ),
            B1AnswerOption(
                text="Играл(а) в «магазин», продажи, обмены, организовывал(а) мини-бизнесы",
                score_changes={"Цитрин": 2},
                inject_questions=["b1_q3_trading_detail"],
            ),
            B1AnswerOption(
                text="Ставил(а) концерты, сценки, выступал(а) перед другими",
                score_changes={"Гранат": 2, "Рубин": 1},
                inject_questions=["b1_q4_stage_detail"],
            ),
            B1AnswerOption(
                text="Возился(ась) с животными, очень их жалел(а), лечил(а), кормил(а)",
                score_changes={"Изумруд": 2},
                inject_questions=["b1_q5_animals_detail"],
            ),
            B1AnswerOption(
                text="Любил(а) разбирать/собирать, конструктор, Лего, мозаики, схемы",
                score_changes={"Янтарь": 1, "Аметист": 1},
                inject_questions=["b1_q8_order_tech_detail"],
            ),
            B1AnswerOption(
                text="Больше болтал(а), общался(ась), придумывал(а) истории и шутки",
                score_changes={"Гелиодор": 2},
                inject_questions=["b1_q6_talk_detail"],
            ),
        ],
    )

    # Q2: Любимые предметы
    q["b1_q2_subjects"] = B1Question(
        id="b1_q2_subjects",
        text="Какие школьные предметы ты любил(а) больше всего?",
        options=[
            B1AnswerOption(
                text="Математика, физика, информатика, логические задачи",
                score_changes={"Аметист": 2, "Янтарь": 1},
            ),
            B1AnswerOption(
                text="Литература, история, обществознание, философия",
                score_changes={"Сапфир": 2, "Аметист": 1},
            ),
            B1AnswerOption(
                text="Музыка, пение, театр, выступления",
                score_changes={"Гранат": 2, "Гелиодор": 1},
                inject_questions=["b1_q9_music_detail"],
            ),
            B1AnswerOption(
                text="Физкультура, любые соревнования",
                score_changes={"Шунгит": 2},
                inject_questions=["b1_q7_sport_detail"],
            ),
            B1AnswerOption(
                text="Рисование, труд, дизайн, всё красивое и творческое",
                score_changes={"Изумруд": 2},
            ),
            B1AnswerOption(
                text="Ничего особо не любил(а), учился(ась) «как надо»",
                score_changes={"Янтарь": 1},
            ),
        ],
    )

    # Q3: Кружки
    q["b1_q3_clubs"] = B1Question(
        id="b1_q3_clubs",
        text="На какие кружки/секции ты ходил(а) в детстве и подростковом возрасте?",
        options=[
            B1AnswerOption(
                text="Спорт (бои, футбол, танцы, гимнастика и др.)",
                score_changes={"Шунгит": 2},
                inject_questions=["b1_q7_sport_detail"],
            ),
            B1AnswerOption(
                text="Музыка/вокал/театр/танцы как сцена и выступления",
                score_changes={"Гранат": 2, "Гелиодор": 1},
                inject_questions=["b1_q4_stage_detail"],
            ),
            B1AnswerOption(
                text="Технические кружки, конструктор, моделирование, робототехника",
                score_changes={"Янтарь": 2, "Аметист": 1},
            ),
            B1AnswerOption(
                text="Рисование, дизайн, hand-made, прикладное творчество",
                score_changes={"Изумруд": 2},
            ),
            B1AnswerOption(
                text="Бизнес/лидерство/дискуссионные клубы, олимпиады по экономике и т.п.",
                score_changes={"Цитрин": 2, "Рубин": 1},
            ),
            B1AnswerOption(
                text="Почти никуда не ходил(а) / всё было «по обязанности», а не по любви",
                score_changes={"Янтарь": 1},
            ),
        ],
    )

    # Q4: Кто выбирал активности
    q["b1_q4_choice"] = B1Question(
        id="b1_q4_choice",
        text="Кто чаще выбирал, чем ты занимаешься в детстве (кружки, секции, занятия)?",
        options=[
            B1AnswerOption(
                text="В основном я сам(а) решал(а), что хочу делать",
                score_changes={"Цитрин": 1, "Аметист": 1},
            ),
            B1AnswerOption(
                text="Чаще родители/взрослые, а я просто соглашался(ась)",
                score_changes={"Янтарь": 1},
            ),
            B1AnswerOption(
                text="Комбинация: я проявлял(а) инициативу, но финальное решение за взрослыми",
                score_changes={"Рубин": 1},
            ),
        ],
    )

    # Q5: Стиль общения
    q["b1_q5_speaking"] = B1Question(
        id="b1_q5_speaking",
        text="Какой ты был(а) в детстве по общению?",
        options=[
            B1AnswerOption(
                text="Много говорил(а), шутил(а), любил(а) быть в компании",
                score_changes={"Гелиодор": 2, "Гранат": 1},
                inject_questions=["b1_q6_talk_detail"],
            ),
            B1AnswerOption(
                text="Скорее тихий(ая), наблюдал(а), больше слушал(а), чем говорил(а)",
                score_changes={"Аметист": 1, "Сапфир": 1},
            ),
            B1AnswerOption(
                text="Было по-разному: с близкими — очень живой(ая), с чужими — закрытый(ая)",
                score_changes={"Гелиодор": 1, "Изумруд": 1},
            ),
        ],
    )

    # Q6: Животные
    q["b1_q6_animals"] = B1Question(
        id="b1_q6_animals",
        text="Были ли у тебя в детстве домашние животные и как ты к ним относился(ась)?",
        options=[
            B1AnswerOption(
                text="Да, и я очень к ним привязывался(ась), ухаживал(а), переживал(а) за них",
                score_changes={"Изумруд": 2},
                inject_questions=["b1_q5_animals_detail"],
            ),
            B1AnswerOption(
                text="Да, но я относился(ась) спокойно, без сильной привязанности",
                score_changes={"Изумруд": 1},
            ),
            B1AnswerOption(
                text="Почти не было животных / мне было не очень интересно",
                score_changes={},
            ),
        ],
    )

    # Q7: Тело и спорт
    q["b1_q7_body"] = B1Question(
        id="b1_q7_body",
        text="Как ты в детстве относился(ась) к физическим нагрузкам и спорту?",
        options=[
            B1AnswerOption(
                text="Обожал(а) двигаться: бег, лазание, соревнования, чувствовать тело",
                score_changes={"Шунгит": 2},
                inject_questions=["b1_q7_sport_detail"],
            ),
            B1AnswerOption(
                text="Спорт был, но больше как «надо», без фанатизма",
                score_changes={"Янтарь": 1},
            ),
            B1AnswerOption(
                text="Скорее избегал(а) физнагрузок, любил(а) спокойные занятия",
                score_changes={"Аметист": 1, "Сапфир": 1},
            ),
        ],
    )

    # Q8: Порядок
    q["b1_q8_order"] = B1Question(
        id="b1_q8_order",
        text="Как ты относился(ась) к порядку в детстве?",
        options=[
            B1AnswerOption(
                text="Любил(а) раскладывать всё по местам, наводить порядок, меня раздражал хаос",
                score_changes={"Янтарь": 2},
                inject_questions=["b1_q8_order_tech_detail"],
            ),
            B1AnswerOption(
                text="Порядок был норм, но я не зацикливался(ась)",
                score_changes={"Янтарь": 1},
            ),
            B1AnswerOption(
                text="Мог(ла) жить и в творческом беспорядке, главное — чтобы было интересно",
                score_changes={"Гранат": 1, "Изумруд": 1},
            ),
        ],
    )

    # Q9: Чтение
    q["b1_q9_reading"] = B1Question(
        id="b1_q9_reading",
        text="Как ты относился(ась) к чтению и обучению в детстве?",
        options=[
            B1AnswerOption(
                text="Много читал(а) сам(а), мне нравилось узнавать новое",
                score_changes={"Аметист": 2, "Сапфир": 1},
                inject_questions=["b1_q2_reading_detail"],
            ),
            B1AnswerOption(
                text="Читал(а), если надо было для школы, сам(а) редко выбирал(а) книги",
                score_changes={"Янтарь": 1},
            ),
            B1AnswerOption(
                text="Чтение вообще не привлекало, старался(ась) избегать",
                score_changes={},
            ),
        ],
    )

    # Q10: «Бизнесы»
    q["b1_q10_money_play"] = B1Question(
        id="b1_q10_money_play",
        text="Были ли у тебя в детстве игры/занятия, связанные с деньгами, продажами, обменом?",
        options=[
            B1AnswerOption(
                text="Да, любил(а) продавать, менять, придумывать «свои правила», торговаться",
                score_changes={"Цитрин": 2},
                inject_questions=["b1_q3_trading_detail"],
            ),
            B1AnswerOption(
                text="Иногда такое было, но не было прям сильной тяги к этому",
                score_changes={"Цитрин": 1},
            ),
            B1AnswerOption(
                text="Нет, меня это мало интересовало",
                score_changes={},
            ),
        ],
    )

    # Q11: Сцена или закулисье
    q["b1_q11_scene_vs_director"] = B1Question(
        id="b1_q11_scene_vs_director",
        text="Представь школьный праздник. Где ты себя чаще видел(а)?",
        options=[
            B1AnswerOption(
                text="На сцене, в центре внимания, выступать, читать, танцевать",
                score_changes={"Гранат": 2},
            ),
            B1AnswerOption(
                text="За кадром: придумывать сценарий, организовать, чтобы всё было идеально",
                score_changes={"Рубин": 2, "Янтарь": 1},
            ),
            B1AnswerOption(
                text="В зале: наблюдать, поддерживать, радоваться за других",
                score_changes={"Изумруд": 1, "Сапфир": 1},
            ),
        ],
    )

    # Q12: Чувствительность
    q["b1_q12_empathy"] = B1Question(
        id="b1_q12_empathy",
        text="Насколько ты был(а) чувствительным ребёнком?",
        options=[
            B1AnswerOption(
                text="Очень: мог(ла) заплакать от фильма, жалко людей и животных",
                score_changes={"Изумруд": 2, "Сапфир": 1},
            ),
            B1AnswerOption(
                text="Скорее средне, иногда трогало, но не часто",
                score_changes={"Изумруд": 1},
            ),
            B1AnswerOption(
                text="Редко что-то задевало, я больше в разуме, чем в чувствах",
                score_changes={"Аметист": 1},
            ),
        ],
    )

    # Q13: Мечты
    q["b1_q13_dreams"] = B1Question(
        id="b1_q13_dreams",
        text="О чём ты больше всего мечтал(а) в детстве/подростковом возрасте?",
        options=[
            B1AnswerOption(
                text="Стать известным(ой): сцена, популярность, признание",
                score_changes={"Гранат": 2, "Гелиодор": 1},
            ),
            B1AnswerOption(
                text="О своём бизнесе, магазине, деле, быть «хозяином своей жизни»",
                score_changes={"Цитрин": 2, "Рубин": 1},
            ),
            B1AnswerOption(
                text="О том, чтобы помогать людям, лечить, учить, делать мир лучше",
                score_changes={"Сапфир": 2, "Изумруд": 1},
            ),
            B1AnswerOption(
                text="О красивой жизни, путешествиях, красоте вокруг",
                score_changes={"Изумруд": 2},
            ),
            B1AnswerOption(
                text="О стабильности и безопасности, доме, семейном уюте",
                score_changes={"Янтарь": 2},
            ),
        ],
    )

    # Q14: Реакция на хаос
    q["b1_q14_chaos_reaction"] = B1Question(
        id="b1_q14_chaos_reaction",
        text="Представь комнату, где всё раскидано. Как ты чаще реагировал(а) в детстве?",
        options=[
            B1AnswerOption(
                text="Хотелось всё разложить по местам, трудно расслабиться в таком хаосе",
                score_changes={"Янтарь": 2},
            ),
            B1AnswerOption(
                text="Мог(ла) навести порядок, но не сразу, сначала делал(а) что-то своё",
                score_changes={"Янтарь": 1},
            ),
            B1AnswerOption(
                text="Меня вообще не напрягало, главное — чтобы было интересно/весело",
                score_changes={"Гранат": 1, "Изумруд": 1},
            ),
        ],
    )

    # Q15: Конкуренция
    q["b1_q15_competition"] = B1Question(
        id="b1_q15_competition",
        text="Как ты относился(ась) к конкуренции и соревнованиям в детстве?",
        options=[
            B1AnswerOption(
                text="Мне очень важно было быть лучшим(ей), первым(ой), сильно переживал(а), если кто-то лучше",
                score_changes={"Гранат": 1, "Шунгит": 1, "Цитрин": 1},
            ),
            B1AnswerOption(
                text="Мне было интересно участвовать, но не критично быть первым(ой)",
                score_changes={"Шунгит": 1},
            ),
            B1AnswerOption(
                text="Не любил(а) дух соревнований, избегал(а) таких ситуаций",
                score_changes={"Сапфир": 1, "Аметист": 1},
            ),
        ],
    )

    # =============== ДОП. ВЕТКИ ===============

    q["b1_q5_animals_detail"] = B1Question(
        id="b1_q5_animals_detail",
        text="Что было для тебя самым важным в отношении к животным?",
        options=[
            B1AnswerOption(
                text="Следить, чтобы им было комфортно, тепло, чтобы они были сыты и ухожены",
                score_changes={"Изумруд": 2, "Янтарь": 1},
            ),
            B1AnswerOption(
                text="Обнимать, играть, чувствовать контакт и любовь",
                score_changes={"Изумруд": 2},
            ),
            B1AnswerOption(
                text="Наблюдать, как они двигаются и ведут себя, просто было интересно",
                score_changes={"Аметист": 1, "Изумруд": 1},
            ),
        ],
    )

    q["b1_q7_sport_detail"] = B1Question(
        id="b1_q7_sport_detail",
        text="Что тебе больше всего нравилось в спорте и движении в детстве?",
        options=[
            B1AnswerOption(
                text="Чувствовать силу тела, усталость, мышцы, преодолевать себя",
                score_changes={"Шунгит": 2},
            ),
            B1AnswerOption(
                text="Соревнования, медали, признание и внимание других",
                score_changes={"Гранат": 1, "Шунгит": 1},
            ),
            B1AnswerOption(
                text="Быть в команде, общение и энергия группы",
                score_changes={"Гелиодор": 1, "Шунгит": 1},
            ),
        ],
    )

    q["b1_q4_stage_detail"] = B1Question(
        id="b1_q4_stage_detail",
        text="Что тебя больше всего притягивало в сцене и выступлениях?",
        options=[
            B1AnswerOption(
                text="Быть в центре внимания, видеть реакцию, эмоции людей",
                score_changes={"Гранат": 2, "Гелиодор": 1},
            ),
            B1AnswerOption(
                text="Придумывать номер, образ, сценарий, как всё будет выглядеть",
                score_changes={"Рубин": 2, "Изумруд": 1},
            ),
            B1AnswerOption(
                text="Проигрывать глубокие эмоции, истории, смысл через выступление",
                score_changes={"Гранат": 1, "Сапфир": 1},
            ),
        ],
    )

    q["b1_q6_talk_detail"] = B1Question(
        id="b1_q6_talk_detail",
        text="Какой формат общения тебе был ближе в детстве?",
        options=[
            B1AnswerOption(
                text="Развлекать, шутить, поднимать настроение другим",
                score_changes={"Гелиодор": 2, "Гранат": 1},
            ),
            B1AnswerOption(
                text="Глубокие разговоры по душам один на один",
                score_changes={"Сапфир": 1, "Изумруд": 1},
            ),
            B1AnswerOption(
                text="Обсуждать идеи, планы, как что-то сделать или улучшить",
                score_changes={"Цитрин": 1, "Рубин": 1, "Аметист": 1},
            ),
        ],
    )

    q["b1_q2_reading_detail"] = B1Question(
        id="b1_q2_reading_detail",
        text="Какие книги/материалы тебе были интереснее всего в детстве/юности?",
        options=[
            B1AnswerOption(
                text="Научно-популярные, энциклопедии, факты, как всё устроено",
                score_changes={"Аметист": 2, "Янтарь": 1},
            ),
            B1AnswerOption(
                text="Истории людей, судьбы, философия, смыслы, психология",
                score_changes={"Сапфир": 2, "Изумруд": 1},
            ),
            B1AnswerOption(
                text="Фантастика, миры, магия, приключения",
                score_changes={"Гранат": 1, "Сапфир": 1},
            ),
            B1AnswerOption(
                text="Практичные вещи: деньги, бизнес, успех, мотивация",
                score_changes={"Цитрин": 2},
            ),
        ],
    )

    q["b1_q3_trading_detail"] = B1Question(
        id="b1_q3_trading_detail",
        text="Что тебе больше всего нравилось в играх «магазин», продажах, обменах?",
        options=[
            B1AnswerOption(
                text="Сам процесс сделки: договориться, уговорить, продать подороже",
                score_changes={"Цитрин": 2, "Гелиодор": 1},
            ),
            B1AnswerOption(
                text="Придумывать, КАК всё оформить: витрина, красивые вещи, упаковка",
                score_changes={"Изумруд": 2, "Рубин": 1},
            ),
            B1AnswerOption(
                text="Считать, планировать, кто сколько должен, вести условную «бухгалтерию»",
                score_changes={"Янтарь": 2, "Аметист": 1},
            ),
        ],
    )

    q["b1_q8_order_tech_detail"] = B1Question(
        id="b1_q8_order_tech_detail",
        text="Что тебе особенно нравилось в порядке/структуре/конструкциях?",
        options=[
            B1AnswerOption(
                text="Когда всё разложено по полочкам, чисто и аккуратно",
                score_changes={"Янтарь": 2},
            ),
            B1AnswerOption(
                text="Собирать/разбирать, понимать, как что устроено внутри",
                score_changes={"Аметист": 1, "Янтарь": 1},
            ),
            B1AnswerOption(
                text="Сделать так, чтобы было и удобно, и красиво",
                score_changes={"Изумруд": 1, "Янтарь": 1},
            ),
        ],
    )

    q["b1_q9_music_detail"] = B1Question(
        id="b1_q9_music_detail",
        text="Какую роль занимала музыка/пение в твоей жизни в детстве/юности?",
        options=[
            B1AnswerOption(
                text="Любил(а) петь сам(а), хотелось выступать, но иногда было страшно",
                score_changes={"Гелиодор": 2, "Гранат": 1},
            ),
            B1AnswerOption(
                text="Больше любил(а) слушать и чувствовать музыку, чем самому выступать",
                score_changes={"Изумруд": 1, "Сапфир": 1},
            ),
            B1AnswerOption(
                text="Музыка была фоном, особой роли не играла",
                score_changes={},
            ),
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


def b1_init_state():
    if st.session_state.get("b1_initialized"):
        return
    st.session_state["b1_initialized"] = True
    st.session_state["b1_questions"] = build_questions_block1()
    st.session_state["b1_scores"] = {p: 0 for p in POTENTIALS}
    st.session_state["b1_answers"] = {}
    st.session_state["b1_current_qid"] = CORE_SEQUENCE_BLOCK1[0]
    st.session_state["b1_core_index"] = 0
    st.session_state["b1_injected_queue"] = []
    st.session_state["b1_finished"] = False


def b1_get_next_question_id() -> Optional[str]:
    # сначала ветви
    while st.session_state["b1_injected_queue"]:
        next_id = st.session_state["b1_injected_queue"].pop(0)
        if next_id not in st.session_state["b1_answers"]:
            return next_id

    # потом основная последовательность
    while True:
        st.session_state["b1_core_index"] += 1
        if st.session_state["b1_core_index"] >= len(CORE_SEQUENCE_BLOCK1):
            return None
        candidate = CORE_SEQUENCE_BLOCK1[st.session_state["b1_core_index"]]
        if candidate not in st.session_state["b1_answers"]:
            return candidate


def b1_apply_answer(question: B1Question, option_index: int):
    st.session_state["b1_answers"][question.id] = option_index
    option = question.options[option_index]

    for pot, delta in option.score_changes.items():
        if pot in st.session_state["b1_scores"]:
            st.session_state["b1_scores"][pot] += delta

    for qid in option.inject_questions:
        if (
            qid not in st.session_state["b1_answers"]
            and qid not in st.session_state["b1_injected_queue"]
        ):
            st.session_state["b1_injected_queue"].append(qid)

    next_id = b1_get_next_question_id()
    st.session_state["b1_current_qid"] = next_id
    if next_id is None:
        st.session_state["b1_finished"] = True


def b1_render_question():
    qid = st.session_state["b1_current_qid"]
    q_dict: Dict[str, B1Question] = st.session_state["b1_questions"]
    question = q_dict[qid]

    st.subheader("Блок 1 · Детство и естественные склонности")
    answered_count = len(st.session_state["b1_answers"])
    approx_total = len(CORE_SEQUENCE_BLOCK1) + 6
    progress = min(1.0, answered_count / approx_total)
    st.progress(progress)

    st.markdown(f"**Вопрос:** {question.text}")

    option_key = f"b1_q_{question.id}"
    current_value = st.session_state["b1_answers"].get(question.id, 0)

    choice_index = st.radio(
        label="Выбери вариант ответа:",
        options=list(range(len(question.options))),
        format_func=lambda i: question.options[i].text,
        index=current_value if question.id in st.session_state["b1_answers"] else 0,
        key=option_key,
    )

    if st.button("Дальше", key=f"b1_next_{question.id}"):
        b1_apply_answer(question, choice_index)


# ================================================================
# БЛОК 2 — ПРОФЕССИИ, ПРОЦЕССЫ, СМЕЩЕНИЯ
# ================================================================

@dataclass
class B2AnswerOption:
    text: str
    score_changes: Dict[str, float] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)


@dataclass
class B2Question:
    id: str
    block: str
    text: str
    options: List[B2AnswerOption]
    allow_multiple: bool = False
    allow_text: bool = False


def build_block2_questions() -> List[B2Question]:
    q: List[B2Question] = []

    # СМЕЩЕНИЯ
    q.append(
        B2Question(
            id="shift_1_trust_intuition",
            block="shifts",
            text="Как вы обычно относитесь к своему внутреннему «чую» и решениям?",
            options=[
                B2AnswerOption(
                    text="Чаще не доверяю себе и спрашиваю других, нуждаюсь в подтверждении.",
                ),
                B2AnswerOption(
                    text="Я уверена(ен), что права(прав), и почти никогда не сомневаюсь в своём мнении.",
                ),
                B2AnswerOption(
                    text="Я слушаю и себя, и других: взвешиваю своё ощущение и факты/советы.",
                ),
            ],
            allow_multiple=False,
            allow_text=True,
        )
    )

    q.append(
        B2Question(
            id="shift_2_interest",
            block="shifts",
            text="Как у вас с интересом и мотивацией к жизни и развитию?",
            options=[
                B2AnswerOption(
                    text="Меня мало что цепляет, часто ощущаю «ничего не хочется».",
                ),
                B2AnswerOption(
                    text="Мне интересно почти всё, хватаюсь за многое и сложно удерживать фокус.",
                ),
                B2AnswerOption(
                    text="Есть несколько тем, которые по-настоящему вдохновляют, и я держу фокус на них.",
                ),
            ],
            allow_multiple=False,
            allow_text=True,
        )
    )

    q.append(
        B2Question(
            id="shift_3_value_self",
            block="shifts",
            text="Как вы ощущаете свою ценность и вклад для других?",
            options=[
                B2AnswerOption(
                    text="Часто обесцениваю себя, сложно поверить, что я реально ценна(ен) для людей.",
                ),
                B2AnswerOption(
                    text="Мне трудно доверять другим — всё лучше сделаю сам(а), держу всё под контролем.",
                ),
                B2AnswerOption(
                    text="В целом свою ценность признаю и могу делегировать, если надо.",
                ),
            ],
            allow_multiple=False,
            allow_text=True,
        )
    )

    # ПРОФЕССИИ

    q.append(
        B2Question(
            id="work_main_activity",
            block="work",
            text="Чем вы в основном занимаетесь сейчас (или чем занимались последние 2–3 года)?",
            options=[
                B2AnswerOption(
                    text="Работаю с людьми: команда, клиенты, обучение, сопровождение, сервис.",
                    score_changes={"Гелиодор": 0.7, "Гранат": 0.5, "Сапфир": 0.3},
                ),
                B2AnswerOption(
                    text="Аналитика, логика, структура: системы, процессы, документы, отчёты.",
                    score_changes={"Аметист": 0.7, "Янтарь": 0.5, "Цитрин": 0.3},
                ),
                B2AnswerOption(
                    text="Красота и визуал: стиль, фото/видео, дизайн, эстетика, атмосфера.",
                    score_changes={"Изумруд": 0.8, "Гранат": 0.4},
                ),
                B2AnswerOption(
                    text="Работа руками или телом: спорт, массаж, производство, ручной труд.",
                    score_changes={"Шунгит": 0.8, "Янтарь": 0.3},
                ),
                B2AnswerOption(
                    text="Операционка и админка: организация, логистика, «чтобы всё работало».",
                    score_changes={"Цитрин": 0.6, "Янтарь": 0.6},
                ),
                B2AnswerOption(
                    text="Я сейчас в поиске/переходе, чёткого рода деятельности нет.",
                    score_changes={"Сапфир": 0.6},
                ),
            ],
            allow_multiple=True,
            allow_text=True,
        )
    )

    q.append(
        B2Question(
            id="work_like_or_not",
            block="work",
            text="Нравится ли вам ваша текущая деятельность в целом?",
            options=[
                B2AnswerOption(
                    text="Да, я чувствую себя на своём месте, чаще вдохновлена(ен), чем выжата(ан).",
                ),
                B2AnswerOption(
                    text="Скорее да, но есть несколько процессов, которые сильно выматывают.",
                ),
                B2AnswerOption(
                    text="Скорее нет, чувствую, что живу не свою жизнь / не свой сценарий.",
                ),
                B2AnswerOption(
                    text="Совсем нет, работаю ради денег / обязанностей.",
                ),
            ],
            allow_multiple=False,
            allow_text=True,
        )
    )

    q.append(
        B2Question(
            id="work_ideal_roles",
            block="work",
            text="Представьте, что вам не нужно думать о деньгах. Какие 2–3 роли/профессии вы бы выбрали из списка?",
            options=[
                B2AnswerOption(
                    text="Сценический артист / ведущий / блогер — быть «лицом» и притягивать внимание.",
                    score_changes={"Гранат": 0.9, "Гелиодор": 0.6},
                ),
                B2AnswerOption(
                    text="Наставник / коуч / психолог / духовный проводник.",
                    score_changes={"Сапфир": 0.9, "Аметист": 0.5, "Гелиодор": 0.4},
                ),
                B2AnswerOption(
                    text="Стратег / аналитик / архитектор систем / методолог.",
                    score_changes={"Аметист": 0.9, "Янтарь": 0.5, "Цитрин": 0.3},
                ),
                B2AnswerOption(
                    text="Создатель красоты: стилист, дизайнер, фотограф, визажист, декоратор.",
                    score_changes={"Изумруд": 0.9, "Гранат": 0.4},
                ),
                B2AnswerOption(
                    text="Бизнес-создатель: предприниматель, продюсер, продакт-менеджер.",
                    score_changes={"Цитрин": 0.9, "Рубин": 0.4},
                ),
                B2AnswerOption(
                    text="Телесные практики, спорт, фитнес, массаж, работа с телом.",
                    score_changes={"Шунгит": 0.9, "Янтарь": 0.4},
                ),
                B2AnswerOption(
                    text="Организатор приключений: туры, экспедиции, события, экстремальные активности.",
                    score_changes={"Рубин": 0.9, "Шунгит": 0.4},
                ),
            ],
            allow_multiple=True,
            allow_text=True,
        )
    )

    # ПРОЦЕССЫ

    q.append(
        B2Question(
            id="process_flow_state",
            block="process",
            text="В каких задачах/процессах вы чаще всего «залипаете» и теряете счёт времени?",
            options=[
                B2AnswerOption(
                    text="Когда разбираюсь в теме до сути, читаю, анализирую, соединяю смыслы.",
                    score_changes={"Аметист": 1.0, "Сапфир": 0.4},
                ),
                B2AnswerOption(
                    text="Когда общаюсь: веду сессии, объясняю, вдохновляю, отвечаю на вопросы.",
                    score_changes={"Гелиодор": 1.0, "Сапфир": 0.3},
                ),
                B2AnswerOption(
                    text="Когда делаю красоту: визуал, стиль, контент, оформление, атмосферу.",
                    score_changes={"Изумруд": 1.0},
                ),
                B2AnswerOption(
                    text="Когда строю порядок: настраиваю систему, структуру, процессы, чек-листы.",
                    score_changes={"Янтарь": 1.0, "Аметист": 0.3},
                ),
                B2AnswerOption(
                    text="Когда считаю деньги, продумываю, как сделать продукт прибыльнее.",
                    score_changes={"Цитрин": 1.0},
                ),
                B2AnswerOption(
                    text="Когда двигаюсь: спорт, пешие прогулки, физическая активность.",
                    score_changes={"Шунгит": 1.0, "Рубин": 0.3},
                ),
                B2AnswerOption(
                    text="Когда организую движ: мероприятие, поездку, запуск, проект.",
                    score_changes={"Рубин": 1.0, "Цитрин": 0.3},
                ),
            ],
            allow_multiple=True,
            allow_text=True,
        )
    )

    q.append(
        B2Question(
            id="process_hate",
            block="process",
            text="Какие типы задач вы больше всего не любите и откладываете до последнего?",
            options=[
                B2AnswerOption(
                    text="Долгие, глубокие размышления и анализ, когда нужно всё продумать заранее.",
                    score_changes={"Аметист": -0.7},
                ),
                B2AnswerOption(
                    text="Порядок, систематизация, рутинные процессы, документы, таблицы.",
                    score_changes={"Янтарь": -0.9},
                ),
                B2AnswerOption(
                    text="Коммуникация и общение, особенно с незнакомыми людьми / публично.",
                    score_changes={"Гелиодор": -0.9, "Гранат": -0.4},
                ),
                B2AnswerOption(
                    text="Творческие задачи без чётких критериев: придумать, как сделать красиво/особенно.",
                    score_changes={"Изумруд": -0.9, "Гранат": -0.3},
                ),
                B2AnswerOption(
                    text="Деньги, финансы, договора, обсуждение стоимости.",
                    score_changes={"Цитрин": -0.9},
                ),
                B2AnswerOption(
                    text="Физический труд, спорт, телесная нагрузка.",
                    score_changes={"Шунгит": -0.9, "Рубин": -0.3},
                ),
            ],
            allow_multiple=True,
            allow_text=True,
        )
    )

    q.append(
        B2Question(
            id="process_decision_style",
            block="process",
            text="Как вы чаще всего принимаете важные решения?",
            options=[
                B2AnswerOption(
                    text="Через анализ: собираю факты, взвешиваю варианты, строю логическую картину.",
                    score_changes={"Аметист": 0.9, "Янтарь": 0.4},
                ),
                B2AnswerOption(
                    text="Через импульс и риск: «почувствовал(а) — сделал(а)», а потом уже разберусь.",
                    score_changes={"Рубин": 0.9},
                ),
                B2AnswerOption(
                    text="Через вдохновение и знак: жду ощущение «кликнуло внутри».",
                    score_changes={"Сапфир": 0.9, "Аметист": 0.3},
                ),
                B2AnswerOption(
                    text="Через эмоции и людей: как это отзовётся у близких/клиентов/аудитории.",
                    score_changes={"Гелиодор": 0.7, "Гранат": 0.5},
                ),
                B2AnswerOption(
                    text="Через выгоду: считаю риски, деньги и долгосрочный эффект.",
                    score_changes={"Цитрин": 0.9},
                ),
                B2AnswerOption(
                    text="Через ощущения в теле и настроения: если тело обмякло/зажалось — это знак.",
                    score_changes={"Изумруд": 0.6, "Шунгит": 0.6},
                ),
            ],
            allow_multiple=True,
            allow_text=True,
        )
    )

    # СЦЕНА / КРАСОТА / ТЕЛО / ДЕНЬГИ

    q.append(
        B2Question(
            id="scene_or_backstage",
            block="process",
            text="Если представить большую сцену / мероприятие, где вы себя видите комфортнее всего?",
            options=[
                B2AnswerOption(
                    text="На сцене: ведущим, спикером, артистом, чтобы на меня смотрели.",
                    score_changes={"Гранат": 1.0, "Гелиодор": 0.5},
                ),
                B2AnswerOption(
                    text="За кулисами: режиссирую, продюсирую, всё соединяю, но не в центре внимания.",
                    score_changes={"Рубин": 0.6, "Аметист": 0.4, "Цитрин": 0.4},
                ),
                B2AnswerOption(
                    text="В визуале: отвечаю за свет, картинку, декор, стиль, красоту кадра.",
                    score_changes={"Изумруд": 1.0},
                ),
                B2AnswerOption(
                    text="В системе: расписание, тайминг, кто за что отвечает, чтобы всё работало.",
                    score_changes={"Янтарь": 1.0},
                ),
                B2AnswerOption(
                    text="В работе с людьми: встречаю гостей, поддерживаю участников, создаю атмосферу.",
                    score_changes={"Гелиодор": 0.8, "Сапфир": 0.3},
                ),
            ],
            allow_multiple=True,
            allow_text=True,
        )
    )

    q.append(
        B2Question(
            id="body_and_sport",
            block="process",
            text="Что вам ближе всего из «телесной» активности?",
            options=[
                B2AnswerOption(
                    text="Я люблю ощущение усталости после тренировки / активного дня — тело «приятно гудит».",
                    score_changes={"Шунгит": 1.0},
                ),
                B2AnswerOption(
                    text="Я больше люблю спокойные телесные практики: йога, растяжка, плавание.",
                    score_changes={"Изумруд": 0.6, "Шунгит": 0.4},
                ),
                B2AnswerOption(
                    text="Мне нравится драйв, скорость, приключения: горы, серф, экстрим.",
                    score_changes={"Рубин": 1.0, "Шунгит": 0.3},
                ),
                B2AnswerOption(
                    text="У меня с телом сложные отношения, спорт — через «надо», а не через «хочу».",
                    score_changes={"Шунгит": -0.5},
                ),
            ],
            allow_multiple=True,
            allow_text=True,
        )
    )

    q.append(
        B2Question(
            id="money_and_value",
            block="process",
            text="Как вы относитесь к деньгам и теме монетизации своих талантов?",
            options=[
                B2AnswerOption(
                    text="Мне реально интересно, как из идеи сделать прибыльный продукт, люблю считать.",
                    score_changes={"Цитрин": 1.0},
                ),
                B2AnswerOption(
                    text="Деньги важны, но я больше про смысл/служение, а не про цифры.",
                    score_changes={"Сапфир": 0.7, "Аметист": 0.4},
                ),
                B2AnswerOption(
                    text="Я знаю, что могу больше зарабатывать, но мне сложно просить / назначать цену.",
                    score_changes={"Цитрин": -0.5},
                ),
                B2AnswerOption(
                    text="Мне ок, если деньги зарабатывает кто-то рядом, а я делаю любимое дело.",
                ),
            ],
            allow_multiple=True,
            allow_text=True,
        )
    )

    return q


def b2_init_state():
    if "b2_questions" not in st.session_state:
        st.session_state["b2_questions"] = build_block2_questions()
    if "b2_index" not in st.session_state:
        st.session_state["b2_index"] = 0
    if "b2_scores" not in st.session_state:
        st.session_state["b2_scores"] = {}
    if "b2_answers" not in st.session_state:
        st.session_state["b2_answers"] = {}
    if "b2_open_text_log" not in st.session_state:
        st.session_state["b2_open_text_log"] = []
    if "b2_finished" not in st.session_state:
        st.session_state["b2_finished"] = False


def b2_apply_answer(question: B2Question, selected_indices: List[int], free_text: str):
    labels = [question.options[i].text for i in selected_indices]
    st.session_state["b2_answers"][question.id] = {
        "question": question.text,
        "selected": labels,
        "free_text": free_text.strip() if free_text else "",
        "block": question.block,
    }

    if free_text and free_text.strip():
        entry = [
            f"--- ВОПРОС [{question.id}] ---",
            question.text,
            "",
            "Ответ (свободный текст):",
            free_text.strip(),
            "",
        ]
        st.session_state["b2_open_text_log"].append("\n".join(entry))

    for idx in selected_indices:
        opt = question.options[idx]
        for pot, delta in opt.score_changes.items():
            if pot not in st.session_state["b2_scores"]:
                st.session_state["b2_scores"][pot] = 0.0
            st.session_state["b2_scores"][pot] += delta


def b2_render_question():
    q_list: List[B2Question] = st.session_state["b2_questions"]
    idx = st.session_state["b2_index"]
    question = q_list[idx]

    if question.block == "shifts":
        st.markdown("#### Часть 0. Состояние и смещения")
    elif question.block == "work":
        st.markdown("#### Часть 1. Профессия и ролевая картинка")
    elif question.block == "process":
        st.markdown("#### Часть 2. Процессы, где вы живые / уставшие")

    st.markdown(f"### {question.text}")

    labels = [opt.text for opt in question.options]

    if question.allow_multiple:
        selected_labels = st.multiselect(
            "Выберите один или несколько вариантов (если откликается):",
            options=labels,
            key=f"b2_multi_{question.id}",
        )
        selected_indices = [labels.index(lbl) for lbl in selected_labels]
    else:
        selected_label = st.radio(
            "Выберите один вариант:",
            options=["— не выбрано —"] + labels,
            key=f"b2_radio_{question.id}",
        )
        if selected_label == "— не выбрано —":
            selected_indices = []
        else:
            selected_indices = [labels.index(selected_label)]

    free_text = ""
    if question.allow_text:
        free_text = st.text_area(
            "Если хотите, опишите живыми словами реальные ситуации / примеры:",
            key=f"b2_text_{question.id}",
            placeholder="Например: «в работе я часто…»",
        )

    disabled = len(selected_indices) == 0 and not (free_text and free_text.strip())
    if st.button("Дальше", key=f"b2_next_{question.id}", disabled=disabled):
        b2_apply_answer(question, selected_indices, free_text)
        st.session_state["b2_index"] += 1
        if st.session_state["b2_index"] >= len(st.session_state["b2_questions"]):
            st.session_state["b2_finished"] = True


# ================================================================
# БЛОК 3 — ЭМОЦИИ, СТОЛБЦЫ, ВОСПРИЯТИЕ–ПРОЦЕСС–РЕЗУЛЬТАТ
# ================================================================

@dataclass
class B3AnswerOption:
    key: str
    text: str
    potentials: List[str]


@dataclass
class B3Question:
    id: str
    group: str  # "c1", "c2", "c3"
    title: str
    text: str
    options: List[B3AnswerOption]
    multi: bool = True
    allow_free_text: bool = True


COLUMN_KEYS = {
    "c1": "Столбец 1 — Восприятие (ВАУ / впечатление)",
    "c2": "Столбец 2 — Процесс (интерес / притяжение)",
    "c3": "Столбец 3 — Результат (победа / триумф)",
}


def b3_init_state():
    if "b3_questions" not in st.session_state:
        st.session_state["b3_questions"] = build_block3_questions()
    if "b3_current_idx" not in st.session_state:
        st.session_state["b3_current_idx"] = 0
    if "b3_scores" not in st.session_state:
        st.session_state["b3_scores"] = {
            pot: {"c1": 0.0, "c2": 0.0, "c3": 0.0} for pot in POTENTIALS
        }
    if "b3_text_log" not in st.session_state:
        st.session_state["b3_text_log"] = []
    if "b3_answers" not in st.session_state:
        st.session_state["b3_answers"] = {}
    if "b3_finished" not in st.session_state:
        st.session_state["b3_finished"] = False


def build_block3_questions() -> List[B3Question]:
    q: List[B3Question] = []

    # C1 — Восприятие
    q.append(
        B3Question(
            id="c1_q1",
            group="c1",
            title="Блок восприятия — ВАУ",
            text="1. Что в людях чаще всего вызывает у вас мгновенный «вау»?",
            options=[
                B3AnswerOption(
                    key="voice",
                    text="Красивый голос, подача, манера говорить",
                    potentials=["Гелиодор", "Гранат"],
                ),
                B3AnswerOption(
                    key="deep_mind",
                    text="Глубокие мысли, нестандартное мышление, мудрость",
                    potentials=["Аметист", "Сапфир"],
                ),
                B3AnswerOption(
                    key="beauty",
                    text="Красота, стиль, эстетика, как человек выглядит / оформляет",
                    potentials=["Изумруд", "Гранат"],
                ),
                B3AnswerOption(
                    key="system",
                    text="Продуманность, система, логика, когда всё разложено по полочкам",
                    potentials=["Янтарь", "Цитрин"],
                ),
                B3AnswerOption(
                    key="risk",
                    text="Смелость, риск, авантюры, готовность делать шаг в неизвестность",
                    potentials=["Рубин", "Шунгит"],
                ),
                B3AnswerOption(
                    key="body",
                    text="Физическая сила, выносливость, владение телом",
                    potentials=["Шунгит"],
                ),
                B3AnswerOption(
                    key="stage",
                    text="Умение зажечь зал, харизма, эмоции на сцене",
                    potentials=["Гранат", "Гелиодор"],
                ),
                B3AnswerOption(
                    key="other",
                    text="Другое (опишу ниже в тексте)",
                    potentials=[],
                ),
            ],
        )
    )

    q.append(
        B3Question(
            id="c1_q2",
            group="c1",
            title="Блок восприятия — картинка мира",
            text="2. Какие сцены / виды / картинки вас завораживают больше всего?",
            options=[
                B3AnswerOption(
                    key="patterns",
                    text="Сложные узоры, свет, тени, отражения, красиво выстроенный кадр",
                    potentials=["Изумруд"],
                ),
                B3AnswerOption(
                    key="nature",
                    text="Природа, вода, горы, закаты, океан",
                    potentials=["Изумруд", "Сапфир"],
                ),
                B3AnswerOption(
                    key="city",
                    text="Город, архитектура, геометрия улиц, линии, мосты",
                    potentials=["Янтарь", "Цитрин"],
                ),
                B3AnswerOption(
                    key="faces",
                    text="Лица людей, эмоции, реакция, мимика",
                    potentials=["Гранат", "Гелиодор"],
                ),
                B3AnswerOption(
                    key="motion",
                    text="Движение: спорт, танец, тела в динамике",
                    potentials=["Шунгит", "Рубин"],
                ),
                B3AnswerOption(
                    key="other",
                    text="Другое (опишу ниже в тексте)",
                    potentials=[],
                ),
            ],
        )
    )

    q.append(
        B3Question(
            id="c1_q3",
            group="c1",
            title="Блок восприятия — сюжеты",
            text="3. Что вас больше всего цепляет в фильмах, книгах и историях?",
            options=[
                B3AnswerOption(
                    key="sense",
                    text="Сильные смыслы, философия, неожиданные выводы",
                    potentials=["Аметист", "Сапфир"],
                ),
                B3AnswerOption(
                    key="feel",
                    text="Глубокие эмоции, переживания героев, драмы",
                    potentials=["Гранат", "Гелиодор"],
                ),
                B3AnswerOption(
                    key="visual",
                    text="Красота мира, картинка, эстетика, костюмы, декор",
                    potentials=["Изумруд"],
                ),
                B3AnswerOption(
                    key="strategy",
                    text="Стратегия, как всё продумано, как герой строит план",
                    potentials=["Янтарь", "Цитрин"],
                ),
                B3AnswerOption(
                    key="hero",
                    text="Преодоление, сила, риск, борьба, экшн",
                    potentials=["Рубин", "Шунгит"],
                ),
                B3AnswerOption(
                    key="other",
                    text="Другое (опишу ниже в тексте)",
                    potentials=[],
                ),
            ],
        )
    )

    q.append(
        B3Question(
            id="c1_q4",
            group="c1",
            title="Блок восприятия — магия и чудо",
            text="4. Что для вас в этом мире выглядит наиболее «магичным» или необъяснимым?",
            options=[
                B3AnswerOption(
                    key="people_mind",
                    text="Как работает человеческий мозг и подсознание",
                    potentials=["Аметист", "Сапфир"],
                ),
                B3AnswerOption(
                    key="voice_music",
                    text="Музыка, голос, влияние звука на людей",
                    potentials=["Гелиодор"],
                ),
                B3AnswerOption(
                    key="nature_magic",
                    text="Природа, циклы, красота мира",
                    potentials=["Изумруд", "Сапфир"],
                ),
                B3AnswerOption(
                    key="order_magic",
                    text="Как системы и структуры могут менять реальность",
                    potentials=["Янтарь", "Цитрин"],
                ),
                B3AnswerOption(
                    key="risk_magic",
                    text="Как одно смелое действие меняет всю жизнь",
                    potentials=["Рубин"],
                ),
                B3AnswerOption(
                    key="body_magic",
                    text="Как тело запоминает и лечит, как работает сила / выносливость",
                    potentials=["Шунгит"],
                ),
                B3AnswerOption(
                    key="other",
                    text="Другое (опишу ниже в тексте)",
                    potentials=[],
                ),
            ],
        )
    )

    q.append(
        B3Question(
            id="c1_q5",
            group="c1",
            title="Блок восприятия — страх / ступор",
            text="5. Что сильнее всего вызывает у вас внутренний страх, ступор или избегание?",
            options=[
                B3AnswerOption(
                    key="public",
                    text="Сцена, публичные выступления, внимание большого количества людей",
                    potentials=["Гранат", "Гелиодор"],
                ),
                B3AnswerOption(
                    key="chaos",
                    text="Полный хаос, отсутствие системы, когда ничего не понятно",
                    potentials=["Янтарь"],
                ),
                B3AnswerOption(
                    key="no_sense",
                    text="Бессмысленность, когда нет логики и смысла в том, что происходит",
                    potentials=["Аметист", "Сапфир"],
                ),
                B3AnswerOption(
                    key="money_resp",
                    text="Большие деньги и ответственность за чужие ресурсы",
                    potentials=["Цитрин"],
                ),
                B3AnswerOption(
                    key="hard_body",
                    text="Очень сильная физическая нагрузка, боль, риск травмы",
                    potentials=["Шунгит", "Рубин"],
                ),
                B3AnswerOption(
                    key="emotional_conflict",
                    text="Громкие конфликты, сильные эмоции людей",
                    potentials=["Гранат"],
                ),
                B3AnswerOption(
                    key="other",
                    text="Другое (опишу ниже в тексте)",
                    potentials=[],
                ),
            ],
        )
    )

    # C2 — ПРОЦЕСС

    q.append(
        B3Question(
            id="c2_q1",
            group="c2",
            title="Блок процесса — чем вы живёте",
            text="6. Какие занятия так втягивают вас, что вы забываете про время?",
            options=[
                B3AnswerOption(
                    key="talk",
                    text="Разговоры, сторис, общение, обсуждения, диалоги",
                    potentials=["Гелиодор", "Гранат"],
                ),
                B3AnswerOption(
                    key="thinking",
                    text="Размышления, чтение, поиск смыслов, саморазборы",
                    potentials=["Аметист", "Сапфир"],
                ),
                B3AnswerOption(
                    key="visual_creative",
                    text="Фото, видео, дизайн, визуал, придумывание красивых образов",
                    potentials=["Изумруд"],
                ),
                B3AnswerOption(
                    key="system_work",
                    text="Раскладывать по полочкам, упорядочивать, строить схемы",
                    potentials=["Янтарь", "Цитрин"],
                ),
                B3AnswerOption(
                    key="body_work",
                    text="Тренировки, движение, спорт, активность телом",
                    potentials=["Шунгит", "Рубин"],
                ),
                B3AnswerOption(
                    key="cook_care",
                    text="Готовить что-то вкусное и радовать других",
                    potentials=["Гелиодор", "Янтарь"],
                ),
                B3AnswerOption(
                    key="other",
                    text="Другое (опишу ниже в тексте)",
                    potentials=[],
                ),
            ],
        )
    )

    q.append(
        B3Question(
            id="c2_q2",
            group="c2",
            title="Блок процесса — для души",
            text="7. Что вы делаете «для души», даже если никто не видит и не оценивает?",
            options=[
                B3AnswerOption(
                    key="write",
                    text="Писать тексты, заметки, дневник, посты в стол",
                    potentials=["Аметист", "Сапфир"],
                ),
                B3AnswerOption(
                    key="visual_order",
                    text="Наводить красоту: перестановка, декор, свечи, детали",
                    potentials=["Изумруд"],
                ),
                B3AnswerOption(
                    key="clean_order",
                    text="Убираться, раскладывать, организовывать пространство",
                    potentials=["Янтарь"],
                ),
                B3AnswerOption(
                    key="ideas",
                    text="Придумывать концепции, проекты, сценарии, форматы",
                    potentials=["Рубин", "Гранат"],
                ),
                B3AnswerOption(
                    key="body_self",
                    text="Растяжка, массаж, уход за телом, осознанные практики",
                    potentials=["Шунгит"],
                ),
                B3AnswerOption(
                    key="listen_watch",
                    text="Слушать музыку / подкасты, смотреть вдохновляющие видео",
                    potentials=["Гелиодор", "Сапфир"],
                ),
                B3AnswerOption(
                    key="other",
                    text="Другое (опишу ниже в тексте)",
                    potentials=[],
                ),
            ],
        )
    )

    q.append(
        B3Question(
            id="c2_q3",
            group="c2",
            title="Блок процесса — выбор задач",
            text="8. Какой тип задач вы чаще всего выбираете добровольно (если есть выбор)?",
            options=[
                B3AnswerOption(
                    key="analyse",
                    text="Разобраться, проанализировать, найти закономерности",
                    potentials=["Аметист", "Янтарь"],
                ),
                B3AnswerOption(
                    key="communicate",
                    text="Поговорить с людьми, созвониться, провести встречу",
                    potentials=["Гелиодор", "Гранат"],
                ),
                B3AnswerOption(
                    key="make_beauty",
                    text="Сделать красиво: презентация, визуал, оформление",
                    potentials=["Изумруд"],
                ),
                B3AnswerOption(
                    key="structure",
                    text="Сделать таблицу, план, схему, чек-лист",
                    potentials=["Янтарь", "Цитрин"],
                ),
                B3AnswerOption(
                    key="move",
                    text="Сделать что-то физически: съездить, привезти, собрать",
                    potentials=["Шунгит", "Рубин"],
                ),
                B3AnswerOption(
                    key="other",
                    text="Другое (опишу ниже в тексте)",
                    potentials=[],
                ),
            ],
        )
    )

    q.append(
        B3Question(
            id="c2_q4",
            group="c2",
            title="Блок процесса — свободный день",
            text="9. Если у вас свободный день без обязательств, что вы с большей вероятностью будете делать?",
            options=[
                B3AnswerOption(
                    key="study_self",
                    text="Учиться чему-то новому, разбираться в себе, читать",
                    potentials=["Аметист", "Сапфир"],
                ),
                B3AnswerOption(
                    key="content",
                    text="Снимать/монтировать контент, сторис, фотосессия",
                    potentials=["Изумруд", "Гранат", "Гелиодор"],
                ),
                B3AnswerOption(
                    key="space",
                    text="Привести в порядок дом / вещи / документы",
                    potentials=["Янтарь"],
                ),
                B3AnswerOption(
                    key="sport_walk",
                    text="Спорт, горы, прогулка, активность",
                    potentials=["Шунгит", "Рубин"],
                ),
                B3AnswerOption(
                    key="money_projects",
                    text="Посчитать / продумать новые проекты, где заработать",
                    potentials=["Цитрин"],
                ),
                B3AnswerOption(
                    key="friends",
                    text="Встретиться с людьми, устроить тёплый круг, общение",
                    potentials=["Гелиодор", "Гранат"],
                ),
                B3AnswerOption(
                    key="other",
                    text="Другое (опишу ниже в тексте)",
                    potentials=[],
                ),
            ],
        )
    )

    q.append(
        B3Question(
            id="c2_q5",
            group="c2",
            title="Блок процесса — любимые моменты в работе",
            text="10. Какие процессы в вашей текущей или прошлой работе вам нравятся сами по себе, без привязки к результату?",
            options=[
                B3AnswerOption(
                    key="explain",
                    text="Объяснять, обучать, доносить сложное простым языком",
                    potentials=["Аметист", "Сапфир", "Гелиодор"],
                ),
                B3AnswerOption(
                    key="design",
                    text="Придумывать формат, упаковку, визуальную концепцию",
                    potentials=["Изумруд", "Гранат"],
                ),
                B3AnswerOption(
                    key="coordination",
                    text="Координировать, сводить людей, быть связующим звеном",
                    potentials=["Гелиодор", "Рубин"],
                ),
                B3AnswerOption(
                    key="backoffice",
                    text="Бэк-офис: документы, регламенты, систематизация",
                    potentials=["Янтарь", "Цитрин"],
                ),
                B3AnswerOption(
                    key="field",
                    text="Полевые задачи: выезды, площадки, логистика, движение",
                    potentials=["Шунгит", "Рубин"],
                ),
                B3AnswerOption(
                    key="other",
                    text="Другое (опишу ниже в тексте)",
                    potentials=[],
                ),
            ],
        )
    )

    # C3 — РЕЗУЛЬТАТ

    q.append(
        B3Question(
            id="c3_q1",
            group="c3",
            title="Блок результата — гордость",
            text="11. Какими своими результатами вы по-настоящему гордитесь?",
            options=[
                B3AnswerOption(
                    key="sport",
                    text="Достижения в спорте, теле, выносливости",
                    potentials=["Шунгит", "Рубин"],
                ),
                B3AnswerOption(
                    key="stage",
                    text="Выступления, сцена, публичные форматы",
                    potentials=["Гранат", "Гелиодор"],
                ),
                B3AnswerOption(
                    key="money",
                    text="Деньги, обороты, сделки, финансовые результаты",
                    potentials=["Цитрин"],
                ),
                B3AnswerOption(
                    key="product",
                    text="Созданный продукт / система, которая работает без меня",
                    potentials=["Янтарь"],
                ),
                B3AnswerOption(
                    key="beauty_res",
                    text="Очень красивый результат: проект, комната, визуал",
                    potentials=["Изумруд"],
                ),
                B3AnswerOption(
                    key="insight_res",
                    text="Сильные инсайты, повороты в жизни, решения, которые всё поменяли",
                    potentials=["Аметист", "Сапфир"],
                ),
                B3AnswerOption(
                    key="impact_people",
                    text="Истории людей: «ты поменял(а) мою жизнь, мышление, путь»",
                    potentials=["Аметист", "Сапфир", "Гелиодор"],
                ),
                B3AnswerOption(
                    key="other",
                    text="Другое (опишу ниже в тексте)",
                    potentials=[],
                ),
            ],
        )
    )

    q.append(
        B3Question(
            id="c3_q2",
            group="c3",
            title="Блок результата — сладкий момент",
            text="12. Какой момент в любом проекте для вас самый сладкий, самый кайфовый?",
            options=[
                B3AnswerOption(
                    key="launch",
                    text="Когда проект выходит в мир, публикация / запуск",
                    potentials=["Гранат", "Гелиодор", "Изумруд"],
                ),
                B3AnswerOption(
                    key="feedback",
                    text="Когда приходят отклики, отзывы, эмоции людей",
                    potentials=["Гранат", "Гелиодор", "Сапфир"],
                ),
                B3AnswerOption(
                    key="numbers",
                    text="Когда вижу цифры, деньги, конкретный результат в цифрах",
                    potentials=["Цитрин"],
                ),
                B3AnswerOption(
                    key="system_done",
                    text="Когда всё собрано в систему и работает как механизм",
                    potentials=["Янтарь"],
                ),
                B3AnswerOption(
                    key="inner_shift",
                    text="Когда понимаю, что внутри я стал(а) другим человеком",
                    potentials=["Аметист", "Сапфир"],
                ),
                B3AnswerOption(
                    key="body_state",
                    text="Когда тело приятно устало, но чувствуется сила и тонус",
                    potentials=["Шунгит", "Рубин"],
                ),
                B3AnswerOption(
                    key="other",
                    text="Другое (опишу ниже в тексте)",
                    potentials=[],
                ),
            ],
        )
    )

    q.append(
        B3Question(
            id="c3_q3",
            group="c3",
            title="Блок результата — «я молодец»",
            text="13. Что вам чаще всего даёт ощущение «я реально молодец»?",
            options=[
                B3AnswerOption(
                    key="complex_task",
                    text="Решил(а) сложную задачу, в которой другие застряли",
                    potentials=["Аметист", "Янтарь"],
                ),
                B3AnswerOption(
                    key="people_saved",
                    text="Кому-то реально помог(ла), поддержал(а), вытащил(а)",
                    potentials=["Сапфир", "Гелиодор"],
                ),
                B3AnswerOption(
                    key="beauty_done",
                    text="Сделал(а) очень красиво, аккуратно, эстетично",
                    potentials=["Изумруд", "Янтарь"],
                ),
                B3AnswerOption(
                    key="deal_closed",
                    text="Закрыл(а) важную сделку / проект, принёс(ла) деньги",
                    potentials=["Цитрин"],
                ),
                B3AnswerOption(
                    key="win",
                    text="Выиграл(а) соревнование, тендер, конкурс, спор",
                    potentials=["Рубин", "Шунгит"],
                ),
                B3AnswerOption(
                    key="other",
                    text="Другое (опишу ниже в тексте)",
                    potentials=[],
                ),
            ],
        )
    )

    q.append(
        B3Question(
            id="c3_q4",
            group="c3",
            title="Блок результата — боевой режим",
            text="14. В каких ситуациях вы чаще всего включаете «боевой режим» и готовы бороться до конца?",
            options=[
                B3AnswerOption(
                    key="protect_people",
                    text="Если нужно защитить близких людей или команду",
                    potentials=["Сапфир", "Шунгит", "Рубин"],
                ),
                B3AnswerOption(
                    key="justice",
                    text="Когда вижу несправедливость, хочется доказать правду",
                    potentials=["Аметист", "Рубин"],
                ),
                B3AnswerOption(
                    key="deal_fight",
                    text="В переговорах, торге, где важно отстоять цену / условия",
                    potentials=["Цитрин", "Рубин"],
                ),
                B3AnswerOption(
                    key="deadline",
                    text="Перед жёстким дедлайном, когда надо срочно додавить проект",
                    potentials=["Янтарь", "Аметист"],
                ),
                B3AnswerOption(
                    key="stage_drive",
                    text="Перед выходом на сцену / в эфир, хочется выдать максимум",
                    potentials=["Гранат", "Гелиодор"],
                ),
                B3AnswerOption(
                    key="other",
                    text="Другое (опишу ниже в тексте)",
                    potentials=[],
                ),
            ],
        )
    )

    q.append(
        B3Question(
            id="c3_q5",
            group="c3",
            title="Блок результата — один тип результата",
            text="15. Если бы вам сказали: «оставь в жизни только один тип результатов, всё остальное уберём», что бы вы оставили?",
            options=[
                B3AnswerOption(
                    key="inner",
                    text="Внутренние изменения: рост, осознанность, глубина",
                    potentials=["Аметист", "Сапфир"],
                ),
                B3AnswerOption(
                    key="money_status",
                    text="Финансовые результаты и статус",
                    potentials=["Цитрин"],
                ),
                B3AnswerOption(
                    key="beauty_world",
                    text="Красивые пространства, визуал, то как выглядит мир вокруг",
                    potentials=["Изумруд"],
                ),
                B3AnswerOption(
                    key="impact",
                    text="Влияние на людей, их истории и судьбы",
                    potentials=["Гелиодор", "Сапфир", "Гранат"],
                ),
                B3AnswerOption(
                    key="body_victory",
                    text="Тело, сила, выносливость, спортивные победы",
                    potentials=["Шунгит", "Рубин"],
                ),
                B3AnswerOption(
                    key="systems_legacy",
                    text="Системы и продукты, которые работают без меня",
                    potentials=["Янтарь"],
                ),
            ],
            multi=False,
        )
    )

    return q


def b3_process_question():
    questions: List[B3Question] = st.session_state["b3_questions"]
    idx = st.session_state["b3_current_idx"]
    question = questions[idx]

    st.subheader(question.title)
    st.markdown(question.text)

    option_texts = [opt.text for opt in question.options]

    select_key = f"b3_select_{question.id}"

    if question.multi:
        selected_texts = st.multiselect(
            "Выберите один или несколько вариантов:",
            options=option_texts,
            key=select_key,
        )
    else:
        selected = st.radio(
            "Выберите один вариант:",
            options=option_texts,
            key=select_key,
        )
        selected_texts = [selected] if selected else []

    free_text_value = ""
    if question.allow_free_text:
        free_text_value = st.text_area(
            "Если хотите, опишите своими словами, что для вас здесь важно:",
            key=f"b3_text_{question.id}",
            height=100,
        )

    if st.button("Дальше ➜", key=f"b3_next_{question.id}"):
        if not selected_texts and not free_text_value.strip():
            st.warning("Пожалуйста, выберите хотя бы один вариант или напишите пару слов.")
            st.stop()

        st.session_state["b3_answers"][question.id] = {
            "selected": selected_texts,
            "free_text": free_text_value.strip(),
        }

        group = question.group
        txt_log_chunks = [f"Вопрос {question.id}: {question.text}"]

        for sel_text in selected_texts:
            try:
                idx_opt = option_texts.index(sel_text)
            except ValueError:
                continue
            opt = question.options[idx_opt]
            txt_log_chunks.append(f"- Выбрано: {opt.text}")
            for pot in opt.potentials:
                if pot in st.session_state["b3_scores"]:
                    st.session_state["b3_scores"][pot][group] += 1.0

        if free_text_value.strip():
            txt_log_chunks.append(f"Текст: {free_text_value.strip()}")

        st.session_state["b3_text_log"].append("\n".join(txt_log_chunks) + "\n")

        st.session_state["b3_current_idx"] += 1
        if st.session_state["b3_current_idx"] >= len(questions):
            st.session_state["b3_finished"] = True


# ================================================================
# РЕЖИМ КЛИЕНТА
# ================================================================

def client_init_state():
    if "client_stage" not in st.session_state:
        st.session_state["client_stage"] = "intro"
    if "client_name" not in st.session_state:
        st.session_state["client_name"] = ""


def render_client_mode():
    client_init_state()

    stage = st.session_state["client_stage"]

    if stage == "intro":
        st.header("Deep Identity · Диагностика потенциалов")
        st.write(
            "Это мягкая диагностика, которая помогает увидеть твои природные потенциалы "
            "и то, как они проявляются в жизни. Отвечай честно, не как «надо»."
        )
        name = st.text_input("Как тебя зовут? (имя, которое будет в отчёте)", key="client_name_input")
        st.session_state["client_name"] = name.strip()

        if st.button("Начать Блок 1"):
            if not st.session_state["client_name"]:
                st.warning("Напиши хотя бы имя 🙏")
            else:
                st.session_state["client_stage"] = "b1"

    elif stage == "b1":
        b1_init_state()
        st.header("Блок 1 — Детство и естественные склонности")

        if not st.session_state["b1_finished"]:
            if st.session_state["b1_current_qid"] is not None:
                b1_render_question()
        else:
            st.success("Блок 1 завершён ✅")
            st.write(
                "Спасибо, что вспомнил(а) своё детство. "
                "Дальше посмотрим на текущую профессию, процессы и смещения."
            )
            if st.button("Перейти к Блоку 2"):
                st.session_state["client_stage"] = "b2"

    elif stage == "b2":
        b2_init_state()
        st.header("Блок 2 — Профессии, процессы, смещения")

        if not st.session_state["b2_finished"]:
            b2_render_question()
        else:
            st.success("Блок 2 завершён ✅")
            st.write(
                "Теперь посмотрим, как ты реагируешь на людей, эмоции, результаты.\n"
                "Это поможет выстроить три столбца: восприятие, процесс и результат."
            )
            if st.button("Перейти к Блоку 3"):
                st.session_state["client_stage"] = "b3"

    elif stage == "b3":
        b3_init_state()
        st.header("Блок 3 — Эмоции, восприятие, процесс, результат")

        questions: List[B3Question] = st.session_state["b3_questions"]
        total_q = len(questions)
        idx = st.session_state["b3_current_idx"]

        if not st.session_state["b3_finished"] and idx < total_q:
            st.markdown(
                f"**Вопрос {idx + 1} из {total_q}** · "
                f"{COLUMN_KEYS[questions[idx].group]}"
            )
            st.progress((idx + 1) / total_q)
            b3_process_question()
        else:
            st.success("Блок 3 завершён ✅")
            st.write(
                "Спасибо, диагностика пройдена. "
                "Ответы и паттерны эмоций будут собраны в единый отчёт, "
                "который ты получишь от мастера."
            )
            if st.button("Завершить диагностику и отправить мастеру"):
                st.session_state["client_stage"] = "final"

    elif stage == "final":
        # Сохранить результат (один раз)
        if not st.session_state.get("result_saved", False):
            all_results = load_results()
            now = datetime.datetime.now().isoformat(timespec="seconds")

            record = {
                "timestamp": now,
                "client_name": st.session_state.get("client_name"),
                "block1": {
                    "scores": st.session_state.get("b1_scores", {}),
                    "answers": st.session_state.get("b1_answers", {}),
                },
                "block2": {
                    "scores": st.session_state.get("b2_scores", {}),
                    "answers": st.session_state.get("b2_answers", {}),
                    "open_text_log": st.session_state.get("b2_open_text_log", []),
                },
                "block3": {
                    "scores": st.session_state.get("b3_scores", {}),
                    "answers": st.session_state.get("b3_answers", {}),
                    "text_log": st.session_state.get("b3_text_log", []),
                },
            }

            all_results.append(record)
            save_results(all_results)
            st.session_state["result_saved"] = True

        st.header("Диагностика отправлена ✅")
        st.write(
            "Спасибо за доверие.\n\n"
            "Мастер Deep Identity аккуратно разберёт твои ответы и вернётся с "
            "персональным отчётом по потенциалам и направлениям реализации."
        )
        st.write("Можно закрывать страницу ❤️")


# ================================================================
# РЕЖИМ МАСТЕРА
# ================================================================

def render_master_mode():
    st.header("Deep Identity · Мастер-панель Асели")

    pwd = st.text_input("Пароль мастера", type="password")
    if pwd != MASTER_PASSWORD:
        st.info("Введи мастер-пароль, чтобы увидеть ответы клиентов.")
        return

    results = load_results()
    if not results:
        st.warning(
            "Пока нет ни одной записи в deep_identity_results.json.\n\n"
            "Проверь, что клиенты проходят диагностику и что файл находится в том же репозитории."
        )
        return

    # список сессий
    options = []
    for i, r in enumerate(results):
        ts = r.get("timestamp", "?")
        name = r.get("client_name") or "Без имени"
        label = f"{i+1}. {name} · {ts}"
        options.append(label)

    selected_label = st.selectbox("Выбери сессию клиента:", options)
    idx = options.index(selected_label)
    rec = results[idx]

    st.subheader(f"Клиент: {rec.get('client_name') or 'Без имени'}")
    st.caption(f"Дата/время: {rec.get('timestamp')}")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Блок 1 — Детство (сырые баллы)**")
        b1_scores = rec.get("block1", {}).get("scores", {})
        if b1_scores:
            for pot, val in b1_scores.items():
                st.write(f"- {pot}: {val}")
        else:
            st.write("_Нет данных_")

    with col2:
        st.markdown("**Блок 2 — Профессии/процессы (сырые баллы)**")
        b2_scores = rec.get("block2", {}).get("scores", {})
        if b2_scores:
            for pot, val in b2_scores.items():
                if pot in POTENTIALS:
                    st.write(f"- {pot}: {round(val, 2)}")
        else:
            st.write("_Нет данных_")

    st.markdown("**Блок 3 — Столбцы (Восприятие / Процесс / Результат)**")
    b3_scores = rec.get("block3", {}).get("scores", {})
    if b3_scores:
        rows = []
        for pot, cols in b3_scores.items():
            rows.append(
                {
                    "Потенциал": pot,
                    "ВАУ": cols.get("c1", 0),
                    "Процесс": cols.get("c2", 0),
                    "Результат": cols.get("c3", 0),
                    "Итого": cols.get("c1", 0) + cols.get("c2", 0) + cols.get("c3", 0),
                }
            )
        # простая таблица
        st.table(rows)
    else:
        st.write("_Нет данных по Блоку 3_")

    with st.expander("Текстовые ответы Блока 2"):
        log2 = rec.get("block2", {}).get("open_text_log", [])
        st.text("\n\n".join(log2) if log2 else "_Нет текстовых ответов_")

    with st.expander("Текстовые ответы Блока 3"):
        log3 = rec.get("block3", {}).get("text_log", [])
        st.text("\n\n".join(log3) if log3 else "_Нет текстовых ответов_")

    st.markdown("---")
    st.subheader("Черновой отчёт для мастера")

    client = get_openai_client()
    if client is None:
        st.error(
            "OpenAI-клиент не инициализирован.\n\n"
            "Проверь, что OPENAI_API_KEY прописан в secrets или переменных окружения."
        )
        return

    if st.button("✨ Сгенерировать черновой отчёт на основе ответов клиента"):
        with st.spinner("Собираю картинку по потенциалам..."):
            # Подготовка контекста для ИИ
            b1_s = rec.get("block1", {}).get("scores", {})
            b2_s = rec.get("block2", {}).get("scores", {})
            b3_s = rec.get("block3", {}).get("scores", {})
            text2 = "\n\n".join(rec.get("block2", {}).get("open_text_log", []))
            text3 = "\n\n".join(rec.get("block3", {}).get("text_log", []))

            potentials_text = "\n".join(
                f"- {p}: {POTENTIALS_DESC.get(p, '')}" for p in POTENTIALS
            )

            prompt = f"""
Ты — аналитик Deep Identity и помогаешь мастеру Аселе составить **черновой отчёт по потенциалам клиента**.

У тебя есть:
1) Список базовых потенциалов и их смыслы:
{potentials_text}

2) Сырые баллы по Блоку 1 (детство):
{json.dumps(b1_s, ensure_ascii=False, indent=2)}

3) Сырые баллы по Блоку 2 (профессии, процессы, смещения):
{json.dumps(b2_s, ensure_ascii=False, indent=2)}

4) Сырые баллы по Блоку 3 (столбцы восприятие/процесс/результат):
{json.dumps(b3_s, ensure_ascii=False, indent=2)}

5) Текстовые ответы клиента (Блок 2):
{text2}

6) Текстовые ответы клиента (Блок 3):
{text3}

Задача: сформировать **структурированный черновой отчёт для мастера**, на русском языке, максимум на 2–3 страницы.

Структура ответа:
1. Кратко: кто перед нами (1–2 абзаца гипотезы по типу личности, мотивации, ключевым темам).
2. Топ-потенциалы по РЯДАМ:
   - 1-й ряд (ядро реализации, куда тянет максимально)
   - 2-й ряд (поддерживающие/обслуживающие потенциалы)
   - 3-й ряд (то, что лучше делегировать или дозировать)
   *Не называй конкретные ряды цифрами клиента, это черновая гипотеза для мастера.*
3. Топ-потенциалы по СТОЛБЦАМ:
   - Восприятие (что восхищает, какие типы людей/сцен/результатов цепляют)
   - Процесс (в каких процессах человек живой, а в каких выгорает)
   - Результат (какие типы побед/результатов дают ощущение «я молодец»)
4. Возможные смещения и ловушки:
   - где человек может себя обесценивать
   - где может путать чужие ожидания и свои истинные желания
   - возможные перекосы по телу, деньгам, сцене, порядку.
5. Рекомендации мастеру по работе:
   - какие темы стоит разобрать глубже в сессии
   - какие вопросы важно задать в живом разговоре
   - на какие зоны опоры и рисков обратить внимание.

Фокус: это **черновик для мастера**, а не готовый текст для клиента.
Можно писать чуть более техническим языком, но без чрезмерного жаргона.
"""

            try:
                resp = client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content": "Ты эксперт по потенциалам Deep Identity и пишешь черновой аналитический отчёт для мастера.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.6,
                )
                report_text = resp.choices[0].message.content
            except Exception as e:
                report_text = f"Ошибка при запросе к OpenAI: {e}"

        st.markdown("### Черновой отчёт (для тебя, Аселя 🧡)")
        st.markdown(report_text)


# ================================================================
# MAIN
# ================================================================

def main():
    st.set_page_config(
        page_title="Deep Identity · Диагностика потенциалов",
        layout="centered",
    )

    mode = st.sidebar.radio("Режим", ["Клиент", "Мастер"])
    if mode == "Клиент":
        render_client_mode()
    else:
        render_master_mode()


if __name__ == "__main__":
    main()

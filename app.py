import streamlit as st
import random
import numpy as np
import pandas as pd
import plotly.express as px
from dataclasses import dataclass, field
from typing import List, Tuple

# ==========================================
# 1. КОНФІГУРАЦІЯ ТА КОНСТАНТИ
# ==========================================
st.set_page_config(
    page_title="Прогноз Вивчення Слів",
    page_icon="🎓",
    layout="wide"
)

# Ваги FSRS v4 (стандартні)
DEFAULT_WEIGHTS = [
    0.4, 0.6, 2.4, 5.8,  
    4.93, 0.94,          
    0.86, 0.01,          
    1.49, 0.14, 0.94,    
    2.18, 0.05, 0.34, 1.26, 0.29, 2.61
]

DECK_SIZE = 65            # Розмір колоди
SIMULATION_DAYS = 365     # Період прогнозу

# ==========================================
# 2. ЛОГІКА ALGORITHM (FSRS)
# ==========================================

@dataclass
class FSRSParams:
    request_retention: float
    initial_stability_good: float
    w: List[float] = field(default_factory=lambda: DEFAULT_WEIGHTS)

class FSRS:
    def __init__(self, params: FSRSParams):
        self.p = params
        self.p.w[2] = self.p.initial_stability_good

    def calculate_retrievability(self, s: float, t: int) -> float:
        if s == 0: return 0.0
        return (1 + 19 * (t / s)) ** -1

    def next_interval(self, s: float, d: float, rating: int, r: float) -> Tuple[float, float]:
        # rating: 1=Забув, 2=Важко, 3=Добре, 4=Легко
        
        # Оновлення Складності (D)
        d_new = d - self.p.w[6] * (rating - 3)
        d_new = self.p.w[5] * self.p.w[4] + (1 - self.p.w[5]) * d_new
        d_new = max(1.0, min(10.0, d_new))

        # Оновлення Міцності пам'яті (S)
        if rating == 1: # Забув (Again)
            s_new = self.p.w[11] * (d_new ** -self.p.w[12]) * ((s + 1) ** self.p.w[13]) * np.exp(self.p.w[14] * (1 - r))
        else:
            if rating == 2: factor = self.p.w[8] 
            elif rating == 3: factor = self.p.w[9] 
            else: factor = self.p.w[10]

            base_growth = 1 + factor * (11 - d) * (s ** -0.9) * (np.exp((1 - r)) - 1)
            
            if rating == 2: base_growth *= 0.8
            if rating == 4: base_growth *= 1.3
            
            s_new = s * base_growth

        return max(s_new, 0.1), d_new

    def initial_params(self, rating: int) -> Tuple[float, float]:
        s = self.p.w[rating - 1]
        d = self.p.w[4] 
        return s, d

@dataclass
class Card:
    id: int
    difficulty: float
    stability: float = 0.0
    state: str = "New" 
    last_review_day: int = -1

# ==========================================
# 3. ІНТЕРФЕЙС ТА СИМУЛЯЦІЯ
# ==========================================

def get_difficulty_range(level_name: str) -> Tuple[float, float]:
    mapping = {
        "Початківець (A1)": (2.0, 3.0),
        "Базовий (A2-B1)": (3.0, 5.0),
        "Просунутий (B2-C1)": (5.0, 7.0),
        "Експерт (C2)": (7.0, 8.0),
    }
    return mapping.get(level_name, (3.0, 5.0))

def main():
    # --- Sidebar: Налаштування ---
    with st.sidebar:
        st.header("⚙️ Налаштування Учня")
        
        level_input = st.selectbox(
            "Ваш рівень англійської", 
            ["Початківець (A1)", "Базовий (A2-B1)", "Просунутий (B2-C1)", "Експерт (C2)"], 
            index=1
        )

        st.divider()
        st.subheader("🎯 Ціль навчання")
        
        # Додав key="mastery_slider_v2", щоб скинути кеш віджета і гарантувати крок 1
        mastery_threshold = st.slider(
            "Вважати слово вивченим, коли інтервал > (днів)",
            min_value=7, max_value=90, value=21, step=1,
            key="mastery_slider_v2", 
            help="Інтервал у днях. Тепер можна вибирати з точністю до 1 дня."
        )

        st.divider()
        st.subheader("📅 Мій розклад")
        
        training_days_per_week = st.slider(
            "Скільки днів на тиждень ви вчитесь?", 
            min_value=1, max_value=7, value=3
        )
        
        max_cards_per_session = st.number_input(
            "Ліміт карток за одне заняття",
            min_value=10, max_value=200, value=30, step=5
        )

        st.divider()
        st.subheader("🧠 Якість навчання")
        
        retention_input = st.slider(
            "Бажана надійність пам'яті (%)", 
            70, 99, 90,
            help="90% означає, що ви хочете пам'ятати слово в 9 випадках з 10 при наступній зустрічі."
        )

        with st.expander("Деталі успішності (Advanced)"):
            st.write("Як часто ви помиляєтесь?")
            prob_again = st.number_input("Забув / Помилка (%)", 0, 100, 15)
            prob_hard = st.number_input("Важко згадати (%)", 0, 100, 15)
            prob_good = st.number_input("Згадав нормально (%)", 0, 100, 55)
            prob_easy = st.number_input("Дуже легко (%)", 0, 100, 15)
            
            if prob_again + prob_hard + prob_good + prob_easy != 100:
                st.error("Сума має бути 100%!")
                st.stop()

        run_btn = st.button("🚀 Запустити прогноз", type="primary")

    # --- Головний екран ---
    st.title("🎓 Прогноз вивчення слів")
    st.markdown(f"""
    **Дано:** Колода з **{DECK_SIZE} слів**.  
    **Ціль:** Закріпити їх у пам'яті (інтервал повторення > **{mastery_threshold} днів**).  
    **Режим:** {training_days_per_week} тренувань на тиждень, максимум {max_cards_per_session} слів за раз.
    """)

    if run_btn:
        with st.spinner("Прораховуємо вашу криву навчання..."):
            params = FSRSParams(
                request_retention=retention_input / 100.0,
                initial_stability_good=4.0
            )
            fsrs = FSRS(params)
            
            min_d, max_d = get_difficulty_range(level_input)
            
            # Генерація колоди
            deck = [Card(id=i, difficulty=random.uniform(min_d, max_d)) for i in range(DECK_SIZE)]
            
            stats_history = []
            
            probs = [prob_again/100, prob_hard/100, prob_good/100, prob_easy/100]
            choices = [1, 2, 3, 4] # Again, Hard, Good, Easy

            total_reviews_log = 0

            for day in range(1, SIMULATION_DAYS + 1):
                is_training_day = (day % 7) < training_days_per_week
                
                mastered_today_count = 0
                reviews_today = 0
                
                if is_training_day:
                    # Due cards
                    due_cards = []
                    for card in deck:
                        if card.state == "New": continue
                        
                        days_elapsed = day - card.last_review_day
                        r = fsrs.calculate_retrievability(card.stability, days_elapsed)
                        
                        if r < params.request_retention:
                            due_cards.append((card, r))
                    
                    due_cards.sort(key=lambda x: x[1])
                    
                    slots_remaining = max_cards_per_session
                    
                    # 1. Reviews
                    for card, r in due_cards:
                        if slots_remaining <= 0: break
                        
                        slots_remaining -= 1
                        reviews_today += 1
                        total_reviews_log += 1
                        
                        rating = np.random.choice(choices, p=probs)
                        
                        was_mastered = card.stability > mastery_threshold
                        
                        new_s, new_d = fsrs.next_interval(card.stability, card.difficulty, rating, r)
                        card.stability = new_s
                        card.difficulty = new_d
                        card.last_review_day = day
                        
                        is_now_mastered = card.stability > mastery_threshold
                        
                        if not was_mastered and is_now_mastered:
                            mastered_today_count += 1
                        if was_mastered and not is_now_mastered:
                            mastered_today_count -= 1 

                    # 2. New Cards
                    new_cards_candidates = [c for c in deck if c.state == "New"]
                    for card in new_cards_candidates:
                        if slots_remaining <= 0: break
                        
                        slots_remaining -= 1
                        reviews_today += 1
                        total_reviews_log += 1
                        
                        rating = np.random.choice(choices, p=probs)
                        init_s, _ = fsrs.initial_params(rating)
                        card.stability = init_s
                        card.difficulty = max(1.0, min(10.0, card.difficulty - 0.5 * (rating - 3)))
                        card.state = "Learning"
                        card.last_review_day = day
                        
                        if card.stability > mastery_threshold:
                            mastered_today_count += 1

                # Stats
                total_mastered = sum(1 for c in deck if c.stability > mastery_threshold)
                
                stats_history.append({
                    "Day": day,
                    "Total Mastered": total_mastered,
                    "Newly Mastered": max(0, mastered_today_count),
                    "Workload": reviews_today
                })

            # --- Visualization ---
            df = pd.DataFrame(stats_history)
            
            final_mastered = df["Total Mastered"].iloc[-1]
            days_to_finish = df[df["Total Mastered"] == DECK_SIZE]["Day"].min()
            
            finish_text = f"{int(days_to_finish)} днів" if not pd.isna(days_to_finish) else "Більше року"

            col1, col2, col3 = st.columns(3)
            col1.metric(f"Вивчено слів (>{mastery_threshold} дн.)", f"{final_mastered} / {DECK_SIZE}")
            col2.metric("Час до повного вивчення", finish_text)
            col3.metric("Всього карток пройдено", total_reviews_log)

            st.divider()

            st.subheader("📈 Скільки слів я буду вивчати щодня?")
            fig_daily = px.bar(
                df, x="Day", y="Newly Mastered",
                title="Нові вивчені слова (по днях)",
                labels={"Newly Mastered": "Слів вивчено", "Day": "День"},
                color_discrete_sequence=["#2ECC71"]
            )
            fig_daily.update_layout(bargap=0.2)
            st.plotly_chart(fig_daily, use_container_width=True)

            st.subheader("🏔️ Загальний прогрес")
            fig_cum = px.area(
                df, x="Day", y="Total Mastered",
                title=f"Слова з міцністю пам'яті > {mastery_threshold} днів",
                labels={"Total Mastered": "Всього вивчено слів", "Day": "День"},
                range_y=[0, DECK_SIZE + 5],
                color_discrete_sequence=["#3498DB"]
            )
            fig_cum.add_hline(y=DECK_SIZE, line_dash="dash", line_color="gray", annotation_text="Ціль (65 слів)")
            st.plotly_chart(fig_cum, use_container_width=True)

            st.subheader("🏋️ Навантаження")
            df_work = df[df["Workload"] > 0]
            if not df_work.empty:
                fig_work = px.bar(
                    df_work, x="Day", y="Workload",
                    title="Кількість повторень на кожному занятті",
                    labels={"Workload": "Карток (Повторення + Нові)", "Day": "День"},
                    color_discrete_sequence=["#F1C40F"]
                )
                fig_work.add_hline(y=max_cards_per_session, line_dash="dot", line_color="red", annotation_text="Ваш ліміт")
                st.plotly_chart(fig_work, use_container_width=True)
            else:
                st.info("Немає даних про навантаження.")

if __name__ == "__main__":
    main()

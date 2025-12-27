import streamlit as st
import random
import numpy as np
import pandas as pd
import plotly.express as px
from dataclasses import dataclass, field
from typing import List, Tuple

# ==========================================
# 1. КОНФІГУРАЦІЯ
# ==========================================
st.set_page_config(
    page_title="Калькулятор часу вивчення слів",
    page_icon="⏱️",
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

# Обмеження щоб симуляція не зависла при поганих параметрах
MAX_SIMULATION_YEARS = 5 

# ==========================================
# 2. МАТЕМАТИКА FSRS
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
        d_new = d - self.p.w[6] * (rating - 3)
        d_new = self.p.w[5] * self.p.w[4] + (1 - self.p.w[5]) * d_new
        d_new = max(1.0, min(10.0, d_new))

        if rating == 1: # Again
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
# 3. ІНТЕРФЕЙС
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
    # --- Sidebar ---
    with st.sidebar:
        st.header("⚙️ Вхідні дані")
        
        # 1. Колода
        deck_size = st.number_input(
            "Кількість слів для вивчення", 
            min_value=10, max_value=5000, value=65, step=5,
            help="Скільки всього слів ви хочете вивчити."
        )

        level_input = st.selectbox(
            "Складність слів", 
            ["Початківець (A1)", "Базовий (A2-B1)", "Просунутий (B2-C1)", "Експерт (C2)"], 
            index=1
        )

        st.divider()
        st.subheader("🎯 Ваша Ціль")
        
        mastery_threshold = st.slider(
            "Вважати слово вивченим, коли інтервал > (днів)",
            min_value=7, max_value=90, value=21, step=1,
            key="mastery_slider_fixed",
            help="Як тільки інтервал повторення слова перевищить це число, воно вважається вивченим і більше не з'являється в тренуваннях."
        )

        st.divider()
        st.subheader("📅 Розклад")
        
        training_days_per_week = st.slider(
            "Днів навчання на тиждень", 
            min_value=1, max_value=7, value=3
        )
        
        max_cards_per_session = st.number_input(
            "Ліміт карток за урок",
            min_value=10, max_value=500, value=30, step=5,
            help="Включає і повторення старих, і вивчення нових."
        )

        st.divider()
        with st.expander("Додаткові налаштування"):
            retention_input = st.slider("Бажана точність пам'яті (%)", 70, 99, 90)
            st.caption("Шанси відповіді учня:")
            prob_again = st.number_input("Помилка (%)", 0, 100, 10)
            prob_hard = st.number_input("Важко (%)", 0, 100, 15)
            prob_good = st.number_input("Добре (%)", 0, 100, 60)
            prob_easy = st.number_input("Легко (%)", 0, 100, 15)

            if prob_again + prob_hard + prob_good + prob_easy != 100:
                st.error("Сума % має бути 100!")
                st.stop()

        run_btn = st.button("🚀 Розрахувати час", type="primary")

    # --- Main Area ---
    st.title("⏱️ Калькулятор часу навчання")
    
    if run_btn:
        with st.spinner("Симулюємо процес навчання день за днем..."):
            
            # Setup
            params = FSRSParams(request_retention=retention_input/100.0, initial_stability_good=4.0)
            fsrs = FSRS(params)
            min_d, max_d = get_difficulty_range(level_input)
            
            # Створюємо колоду
            deck = [Card(id=i, difficulty=random.uniform(min_d, max_d)) for i in range(deck_size)]
            
            # Змінні циклу
            day = 0
            total_reps = 0
            mastered_count = 0
            stats_history = []
            
            probs = [prob_again/100, prob_hard/100, prob_good/100, prob_easy/100]
            choices = [1, 2, 3, 4]

            # --- ГОЛОВНИЙ ЦИКЛ (Йдемо поки не вивчимо все) ---
            while mastered_count < deck_size:
                day += 1
                
                # Запобіжник вічного циклу
                if day > 365 * MAX_SIMULATION_YEARS:
                    st.warning(f"Симуляцію зупинено на {day} дні. Схоже, параметри занадто складні (мало уроків або занадто висока ціль).")
                    break

                # Чи сьогодні тренування?
                is_training_day = (day % 7) < training_days_per_week
                
                if is_training_day:
                    # 1. Знаходимо картки для повторення (Тільки ті, що НЕ вивчені)
                    # "Вивчено" означає stability > threshold. 
                    # Ми їх взагалі ігноруємо, ніби відклали в архів "Done".
                    
                    due_cards = []
                    active_cards_count = 0 # Скільки карток ще в грі
                    
                    for card in deck:
                        # Якщо вже вивчено - пропускаємо
                        if card.stability > mastery_threshold:
                            continue
                            
                        active_cards_count += 1
                        
                        if card.state == "New":
                            continue
                        
                        days_elapsed = day - card.last_review_day
                        r = fsrs.calculate_retrievability(card.stability, days_elapsed)
                        
                        if r < params.request_retention:
                            due_cards.append((card, r))
                    
                    # Якщо немає активних карток і всі вивчені -> кінець
                    if active_cards_count == 0:
                        mastered_count = deck_size # Fix count just in case
                        break

                    # Сортуємо: спочатку ті, що найбільше забули
                    due_cards.sort(key=lambda x: x[1])
                    
                    slots = max_cards_per_session
                    
                    # --- Етап А: Повторення (Reviews) ---
                    for card, r in due_cards:
                        if slots <= 0: break
                        
                        slots -= 1
                        total_reps += 1
                        
                        rating = np.random.choice(choices, p=probs)
                        new_s, new_d = fsrs.next_interval(card.stability, card.difficulty, rating, r)
                        
                        card.stability = new_s
                        card.difficulty = new_d
                        card.last_review_day = day

                    # --- Етап Б: Нові слова (New cards) ---
                    # Беремо тільки якщо лишилось місце
                    if slots > 0:
                        new_candidates = [c for c in deck if c.state == "New"]
                        for card in new_candidates:
                            if slots <= 0: break
                            
                            slots -= 1
                            total_reps += 1
                            
                            rating = np.random.choice(choices, p=probs)
                            init_s, _ = fsrs.initial_params(rating)
                            
                            card.stability = init_s
                            card.difficulty = max(1.0, min(10.0, card.difficulty - 0.5 * (rating - 3)))
                            card.state = "Learning"
                            card.last_review_day = day

                # Підрахунок вивчених на кінець дня
                current_mastered = sum(1 for c in deck if c.stability > mastery_threshold)
                mastered_count = current_mastered
                
                # Записуємо статистику для графіка (але не кожен день, щоб не перевантажити, якщо дуже довго)
                # Якщо днів < 1000 - кожен день, якщо більше - рідше.
                stats_history.append({
                    "День": day,
                    "Вивчено слів": current_mastered
                })
                
                if current_mastered >= deck_size:
                    break

            # --- РЕЗУЛЬТАТИ ---
            
            # Метрики
            col1, col2, col3 = st.columns(3)
            
            # 1. Час
            years = day // 365
            rem_days = day % 365
            time_str = f"{day} днів"
            if years > 0:
                time_str += f" ({years} р. {rem_days} дн.)"
                
            col1.metric("⏳ Час до повного вивчення", time_str, help="Скільки часу пройде від старту до моменту, коли ОСТАННЄ слово перетне поріг вивченого.")
            
            # 2. Кількість вправ
            col2.metric("👆 Всього виконано вправ", f"{total_reps}", help="Загальна кількість разів, коли ви тренували картки (сума всіх відповідей).")
            
            # 3. Ефективність
            avg_reps = total_reps / deck_size
            col3.metric("📊 Середня кількість повторень", f"{avg_reps:.1f} на слово", help="Скільки разів в середньому треба повторити одне слово, щоб вивчити його.")

            st.divider()

            # Графік Прогресу
            st.subheader("📈 Графік досягнення мети")
            df = pd.DataFrame(stats_history)
            
            fig = px.area(
                df, x="День", y="Вивчено слів",
                title=f"Динаміка вивчення {deck_size} слів",
                color_discrete_sequence=["#3498DB"]
            )
            # Лінія мети
            fig.add_hline(y=deck_size, line_dash="dash", line_color="green", annotation_text="Ціль")
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.success(f"""
            **Що це означає?**
            Щоб вивчити **{deck_size} слів** так, щоб пам'ятати кожне мінімум **{mastery_threshold} днів**, 
            вам знадобиться займатися **{day} днів** за вашим розкладом.
            """)

    else:
        st.info("👈 Введіть параметри зліва та натисніть кнопку 'Розрахувати час'")

if __name__ == "__main__":
    main()

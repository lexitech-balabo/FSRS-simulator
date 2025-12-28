import streamlit as st
import random
import numpy as np
import pandas as pd
import plotly.express as px
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

# ==========================================
# 1. КОНФІГУРАЦІЯ
# ==========================================
st.set_page_config(
    page_title="FSRS Simulator: Smart Allocation",
    page_icon="🧠",
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

    def get_ideal_interval(self, s: float) -> float:
        """
        Розраховує інтервал, при якому R дорівнює request_retention.
        Формула обернена до R = (1 + 19 * t/s)^-1
        """
        if s == 0: return 0.0
        r = self.p.request_retention
        # R^-1 = 1 + 19 * I / S  =>  I = (S / 19) * (1/R - 1)
        return (s / 19.0) * ((1.0 / r) - 1.0)

    def next_interval(self, s: float, d: float, rating: int, r: float) -> Tuple[float, float]:
        # Оновлення Difficulty
        d_new = d - self.p.w[6] * (rating - 3)
        d_new = self.p.w[5] * self.p.w[4] + (1 - self.p.w[5]) * d_new
        d_new = max(1.0, min(10.0, d_new))

        # Оновлення Stability
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
    state: str = "New" # New, Learning, Mastered
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
        
        deck_size = st.number_input("Кількість слів", 10, 5000, 65, 5)
        level_input = st.selectbox(
            "Складність слів (CEFR)", 
            ["Початківець (A1)", "Базовий (A2-B1)", "Просунутий (B2-C1)", "Експерт (C2)"], 
            index=1
        )

        st.divider()
        st.subheader("🎯 Ціль")
        mastery_threshold = st.slider(
            "Вважати слово вивченим, коли інтервал > (днів)",
            7, 90, 21, 1, key="mastery_slider_fixed"
        )

        st.divider()
        st.subheader("📅 Розклад")
        training_days_per_week = st.slider("Днів навчання на тиждень", 1, 7, 3)
        max_cards_per_session = st.number_input("Ліміт карток за урок", 10, 500, 30, 5)

        st.divider()
        with st.expander("Додаткові налаштування"):
            retention_input = st.slider("Бажана точність пам'яті (%)", 70, 99, 90)
            st.caption("Шанси відповіді:")
            prob_again = st.number_input("Again (%)", 0, 100, 10)
            prob_hard = st.number_input("Hard (%)", 0, 100, 15)
            prob_good = st.number_input("Good (%)", 0, 100, 60)
            prob_easy = st.number_input("Easy (%)", 0, 100, 15)

            if prob_again + prob_hard + prob_good + prob_easy != 100:
                st.error("Сума % має бути 100!")
                st.stop()

        run_btn = st.button("🚀 Розрахувати час", type="primary")

    # --- Main Area ---
    st.title("⏱️ Симулятор FSRS: Smart Allocation")
    
    if run_btn:
        with st.spinner("Симулюємо процес навчання..."):
            
            # Setup
            params = FSRSParams(request_retention=retention_input/100.0, initial_stability_good=4.0)
            fsrs = FSRS(params)
            min_d, max_d = get_difficulty_range(level_input)
            
            # Генерація колоди (Difficulty based on CEFR Level)
            deck = [Card(id=i, difficulty=random.uniform(min_d, max_d)) for i in range(deck_size)]
            
            day = 0
            total_reps = 0
            mastered_count = 0
            stats_history = []
            
            probs = [prob_again/100, prob_hard/100, prob_good/100, prob_easy/100]
            choices = [1, 2, 3, 4]

            # Цільові квоти
            target_new_ratio = 0.3
            
            # --- MAIN LOOP ---
            while mastered_count < deck_size:
                day += 1
                if day > 365 * MAX_SIMULATION_YEARS:
                    st.warning("Ліміт часу перевищено.")
                    break

                is_training_day = (day % 7) < training_days_per_week
                
                if is_training_day:
                    # 1. Identify Candidates
                    review_candidates = [] # List of dicts with sorting info
                    new_candidates = []
                    
                    active_cards_count = 0 
                    
                    for card in deck:
                        # Skip mastered
                        if card.stability > mastery_threshold:
                            continue
                        
                        active_cards_count += 1
                        
                        if card.state == "New":
                            new_candidates.append(card)
                        else:
                            # Check Review (Due) Status
                            days_elapsed = day - card.last_review_day
                            r = fsrs.calculate_retrievability(card.stability, days_elapsed)
                            
                            if r < params.request_retention:
                                # --- URGENCY SCORE LOGIC ---
                                # Urgency = (overdue_days * 10) + difficulty
                                ideal_interval = fsrs.get_ideal_interval(card.stability)
                                overdue_days = (day - card.last_review_day) - ideal_interval
                                
                                # Overdue може бути трохи менше 0, якщо R впало нижче порогу раніше (через округлення),
                                # але зазвичай позитивне для прострочених карток.
                                urgency = (overdue_days * 10) + card.difficulty
                                
                                review_candidates.append({
                                    "card": card,
                                    "r": r,
                                    "urgency": urgency
                                })

                    if active_cards_count == 0:
                        mastered_count = deck_size 
                        break

                    # 2. Sort Review Candidates by Urgency (Highest first)
                    review_candidates.sort(key=lambda x: x["urgency"], reverse=True)
                    
                    # 3. Smart 70/30 Allocation with Backfill
                    limit = max_cards_per_session
                    
                    # Розрахунок цілей
                    target_new = round(limit * target_new_ratio)
                    target_review = limit - target_new
                    
                    # Вибірка
                    selected_reviews_wrappers = review_candidates[:target_review]
                    selected_new_cards = new_candidates[:target_new]
                    
                    # Backfill Logic
                    # Якщо не вистачає New, заповнюємо Review
                    if len(selected_new_cards) < target_new:
                        shortage = target_new - len(selected_new_cards)
                        extra_reviews = review_candidates[target_review : target_review + shortage]
                        selected_reviews_wrappers.extend(extra_reviews)
                        
                    # Якщо не вистачає Review, заповнюємо New (якщо є)
                    # Перераховуємо поточну кількість Reviews, бо ми могли додати вище
                    current_reviews_count = len(selected_reviews_wrappers)
                    if current_reviews_count < (limit - len(selected_new_cards)):
                        # Скільки слотів ще вільно?
                        slots_left = limit - current_reviews_count - len(selected_new_cards)
                        if slots_left > 0:
                            start_idx = len(selected_new_cards) # ми вже взяли цю кількість
                            extra_new = new_candidates[start_idx : start_idx + slots_left]
                            selected_new_cards.extend(extra_new)

                    # Формуємо фінальний список сесії
                    # Розпаковуємо review wrappers назад в об'єкти карток
                    session_cards = [item["card"] for item in selected_reviews_wrappers] + selected_new_cards
                    
                    # Обробка сесії
                    for card in session_cards:
                        total_reps += 1
                        
                        # Calculate current R for algorithm (needs fresh calculation)
                        if card.state == "New":
                            r_current = 0.0 # Не використовується для init
                        else:
                            days_elapsed = day - card.last_review_day
                            r_current = fsrs.calculate_retrievability(card.stability, days_elapsed)
                            
                        # Sim Rating
                        rating = np.random.choice(choices, p=probs)
                        
                        if card.state == "New":
                            # Init
                            init_s, _ = fsrs.initial_params(rating)
                            card.stability = init_s
                            # Slight difficulty adjust
                            card.difficulty = max(1.0, min(10.0, card.difficulty - 0.5 * (rating - 3)))
                            card.state = "Learning"
                        else:
                            # Review Update
                            new_s, new_d = fsrs.next_interval(card.stability, card.difficulty, rating, r_current)
                            card.stability = new_s
                            card.difficulty = new_d
                            
                        card.last_review_day = day

                # Stats Recording
                current_mastered = sum(1 for c in deck if c.stability > mastery_threshold)
                mastered_count = current_mastered
                
                # Логуємо рідше, якщо симуляція довга, для продуктивності графіка
                if day < 1000 or day % 7 == 0:
                    stats_history.append({"День": day, "Вивчено слів": current_mastered})
                
                if current_mastered >= deck_size:
                    break

            # --- OUTPUT ---
            years = day // 365
            rem_days = day % 365
            time_str = f"{day} днів"
            if years > 0: time_str += f" ({years} р. {rem_days} дн.)"
                
            col1, col2, col3 = st.columns(3)
            col1.metric("⏳ Час до фінішу", time_str)
            col2.metric("👆 Всього вправ", f"{total_reps}")
            col3.metric("📊 Reps / Word", f"{total_reps / deck_size:.1f}")

            st.divider()
            st.subheader("📈 Динаміка вивчення")
            
            df = pd.DataFrame(stats_history)
            fig = px.area(df, x="День", y="Вивчено слів", color_discrete_sequence=["#3498DB"])
            fig.add_hline(y=deck_size, line_dash="dash", line_color="green", annotation_text="Ціль")
            st.plotly_chart(fig, use_container_width=True)

            st.info(f"Алгоритм: FSRS v4 | Allocation: ~30% New / 70% Review (Smart Backfill) | Sorting: Urgency Score")

if __name__ == "__main__":
    main()

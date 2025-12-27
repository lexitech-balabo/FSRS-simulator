import streamlit as st
import random
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# ==========================================
# 1. CONFIG & CONSTANTS
# ==========================================
st.set_page_config(
    page_title="FSRS v4 Simulator",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# FSRS v4 Default Weights (approximate standard values)
# w[0-3]: Initial Stability for Again, Hard, Good, Easy
# w[4-5]: Difficulty normalization
# w[6-7]: Difficulty update
# w[8-16]: Stability update
DEFAULT_WEIGHTS = [
    0.4, 0.6, 2.4, 5.8,  # Initial Stability
    4.93, 0.94,          # Difficulty mean/reversion
    0.86, 0.01,          # Difficulty update
    1.49, 0.14, 0.94,    # Stability update (Hard/Good/Easy factors)
    2.18, 0.05, 0.34, 1.26, 0.29, 2.61 # Other constants
]

MASTERY_THRESHOLD = 21.0
DECK_SIZE = 65
SIMULATION_DAYS = 365

# ==========================================
# 2. CORE LOGIC (FSRS ALGORITHM)
# ==========================================

@dataclass
class FSRSParams:
    request_retention: float
    initial_stability_good: float
    w: List[float] = field(default_factory=lambda: DEFAULT_WEIGHTS)

class FSRS:
    """
    Реалізація математичної моделі FSRS v4.
    """
    def __init__(self, params: FSRSParams):
        self.p = params
        # Override default weight for 'Good' start based on user input
        self.p.w[2] = self.p.initial_stability_good

    def calculate_retrievability(self, s: float, t: int) -> float:
        """
        R = (1 + factor * t / S) ^ decay
        Для FSRS v4 зазвичай використовується спрощена формула з фактором 19 (при decay -1 або близько того)
        або точна формула: R = (1 + 19 * (t / s)) ^ -1
        """
        if s == 0: return 0.0
        return (1 + 19 * (t / s)) ** -1

    def next_interval(self, s: float, d: float, rating: int, r: float) -> Tuple[float, float]:
        """
        Розрахунок нових S та D.
        rating: 1=Again, 2=Hard, 3=Good, 4=Easy
        Повертає: (new_s, new_d)
        """
        # 1. Update Difficulty (D)
        # D_new = D - w6 * (rating - 3)
        # Потім застосовуємо Mean Reversion до w4 (default D)
        # D_new = w7 * w4 + (1 - w7) * D_new
        
        # Рейтинг для формули D: Again=1 .. Easy=4. 
        # Але в формулі часто використовується (grade - 3), де grade 1..4.
        
        d_new = d - self.p.w[6] * (rating - 3)
        d_new = self.p.w[5] * self.p.w[4] + (1 - self.p.w[5]) * d_new
        
        # Clamp D (1 to 10)
        d_new = max(1.0, min(10.0, d_new))

        # 2. Update Stability (S)
        if rating == 1: # Again
            # Stability decreases
            # S_new = S * w8 * exp(w9 * (11-D)) * S^(-w10) ... (прощена логіка для симулятора)
            # Використаємо стандартну формулу скорочення для Again з v4
            s_new = self.p.w[11] * (d_new ** -self.p.w[12]) * ((s + 1) ** self.p.w[13]) * np.exp(self.p.w[14] * (1 - r))
            # Для Again стабільність зазвичай різко падає, часто до 10-50% від попередньої
            # Але в FSRS v4 є окрема формула. Заради спрощення і робастності симулятора:
            # Використовуємо наближення, якщо формула дає збій, але формула вище - з параметрів.
            # Зауваження: в clean FSRS v4 formula для Again інша.
            # Правильна формула v4 для Again:
            # S_new = w11 * D^(-w12) * (S+1)^w13 * exp(w14 * (1-R))
            pass 
        else:
            # Hard / Good / Easy
            # S_new = S * (1 + factor)
            # Factor depends on D, S, R and Rating weights
            
            # w8 exp(w9(11-D)) S^(-w10) (exp(w11(1-R))-1) - це структура v5/optimizer.
            # Для v4 structure:
            # S_new = S * (1 + exp(w8) * (11-D) * S^(-w9) * (exp(w10*(1-R)) - 1))
            
            # Відповідно до індексу в DEFAULT_WEIGHTS (mapping може відрізнятись в різних версіях):
            # w[8] base multiplier factor
            # w[9] difficulty factor power (usually negative or handled in exp)
            # w[10] retrievability factor
            
            # Реалізація поширеної версії v4:
            if rating == 2: # Hard
                factor = self.p.w[8] 
            elif rating == 3: # Good
                factor = self.p.w[9] 
            else: # Easy
                factor = self.p.w[10]

            # FSRS v4 scaling formula approximation:
            # Next Stability = S * (1 + exp(w8) * (11-D) * S^(-w9) * (exp(w10 * (1-R)) - 1))
            # Але оскільки ми маємо масив weights, давайте використаємо простішу logic v4:
            # S_new = S * (1 + factor * (11 - d) * (s ** -0.5) * (np.exp(0.5 * (1 - r)) - 1)) * tuning
            
            # Для цього симулятора, щоб гарантувати коректне зростання S > 21:
            base_growth = 1 + factor * (11 - d) * (s ** -0.9) * (np.exp((1 - r)) - 1)
            
            # Hard penalty / Easy bonus hardcoded coefficients for stability
            if rating == 2: base_growth *= 0.8
            if rating == 4: base_growth *= 1.3
            
            s_new = s * base_growth

        return max(s_new, 0.1), d_new

    def initial_params(self, rating: int) -> Tuple[float, float]:
        """Початкові S та D для нового слова"""
        # S base on rating (index 0-3 for ratings 1-4)
        s = self.p.w[rating - 1]
        # D base on w4 but we override D externally based on Level
        d = self.p.w[4] 
        return s, d

@dataclass
class Card:
    id: int
    difficulty: float
    stability: float = 0.0
    state: str = "New" # New, Learning, Review, Mastered
    last_review_day: int = -1
    history: List[dict] = field(default_factory=list)

    @property
    def is_mastered(self):
        return self.stability > MASTERY_THRESHOLD

# ==========================================
# 3. STREAMLIT UI & SIMULATION LOOP
# ==========================================

def get_difficulty_range(level: str) -> Tuple[float, float]:
    if level == "A1": return (2.0, 3.0)
    if level == "A2-B1": return (3.0, 5.0)
    if level == "B2-C1": return (5.0, 7.0)
    if level == "C2": return (7.0, 8.0)
    return (3.0, 5.0)

def main():
    # --- Sidebar ---
    with st.sidebar:
        st.header("Налаштування Симуляції")
        
        # Reset Button Logic (Trick with key)
        if st.button("Reset to Defaults"):
            st.session_state.clear()
            st.rerun()

        level = st.selectbox(
            "Рівень Курсу (Difficulty)", 
            ["A1", "A2-B1", "B2-C1", "C2"], 
            index=1,
            help="Визначає початкову складність (D) слів."
        )

        request_retention = st.slider(
            "Request Retention (R)", 
            0.70, 0.99, 0.90, 0.01,
            help="Цільовий рівень пам'яті. Чим вище, тим частіше повторення."
        )

        initial_s_good = st.number_input(
            "Initial Stability (Good)", 
            value=4.0, step=0.5,
            help="Інтервал (в днях) після першого 'Good'."
        )

        st.subheader("Профіль Учня (Ймовірності)")
        col1, col2 = st.columns(2)
        prob_again = col1.number_input("Again %", 0, 100, 10)
        prob_hard = col2.number_input("Hard %", 0, 100, 15)
        prob_good = col1.number_input("Good %", 0, 100, 60)
        prob_easy = col2.number_input("Easy %", 0, 100, 15)

        total_prob = prob_again + prob_hard + prob_good + prob_easy
        if total_prob != 100:
            st.error(f"Сума ймовірностей має бути 100%. Поточна: {total_prob}%")
            run_sim = False
        else:
            st.success("Профіль валідний")
            run_sim = st.button("Run Simulation", type="primary")

    # --- Main Content ---
    st.title("🧩 FSRS v4 Simulator")
    st.markdown(f"**Scenario:** Learning **{DECK_SIZE} words** over **{SIMULATION_DAYS} days** | **Level:** {level}")

    if run_sim:
        with st.spinner("Симуляція навчання..."):
            # 1. Setup Simulation
            params = FSRSParams(
                request_retention=request_retention,
                initial_stability_good=initial_s_good
            )
            fsrs = FSRS(params)
            
            min_d, max_d = get_difficulty_range(level)
            
            # Generate Deck
            deck = [Card(id=i, difficulty=random.uniform(min_d, max_d)) for i in range(DECK_SIZE)]
            
            # Stats Containers
            daily_reviews_count = np.zeros(SIMULATION_DAYS)
            daily_mastered_count = np.zeros(SIMULATION_DAYS)
            
            # Probabilities for np.random.choice
            rating_probs = [prob_again/100, prob_hard/100, prob_good/100, prob_easy/100]
            rating_choices = [1, 2, 3, 4]

            # 2. Simulation Loop (Day by Day)
            # Щоб симуляція була цікавою, вводимо слова поступово (наприклад, 5 нових в день), 
            # або всі відразу. Згідно з задачею "колода з 65 слів", припустимо, що користувач 
            # починає вчити їх всі в перші дні (e.g., 10 new cards/day).
            new_cards_per_day = 10 
            
            for day in range(SIMULATION_DAYS):
                reviews_today = 0
                
                # A. Process Reviews (Due Cards)
                for card in deck:
                    if card.state == "New":
                        continue
                    
                    # Calculate R
                    days_elapsed = day - card.last_review_day
                    r = fsrs.calculate_retrievability(card.stability, days_elapsed)
                    
                    # Check if due
                    if r < request_retention:
                        # Review happens
                        reviews_today += 1
                        
                        # Simulate User Rating
                        rating = np.random.choice(rating_choices, p=rating_probs)
                        
                        # Store old state for log (if selected for tracking)
                        old_s = card.stability
                        old_d = card.difficulty
                        
                        # Update S and D
                        new_s, new_d = fsrs.next_interval(card.stability, card.difficulty, rating, r)
                        
                        card.stability = new_s
                        card.difficulty = new_d
                        card.last_review_day = day
                        card.state = "Mastered" if card.stability > MASTERY_THRESHOLD else "Review"
                        
                        # Log history
                        card.history.append({
                            "Day": day, "Action": ["Again", "Hard", "Good", "Easy"][rating-1],
                            "R": round(r, 2), "Old S": round(old_s, 2), "New S": round(new_s, 2),
                            "D": round(new_d, 2)
                        })

                # B. Introduce New Cards
                new_cards_reviewed = 0
                for card in deck:
                    if card.state == "New" and new_cards_reviewed < new_cards_per_day:
                        # Initial Learning
                        reviews_today += 1
                        new_cards_reviewed += 1
                        
                        # For simplicity, assume first rating follows prob distribution, 
                        # but mostly usually Good/Easy for new easy words. Let's use same prob.
                        rating = np.random.choice(rating_choices, p=rating_probs)
                        
                        # Init S, D
                        # Note: D is already set by level, but S needs init.
                        # If rating is 'Good' (3), S becomes initial_stability_good (user param)
                        init_s, _ = fsrs.initial_params(rating)
                        
                        card.stability = init_s
                        # D updates slightly on first interaction too in full FSRS, but let's keep Level D dominant
                        # or update slightly:
                        card.difficulty = max(1.0, min(10.0, card.difficulty - 0.5 * (rating - 3)))

                        card.state = "Review"
                        card.last_review_day = day
                        
                        card.history.append({
                            "Day": day, "Action": f"New ({['Again', 'Hard', 'Good', 'Easy'][rating-1]})",
                            "R": 0.0, "Old S": 0.0, "New S": round(init_s, 2),
                            "D": round(card.difficulty, 2)
                        })

                daily_reviews_count[day] = reviews_today
                
                # Count Mastered
                mastered_cnt = sum(1 for c in deck if c.is_mastered)
                daily_mastered_count[day] = mastered_cnt

            # ==========================================
            # 4. VISUALIZATION & OUTPUT
            # ==========================================
            
            # --- KPI Metrics ---
            total_reviews = int(np.sum(daily_reviews_count))
            final_mastered = daily_mastered_count[-1]
            avg_stability = np.mean([c.stability for c in deck])
            mastered_pct = (final_mastered / DECK_SIZE) * 100

            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            kpi1.metric("Total Reviews", f"{total_reviews}")
            kpi2.metric("Mastered Words", f"{int(final_mastered)} / {DECK_SIZE}", f"{mastered_pct:.1f}%")
            kpi3.metric("Avg Stability (Days)", f"{avg_stability:.1f}")
            kpi4.metric("Avg Reviews/Card", f"{total_reviews / DECK_SIZE:.1f}")

            # --- Chart 1: Reviews per Day (Bar) ---
            st.subheader("📊 Review Load (Навантаження)")
            df_reviews = pd.DataFrame({
                "Day": range(1, SIMULATION_DAYS + 1),
                "Reviews": daily_reviews_count
            })
            fig_bar = px.bar(df_reviews, x="Day", y="Reviews", title="Кількість повторень на день")
            fig_bar.update_traces(marker_color='#4A90E2')
            st.plotly_chart(fig_bar, use_container_width=True)

            # --- Chart 2: Learning Progress (Area) ---
            st.subheader("📈 Learning Progress (Mastery)")
            df_mastery = pd.DataFrame({
                "Day": range(1, SIMULATION_DAYS + 1),
                "Mastered Words": daily_mastered_count
            })
            fig_area = px.area(
                df_mastery, x="Day", y="Mastered Words", 
                title="Зростання кількості вивчених слів (S > 21 days)",
                range_y=[0, DECK_SIZE]
            )
            fig_area.update_traces(line_color='#2ECC71', fillcolor='rgba(46, 204, 113, 0.3)')
            st.plotly_chart(fig_area, use_container_width=True)

            # --- Detailed Card Log ---
            st.subheader("🔍 Детальний лог випадкової картки")
            # Filter cards that have history
            active_cards = [c for c in deck if c.history]
            if active_cards:
                sample_card = random.choice(active_cards)
                st.markdown(f"**Card ID:** {sample_card.id} | **Final Difficulty:** {sample_card.difficulty:.2f} | **Final Stability:** {sample_card.stability:.2f}")
                
                df_log = pd.DataFrame(sample_card.history)
                # Calculate Next Interval for display
                df_log["Interval"] = (df_log["New S"]).astype(float).round(1)
                
                st.dataframe(
                    df_log[["Day", "Action", "D", "Old S", "New S", "Interval"]],
                    use_container_width=True
                )
            else:
                st.info("Ще не було взаємодій з картками.")

    else:
        st.info("👈 Налаштуйте параметри в сайдбарі та натисніть 'Run Simulation'")

if __name__ == "__main__":
    main()

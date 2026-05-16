import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings("ignore")

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EduPro – Instructor Performance Dashboard",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .block-container { padding-top: 1.5rem; }
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 18px 22px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 5px solid;
        margin-bottom: 10px;
    }
    .metric-value { font-size: 2rem; font-weight: 700; margin: 0; }
    .metric-label { font-size: 0.85rem; color: #666; margin: 0; }
    .section-title {
        font-size: 1.3rem; font-weight: 700;
        color: #1F3864; margin-bottom: 0.5rem;
        border-bottom: 2px solid #2E75B6;
        padding-bottom: 6px;
    }
    .insight-box {
        background: #EEF4FB;
        border-left: 4px solid #2E75B6;
        padding: 12px 16px;
        border-radius: 6px;
        margin: 8px 0;
        font-size: 0.92rem;
    }
    .stSelectbox label, .stSlider label, .stMultiSelect label {
        font-weight: 600; color: #1F3864;
    }
</style>
""", unsafe_allow_html=True)

# ── Load Data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    xl = pd.ExcelFile("EduPro_Online_Platform.xlsx")
    teachers    = pd.read_excel(xl, "Teachers")
    courses     = pd.read_excel(xl, "Courses")
    transactions= pd.read_excel(xl, "Transactions")
    merged = transactions.merge(teachers, on="TeacherID").merge(courses, on="CourseID")
    bins   = [0, 2, 3, 4, 5]
    labels = ["Low (0-2)", "Mid (2-3)", "Good (3-4)", "Top (4-5)"]
    teachers["RatingTier"] = pd.cut(teachers["TeacherRating"], bins=bins, labels=labels)
    merged["RatingTier"]   = pd.cut(merged["TeacherRating"],   bins=bins, labels=labels)
    return teachers, courses, transactions, merged

teachers, courses, transactions, merged = load_data()

PALETTE = {
    "Low (0-2)": "#e74c3c", "Mid (2-3)": "#f39c12",
    "Good (3-4)": "#3498db", "Top (4-5)": "#2ecc71"
}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/graduation-cap.png", width=60)
    st.title("EduPro Analytics")
    st.markdown("**Instructor Performance Dashboard**")
    st.markdown("---")

    st.markdown("### 🔍 Filters")

    all_expertise = sorted(teachers["Expertise"].unique())
    selected_expertise = st.multiselect(
        "Instructor Expertise",
        options=all_expertise,
        default=all_expertise,
        help="Filter by instructor expertise domain"
    )

    all_categories = sorted(courses["CourseCategory"].unique())
    selected_categories = st.multiselect(
        "Course Category",
        options=all_categories,
        default=all_categories,
        help="Filter by course category"
    )

    all_levels = sorted(courses["CourseLevel"].unique())
    selected_levels = st.multiselect(
        "Course Level",
        options=all_levels,
        default=all_levels
    )

    rating_range = st.slider(
        "Teacher Rating Range",
        min_value=0.0, max_value=5.0,
        value=(0.0, 5.0), step=0.1
    )

    gender_filter = st.multiselect(
        "Gender",
        options=["Male", "Female"],
        default=["Male", "Female"]
    )

    st.markdown("---")
    st.markdown("**👤 Author:** Uday Kumar")
    st.markdown("**🏫** Unified Mentor Internship")
    st.markdown("**📅** May 2026")

# ── Apply Filters ─────────────────────────────────────────────────────────────
t_filtered = teachers[
    (teachers["Expertise"].isin(selected_expertise)) &
    (teachers["TeacherRating"].between(*rating_range)) &
    (teachers["Gender"].isin(gender_filter))
]

c_filtered = courses[
    (courses["CourseCategory"].isin(selected_categories)) &
    (courses["CourseLevel"].isin(selected_levels))
]

m_filtered = merged[
    (merged["Expertise"].isin(selected_expertise)) &
    (merged["CourseCategory"].isin(selected_categories)) &
    (merged["CourseLevel"].isin(selected_levels)) &
    (merged["TeacherRating"].between(*rating_range)) &
    (merged["Gender"].isin(gender_filter))
]

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("## 📚 EduPro – Instructor Performance & Course Quality Dashboard")
st.markdown("*Data-driven evaluation of instructor effectiveness and course quality on the EduPro platform*")
st.markdown("---")

# ── KPI Cards ─────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5, k6 = st.columns(6)

kpis = [
    (k1, "Avg Teacher Rating", f"{t_filtered['TeacherRating'].mean():.2f}", "#3498db", "⭐"),
    (k2, "Avg Course Rating",  f"{c_filtered['CourseRating'].mean():.2f}",  "#2ecc71", "📘"),
    (k3, "Total Instructors",  f"{len(t_filtered)}",                         "#9b59b6", "👩‍🏫"),
    (k4, "Total Courses",      f"{len(c_filtered)}",                         "#e67e22", "📖"),
    (k5, "Total Enrollments",  f"{len(m_filtered):,}",                       "#1abc9c", "🎓"),
    (k6, "Exp-Rating Corr.",   f"{t_filtered['YearsOfExperience'].corr(t_filtered['TeacherRating']):.3f}", "#e74c3c", "📈"),
]

for col, label, value, color, icon in kpis:
    with col:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color:{color}">
            <p class="metric-value" style="color:{color}">{icon} {value}</p>
            <p class="metric-label">{label}</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# ── Tab Layout ────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏆 Leaderboard",
    "📊 Instructor Analysis",
    "📘 Course Quality",
    "🔗 Correlations",
    "💡 Insights"
])

# ══════════════════════════════════════════════════════════
# TAB 1 – LEADERBOARD
# ══════════════════════════════════════════════════════════
with tab1:
    st.markdown('<p class="section-title">🏆 Instructor Performance Leaderboard</p>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        leaderboard = t_filtered[["TeacherName","Expertise","Gender","YearsOfExperience","TeacherRating","RatingTier"]].copy()
        leaderboard = leaderboard.sort_values("TeacherRating", ascending=False).reset_index(drop=True)
        leaderboard.index += 1
        leaderboard.columns = ["Name","Expertise","Gender","Experience (yrs)","Rating","Tier"]

        def color_tier(val):
            colors = {"Low (0-2)":"#ffe0e0","Mid (2-3)":"#fff3cd","Good (3-4)":"#d0eaff","Top (4-5)":"#d4f5e2"}
            return f"background-color: {colors.get(val,'white')}"

        def color_rating(val):
            if val >= 4: return "color: #2ecc71; font-weight: bold"
            elif val >= 3: return "color: #3498db; font-weight: bold"
            elif val >= 2: return "color: #f39c12; font-weight: bold"
            else: return "color: #e74c3c; font-weight: bold"

        styled = leaderboard.style\
            .map(color_tier, subset=["Tier"])\
            .applymap(color_rating, subset=["Rating"])\
            .format({"Rating": "{:.2f}", "Experience (yrs)": "{:.0f}"})
        st.dataframe(styled, use_container_width=True, height=500)

    with col2:
        st.markdown("**🥇 Top 5 Instructors**")
        top5 = t_filtered.nlargest(5, "TeacherRating")[["TeacherName","Expertise","TeacherRating"]]
        for i, row in top5.iterrows():
            medal = ["🥇","🥈","🥉","4️⃣","5️⃣"][list(top5.index).index(i)]
            st.markdown(f"""
            <div class="insight-box">
                {medal} <b>{row['TeacherName']}</b><br>
                {row['Expertise']} | ⭐ {row['TeacherRating']:.2f}
            </div>
            """, unsafe_allow_html=True)

        st.markdown("**⚠️ Needs Attention**")
        bot3 = t_filtered.nsmallest(3, "TeacherRating")[["TeacherName","Expertise","TeacherRating"]]
        for _, row in bot3.iterrows():
            st.markdown(f"""
            <div style="background:#ffe0e0;border-left:4px solid #e74c3c;padding:10px 14px;border-radius:6px;margin:6px 0;font-size:0.9rem">
                🔴 <b>{row['TeacherName']}</b><br>
                {row['Expertise']} | ⭐ {row['TeacherRating']:.2f}
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# TAB 2 – INSTRUCTOR ANALYSIS
# ══════════════════════════════════════════════════════════
with tab2:
    st.markdown('<p class="section-title">📊 Instructor Profile Analysis</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # Rating Distribution
        fig, ax = plt.subplots(figsize=(7, 4))
        tier_counts = t_filtered["RatingTier"].value_counts().sort_index()
        colors = [PALETTE.get(str(t), "#aaa") for t in tier_counts.index]
        bars = ax.bar(tier_counts.index, tier_counts.values, color=colors, edgecolor="white", linewidth=1.5)
        for bar, val in zip(bars, tier_counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(val), ha="center", fontsize=11, fontweight="bold")
        ax.set_title("Instructor Rating Tier Distribution", fontweight="bold", pad=12)
        ax.set_xlabel("Rating Tier"); ax.set_ylabel("Count")
        ax.set_ylim(0, tier_counts.max() + 4)
        sns.despine()
        st.pyplot(fig, use_container_width=True)

    with col2:
        # Expertise avg rating
        fig, ax = plt.subplots(figsize=(7, 4))
        exp_rating = t_filtered.groupby("Expertise")["TeacherRating"].mean().sort_values()
        colors = ["#2ecc71" if v >= 3.5 else "#f39c12" if v >= 2.5 else "#e74c3c" for v in exp_rating.values]
        ax.barh(exp_rating.index, exp_rating.values, color=colors, edgecolor="white")
        ax.axvline(t_filtered["TeacherRating"].mean(), color="#e74c3c",
                   linestyle="--", linewidth=1.5, label=f"Avg={t_filtered['TeacherRating'].mean():.2f}")
        ax.set_title("Avg Teacher Rating by Expertise", fontweight="bold", pad=12)
        ax.set_xlabel("Avg Rating"); ax.legend(fontsize=9)
        sns.despine()
        st.pyplot(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        # Gender comparison
        fig, ax = plt.subplots(figsize=(7, 4))
        gender_data = t_filtered.groupby("Gender")["TeacherRating"].mean()
        colors = ["#e91e8c", "#3498db"][:len(gender_data)]
        bars = ax.bar(gender_data.index, gender_data.values, color=colors, edgecolor="white", width=0.4)
        for bar, val in zip(bars, gender_data.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                    f"{val:.2f}", ha="center", fontsize=12, fontweight="bold")
        ax.set_title("Avg Teacher Rating by Gender", fontweight="bold", pad=12)
        ax.set_ylabel("Avg Rating")
        ax.set_ylim(0, gender_data.max() + 0.6)
        sns.despine()
        st.pyplot(fig, use_container_width=True)

    with col4:
        # Rating histogram
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(t_filtered["TeacherRating"], bins=15, color="#3498db", edgecolor="white", alpha=0.85)
        ax.axvline(t_filtered["TeacherRating"].mean(), color="#e74c3c",
                   linestyle="--", linewidth=2, label=f"Mean={t_filtered['TeacherRating'].mean():.2f}")
        ax.set_title("Teacher Rating Distribution", fontweight="bold", pad=12)
        ax.set_xlabel("Rating"); ax.set_ylabel("Frequency"); ax.legend()
        sns.despine()
        st.pyplot(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════
# TAB 3 – COURSE QUALITY
# ══════════════════════════════════════════════════════════
with tab3:
    st.markdown('<p class="section-title">📘 Course Quality Evaluation</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # Course rating by category
        fig, ax = plt.subplots(figsize=(7, 5))
        cat_rating = c_filtered.groupby("CourseCategory")["CourseRating"].mean().sort_values()
        colors = ["#2ecc71" if v >= 3.5 else "#f39c12" if v >= 2.5 else "#e74c3c" for v in cat_rating.values]
        ax.barh(cat_rating.index, cat_rating.values, color=colors, edgecolor="white")
        ax.axvline(c_filtered["CourseRating"].mean(), color="#e74c3c",
                   linestyle="--", linewidth=1.5, label=f"Avg={c_filtered['CourseRating'].mean():.2f}")
        ax.set_title("Avg Course Rating by Category", fontweight="bold", pad=12)
        ax.set_xlabel("Avg Rating"); ax.legend(fontsize=9)
        sns.despine()
        st.pyplot(fig, use_container_width=True)

    with col2:
        # Course rating by level
        fig, ax = plt.subplots(figsize=(7, 5))
        level_rating = c_filtered.groupby("CourseLevel")["CourseRating"].mean().sort_values(ascending=False)
        colors = ["#3498db", "#2ecc71", "#e74c3c"][:len(level_rating)]
        bars = ax.bar(level_rating.index, level_rating.values, color=colors, edgecolor="white", linewidth=1.2)
        for bar, val in zip(bars, level_rating.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f"{val:.2f}", ha="center", fontsize=12, fontweight="bold")
        ax.axhline(c_filtered["CourseRating"].mean(), color="#e74c3c",
                   linestyle="--", linewidth=1.5, label=f"Avg={c_filtered['CourseRating'].mean():.2f}")
        ax.set_title("Avg Course Rating by Level", fontweight="bold", pad=12)
        ax.set_ylabel("Avg Rating"); ax.legend()
        ax.set_ylim(0, level_rating.max() + 0.5)
        sns.despine()
        st.pyplot(fig, use_container_width=True)

    # Enrollment by tier
    st.markdown('<p class="section-title">🎓 Enrollment by Instructor Rating Tier</p>', unsafe_allow_html=True)
    col3, col4 = st.columns(2)

    with col3:
        fig, ax = plt.subplots(figsize=(7, 4))
        tier_enroll = m_filtered["RatingTier"].value_counts().sort_index()
        colors = [PALETTE.get(str(t), "#aaa") for t in tier_enroll.index]
        bars = ax.bar(tier_enroll.index, tier_enroll.values, color=colors, edgecolor="white")
        for bar, val in zip(bars, tier_enroll.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                    f"{val:,}", ha="center", fontsize=10, fontweight="bold")
        ax.set_title("Enrollment Count by Teacher Tier", fontweight="bold", pad=12)
        ax.set_ylabel("Enrollments")
        ax.tick_params(axis='x', labelsize=8)
        sns.despine()
        st.pyplot(fig, use_container_width=True)

    with col4:
        # Pie chart enrollment share
        fig, ax = plt.subplots(figsize=(7, 4))
        tier_enroll2 = m_filtered["RatingTier"].value_counts().sort_index()
        colors2 = [PALETTE.get(str(t), "#aaa") for t in tier_enroll2.index]
        ax.pie(tier_enroll2.values, labels=tier_enroll2.index, colors=colors2,
               autopct="%1.1f%%", startangle=90, pctdistance=0.8,
               wedgeprops={"edgecolor":"white","linewidth":2})
        ax.set_title("Enrollment Share by Teacher Tier", fontweight="bold", pad=12)
        st.pyplot(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════
# TAB 4 – CORRELATIONS
# ══════════════════════════════════════════════════════════
with tab4:
    st.markdown('<p class="section-title">🔗 Correlation & Scatter Analysis</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # Experience vs Teacher Rating
        fig, ax = plt.subplots(figsize=(7, 5))
        scatter_colors = [PALETTE.get(str(t), "#aaa") for t in t_filtered["RatingTier"]]
        ax.scatter(t_filtered["YearsOfExperience"], t_filtered["TeacherRating"],
                   c=scatter_colors, alpha=0.85, s=90, edgecolors="white", linewidth=0.8)
        if len(t_filtered) > 1:
            m, b = np.polyfit(t_filtered["YearsOfExperience"], t_filtered["TeacherRating"], 1)
            x_line = np.linspace(t_filtered["YearsOfExperience"].min(),
                                 t_filtered["YearsOfExperience"].max(), 100)
            corr = t_filtered["YearsOfExperience"].corr(t_filtered["TeacherRating"])
            ax.plot(x_line, m*x_line + b, color="#e74c3c", linewidth=2,
                    linestyle="--", label=f"r = {corr:.3f}")
        ax.set_title("Experience vs Teacher Rating", fontweight="bold", pad=12)
        ax.set_xlabel("Years of Experience"); ax.set_ylabel("Teacher Rating")
        ax.legend(fontsize=10)
        sns.despine()
        st.pyplot(fig, use_container_width=True)

        corr_val = t_filtered["YearsOfExperience"].corr(t_filtered["TeacherRating"])
        st.markdown(f"""
        <div class="insight-box">
            📈 <b>Correlation: r = {corr_val:.3f}</b> — Moderate positive relationship.
            More experienced instructors tend to earn higher ratings.
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # Teacher Rating vs Course Rating
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(m_filtered["TeacherRating"], m_filtered["CourseRating"],
                   alpha=0.3, s=20, color="#3498db", edgecolors="none")
        if len(m_filtered) > 1:
            m2, b2 = np.polyfit(m_filtered["TeacherRating"], m_filtered["CourseRating"], 1)
            x2 = np.linspace(m_filtered["TeacherRating"].min(), m_filtered["TeacherRating"].max(), 100)
            corr2 = m_filtered["TeacherRating"].corr(m_filtered["CourseRating"])
            ax.plot(x2, m2*x2 + b2, color="#e74c3c", linewidth=2,
                    linestyle="--", label=f"r = {corr2:.3f}")
        ax.set_title("Teacher Rating vs Course Rating", fontweight="bold", pad=12)
        ax.set_xlabel("Teacher Rating"); ax.set_ylabel("Course Rating")
        ax.legend(fontsize=10)
        sns.despine()
        st.pyplot(fig, use_container_width=True)

        corr2_val = m_filtered["TeacherRating"].corr(m_filtered["CourseRating"])
        st.markdown(f"""
        <div class="insight-box">
            🔍 <b>Correlation: r = {corr2_val:.3f}</b> — Near-zero relationship.
            Instructor rating does NOT predict course quality — curriculum design matters more.
        </div>
        """, unsafe_allow_html=True)

    # Heatmap
    st.markdown('<p class="section-title">🗺️ Course Rating Heatmap (Category × Level)</p>', unsafe_allow_html=True)
    if len(m_filtered) > 0:
        heatmap_data = m_filtered.groupby(["CourseCategory", "CourseLevel"])["CourseRating"].mean().unstack()
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="RdYlGn",
                    linewidths=0.5, ax=ax, vmin=1, vmax=5,
                    cbar_kws={"label": "Avg Course Rating"})
        ax.set_title("Avg Course Rating: Category × Level", fontweight="bold", pad=12)
        ax.set_xlabel("Course Level"); ax.set_ylabel("Course Category")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════
# TAB 5 – INSIGHTS
# ══════════════════════════════════════════════════════════
with tab5:
    st.markdown('<p class="section-title">💡 Key Insights & Recommendations</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🔑 Key Findings")
        insights = [
            ("📈", "Experience matters (r=0.598)", "More experienced instructors consistently earn higher ratings — the strongest driver of teacher quality."),
            ("🔍", "Teacher ≠ Course Quality (r≈0)", "Near-zero correlation between instructor rating and course rating. Curriculum design drives course quality independently."),
            ("🎓", "Top-tier teachers dominate", "68% of all enrollments go to Top-tier instructors — only 18% of the teacher pool."),
            ("✅", "Marketing & ML lead quality", "Marketing (4.29) and Machine Learning (4.23) have the highest average instructor ratings."),
            ("⚠️", "Advanced courses underperform", "Advanced courses average 2.81 vs 3.32 for Intermediate — curriculum review needed."),
            ("❌", "9 instructors in Low tier", "15% of instructors rated below 2.0 — immediate mentoring intervention required."),
            ("👩‍🏫", "Female instructors outperform", "Female avg rating 3.23 vs male 3.04 — a consistent pattern across experience levels."),
        ]
        for icon, title, desc in insights:
            st.markdown(f"""
            <div class="insight-box">
                {icon} <b>{title}</b><br>
                <span style="color:#444">{desc}</span>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown("#### ✅ Recommendations")
        recs = [
            ("🚨", "Immediate (0-3 months)", [
                "Fast-track mentoring for 9 instructors rated below 2.0",
                "Conduct Advanced course audit across all categories",
                "Establish rating < 1.5 triggers a formal performance review"
            ]),
            ("📋", "Medium-term (3-6 months)", [
                "Structured onboarding curriculum for new instructors",
                "Curriculum Design Workshop for Business & Design categories",
                "Course Design Certification program for instructors"
            ]),
            ("🚀", "Strategic (6-12 months)", [
                "Build monthly Instructor Leaderboard for administrators",
                "Diversify top-tier talent — reduce over-dependence on 11 instructors",
                "Pilot Co-Teaching Model: top instructors mentor mid-tier teachers"
            ]),
        ]
        for icon, phase, items in recs:
            st.markdown(f"**{icon} {phase}**")
            for item in items:
                st.markdown(f"""
                <div style="background:white;border:1px solid #ddd;border-radius:6px;
                padding:8px 14px;margin:4px 0;font-size:0.88rem">
                    ▶ {item}
                </div>
                """, unsafe_allow_html=True)
            st.markdown("")

    st.markdown("---")
    st.markdown("""
    <div style="text-align:center;color:#888;font-size:0.85rem;padding:10px">
        📚 EduPro Instructor Performance Dashboard &nbsp;|&nbsp;
        👤 Uday Kumar &nbsp;|&nbsp;
        🏫 Unified Mentor Internship Program &nbsp;|&nbsp;
        📅 May 2026
    </div>
    """, unsafe_allow_html=True)

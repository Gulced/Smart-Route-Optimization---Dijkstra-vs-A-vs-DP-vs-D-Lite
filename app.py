import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import altair as alt
import random

from grid import GridMap
from dijkstra import dijkstra
from astar import astar
from dp_path import dp_shortest_path
from dstar_lite import DStarLite

from sort_algorithms import merge_sort, quick_sort

# =================================================
# STREAMLIT CONFIG
# =================================================
st.set_page_config(
    page_title="Smart Route Optimization",
    layout="wide"
)

st.title("🚗 Akıllı Ulaşımda En Kısa Rota")

tab1, tab2 = st.tabs([
    "🧭 Rota Planlama (Orijinal)",
    "⚔️ Algoritma Arenası: Sort & Search"
])

# =================================================
# SESSION STATE
# =================================================
if "gridmap" not in st.session_state:
    st.session_state.gridmap = None
if "path" not in st.session_state:
    st.session_state.path = None
if "runs" not in st.session_state:
    st.session_state.runs = []
if "dstar" not in st.session_state:
    st.session_state.dstar = None
if "synthetic_data" not in st.session_state:
    st.session_state.synthetic_data = []
if "sort_results" not in st.session_state:
    st.session_state.sort_results = None

# =================================================
# GRID DRAW
# =================================================
def draw_grid(gridmap, path=None):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(gridmap.grid, cmap="binary")

    r0, c0 = gridmap.start
    r1, c1 = gridmap.goal
    ax.scatter(c0, r0, c="green", s=60)
    ax.scatter(c1, r1, c="red", s=60)

    if path:
        xs = [c for (r, c) in path]
        ys = [r for (r, c) in path]
        ax.plot(xs, ys, c="blue", linewidth=2)

    ax.invert_yaxis()
    ax.set_title("Grid Haritası")
    ax.axis("off")
    st.pyplot(fig)

# =================================================
# TAB 1 — SENİN KODUN (SADECE GRID KÜÇÜK)
# =================================================
with tab1:
    st.sidebar.header("Grid Ayarları")

    n = st.sidebar.slider("Grid boyutu (n x n)", 10, 80, 30, step=2)
    obs = st.sidebar.slider("Engel oranı", 0.0, 0.6, 0.22, step=0.02)

    st.sidebar.markdown("---")

    algo = st.sidebar.radio(
        "Algoritma Seç",
        ["Dijkstra", "A*", "DP", "D* Lite"]
    )

    heuristic = st.sidebar.selectbox(
        "A* sezgisi",
        ["manhattan", "euclidean", "chebyshev"]
    )

    st.sidebar.markdown("---")
    btn_generate = st.sidebar.button("🧱 Grid Oluştur / Yenile")
    btn_run = st.sidebar.button("🏃‍♂️ Algoritmayı Çalıştır")
    btn_dynamic = st.sidebar.button("⚡ Dinamik Güncelle (D* Lite)")
    btn_clear_runs = st.sidebar.button("🗑 Tüm Run Sonuçlarını Temizle")

    if btn_generate:
        gm = GridMap(n, obs)
        gm.generate()
        st.session_state.gridmap = gm
        st.session_state.path = None
        st.session_state.dstar = None
        st.success("Yeni grid oluşturuldu")

    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        st.subheader("🗺 Grid ve Rota")

        if st.session_state.gridmap:
            draw_grid(st.session_state.gridmap, st.session_state.path)
        else:
            st.info("Önce grid oluştur")

        if btn_run and st.session_state.gridmap:
            gm = st.session_state.gridmap

            if algo == "Dijkstra":
                path, cost, expanded, runtime, _ = dijkstra(gm.grid, gm.start, gm.goal)
            elif algo == "A*":
                path, cost, expanded, runtime, _ = astar(
                    gm.grid, gm.start, gm.goal, heuristic
                )
            elif algo == "DP":
                path, cost, expanded, runtime, _ = dp_shortest_path(
                    gm.grid, gm.start, gm.goal
                )
            else:
                if st.session_state.dstar is None:
                    st.session_state.dstar = DStarLite(
                        gm.grid, gm.start, gm.goal
                    )
                path, cost, expanded, runtime, _ = st.session_state.dstar.find_path()

            if path:
                st.session_state.path = path
                st.session_state.runs.append({
                    "algo": algo,
                    "time_ms": round(runtime * 1000, 3),
                    "expanded": expanded,
                    "cost": cost
                })
                draw_grid(gm, path)
            else:
                st.warning("Yol bulunamadı")

    with col_right:
        st.subheader("📊 Run Sonuçları")

        if btn_clear_runs:
            st.session_state.runs = []

        if st.session_state.runs:
            df = pd.DataFrame(st.session_state.runs)
            st.dataframe(df, use_container_width=True)

# =================================================
# TAB 2 — ALGORİTMA ARENASI
# =================================================
with tab2:
    st.markdown("## ⚔️ Algoritma Arenası: Sort & Search")
    st.info("Bu bölümde sıralama ve arama algoritmaları kendi iç performanslarıyla analiz edilir.")

    st.sidebar.markdown("### 📊 Veri Ayarları (Tab 2)")
    data_source = st.sidebar.radio(
        "Veri Kaynağı:",
        ["Tab 1'den Gelen Runlar", "Sentetik Veri"],
        index=1
    )

    dataset = []

    if data_source == "Tab 1'den Gelen Runlar":
        if st.session_state.runs:
            dataset = st.session_state.runs
            st.success(f"Tab 1'den {len(dataset)} kayıt alındı.")
            sort_key = "cost"
        else:
            st.warning("Tab 1'de henüz run yok.")
    else:
        n_syn = st.sidebar.slider("Sentetik Veri Boyutu", 100, 5000, 1000, step=100)
        if st.sidebar.button("🎲 Rastgele Veri Üret"):
            st.session_state.synthetic_data = [
                {"id": i, "cost": random.randint(1, 10000), "time_ms": random.random() * 100}
                for i in range(n_syn)
            ]
        if st.session_state.synthetic_data:
            dataset = st.session_state.synthetic_data
            sort_key = st.sidebar.selectbox("Sıralama Anahtarı", ["cost", "time_ms"])
            st.success(f"{len(dataset)} sentetik veri hazır.")

    if dataset:
        st.markdown("---")
        st.subheader("1️⃣ Sorting – Algoritma İçi Performans")

        if st.button("🔥 Sort Analizini Başlat"):
            _, t_merge, ops_merge = merge_sort(dataset.copy(), sort_key)
            _, t_quick, ops_quick = quick_sort(dataset.copy(), sort_key)

            st.session_state.sort_results = {
                "Merge Sort": (t_merge, ops_merge),
                "Quick Sort": (t_quick, ops_quick)
            }

        if st.session_state.sort_results:
            res = st.session_state.sort_results

            c1, c2 = st.columns(2)
            with c1:
                st.metric("Merge Sort Süre (ms)", f"{res['Merge Sort'][0]:.3f}")
                st.metric("Merge Sort Op", res['Merge Sort'][1])
            with c2:
                st.metric("Quick Sort Süre (ms)", f"{res['Quick Sort'][0]:.3f}")
                st.metric("Quick Sort Op", res['Quick Sort'][1])

            chart_df = pd.DataFrame({
                "Algoritma": ["Merge Sort", "Quick Sort"],
                "Süre (ms)": [res["Merge Sort"][0], res["Quick Sort"][0]]
            })

            chart = alt.Chart(chart_df).mark_bar().encode(
                x="Algoritma",
                y="Süre (ms)",
                color=alt.Color(
                    "Algoritma",
                    scale=alt.Scale(
                        domain=["Merge Sort", "Quick Sort"],
                        range=["#1f77b4", "#d62728"]
                    ),
                    legend=None
                ),
                tooltip=["Algoritma", "Süre (ms)"]
            ).properties(height=300)

            st.altair_chart(chart, use_container_width=True)

        st.markdown("---")
        st.subheader("2️⃣ Search – Algoritma İçi Performans")

        def linear_search_perf(arr, target):
            steps = 0
            t0 = time.perf_counter()
            for i, v in enumerate(arr):
                steps += 1
                if v == target:
                    return i, steps, (time.perf_counter() - t0) * 1000
            return -1, steps, (time.perf_counter() - t0) * 1000

        def binary_search_perf(arr, target):
            low, high = 0, len(arr) - 1
            steps = 0
            t0 = time.perf_counter()
            while low <= high:
                steps += 1
                mid = (low + high) // 2
                if arr[mid] == target:
                    return mid, steps, (time.perf_counter() - t0) * 1000
                elif arr[mid] < target:
                    low = mid + 1
                else:
                    high = mid - 1
            return -1, steps, (time.perf_counter() - t0) * 1000

        values = [d[sort_key] for d in dataset]
        target = random.choice(values)

        if st.button("🔍 Search Analizini Başlat"):
            idx_l, steps_l, t_l = linear_search_perf(values, target)
            idx_b, steps_b, t_b = binary_search_perf(sorted(values), target)

            s1, s2 = st.columns(2)
            with s1:
                st.metric("Linear Search Süre (ms)", f"{t_l:.5f}")
                st.metric("Linear Search Adım", steps_l)
            with s2:
                st.metric("Binary Search Süre (ms)", f"{t_b:.5f}")
                st.metric("Binary Search Adım", steps_b)

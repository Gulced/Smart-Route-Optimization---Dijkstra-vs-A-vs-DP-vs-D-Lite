import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from grid import GridMap
from dijkstra import dijkstra
from astar import astar
from dp_path import dp_shortest_path
from dstar_lite import DStarLite

# -------------------------------------------------
# STREAMLIT PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Smart Route Optimization - Dijkstra vs A* vs DP vs D* Lite",
    layout="wide"
)

st.title("🚗 Akıllı Ulaşımda En Kısa Rota")
st.markdown(
    """
    **Dijkstra, A\*, Dinamik Programlama ve Dynamic A\* (D\* Lite) algoritmalarını**
    ızgara tabanlı (grid) bir şehir modelinde karşılaştırıyoruz.

    - Sol taraftan grid boyutu ve engel oranını ayarla  
    - Algoritmayı seç ve çalıştır  
    - Her run için: süre (ms), genişletilen düğüm sayısı, yol maliyeti kaydedilir  
    - Aşağıda grid + rota ve performans grafikleri gösterilir
    """
)

# -------------------------------------------------
# SESSION STATE INIT
# -------------------------------------------------
if "gridmap" not in st.session_state:
    st.session_state.gridmap = None

if "path" not in st.session_state:
    st.session_state.path = None

if "runs" not in st.session_state:
    st.session_state.runs = []  # her biri: dict(algo, time_ms, expanded, cost)

if "dstar" not in st.session_state:
    st.session_state.dstar = None

# -------------------------------------------------
# SIDEBAR - KONTROLLER
# -------------------------------------------------
st.sidebar.header("Grid Ayarları")

n = st.sidebar.slider("Grid boyutu (n x n)", min_value=10, max_value=80, value=30, step=2)
obs = st.sidebar.slider("Engel oranı", min_value=0.0, max_value=0.6, value=0.22, step=0.02)

st.sidebar.markdown("---")

algo = st.sidebar.radio(
    "Algoritma Seç",
    ["Dijkstra", "A*", "DP", "D* Lite"],
    help="Rota planlama algoritmasını seç."
)

heuristic = st.sidebar.selectbox(
    "A* sezgisi (heuristic)",
    ["manhattan", "euclidean", "chebyshev"],
    index=0,
    help="A* için kullanılacak sezgi fonksiyonu."
)

st.sidebar.markdown("---")
btn_generate = st.sidebar.button("🧱 Grid Oluştur / Yenile")
btn_run = st.sidebar.button("🏃‍♂️ Algoritmayı Çalıştır")
btn_dynamic = st.sidebar.button("⚡ Dinamik Güncelle (sadece D* Lite)")
btn_clear_runs = st.sidebar.button("🗑 Tüm Run Sonuçlarını Temizle")


# -------------------------------------------------
# YARDIMCI: GRID + PATH ÇİZİMİ
# -------------------------------------------------
def draw_grid(gridmap, path=None):
    fig, ax = plt.subplots(figsize=(6, 6))

    grid = gridmap.grid
    ax.imshow(grid, cmap="binary")

    # start / goal
    r0, c0 = gridmap.start
    r1, c1 = gridmap.goal
    ax.scatter(c0, r0, c="green", s=80, label="Start")
    ax.scatter(c1, r1, c="red", s=80, label="Goal")

    # path
    if path:
        xs = [c for (r, c) in path]
        ys = [r for (r, c) in path]
        ax.plot(xs, ys, c="blue", linewidth=2, label="Path")

    ax.set_title("Grid Haritası")
    ax.invert_yaxis()
    ax.legend(loc="upper right", fontsize=8)
    st.pyplot(fig)


# -------------------------------------------------
# 1) GRID OLUŞTURMA
# -------------------------------------------------
if btn_generate:
    gm = GridMap(n, obs)
    gm.generate()
    st.session_state.gridmap = gm
    st.session_state.path = None
    st.session_state.dstar = None  # dynamic planner reset
    st.success(f"Yeni grid oluşturuldu: {n}x{n}, engel oranı={obs:.2f}")

# -------------------------------------------------
# SOL/SAĞ KOLONLAR
# -------------------------------------------------
col_left, col_right = st.columns([1.2, 1])

# -------------------------------------------------
# SOL TARAF: GRID + RUN MESAJLARI
# -------------------------------------------------
with col_left:
    st.subheader("🗺 Grid ve Rota Görselleştirme")

    if st.session_state.gridmap is None:
        st.info("Önce sol taraftan **Grid Oluştur** butonuna bas.")
    else:
        draw_grid(st.session_state.gridmap, st.session_state.path)

    # -------------------------------------------------
    # 2) ALGORİTMAYI ÇALIŞTIR
    # -------------------------------------------------
    if btn_run:
        gm = st.session_state.gridmap
        if gm is None:
            st.error("Önce grid oluşturmalısın.")
        else:
            grid = gm.grid
            start = gm.start
            goal = gm.goal

            algo_name = algo  # string

            try:
                if algo == "Dijkstra":
                    path, cost, expanded, runtime, visited = dijkstra(grid, start, goal)

                elif algo == "A*":
                    path, cost, expanded, runtime, visited = astar(grid, start, goal, heuristic)

                elif algo == "DP":
                    path, cost, expanded, runtime, visited = dp_shortest_path(grid, start, goal)

                elif algo == "D* Lite":
                    if st.session_state.dstar is None:
                        st.session_state.dstar = DStarLite(grid, start, goal)
                    path, cost, expanded, runtime, updates = st.session_state.dstar.find_path()
                else:
                    st.error("Bilinmeyen algoritma.")
                    path = None

            except Exception as e:
                st.error(f"Algoritma çalışırken hata: {e}")
                path = None

            if path is None:
                st.warning(f"❌ {algo_name} bir yol bulamadı.")
            else:
                st.session_state.path = path
                time_ms = round(runtime * 1000, 3)
                st.success(
                    f"✅ {algo_name} yol buldu! "
                    f" Maliyet = {cost}, Genişletilen düğüm = {expanded}, Süre = {time_ms} ms"
                )
                # Run kaydı
                st.session_state.runs.append({
                    "algo": algo_name,
                    "heuristic": heuristic if algo == "A*" else "",
                    "time_ms": time_ms,
                    "expanded": expanded,
                    "cost": cost
                })
                # grid + path tekrar çiz
                draw_grid(gm, path)

    # Dinamik güncelleme (D* Lite)
    if btn_dynamic:
        gm = st.session_state.gridmap
        if gm is None:
            st.error("Önce grid oluştur ve en az bir kez D* Lite çalıştır.")
        elif algo != "D* Lite":
            st.info("Dinamik güncelleme sadece **D* Lite** için geçerli.")
        elif st.session_state.dstar is None:
            st.error("Önce D* Lite ile bir yol hesapla.")
        else:
            # Rastgele bir hücre seç, engel durumunu değiştir
            rows, cols = gm.grid.shape
            import random as _rand

            r = _rand.randint(0, rows - 1)
            c = _rand.randint(0, cols - 1)

            old = gm.grid[r, c]
            new_val = 1 - old  # 0->1 veya 1->0
            gm.grid[r, c] = new_val

            # D* Lite'a bildir
            try:
                updated = st.session_state.dstar.update_cell((r, c), new_val)
                path, cost, expanded, runtime, updates = st.session_state.dstar.find_path()
            except Exception as e:
                st.error(f"D* Lite güncelleme hatası: {e}")
                path = None

            if path is None:
                st.warning(f"Dinamik güncellemeden sonra yol kalmadı. Hücre: ({r},{c}), eski={old}, yeni={new_val}")
            else:
                st.session_state.path = path
                st.info(
                    f"Dinamik güncelleme: hücre ({r},{c}) {old} → {new_val}. "
                    f"Yeni yol maliyeti={cost}, genişletilen={expanded}."
                )
                draw_grid(gm, path)

# -------------------------------------------------
# SAĞ TARAF: RUN SONUÇLARI & GRAFİKLER
# -------------------------------------------------
with col_right:
    st.subheader("📊 Run Sonuçları")

    if btn_clear_runs:
        st.session_state.runs = []
        st.success("Kayıtlı tüm run sonuçları temizlendi.")

    runs = st.session_state.runs

    if not runs:
        st.info("Henüz hiç algoritma çalıştırılmadı.")
    else:
        # Tablo
        st.markdown("**Run Tablosu**")
        import pandas as pd

        df = pd.DataFrame(runs)
        st.dataframe(df, use_container_width=True)

        st.markdown("---")
        st.markdown("**Performans Karşılaştırma Grafikleri**")

        # Runtime grafiği
        fig1, ax1 = plt.subplots(figsize=(4, 3))
        ax1.bar(range(len(runs)), [r["time_ms"] for r in runs])
        ax1.set_xticks(range(len(runs)))
        labels = [f"{r['algo']}{' (' + r['heuristic'] + ')' if r['algo'] == 'A*' else ''}" for r in runs]
        ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax1.set_ylabel("Süre (ms)")
        ax1.set_title("Çalışma Süresi")
        st.pyplot(fig1)

        # Expanded nodes grafiği
        fig2, ax2 = plt.subplots(figsize=(4, 3))
        ax2.bar(range(len(runs)), [r["expanded"] for r in runs], color="orange")
        ax2.set_xticks(range(len(runs)))
        ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax2.set_ylabel("Genişletilen Düğüm")
        ax2.set_title("Arama Uzayı (Expanded Nodes)")
        st.pyplot(fig2)

        # Cost grafiği
        fig3, ax3 = plt.subplots(figsize=(4, 3))
        ax3.bar(range(len(runs)), [r["cost"] for r in runs], color="green")
        ax3.set_xticks(range(len(runs)))
        ax3.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax3.set_ylabel("Yol Maliyeti (adım sayısı)")
        ax3.set_title("Yol Maliyeti Karşılaştırması")
        st.pyplot(fig3)

        st.markdown(
            """
            - **Süre (ms)**: Algoritmanın çalışma süresi  
            - **Expanded Nodes**: Ziyaret edilen / genişletilen düğüm sayısı  
            - **Cost**: Bulunan en kısa yolun adım sayısı (grid üzerinde)
            """
        )

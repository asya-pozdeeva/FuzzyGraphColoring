import streamlit as st
import pandas as pd
import numpy as np
import math
import uuid
import streamlit.components.v1 as components
from pyvis.network import Network
import matplotlib.colors as mcolors

from fuzzilib import FuzzyGraph, Node, FuzzyValue
from graph_algorithms import (
    get_heuristic_chromatic_number, 
    get_exact_chromatic_number,
    get_heuristic_chromatic_number_with_constraints,
    get_exact_chromatic_number_with_constraints
)
from genetic_graph_algorithm import get_genetic_chromatic_number_with_constraints

# Настройка страницы
st.set_page_config(layout="wide", page_title="Назначение радиочастот")

# ---------------------------- #
# Инициализация State
# ---------------------------- #
if "transmitter_nodes" not in st.session_state:
    st.session_state.transmitter_nodes = []

if "generated_edges" not in st.session_state:
    st.session_state.generated_edges = []

if "node_colors" not in st.session_state:
    st.session_state.node_colors = {}
    
if "fuzzy_graph" not in st.session_state:
    st.session_state.fuzzy_graph = FuzzyGraph()

# ---------------------------- #
# Вспомогательные функции
# ---------------------------- #
def reset_calculations():
    """Сброс рассчитанных ребер и цветов при изменении узлов."""
    st.session_state.generated_edges = []
    st.session_state.node_colors = {}
    
import matplotlib as mpl

def generate_pastel_colors(n):
    """Генерация контрастных пастельных цветов для раскраски вершин."""
    colors = []
    cmap = mpl.colormaps['hsv']
    for i in range(n):
        rgba = cmap(i / max(n, 1))
        hex_col = mcolors.to_hex(rgba)
        colors.append(hex_col)
    return colors

# ---------------------------- #
# Сайдбар: Ввод данных
# ---------------------------- #
st.sidebar.header("1. Передатчики (Узлы)")

add_mode = st.sidebar.radio("Способ добавления:", ["Ручной ввод", "Загрузка CSV"])

if add_mode == "Ручной ввод":
    with st.sidebar.form("add_node_form"):
        t_name = st.text_input("Название", value=f"T{len(st.session_state.transmitter_nodes)+1}")
        col1, col2 = st.columns(2)
        t_x = col1.number_input("Координата X", value=0.0)
        t_y = col2.number_input("Координата Y", value=0.0)
        t_power = st.number_input("Мощность (Вт)", value=100.0, min_value=0.1)
        t_radius = st.number_input("Радиус покрытия", value=50.0, min_value=1.0)
        t_forbidden = st.text_input("Запрещенные частоты (через запятую)", value="", placeholder="Например: 1, 3")
        
        submitted = st.form_submit_button("Добавить передатчик")
        if submitted:
            if any(n['name'] == t_name for n in st.session_state.transmitter_nodes):
                st.sidebar.error("Передатчик с таким именем уже существует!")
            else:
                st.session_state.transmitter_nodes.append({
                    "name": t_name, "x": t_x, "y": t_y, "power": t_power, "radius": t_radius, "forbidden": t_forbidden
                })
                reset_calculations()
                st.sidebar.success(f"Добавлен {t_name}")

elif add_mode == "Загрузка CSV":
    uploaded_file = st.sidebar.file_uploader("Загрузить CSV с передатчиками", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            expected_cols = {'Name', 'X', 'Y', 'Power', 'Radius'}
            if not expected_cols.issubset(set(df.columns)):
                st.sidebar.error(f"Файл должен содержать столбцы: {', '.join(expected_cols)}")
            else:
                if st.sidebar.button("Загрузить данные"):
                    nodes_list = []
                    for _, row in df.iterrows():
                        nodes_list.append({
                            "name": str(row['Name']),
                            "x": float(row['X']),
                            "y": float(row['Y']),
                            "power": float(row['Power']),
                            "radius": float(row['Radius']),
                            "forbidden": str(row['Forbidden']) if 'Forbidden' in row else ""
                        })
                    st.session_state.transmitter_nodes = nodes_list
                    reset_calculations()
                    st.sidebar.success("Данные успешно загружены!")
        except Exception as e:
            st.sidebar.error(f"Ошибка чтения файла: {str(e)}")

if st.session_state.transmitter_nodes:
    with st.sidebar.expander("Текущий список передатчиков (Редактируемый)", expanded=False):
        df_nodes = pd.DataFrame(st.session_state.transmitter_nodes)
        edited_df = st.data_editor(df_nodes, use_container_width=True, num_rows="dynamic", key="nodes_editor")
        
        col_b1, col_b2 = st.columns(2)
        with col_b1:
            if st.button("Сохранить"):
                st.session_state.transmitter_nodes = edited_df.to_dict('records')
                reset_calculations()
                st.rerun()
        with col_b2:
            if st.button("Очистить все"):
                st.session_state.transmitter_nodes = []
                reset_calculations()
                st.rerun()

# ---------------------------- #
# Сайдбар: Расчет связей (ребер)
# ---------------------------- #
st.sidebar.header("2. Генерация связей (Интерференция)")

calc_method = st.sidebar.selectbox(
    "Метод расчета нечеткости ребер:",
    ["По расстоянию", "По перекрытию радиусов покрытия"]
)

if st.sidebar.button("Сгенерировать связи", type="primary"):
    if len(st.session_state.transmitter_nodes) < 2:
        st.sidebar.warning("Нужно минимум 2 передатчика!")
    else:
        graph = FuzzyGraph()
        # Добавляем вершины в граф
        for nd in st.session_state.transmitter_nodes:
            n = Node(name=nd['name'], x=nd['x'], y=nd['y'], power=nd['power'], radius=nd['radius'])
            graph.add_node(n)
        
        edges = []
        nodes_list = list(graph.nodes.values())
        
        if calc_method == "По расстоянию":
            # Находим максимальное расстояние
            max_d = 0.001
            for i in range(len(nodes_list)):
                for j in range(i+1, len(nodes_list)):
                    d = math.hypot(nodes_list[i].x - nodes_list[j].x, nodes_list[i].y - nodes_list[j].y)
                    if d > max_d: max_d = d
            
            for i in range(len(nodes_list)):
                for j in range(i+1, len(nodes_list)):
                    n1, n2 = nodes_list[i], nodes_list[j]
                    d = math.hypot(n1.x - n2.x, n1.y - n2.y)
                    val = 1.0 - (d / max_d)
                    if val > 0:
                        graph.add_edge((n1.name, n2.name, val))
                        edges.append((n1.name, n2.name, val))
                        
        elif calc_method == "По перекрытию радиусов покрытия":
            for i in range(len(nodes_list)):
                for j in range(i+1, len(nodes_list)):
                    n1, n2 = nodes_list[i], nodes_list[j]
                    d = math.hypot(n1.x - n2.x, n1.y - n2.y)
                    # Если сумма радиусов больше расстояния, зоны пересекаются
                    sum_r = n1.radius + n2.radius
                    if sum_r > d:
                        val = 1.0 - (d / sum_r)
                        if val > 0:
                            graph.add_edge((n1.name, n2.name, val))
                            edges.append((n1.name, n2.name, val))
                            
        st.session_state.generated_edges = edges
        st.session_state.fuzzy_graph = graph
        st.session_state.node_colors = {}
        st.sidebar.success(f"Сгенерировано {len(edges)} связей!")


# ---------------------------- #
# Сайдбар: Раскраска графа
# ---------------------------- #
st.sidebar.header("3. Назначение частот")

alpha_cut = st.sidebar.slider("Порог нечеткости (α-срез):", 0.0, 1.0, 0.0, 0.05)
algo_name = st.sidebar.selectbox("Алгоритм раскраски:", ["Уэлш-Пауэлл (Эвристический)", "Точный (Backtracking)", "Генетический алгоритм"])

if st.sidebar.button("Назначить частоты", type="primary"):
    if not st.session_state.transmitter_nodes:
        st.sidebar.warning("Список вершин пуст!")
    elif not st.session_state.generated_edges:
        st.sidebar.warning("Сначала сгенерируйте связи!")
    else:
        graph = st.session_state.fuzzy_graph
        # Получаем четкий граф по альфа-срезу
        crisp_graph = graph.get_alpha_cut_subgraph(alpha_cut)
        
        # Получаем матрицу смежности (crisp_graph.get_adjacency_matrix возвращает (матрица, список имен))
        adj_matrix, node_names = crisp_graph.get_adjacency_matrix()
        # Собираем ограничения для графа
        forbidden_dict = {}
        for idx, name in enumerate(node_names):
            node_info = next((n for n in st.session_state.transmitter_nodes if n['name'] == name), None)
            if node_info and 'forbidden' in node_info and str(node_info['forbidden']).strip() and str(node_info['forbidden']).lower() != 'nan':
                try:
                    f_colors = {int(x.strip()) for x in str(node_info['forbidden']).split(",") if x.strip().isdigit()}
                    if f_colors:
                        forbidden_dict[idx] = f_colors
                except Exception:
                    pass

        if algo_name == "Уэлш-Пауэлл (Эвристический)":
            num_colors, color_dict = get_heuristic_chromatic_number_with_constraints(adj_matrix, forbidden_dict)
        elif algo_name == "Точный (Backtracking)":
            num_colors, color_dict = get_exact_chromatic_number_with_constraints(adj_matrix, forbidden_dict)
        elif algo_name == "Генетический алгоритм":
            num_colors, color_dict = get_genetic_chromatic_number_with_constraints(adj_matrix, forbidden_dict)

        # Сохраняем цвета
        colors_palette = generate_pastel_colors(max(num_colors, 1))
        node_colors = {}
        
        # Индексы в color_dict соответствуют индексами в node_names
        for idx_vertex, c_id in color_dict.items():
            if c_id - 1 < len(colors_palette):
                color_hex = colors_palette[c_id - 1]
            else:
                color_hex = "#cccccc" # Фолбэк на серый
            node_colors[node_names[idx_vertex]] = {
                "color_id": c_id,
                "hex": color_hex
            }
            
        st.session_state.node_colors = node_colors
        st.sidebar.success(f"Частоты назначены! Использовано частот: {num_colors}")


# ---------------------------- #
# Основной экран: Визуализация
# ---------------------------- #
st.title("Назначение радиочастот")

# Статистика
col1, col2, col3 = st.columns(3)
col1.metric("Всего передатчиков", len(st.session_state.transmitter_nodes))
col2.metric("Связей (интерференция)", len(st.session_state.generated_edges))

if st.session_state.node_colors:
    unique_colors = set(v["color_id"] for v in st.session_state.node_colors.values())
    
    # Ищем изолированные вершины - это вершины, у которых нет конфликтов
    # В контексте альфа-среза это те вершины, у которых степень в четком графе = 0.
    crisp_g = st.session_state.fuzzy_graph.get_alpha_cut_subgraph(alpha_cut)
    isolated_count = 0
    adj, names = crisp_g.get_adjacency_matrix()
    if len(names) > 0:
        degrees = np.sum(adj > 0, axis=1)
        isolated_count = int(np.sum(degrees == 0))

    col3.metric("Использовано частот", len(unique_colors))
    st.info(f"Количество изолированных (без ограничений) передатчиков: **{isolated_count}**", icon="ℹ️")

# Отрисовка PyVis
if st.session_state.transmitter_nodes:
    cols = st.columns([2, 1, 1])
    graph_mode = cols[0].radio("Режим графа:", ["По координатной сетке (Статичный)", "Интерактивный (Пружинная модель)"], horizontal=True, label_visibility="collapsed")
    is_fixed = (graph_mode == "По координатной сетке (Статичный)")
    
    show_radius = cols[1].checkbox("Отображать радиусы узлов", value=False)
    show_grid = cols[2].checkbox("Координатная сетка", value=True) if is_fixed else False

    net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="#333333")
    
    # Отключаем или включаем физику в зависимости от режима
    net.toggle_physics(not is_fixed)

    # Добавляем узлы
    for nd in st.session_state.transmitter_nodes:
        n_name = nd['name']
        n_x, n_y = nd['x'], nd['y']
        
        # Настраиваем tooltip
        color_info = ""
        n_color = "#97C2FC" # Default blue
        
        if n_name in st.session_state.node_colors:
            c_data = st.session_state.node_colors[n_name]
            color_info = f"\nЧастота (Цвет): {c_data['color_id']}"
            n_color = c_data["hex"]
            
        title = f"Передатчик: {n_name}\nПозиция: ({n_x}, {n_y})\nМощность: {nd['power']}\nРадиус: {nd['radius']}{color_info}"
        
        # Чтобы граф не расплывался слишком сильно или мелко, масштабируем визуальные координаты
        # (Отрисовка Pyvis чувствительна к диапазону, мы используем множитель 5 для наглядности)
        scaled_x = n_x * 5
        scaled_y = -n_y * 5  # y инверсирован в pyvis (canvas)
        
        # Устанавливаем размер узла
        n_size = nd['radius'] * 5 if show_radius else 20
        
        # Если включены радиусы, делаем внутреннюю заливку узла почти прозрачной
        if show_radius:
            rgb = mcolors.to_rgb(n_color)
            r, g, b = int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255)
            node_color_prop = {
                "background": f"rgba({r}, {g}, {b}, 0.15)",
                "border": f"rgba({r}, {g}, {b}, 1.0)",
                "highlight": {
                    "background": f"rgba({r}, {g}, {b}, 0.4)",
                    "border": f"rgba({r}, {g}, {b}, 1.0)"
                }
            }
        else:
            node_color_prop = n_color
        
        if is_fixed:
            net.add_node(n_name, label=n_name, title=title, x=scaled_x, y=scaled_y, color=node_color_prop, shape="dot", size=n_size, fixed=True)
        else:
            net.add_node(n_name, label=n_name, title=title, color=node_color_prop, shape="dot", size=n_size)
        
    # Добавляем ребра
    for edge in st.session_state.generated_edges:
        n1, n2, val = edge
        if val > 0: # Отрисовываем вообще все сгенерированные, либо те, которые прошли срез?
            # В задаче: "толщина или прозрачность зависит от нечеткого значения"
            # Отображаем оригинальный граф интерференции. Альфа-срез влияет только на раскраску.
            width = val * 5.0  # Масштабируем толщину (0 - 5px)
            # Прозрачность rgba
            alpha_hex = hex(int(val * 255))[2:].zfill(2)
            edge_color = f"#888888{alpha_hex}"
            
            # Подсветка связей, если они не проходят alpha_cut: делаем их полупрозрачными/пунктирными (opationally)
            if val < alpha_cut and st.session_state.node_colors:
                edge_color = "#e0e0e0"
                width = 1.0

            net.add_edge(n1, n2, value=val, title=f"Интерференция: {val:.2f}", width=width, color=edge_color)
            
    # Сохраняем граф
    drag_nodes = "false" if is_fixed else "true"
    net.set_options(f"""
    {{
      "nodes": {{
        "borderWidth": 2,
        "borderWidthSelected": 4
      }},
      "edges": {{
        "smooth": {{
          "type": "continuous",
          "roundness": 0.5
        }}
      }},
      "interaction": {{
        "hover": true,
        "zoomView": true,
        "dragView": true,
        "dragNodes": {drag_nodes}
      }}
    }}
    """)
    
    html_path = "pyvis_graph.html"
    net.save_graph(html_path)
    
    with open(html_path, 'r', encoding='utf-8') as f:
        html_data = f.read()
        
    if is_fixed and show_grid:
        grid_js = """
        network.on("beforeDrawing", function(ctx) {
            ctx.strokeStyle = '#e8e8e8';
            ctx.lineWidth = 1;
            var step = 50 * 5; // шаг координатной сетки
            ctx.beginPath();
            for (var x = -10000; x <= 10000; x += step) {
                ctx.moveTo(x, -10000);
                ctx.lineTo(x, 10000);
            }
            for (var y = -10000; y <= 10000; y += step) {
                ctx.moveTo(-10000, y);
                ctx.lineTo(10000, y);
            }
            ctx.stroke();
        });
        """
        html_data = html_data.replace("return network;", f"{grid_js}\n            return network;")
        
    components.html(html_data, height=620)
else:
    st.info("Пожалуйста, добавьте передатчики в меню слева для начала работы.")


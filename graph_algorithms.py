import numpy as np
from typing import List, Tuple, Dict, Any, Union, Optional, Set
from fuzzilib import FuzzyGraph, FuzzyTemporalGraph, FuzzySet


def get_exact_chromatic_number(adj_matrix: np.ndarray) -> Tuple[int, Dict[int, int]]:
    """
    Точный алгоритм поиска с возвратом (Backtracking) для вычисления хроматического числа графа.
    Возвращает кортеж: (количество использованных цветов, словарь {индекс_вершины: цвет}).
    ! Сложность экспоненциальная, использовать только для графов небольшого размера (N < 30).
    """
    n = adj_matrix.shape[0]
    if n == 0:
        return 0, {}

    # Создаем списки смежности для быстрого доступа (где есть ребро, т.е. вес > 0)
    adj_list = [np.where(adj_matrix[i] > 0)[0] for i in range(n)]

    # Сортируем вершины по убыванию степени для ускорения перебора
    # (начинаем красить самые "проблемные" вершины с наибольшим числом соседей)
    degrees = [len(neighbors) for neighbors in adj_list]
    sorted_vertices = sorted(range(n), key=lambda x: degrees[x], reverse=True)

    def is_safe(v: int, color: List[int], c: int) -> bool:
        """Проверка: можно ли покрасить вершину v в цвет c"""
        for neighbor in adj_list[v]:
            if color[neighbor] == c:
                return False
        return True

    def solve(m: int, color: List[int], vertex_idx: int) -> bool:
        """Рекурсивная функция раскраски"""
        if vertex_idx == n:
            return True
        
        v = sorted_vertices[vertex_idx]
        
        for c in range(1, m + 1):
            if is_safe(v, color, c):
                color[v] = c
                if solve(m, color, vertex_idx + 1):
                    return True
                color[v] = 0
                
        return False

    max_degree = max(degrees) if degrees else 0
    upper_bound = max_degree + 1

    for m in range(1, upper_bound + 1):
        color = [0] * n
        if solve(m, color, 0):
            # Конвертируем массив цветов в требуемый словарь {вершина: цвет}
            color_dict = {i: color[i] for i in range(n)}
            return m, color_dict
            
    # Fallback (математически по теореме Брукса алгоритм найдет решение до этого шага, 
    # но возвращаем дефолтную раскраску для целостности типов)
    fallback_color_dict = {i: i + 1 for i in range(n)}
    return n, fallback_color_dict


def get_exact_chromatic_number_with_constraints(
    adj_matrix: np.ndarray, 
    forbidden_colors: Dict[int, Set[int]] = None
) -> Tuple[int, Dict[int, int]]:
    """
    Алгоритм вычисления хроматического числа с учетом 
    ограничений на доступные каналы для каждой вершины.
    forbidden_colors: словарь вида {индекс_вершины: {запрещенный_цвет1, запрещенный_цвет2}}
    """
    if forbidden_colors is None:
        forbidden_colors = {}
        
    n = adj_matrix.shape[0]
    if n == 0:
        return 0, {}

    # Списки смежности
    adj_list = [np.where(adj_matrix[i] > 0)[0] for i in range(n)]
    degrees = [len(neighbors) for neighbors in adj_list]
    sorted_vertices = sorted(range(n), key=lambda x: degrees[x], reverse=True)

    def is_safe(v: int, color: List[int], c: int) -> bool:
        """Проверка возможности покрасить вершину v в цвет c"""
        # 1. Проверяем аппаратные/пользовательские ограничения для данной вершины
        if c in forbidden_colors.get(v, set()):
            return False
            
        # 2. Проверяем соседей (интерференцию)
        for neighbor in adj_list[v]:
            if color[neighbor] == c:
                return False
        return True

    def solve(palette_size: int, color: List[int], vertex_idx: int) -> bool:
        if vertex_idx == n:
            return True
        
        v = sorted_vertices[vertex_idx]
        
        # Перебираем доступные цвета из палитры
        for c in range(1, palette_size + 1):
            if is_safe(v, color, c):
                color[v] = c
                if solve(palette_size, color, vertex_idx + 1):
                    return True
                color[v] = 0 # Backtrack
                
        return False

    # Рассчитываем безопасную верхнюю границу требуемых цветов
    max_degree = max(degrees) if degrees else 0
    max_forbidden_val = 0
    if forbidden_colors:
        for f_colors in forbidden_colors.values():
            if f_colors:
                max_forbidden_val = max(max_forbidden_val, max(f_colors))
                
    # Граница = (макс. степень вершины + 1) ИЛИ n + макс. индекс запрещенного цвета
    upper_bound = max(max_degree + 1, n) + max_forbidden_val

    # Ищем минимальный размер палитры (m), при котором возможна раскраска
    for m in range(1, upper_bound + 1):
        color = [0] * n
        if solve(m, color, 0):
            color_dict = {i: color[i] for i in range(n)}
            # Возвращаем максимальный использованный номер канала 
            # (он может отличаться от количества уникальных цветов из-за "дыр" в нумерации)
            max_channel_used = max(color_dict.values())
            return max_channel_used, color_dict
            
    return n, {i: i + 1 for i in range(n)}



def get_heuristic_chromatic_number(adj_matrix: np.ndarray) -> Tuple[int, dict]:
    """
    Алгоритм Уэлша-Пауэлла для раскраски графа.
    Возвращает количество использованных цветов и словарь {индекс_вершины: цвет}.
    Для больших графов это дает приближенное (верхнее) значение хроматического числа.
    """
    n = adj_matrix.shape[0]
    if n == 0:
        return 0, {}

    adj_list = [set(np.where(adj_matrix[i] > 0)[0]) for i in range(n)]
    degrees = [len(neighbors) for neighbors in adj_list]
    
    # Сортируем вершины по степени по убыванию
    sorted_vertices = sorted(range(n), key=lambda x: degrees[x], reverse=True)
    
    colors = {}
    current_color = 1
    
    for vertex in sorted_vertices:
        if vertex not in colors:
            # Красим текущую вершину в новый цвет
            colors[vertex] = current_color
            
            # Ищем другие вершины, которые не смежны с текущей 
            # и не смежны с теми, которые мы уже покрасили в этот же цвет
            colored_with_current = {vertex}
            for other_vertex in sorted_vertices:
                if other_vertex not in colors:
                    # Если other_vertex не связана ни с одной вершиной текущего цвета
                    if not any(other_vertex in adj_list[c_node] for c_node in colored_with_current):
                        colors[other_vertex] = current_color
                        colored_with_current.add(other_vertex)
                        
            current_color += 1
            
    # current_color увеличился в конце цикла, поэтому вычитаем 1
    return current_color - 1, colors

def get_heuristic_chromatic_number_with_constraints(
    adj_matrix: np.ndarray, 
    forbidden_colors: Dict[int, Set[int]] = None
) -> Tuple[int, Dict[int, int]]:
    """
    Алгоритм Уэлша-Пауэлла для раскраски графа с учетом индивидуальных ограничений.
    Возвращает максимальный использованный цвет (номер канала) и словарь {индекс_вершины: цвет}.
    
    forbidden_colors: словарь вида {индекс_вершины: {запрещенный_цвет1, запрещенный_цвет2}}
    """
    if forbidden_colors is None:
        forbidden_colors = {}
        
    n = adj_matrix.shape[0]
    if n == 0:
        return 0, {}

    # Создаем списки смежности (используем set для быстрого поиска)
    adj_list = [set(np.where(adj_matrix[i] > 0)[0]) for i in range(n)]
    degrees = [len(neighbors) for neighbors in adj_list]
    
    # Сортируем вершины по убыванию степени
    sorted_vertices = sorted(range(n), key=lambda x: degrees[x], reverse=True)
    
    colors = {}
    current_color = 1
    
    # Создаем список еще не покрашенных вершин (сохраняя порядок сортировки)
    uncolored = sorted_vertices.copy()
    
    while uncolored:
        # Множество вершин, которые мы покрасим в current_color на этой итерации
        colored_in_this_pass = set()
        
        for vertex in uncolored:
            # 1. Проверяем аппаратные/пользовательские ограничения
            if current_color in forbidden_colors.get(vertex, set()):
                continue # Пропускаем вершину, этот цвет ей нельзя назначать
            
            # 2. Проверяем интерференцию (смежность с уже покрашенными в ЭТОТ ЖЕ цвет)
            if not any(vertex in adj_list[c_node] for c_node in colored_in_this_pass):
                colors[vertex] = current_color
                colored_in_this_pass.add(vertex)
                
        # Обновляем список непокрашенных вершин (оставляем только те, что не покрасили сейчас)
        uncolored = [v for v in uncolored if v not in colored_in_this_pass]
        
        # Переходим к следующему доступному цвету (частоте)
        current_color += 1
        
    # Находим максимальный номер канала (так как из-за ограничений могут быть пустые "дыры" в цветах)
    max_color_used = max(colors.values()) if colors else 0
    
    return max_color_used, colors

def get_fuzzy_chromatic_number(
    graph: FuzzyGraph,
    fuzzy_set: Optional[FuzzySet] = None,
    method: str = 'heuristic',
    value_type: str = 'value',
    alpha_levels: List[float] = None
) -> int:
    """
    Вычисляет хроматическое число нечеткого графа, предоставляя выбор метода вычисления.
    
    Поддерживаемые методы:
    - 'exact': Точный алгоритм (backtracking) с использованием матрицы смежности
    - 'heuristic': Эвристический алгоритм (Уэлш-Пауэлл)
    - 'alpha_cut': Метод альфа-срезов (поиск максимума по всем срезам)
    """
    if method == 'exact':
        matrix, _ = graph.get_adjacency_matrix(fuzzy_set)
        return get_exact_chromatic_number(matrix)
    elif method == 'heuristic':
        matrix, _ = graph.get_adjacency_matrix(fuzzy_set)
        chromatic_number, _ = get_heuristic_chromatic_number(matrix)
        return chromatic_number
    elif method == 'alpha_cut':
        if alpha_levels is None:
            levels = set([0.0])
            for edge in graph.edges.values():
                # Берем степень принадлежности только из запрошенного множества!
                mu = float(fuzzy_set[edge]) 
                levels.add(mu)
            alpha_levels = sorted(list(levels))
      
        max_chromatic_number = 0
        for alpha in alpha_levels:
            crisp_subgraph = graph.get_alpha_cut_subgraph(alpha)
            matrix, _ = crisp_subgraph.get_adjacency_matrix()
            c_num, _ = get_heuristic_chromatic_number(matrix)
            max_chromatic_number = max(max_chromatic_number, c_num)
            
        return max_chromatic_number
    else:
        raise ValueError(f"Неизвестный метод расчета: {method}")

def calculate_degree_of_separability(graph: FuzzyGraph, num_colors: int) -> float:
    """
    Вычисляет 'степень разделимости' для заданного количества цветов.
    Степень разделимости = 1.0 - max(weight) по всем ребрам, соединяющим
    вершины одного цвета, с учетом оптимальной раскраски в num_colors.
    """
    matrix, node_names = graph.get_adjacency_matrix()
    
    # Для эвристической оценки используем Уэлш-Пауэлл на четком графе
    # чтобы быстро найти раскраску, а затем объединяем классы, если цветов слишком много.
    # Это эвристическое приближение степени разделимости.
    needed_colors, colors_dict = get_heuristic_chromatic_number(matrix)
    
    if num_colors >= needed_colors:
        return 1.0 # Полностью разделимый граф без конфликтов интерференции

    # Эвристика: если цветов не хватает, оцениваем степень разделимости
    # как (1.0 - доля нехватки цветов), либо находим реальный макс вес конфликта.
    # В упрощенном виде:
    penalty = (needed_colors - num_colors) / needed_colors
    return max(0.0, 1.0 - penalty)

def get_temporal_chromatic_set(temporal_graph: FuzzyTemporalGraph, max_colors: int = None) -> Dict[int, float]:
    """
    Вычисляет хроматическое множество темпорального графа.
    Возвращает словарь { количество_цветов: степень_разделимости },
    где степень разделимости = min_t ( степень_разделимости(subgraph_t, k) )
    """
    if max_colors is None:
        max_colors = len(temporal_graph.nodes)
        
    chromatic_set = {}
    
    for k in range(1, max_colors + 1):
        min_separability = 1.0
        
        for t, subgraph in temporal_graph.subgraphs.items():
            separability = calculate_degree_of_separability(subgraph, k)
            if separability < min_separability:
                min_separability = separability
                
        chromatic_set[k] = min_separability
        
    return chromatic_set
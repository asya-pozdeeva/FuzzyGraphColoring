import numpy as np
import random
from typing import List, Tuple, Dict, Set

def get_genetic_chromatic_number_with_constraints(
    adj_matrix: np.ndarray,
    forbidden_colors: Dict[int, Set[int]] = None,
    pop_size: int = 100,
    generations: int = 400,
    mutation_rate: float = 0.2
) -> Tuple[int, Dict[int, int]]:
    """
    Генетический алгоритм для вычисления хроматического числа графа с учетом 
    ограничений на доступные каналы для каждой вершины.
    
    Алгоритм минимизирует функцию штрафа, состоящую из:
    - Конфликтов между соседними вершинами.
    - Назначений запрещенных (forbidden) цветов.
    - Общего количества используемых цветов (минимизация максимального номера цвета).
    """
    if forbidden_colors is None:
        forbidden_colors = {}
        
    n = adj_matrix.shape[0]
    if n == 0:
        return 0, {}

    # Строим список смежности
    adj_list = [np.where(adj_matrix[i] > 0)[0] for i in range(n)]
    degrees = [len(neighbors) for neighbors in adj_list]
    max_degree = max(degrees) if degrees else 0
    
    # Максимальный возможный цвет = макс. степень + кол-во запрещенных
    max_forbidden_val = 0
    if forbidden_colors:
        for f_colors in forbidden_colors.values():
            if f_colors:
                max_forbidden_val = max(max_forbidden_val, max(f_colors))
                
    upper_bound = max(max_degree + 1, n) + max_forbidden_val
    available_colors = list(range(1, upper_bound + 1))
    
    # Весовые коэффициенты целевой функции
    W_CONFLICT = 1000   # Штраф за две соседние вершины одного цвета
    W_FORBIDDEN = 1000  # Штраф за назначение запрещенного цвета
    W_COLORS = 1        # Штраф за размер палитры (стимул сжимать раскраску)
    
    def calculate_fitness(chromosome: List[int]) -> int:
        conflicts = 0
        forbidden_count = 0
        
        # 1. Конфликты интерференции
        for i in range(n):
            for neighbor in adj_list[i]:
                # Проверяем только neighbor > i чтобы не считать дважды
                if neighbor > i and chromosome[i] == chromosome[neighbor]:
                    conflicts += 1
                    
        # 2. Нарушение ограничений (запрещенные частоты)
        for i in range(n):
            if chromosome[i] in forbidden_colors.get(i, set()):
                forbidden_count += 1
                
        # 3. Штраф за макс. цвет
        max_color = max(chromosome)
        
        return conflicts * W_CONFLICT + forbidden_count * W_FORBIDDEN + max_color * W_COLORS

    def create_individual() -> List[int]:
        # Случайная особь (стараемся брать цвета из начала палитры)
        individual = []
        for i in range(n):
            forbidden = forbidden_colors.get(i, set())
            valid_colors = [c for c in available_colors if c not in forbidden]
            if not valid_colors:
                valid_colors = available_colors
            # Эвристика: чаще берем мелкие цвета
            pool = valid_colors[:max(max_degree, 3)]
            individual.append(random.choice(pool))
        return individual
        
    # Создание начальной популяции
    population = [create_individual() for _ in range(pop_size)]
    best_fitness = float('inf')
    best_chromosome = None
    
    for generation in range(generations):
        # Оцениваем популяцию
        fitness_scores = [calculate_fitness(ind) for ind in population]
        
        # Сохраняем лучшее решение
        min_fitness_idx = np.argmin(fitness_scores)
        if fitness_scores[min_fitness_idx] < best_fitness:
            best_fitness = fitness_scores[min_fitness_idx]
            best_chromosome = population[min_fitness_idx].copy()
                
        # Селекция (Турнирная)
        new_population = []
        # Элитизм (гарантируем выживание лучшей особи)
        new_population.append(best_chromosome.copy())
        
        while len(new_population) < pop_size:
            # Турнир 1
            i1, i2 = random.sample(range(pop_size), 2)
            parent1 = population[i1] if fitness_scores[i1] < fitness_scores[i2] else population[i2]
            
            # Турнир 2
            i3, i4 = random.sample(range(pop_size), 2)
            parent2 = population[i3] if fitness_scores[i3] < fitness_scores[i4] else population[i4]
            
            # Скрещивание (Равномерное - Uniform Crossover)
            child = []
            for i in range(n):
                if random.random() < 0.5:
                    child.append(parent1[i])
                else:
                    child.append(parent2[i])
                    
            # Мутация
            if random.random() < mutation_rate:
                # Выбираем случайную вершину
                mutate_idx = random.randint(0, n - 1)
                forbidden = forbidden_colors.get(mutate_idx, set())
                valid_colors = [c for c in available_colors if c not in forbidden]
                
                # Мутируем в случайный допустимый цвет из суженного пула
                if valid_colors:
                    pool = valid_colors[:max(best_chromosome[mutate_idx] + 2, 3)]
                    child[mutate_idx] = random.choice(pool)
                    
            new_population.append(child)
            
        population = new_population

    color_dict = {i: best_chromosome[i] for i in range(n)}
    max_used = max(best_chromosome)
    return max_used, color_dict

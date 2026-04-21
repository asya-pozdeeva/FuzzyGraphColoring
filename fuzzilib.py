from typing import Callable, Optional, Union, Dict, List, Any, Tuple
import numpy as np


class FuzzyValue(float):
    """Класс для представления нечеткого значения в диапазоне [0, 1]"""
    def __new__(cls, value):
        val = float(value)
        if not (0.0 <= val <= 1.0):
            raise ValueError(f"Нечеткое значение должно быть в диапазоне [0, 1], получено: {val}")
        return super().__new__(cls, val)

    def __and__(self, other):
        return FuzzyValue(min(self, other))

    def __or__(self, other):
        return FuzzyValue(max(self, other))

    def __invert__(self):
        return FuzzyValue(1.0 - self)

    def __repr__(self):
        return f"{self.__class__.__name__}({super().__repr__()})"

class MembershipFunction:
    """Обертка для функций принадлежности"""
    
    def __init__(self, name: str, func: Callable[[float], float]):
        self.name = name
        self.func = func
        
    def __call__(self, x: float) -> FuzzyValue:
        return FuzzyValue(self.func(x))
        
    def __str__(self):
        return f"Функция принадлежности: {self.name}"

    @staticmethod
    def create_triangular(name: str, a: float, b: float, c: float) -> 'MembershipFunction':
        """Создает треугольную функцию принадлежности (a, b, c)"""
        def tri_func(x: float) -> FuzzyValue:
            return FuzzyValue(np.maximum(0.0, np.minimum((x - a) / (b - a + 1e-9), (c - x) / (c - b + 1e-9))))
        return MembershipFunction(name, tri_func)

    # todo: add trapezoidal, gaussian, etc.

class FuzzyAccessor:
    def __init__(self, owner):
        self._owner = owner
        self._sets: Dict[str, 'FuzzySet'] = {}

    def _register(self, fuzzy_set: 'FuzzySet'):
        self._sets[fuzzy_set.name] = fuzzy_set

    def __getattr__(self, set_name: str) -> FuzzyValue:
        if set_name in self._sets:
            return self._sets[set_name][self._owner]
        raise AttributeError(f"Нечеткое множество не найдено '{set_name}'")

    def __getitem__(self, set_name: str) -> FuzzyValue:
        if set_name in self._sets:
            return self._sets[set_name][self._owner]
        raise KeyError(f"Нечеткое множество не найдено '{set_name}'")

    def get_sets(self) -> List['FuzzySet']:
        return list(self._sets.values())

    def __repr__(self):
        vals = {name: float(fs[self._owner]) for name, fs in self._sets.items()}
        return f"FuzzyValues({vals})"

class FuzzySet:
    def __init__(self, name: str, membership_func: MembershipFunction = None, attribute_name: str = None):
        """
        Инициализация нечеткого множества.
        :param name: Название множества.
        :param membership_func: Функция принадлежности (опционально).
        :param attribute_name: Название атрибута на основе которого считается степень принадлежности (опционально).
        !Объекты нечеткого множества должны быть Hashable
        """
        self.name = name
        self.membership_func = membership_func
        if attribute_name is not None and membership_func is None:
            raise ValueError("Нельзя задать attribute_name без membership_func.")
        self.attribute_name = attribute_name
        self._elements: Dict[Any, FuzzyValue] = {}
        self._elements_names: Dict[str, Any] = {}


    def add(self, obj: Any, crisp_value: Union[float, int] = None, degree: Union[float, int, FuzzyValue] = None):
        """Добавляет объект в множество
        :param obj: Объект, добавляемый в множество.
        :param crisp_value: Четкое значение атрибута объекта (опционально).
        :param degree: Степень принадлежности объекта к множеству (опционально).
        
        Если задан attribute_name, то crisp_value будет взят из атрибута объекта.
        Если задан crisp_value, то degree будет вычислен с помощью membership_func.
        Если не задано ни crisp_value, ни attribute_name, то degree должен быть передан явно.
        """
        if self.attribute_name is not None:
            if hasattr(obj, self.attribute_name):
                crisp_value = getattr(obj, self.attribute_name)
            else:
                raise ValueError(f"У объекта {obj} нет атрибута {self.attribute_name}")
            degree = self.membership_func(crisp_value)
        elif crisp_value is not None:
            if self.membership_func is None:
                raise ValueError(f"Не задана membership_func.")
            degree = self.membership_func(crisp_value)
        elif degree is None:
            raise ValueError(f"Не задана степень принадлежности.")
        else:
            degree = FuzzyValue(degree)

        self._elements[obj] = degree
        
        if hasattr(obj, "name"):
            self._elements_names[obj.name] = obj
        
        if hasattr(obj, 'fuzzy') and hasattr(obj.fuzzy, '_register'):
            obj.fuzzy._register(self)

    def __setitem__(self, key: Any, degree: float):
        """
        Позволяет добавлять элементы через индексацию: fuzzy_set[obj] = 0.8
        Если заданы attribute_name или membership_func, то нельзя задавать степень принадлежности вручную!
        """
        if self.attribute_name is not None:
            raise ValueError(f"Нельзя задать степень принадлежности вручную, так как задан attribute_name.")
        if self.membership_func is not None:
            raise ValueError(f"Нельзя задать степень принадлежности вручную, так как задана membership_func.")
        self.add(obj=key, degree=degree)

    def __getitem__(self, key: Any) -> FuzzyValue:
        """
        Получение степени принадлежности.
        Поддерживает поиск по самому объекту, по строке (совпадающей с object.name)
        или по объекту, у которого совпадает атрибут name.
        """
        if key in self._elements:
            return self._elements[key]
        
        if isinstance(key, str):
            obj = self._elements_names.get(key)
            return self._elements.get(obj, FuzzyValue(0.0)) if obj is not None else FuzzyValue(0.0)
        
        if hasattr(key, 'name'):
            obj = self._elements_names.get(key.name)
            return self._elements.get(obj, FuzzyValue(0.0)) if obj is not None else FuzzyValue(0.0)
        

    def __or__(self, other: 'FuzzySet') -> 'FuzzySet':
        """Объединение (Union): max(A(x), B(x))"""
        result = FuzzySet(f"({self.name} ∪ {other.name})")
        all_keys = set(self._elements.keys()) | set(other._elements.keys())
        
        for key in all_keys:
            result.add(key, max(self[key], other[key]))
        return result

    def __and__(self, other: 'FuzzySet') -> 'FuzzySet':
        """Пересечение (Intersection): min(A(x), B(x))"""
        result = FuzzySet(f"({self.name} ∩ {other.name})")
        all_keys = set(self._elements.keys()) | set(other._elements.keys())
        
        for key in all_keys:
            result.add(key, min(self[key], other[key]))
        return result

    def __invert__(self) -> 'FuzzySet':
        """Дополнение (Complement): 1 - A(x)"""
        result = FuzzySet(f"¬{self.name}")
        for key, val in self._elements.items():
            result.add(key, ~val)
        return result

    def __str__(self):
        elements_str = ", ".join([
            f"'{getattr(k, 'name', str(k))}': {v:.2f}" 
            for k, v in self._elements.items()
        ])
        return f"FuzzySet '{self.name}' = {{{elements_str}}}"

    def alpha_cut(self, alpha: float) -> List[Any]:
        result = []
        for key, val in self._elements.items():
            if val >= alpha:
                result.append(key)
        return result

class Node:
    """Класс для представления узла
        name - имя узла
        kwargs - дополнительные атрибуты
    """
    def __init__(self, name: str, **kwargs):
        self.name = name
        self.fuzzy = FuzzyAccessor(self)
        for key, value in kwargs.items():
            setattr(self, key, value)
        
    def __repr__(self):
        return f"Node('{self.name}')"

    def __str__(self):
        return f"Вершина {self.name}"

class Edge:
    """Класс для представления ребра
        node1 - первая вершина
        node2 - вторая вершина
        value - значение ребра по умолчанию
        kwargs - дополнительные атрибуты
    """
    def __init__(self, node1: Node, node2: Node, value: float = 0.0, **kwargs):
        self.nodes_key= frozenset({node1.name, node2.name})
        self.nodes = {node1, node2}
        self.value = value
        self.fuzzy = FuzzyAccessor(self)
        for key, val in kwargs.items():
            setattr(self, key, val)

    def get_opposite_node(self, node: Node):
        if node in self.nodes:
            remaining = list(self.nodes - {node})
            return remaining[0]
        else:
            print(f"Вершина {node} не соединена с этим ребром")

    def __eq__(self, other):
        if not isinstance(other, Edge):
            return NotImplemented
        return self.nodes_key == other.nodes_key
    
    def __hash__(self):
        return hash(self.nodes_key)
        
    
    def __repr__(self):
        names = [n.name for n in self.nodes]
        return f"Edge({names[0]}, {names[1]}): {self.value}"
    
    def __str__(self):
        names = [n.name for n in self.nodes]
        return f"Ребро ({names[0]}, {names[1]}): {self.value}"

class FuzzyGraph:
    """Класс для представления нечеткого графа.
        fuzzy_nodes_sets - список нечетких множеств (FuzzySet) для вершин
        fuzzy_edges_sets - список нечетких множеств (FuzzySet) для ребер
        nodes - список вершин (Node)
        edges - список ребер (Edge)
    """
    fuzzy_nodes_sets: Optional[List[FuzzySet]] = None
    fuzzy_edges_sets: Optional[List[FuzzySet]] = None
    
    def __init__(self,
                 fuzzy_nodes_sets: Optional[list[FuzzySet] | FuzzySet] = None, 
                 fuzzy_edges_sets: Optional[list[FuzzySet] | FuzzySet] = None, 
                 nodes: Optional[List[Union[str, Node]]] = None, 
                 edges: Optional[List[Union[tuple, Edge]]] = None):
        self.nodes: Dict[str, Node] = {}
        self.edges: Dict[frozenset, Edge] = {}
        
        # Список смежности Формат: { "имя_вершины_1": { "имя_соседа_1": Edge, "имя_соседа_2": Edge } }
        self.adj: Dict[str, Dict[str, Edge]] = {}

        if nodes is not None:
            self.add_nodes(nodes)
        if edges is not None:
            self.add_edges(edges)

        if fuzzy_nodes_sets is not None:
            self.fuzzy_nodes_sets = []
            if isinstance(fuzzy_nodes_sets, FuzzySet):
                self.fuzzy_nodes_sets.append(fuzzy_nodes_sets)
            elif isinstance(fuzzy_nodes_sets, list):
                for fuzzy_set in fuzzy_nodes_sets:
                    self.fuzzy_nodes_sets.append(fuzzy_set)
            else:
                raise ValueError("Неверный формат fuzzy_nodes_sets")

        if fuzzy_edges_sets is not None:
            self.fuzzy_edges_sets = []
            if isinstance(fuzzy_edges_sets, FuzzySet):
                self.fuzzy_edges_sets.append(fuzzy_edges_sets)
            elif isinstance(fuzzy_edges_sets, list):
                for fuzzy_set in fuzzy_edges_sets:
                    self.fuzzy_edges_sets.append(fuzzy_set)
            else:
                raise ValueError("Неверный формат fuzzy_edges_sets")
        

    def _fuzzify_sets_add(self, obj):
        # Реализовать логику если не задан attribute в fuzzy_set
        if isinstance(obj, Node) and self.fuzzy_nodes_sets is not None:
            for fuzzy_set in self.fuzzy_nodes_sets:
                fuzzy_set.add(obj)
        elif isinstance(obj, Edge) and self.fuzzy_edges_sets is not None:
            for fuzzy_set in self.fuzzy_edges_sets:
                fuzzy_set.add(obj)
        else:
            raise ValueError("Объект должен быть Node или Edge")

    def add_nodes(self, nodes: List[Union[str, Node]]):
        for node in nodes:
            self.add_node(node)

    def add_node(self, node: Union[str, Node]):
        if isinstance(node, str):
            new_node = Node(name=node)
        elif isinstance(node, Node):
            new_node = node
        else:
            raise ValueError("Узел должен быть строкой или объектом Node")

        if new_node.name not in self.nodes:
            if self.fuzzy_nodes_sets is not None:
                self._fuzzify_sets_add(new_node)
            self.nodes[new_node.name] = new_node
            self.adj[new_node.name] = {}
        else:
            print(f"Узел {new_node.name} уже существует")

    def add_edges(self, edges: List[Union[tuple[str, str, float], Edge]]):
        for edge in edges:
            self.add_edge(edge)

    def add_edge(self, edge: Union[Edge, tuple[str, str, float]]):
        if isinstance(edge, tuple) and len(edge) == 3:
            if edge[0] not in self.nodes:
                print(f"Вершина {edge[0]} не найдена в графе")
                return
            if edge[1] not in self.nodes:
                print(f"Вершина {edge[1]} не найдена в графе")
                return
            node1 = self.nodes[edge[0]]
            node2 = self.nodes[edge[1]]
            value = edge[2]
            new_edge = Edge(node1, node2, value)
        elif isinstance(edge, Edge):
            new_edge = edge
        else:
            raise ValueError("Ребро должно быть кортежем (u, v, w) или объектом Edge")
        if new_edge.nodes_key not in self.edges:
            if self.fuzzy_edges_sets is not None:
                self._fuzzify_sets_add(new_edge)
            self.edges[new_edge.nodes_key] = new_edge
            # Добавляем ребро в список смежности для обеих вершин
            n1_name, n2_name = list(new_edge.nodes_key)
            self.adj[n1_name][n2_name] = new_edge
            self.adj[n2_name][n1_name] = new_edge
        else:
            print(f"Ребро {new_edge} уже существует")
            
    def calculate_edges_by_condition(self, condition: Callable[[Node, Node], FuzzyValue], alpha: float = 0):
        """
        Вычисляет ребра неориентированного графа на основе условия.
        Очищает старые ребра перед расчетом.
        ! Сложность O(N^2)
        """
        self.edges.clear()
        node_list = list(self.nodes.values())
        n = len(node_list)

        for i in range(n):
            for j in range(i + 1, n):
                n1 = node_list[i]
                n2 = node_list[j]
                fuzzy_value = condition(n1, n2)
                if fuzzy_value.value > alpha:
                    self.add_edge((n1.name, n2.name, fuzzy_value))

        return f"Вычисленно ребер: {len(self.edges)}"

    def calculate_edges_threshold_matrix(self, threshold_matrix: np.ndarray, alpha: float = 0):
        """
        Быстрый метод для вычисления ребер на основе матрицы смежности.
        threshold_matrix - матрица NxN (например, расстояния или корреляции)
        """
        self.edges.clear()
        node_names = list(self.nodes.keys())
        rows, cols = np.where(np.triu(threshold_matrix > alpha, k=1))
        
        for r, c in zip(rows, cols):
            val = float(threshold_matrix[r, c])
            self.add_edge((node_names[r], node_names[c], val))
            
        return f"Вычисленно ребер: {len(self.edges)}"
    
    def get_adjacency_matrix(self, fuzzy_set: Union['FuzzySet', str, None] = None) -> Tuple[np.ndarray, List[str]]:
        """
        Строит матрицу смежности графа.
        :param fuzzy_set: Если задано, матрица строится по степеням принадлежности 
                          ребер к этому множеству. Если None — используются четкие значения (edge.value).
        :return: Кортеж (матрица_смежности, список_имен_вершин_по_порядку)
        """
        actual_fuzzy_set = None
        if fuzzy_set is not None:
            if isinstance(fuzzy_set, str):
                if self.fuzzy_edges_sets is None:
                    raise ValueError("В графе нет нечетких множеств для ребер.")
                for fs in self.fuzzy_edges_sets:
                    if fs.name == fuzzy_set:
                        actual_fuzzy_set = fs
                        break
                else:
                    raise ValueError(f"Нечеткое множество '{fuzzy_set}' не найдено.")
            elif hasattr(fuzzy_set, '__getitem__'):
                actual_fuzzy_set = fuzzy_set
            else:
                raise ValueError("fuzzy_set должен быть объектом FuzzySet или строкой")

        node_names = list(self.nodes.keys())
        node_to_idx = {name: idx for idx, name in enumerate(node_names)}
        n = len(node_names)
        matrix = np.zeros((n, n), dtype=float)

        for edge in self.edges.values():
            nodes_list = list(edge.nodes_key)
            if len(nodes_list) == 2:
                n1, n2 = nodes_list
            else:
                n1 = n2 = nodes_list[0]

            idx1, idx2 = node_to_idx[n1], node_to_idx[n2]
            if actual_fuzzy_set is not None:
                val = float(actual_fuzzy_set[edge])
            else:
                val = float(edge.value)

            matrix[idx1, idx2] = val
            matrix[idx2, idx1] = val

        return matrix, node_names

    def _get_node_name(self, node: Union[str, Node]) -> str:
        node_name = node.name if isinstance(node, Node) else node

        if not isinstance(node_name, str):
            raise ValueError("Вершина должна быть строкой или объектом Node")
        if node_name not in self.nodes: 
            raise ValueError(f"Вершина {node_name} не найдена в графе.")
        return node_name

    def get_edges_of_node(self, node: Union[str, Node]) -> List[Edge]:
        """Получение всех ребер вершины."""
        node_name = self._get_node_name(node)
        return list(self.adj[node_name].values())

    def get_neighbors(self, node: Union[str, Node]) -> List[Node]:
        """Получение всех вершин, соединенных с вершиной."""
        node_name = self._get_node_name(node)
        if node_name not in self.adj:
            return []
        return [self.nodes[neighbor_name] for neighbor_name in self.adj[node_name].keys()]
    
    def get_neighbors_with_alpha(self, node: Union[str, Node], fuzzy_set: Union[FuzzySet, str], alpha: float) -> List[Node]:
        """
        Получение всех вершин, соединенных с вершиной ребрами,
        степень принадлежности которых к заданному нечеткому множеству > alpha.
        """
        node_name = self._get_node_name(node)
        if node_name not in self.adj:
            return []
        
        if isinstance(fuzzy_set, str):
            fuzzy_set_name = fuzzy_set
            for fs in self.fuzzy_edges_sets:
                if fs.name == fuzzy_set_name:
                    fuzzy_set = fs
                    break
            else:
                raise ValueError("Нечеткое множество не найдено в графе")
        elif not isinstance(fuzzy_set, FuzzySet):
            raise ValueError("Нечеткое множество должно быть объектом FuzzySet или строкой")

        result = []
        for neighbor_name, edge in self.adj[node_name].items():
            if fuzzy_set[edge] > alpha:
                result.append(self.nodes[neighbor_name])             
        return result

    def get_node_sigma(self, node: Union[str, Node], aggregation_func: Callable[[List[float]], float] = min) -> float:
        """
        Вычисляет степень принадлежности вершины (sigma).
        Если есть списки нечетких множеств вершин (fuzzy_nodes_sets), 
        собирает все принадлежности и агрегирует их (по умолчанию min).
        Если множеств нет, возвращает 1.0 (четкая вершина).
        """
        node_name = self._get_node_name(node)
        n = self.nodes[node_name]
        
        if not self.fuzzy_nodes_sets:
            return 1.0
            
        degrees = [float(fs[n]) for fs in self.fuzzy_nodes_sets]
        if not degrees:
            return 1.0
            
        return float(aggregation_func(degrees))

    def is_strong_edge(self, edge: Edge, aggregation_func: Callable[[List[float]], float] = min) -> bool:
        """
        Проверяет, является ли ребро сильным.
        Ребро сильное, если: 0.5 * min(sigma(u), sigma(v)) <= mu(u, v)
        """
        nodes = list(edge.nodes)
        if len(nodes) != 2:
            return False
            
        u, v = nodes[0], nodes[1]
        sigma_u = self.get_node_sigma(u, aggregation_func)
        sigma_v = self.get_node_sigma(v, aggregation_func)
        
        # Получаем вес ребра
        mu = float(edge.value)
        if self.fuzzy_edges_sets:
             degrees = [float(fs[edge]) for fs in self.fuzzy_edges_sets]
             if degrees:
                 mu = max(degrees)
        
        return 0.5 * min(sigma_u, sigma_v) <= mu

    def get_alpha_cut_subgraph(self, alpha: float, fuzzy_set: Union['FuzzySet', str, None] = None) -> 'FuzzyGraph':
        """
        Возвращает четкий подграф (FuzzyGraph, где оставшиеся ребра имеют вес 1.0),
        оставляя только ребра со значением (mu) >= alpha.
        
        :param alpha: Порог отсечения (0.0 - 1.0)
        :param fuzzy_set: Объект FuzzySet или строка с именем множества. 
                          Если передано, срез строится ТОЛЬКО по этому множеству.
        """
        if alpha < 0 or alpha > 1:
            raise ValueError("Альфа должна быть в диапазоне [0, 1]")
            
        actual_fuzzy_set = None
        if fuzzy_set is not None:
            if isinstance(fuzzy_set, str):
                if self.fuzzy_edges_sets is None:
                    raise ValueError("В графе нет нечетких множеств для ребер.")
                for fs in self.fuzzy_edges_sets:
                    if fs.name == fuzzy_set:
                        actual_fuzzy_set = fs
                        break
                else:
                    raise ValueError(f"Нечеткое множество '{fuzzy_set}' не найдено.")
            elif hasattr(fuzzy_set, '__getitem__'):
                actual_fuzzy_set = fuzzy_set
            else:
                raise ValueError("fuzzy_set должен быть объектом FuzzySet или строкой")

        new_nodes = list(self.nodes.values())
        crisp_graph = FuzzyGraph(nodes=new_nodes)
        crisp_graph.fuzzy_nodes_sets = self.fuzzy_nodes_sets 
        
        for froz_key, edge in self.edges.items():
            nodes = list(edge.nodes)
            if len(nodes) == 2:
                u, v = nodes
            else:
                u = v = nodes[0]
            if actual_fuzzy_set is not None:
                mu = float(actual_fuzzy_set[edge])
            else:
                # Если множество не указанно берется наибольшее значение
                mu = float(edge.value)
                if self.fuzzy_edges_sets:
                     degrees = [float(fs[edge]) for fs in self.fuzzy_edges_sets]
                     if degrees:
                         mu = max(degrees)
                         
            # альфа-срез
            if mu >= alpha:
                crisp_graph.add_edge((u.name, v.name, 1.0))
                
        return crisp_graph

class FuzzyTemporalGraph:
    """
    Класс для представления нечеткого темпорального графа.
    Представляет собой упорядоченный во времени набор нечетких подграфов 
    с одним и тем же множеством вершин.
    """
    def __init__(self, time_steps: List[str], subgraphs: List[FuzzyGraph]):
        if len(time_steps) != len(subgraphs):
            raise ValueError("Количество временных шагов должно совпадать с количеством подграфов")
        
        self.time_steps = time_steps
        self.subgraphs = {t: g for t, g in zip(time_steps, subgraphs)}
        
        # Предполагаем, что множество вершин одинаково для всех подграфов
        if subgraphs:
            self.nodes = list(subgraphs[0].nodes.values())
        else:
            self.nodes = []
            
    def get_subgraph(self, t: str) -> FuzzyGraph:
        if t not in self.subgraphs:
            raise KeyError(f"Временной срез {t} не найден")
        return self.subgraphs[t]
        
    def __str__(self):
        return f"FuzzyTemporalGraph (Time steps: {len(self.time_steps)}, Nodes: {len(self.nodes)})"




if __name__ == "__main__":
    print("Hello from fuzzilib!")

    mf_1 = MembershipFunction.create_triangular('test mf 1', 50,75,100)
    mf_2 = MembershipFunction.create_triangular('test mf 2', 25,50,75)
    mf_3 = MembershipFunction.create_triangular('test mf 3', 0,25,50)
    
    mfs_1 = FuzzySet(name='high', membership_func=mf_1, attribute_name='value')
    mfs_2 = FuzzySet(name='medium', membership_func=mf_2, attribute_name='value')
    mfs_3 = FuzzySet(name='low', membership_func=mf_3, attribute_name='value')

    fuzzy_edges_sets = [mfs_1, mfs_2, mfs_3]

    nodes=['1','2','3','4','5','6','7','8','9','10']
    edges=[('1','2',52),('1','3',80), ('1','4',20), ('4','5',80)]

    graph = FuzzyGraph(fuzzy_edges_sets=fuzzy_edges_sets, nodes=nodes, edges=edges)

    # print(graph.edges)
    # print(graph.nodes)
    # print(graph.fuzzy_edges_attributes)

    print(graph.edges.values())
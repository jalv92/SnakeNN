# SnakeRL.py - Implementación recreada basada en el código original
import pygame
import numpy as np
import random
from collections import deque

# ------------------- Configuración del Juego -------------------
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
GRID_SIZE = 20
GRID_WIDTH = SCREEN_WIDTH // GRID_SIZE
GRID_HEIGHT = SCREEN_HEIGHT // GRID_SIZE
FPS = 10

# Colores
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
BRIGHT_GRAY = (230, 230, 230)
GRID_COLOR = (50, 50, 50)

# Flags para depuración
DEBUG_MODE = False

# Direcciones
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

class SnakeGame:
    """Juego Snake con laberintos y compatibilidad para RL"""
    def __init__(self, render=True, difficulty=2, num_workers=0):
        self.render_mode = render
        self.difficulty = difficulty  # 0: Sin laberinto, 1: Laberinto simple, 2: Laberinto complejo
        self.num_workers = num_workers  # Número de workers en background
        
        if self.render_mode:
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption('Snake RL')
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont(None, 24)
            
        self.steps_without_food = 0
        self.max_steps_without_food = GRID_WIDTH * GRID_HEIGHT * 2
        self.last_positions = deque(maxlen=10)
        self.reset()
    
    def close(self):
        """Cierra y libera recursos de Pygame"""
        if self.render_mode:
            pygame.quit()

    def reset(self):
        """Reinicia el juego"""
        self.starting_position = (GRID_WIDTH // 2, GRID_HEIGHT // 2)
        
        # Inicializar las paredes - estructura como matrices de booleanos
        self.vertical_walls = [[False for _ in range(GRID_WIDTH + 1)] for _ in range(GRID_HEIGHT)]
        self.horizontal_walls = [[False for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT + 1)]
        
        # Bordes del mapa siempre presentes
        for y in range(GRID_HEIGHT):
            self.vertical_walls[y][0] = True  # Borde izquierdo
            self.vertical_walls[y][GRID_WIDTH] = True  # Borde derecho
            
        for x in range(GRID_WIDTH):
            self.horizontal_walls[0][x] = True  # Borde superior
            self.horizontal_walls[GRID_HEIGHT][x] = True  # Borde inferior
        
        # Generar laberinto según dificultad
        if self.difficulty == 0:
            pass  # Sin laberinto adicional
        elif self.difficulty == 1:
            self._generate_simple_maze()
        else:
            self._generate_complex_maze()
        
        # Inicializar serpiente
        self.snake = [self.starting_position]
        self.direction = RIGHT
        self.food = self._place_food()
        self.score = 0
        self.game_over = False
        self.steps_without_food = 0
        self.last_positions.clear()
        self.last_positions.append(self.starting_position)
        
        # Para rastreo de exploración
        self.visited_cells = set()
        self.visited_cells.add(self.starting_position)
        
        # Estado para el algoritmo de aprendizaje
        return self._get_state()
    
    def _generate_simple_maze(self):
        """Genera un laberinto simple con 4 paredes grandes y visibles"""
        # 1. Pared vertical en el tercio izquierdo con un hueco
        third_x = GRID_WIDTH // 3
        for y in range(5, GRID_HEIGHT - 5):
            if y != GRID_HEIGHT // 2:  # Dejar un hueco en el medio
                self.vertical_walls[y][third_x] = True
        
        # 2. Pared vertical en el tercio derecho con un hueco
        two_third_x = 2 * GRID_WIDTH // 3
        for y in range(5, GRID_HEIGHT - 5):
            if y != GRID_HEIGHT // 3:  # Hueco en posición diferente
                self.vertical_walls[y][two_third_x] = True
        
        # 3. Pared horizontal en el tercio superior con un hueco
        third_y = GRID_HEIGHT // 3
        for x in range(5, GRID_WIDTH - 5):
            if x != GRID_WIDTH // 4:  # Hueco a un cuarto
                self.horizontal_walls[third_y][x] = True
        
        # 4. Pared horizontal en el tercio inferior con un hueco
        two_third_y = 2 * GRID_HEIGHT // 3
        for x in range(5, GRID_WIDTH - 5):
            if x != 3 * GRID_WIDTH // 4:  # Hueco a tres cuartos
                self.horizontal_walls[two_third_y][x] = True
    
    def _generate_complex_maze(self):
        """Genera un laberinto más complejo pero manejable"""
        # Crear estructura central en forma de cruz
        mid_x = GRID_WIDTH // 2
        mid_y = GRID_HEIGHT // 2
        
        # Cruz horizontal (con huecos)
        for x in range(5, GRID_WIDTH - 5):
            if x % 5 != 0:  # Dejar huecos cada 5 celdas
                self.horizontal_walls[mid_y][x] = True
        
        # Cruz vertical (con huecos)
        for y in range(5, GRID_HEIGHT - 5):
            if y % 5 != 0:  # Dejar huecos cada 5 celdas
                self.vertical_walls[y][mid_x] = True
        
        # Añadir 4 "habitaciones" en las esquinas
        room_size = 8
        
        # Habitación superior izquierda
        start_x = mid_x - 15
        start_y = mid_y - 15
        for x in range(start_x, start_x + room_size):
            self.horizontal_walls[start_y][x] = True
            self.horizontal_walls[start_y + room_size][x] = True
        for y in range(start_y, start_y + room_size):
            self.vertical_walls[y][start_x] = True
            self.vertical_walls[y][start_x + room_size] = True
        # Hacer una puerta
        self.horizontal_walls[start_y + room_size][start_x + room_size//2] = False
        
        # Habitación superior derecha
        start_x = mid_x + 7
        for x in range(start_x, start_x + room_size):
            self.horizontal_walls[start_y][x] = True
            self.horizontal_walls[start_y + room_size][x] = True
        for y in range(start_y, start_y + room_size):
            self.vertical_walls[y][start_x] = True
            self.vertical_walls[y][start_x + room_size] = True
        # Hacer una puerta
        self.vertical_walls[start_y + room_size//2][start_x] = False
        
        # Habitación inferior izquierda
        start_y = mid_y + 7
        start_x = mid_x - 15
        for x in range(start_x, start_x + room_size):
            self.horizontal_walls[start_y][x] = True
            self.horizontal_walls[start_y + room_size][x] = True
        for y in range(start_y, start_y + room_size):
            self.vertical_walls[y][start_x] = True
            self.vertical_walls[y][start_x + room_size] = True
        # Hacer una puerta
        self.vertical_walls[start_y + room_size//2][start_x + room_size] = False
        
        # Habitación inferior derecha
        start_x = mid_x + 7
        for x in range(start_x, start_x + room_size):
            self.horizontal_walls[start_y][x] = True
            self.horizontal_walls[start_y + room_size][x] = True
        for y in range(start_y, start_y + room_size):
            self.vertical_walls[y][start_x] = True
            self.vertical_walls[y][start_x + room_size] = True
        # Hacer una puerta
        self.horizontal_walls[start_y][start_x + room_size//2] = False
    
    def _place_food(self):
        """Coloca la comida en una posición accesible desde la serpiente"""
        queue = deque([self.snake[0]])
        visited = {self.snake[0]}
        
        while queue:
            cx, cy = queue.popleft()
            for dx, dy, check_wall in [
                (-1, 0, lambda x, y: self.vertical_walls[y][x]),
                (1, 0, lambda x, y: self.vertical_walls[y][x + 1]),
                (0, -1, lambda x, y: self.horizontal_walls[y][x]),
                (0, 1, lambda x, y: self.horizontal_walls[y + 1][x]),
            ]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT and not check_wall(cx, cy):
                    if (nx, ny) not in visited:
                        visited.add((nx, ny))
                        queue.append((nx, ny))
        
        free_cells = [cell for cell in visited if cell not in self.snake]
        return random.choice(free_cells) if free_cells else (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))
    
    def _get_state(self):
        """Devuelve el estado actual como un vector para el agente RL"""
        head_x, head_y = self.snake[0]
        
        # Detectar peligro en diferentes direcciones
        danger_straight = False
        danger_right = False
        danger_left = False
        
        # Determinar qué hay en cada dirección relativa a la orientación actual
        if self.direction == UP:
            # Peligro al frente (arriba)
            danger_straight = head_y <= 0 or self.horizontal_walls[head_y][head_x] or (head_x, head_y - 1) in self.snake[1:]
            # Peligro a la derecha
            danger_right = head_x >= GRID_WIDTH - 1 or self.vertical_walls[head_y][head_x + 1] or (head_x + 1, head_y) in self.snake[1:]
            # Peligro a la izquierda
            danger_left = head_x <= 0 or self.vertical_walls[head_y][head_x] or (head_x - 1, head_y) in self.snake[1:]
        
        elif self.direction == RIGHT:
            # Peligro al frente (derecha)
            danger_straight = head_x >= GRID_WIDTH - 1 or self.vertical_walls[head_y][head_x + 1] or (head_x + 1, head_y) in self.snake[1:]
            # Peligro a la derecha
            danger_right = head_y >= GRID_HEIGHT - 1 or self.horizontal_walls[head_y + 1][head_x] or (head_x, head_y + 1) in self.snake[1:]
            # Peligro a la izquierda
            danger_left = head_y <= 0 or self.horizontal_walls[head_y][head_x] or (head_x, head_y - 1) in self.snake[1:]
        
        elif self.direction == DOWN:
            # Peligro al frente (abajo)
            danger_straight = head_y >= GRID_HEIGHT - 1 or self.horizontal_walls[head_y + 1][head_x] or (head_x, head_y + 1) in self.snake[1:]
            # Peligro a la derecha
            danger_right = head_x <= 0 or self.vertical_walls[head_y][head_x] or (head_x - 1, head_y) in self.snake[1:]
            # Peligro a la izquierda
            danger_left = head_x >= GRID_WIDTH - 1 or self.vertical_walls[head_y][head_x + 1] or (head_x + 1, head_y) in self.snake[1:]
        
        elif self.direction == LEFT:
            # Peligro al frente (izquierda)
            danger_straight = head_x <= 0 or self.vertical_walls[head_y][head_x] or (head_x - 1, head_y) in self.snake[1:]
            # Peligro a la derecha
            danger_right = head_y <= 0 or self.horizontal_walls[head_y][head_x] or (head_x, head_y - 1) in self.snake[1:]
            # Peligro a la izquierda
            danger_left = head_y >= GRID_HEIGHT - 1 or self.horizontal_walls[head_y + 1][head_x] or (head_x, head_y + 1) in self.snake[1:]
        
        # Estado base 
        state = [
            int(danger_straight),
            int(danger_right),
            int(danger_left),
            int(self.direction == LEFT),
            int(self.direction == RIGHT),
            int(self.direction == UP),
            int(self.direction == DOWN),
            int(self.food[0] < head_x),  # Comida a la izquierda
            int(self.food[0] > head_x),  # Comida a la derecha
            int(self.food[1] < head_y),  # Comida arriba
            int(self.food[1] > head_y),  # Comida abajo
            self._manhattan_distance(self.snake[0], self.food) / (GRID_WIDTH + GRID_HEIGHT),  # Distancia normalizada
            len(self.snake) / 20,  # Longitud normalizada
            self.steps_without_food / self.max_steps_without_food  # Pasos sin comer normalizados
        ]
        
        # Visión en 8 direcciones (N, NE, E, SE, S, SW, W, NW)
        for dx, dy in [(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]:
            found_wall = False
            found_food = False
            found_body = False
            
            for dist in range(1, max(GRID_WIDTH, GRID_HEIGHT)):
                nx, ny = head_x + dx * dist, head_y + dy * dist
                
                # Si está fuera de los límites
                if not (0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT):
                    found_wall = True
                    break
                
                # Verificar paredes
                if (dx < 0 and self.vertical_walls[ny][nx]) or \
                   (dx > 0 and self.vertical_walls[ny][nx + 1]) or \
                   (dy < 0 and self.horizontal_walls[ny][nx]) or \
                   (dy > 0 and self.horizontal_walls[ny + 1][nx]):
                    found_wall = True
                    break
                
                # Verificar si hay comida
                if (nx, ny) == self.food:
                    found_food = True
                    break
                
                # Verificar si hay cuerpo de serpiente
                if (nx, ny) in self.snake:
                    found_body = True
                    break
            
            # Agregar información normalizada
            state.append(1.0 if found_wall else 0.0)
            state.append(1.0 if found_food else 0.0)
            state.append(1.0 if found_body else 0.0)
        
        return np.array(state, dtype=np.float32)
    
    def _manhattan_distance(self, point1, point2):
        """Calcula la distancia Manhattan entre dos puntos"""
        return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])
    
    def _detect_circles(self):
        """Detecta si la serpiente está moviéndose en círculos"""
        if len(self.last_positions) < 10:
            return False
        
        # Si la cabeza ha regresado a una posición previa reciente
        repeats = self.last_positions.count(self.snake[0])
        return repeats > 1
    
    def _is_in_corridor(self):
        """Detecta si la serpiente está en un pasillo (rodeada por paredes en dos lados opuestos)"""
        head_x, head_y = self.snake[0]
        
        # Verificar si hay paredes en dos lados opuestos (horizontal o vertical)
        horizontal_walls = 0
        vertical_walls = 0
        
        # Verificar paredes a la izquierda y derecha
        if head_x > 0 and self.vertical_walls[head_y][head_x]:
            horizontal_walls += 1
        if head_x < GRID_WIDTH and self.vertical_walls[head_y][head_x + 1]:
            horizontal_walls += 1
        
        # Verificar paredes arriba y abajo
        if head_y > 0 and self.horizontal_walls[head_y][head_x]:
            vertical_walls += 1
        if head_y < GRID_HEIGHT and self.horizontal_walls[head_y + 1][head_x]:
            vertical_walls += 1
        
        # Estamos en un pasillo si hay paredes en dos lados opuestos
        return horizontal_walls == 2 or vertical_walls == 2
    
    def _path_efficiency_ratio(self):
        """Calcula qué tan eficientemente se ha movido la serpiente hacia la comida"""
        if len(self.snake) <= 1:
            return 1.0
            
        # Distancia ideal (Manhattan) desde posición inicial a comida
        ideal_distance = self._manhattan_distance(self.starting_position, self.food)
        
        # Número de pasos tomados
        steps_taken = len(self.snake) - 1
        
        # Si ideal_distance es 0, estamos en la comida
        if ideal_distance == 0:
            return 1.0
            
        # Ratio de eficiencia: 1.0 es óptimo, menor es menos eficiente
        return min(1.0, ideal_distance / (steps_taken + 1))
    
    def step(self, action):
        """Ejecuta un paso del juego basado en la acción"""
        clock_wise = [UP, RIGHT, DOWN, LEFT]
        idx = clock_wise.index(self.direction)
        
        # Interpretación de acciones (0=recto, 1=izquierda, 2=derecha)
        if action == 0:  # Recto
            self.direction = clock_wise[idx]
        elif action == 1:  # Izquierda
            self.direction = clock_wise[(idx - 1) % 4]
        elif action == 2:  # Derecha
            self.direction = clock_wise[(idx + 1) % 4]
        
        # Posición actual y distancia a la comida
        prev_distance = self._manhattan_distance(self.snake[0], self.food)
        x, y = self.snake[0]
        
        # Mover según dirección
        if self.direction == UP:
            if self.horizontal_walls[y][x]:
                self.game_over = True
                return self._get_state(), -10, True
            y -= 1
        elif self.direction == DOWN:
            if self.horizontal_walls[y + 1][x]:
                self.game_over = True
                return self._get_state(), -10, True
            y += 1
        elif self.direction == LEFT:
            if self.vertical_walls[y][x]:
                self.game_over = True
                return self._get_state(), -10, True
            x -= 1
        elif self.direction == RIGHT:
            if self.vertical_walls[y][x + 1]:
                self.game_over = True
                return self._get_state(), -10, True
            x += 1
        
        # Nueva posición de la cabeza
        new_head = (x, y)
        self.snake.insert(0, new_head)
        self.last_positions.append(new_head)
        
        # Recompensa base
        reward = 0
        
        # Recompensar exploración de nuevas celdas
        if new_head not in self.visited_cells:
            self.visited_cells.add(new_head)
            reward += 0.05  # Pequeña recompensa por explorar
        
        # Verificar colisión con el cuerpo
        if new_head in self.snake[1:]:
            self.game_over = True
            return self._get_state(), -10, True
        
        # Si come comida
        if new_head == self.food:
            self.score += 1
            
            # Base 10 + bonificación por longitud (hasta +10 adicionales)
            reward = 10 + min(10, len(self.snake) // 3)
            
            # Bonificación por eficiencia
            efficiency = self._path_efficiency_ratio()
            if efficiency > 0.7:
                reward += 5 * efficiency
                
            self.food = self._place_food()
            self.steps_without_food = 0
        else:
            # Eliminar la cola si no come
            self.snake.pop()
            self.steps_without_food += 1
            
            # Verificar si se ha excedido el límite de pasos sin comer
            if self.steps_without_food >= self.max_steps_without_food:
                # Penalización proporcional a la cercanía de la comida
                penalty = -2 - 3 * (1 - self._manhattan_distance(self.snake[0], self.food) / (GRID_WIDTH + GRID_HEIGHT))
                self.game_over = True
                return self._get_state(), penalty, True
            
            # Recompensa basada en distancia
            new_distance = self._manhattan_distance(self.snake[0], self.food)
            # Versión mejorada - más tolerante en laberintos
            if self.difficulty > 0:  # Si hay laberinto
                # Reducir sensibilidad en laberintos
                distance_factor = 0.5
            else:
                distance_factor = 1.0
            distance_reward = (prev_distance - new_distance) * distance_factor * (0.9 ** len(self.snake))
            reward += distance_reward
            
            # Penalizar movimientos en círculos sólo en espacios abiertos
            if self._detect_circles() and not self._is_in_corridor():
                reward -= 0.5
        
        # Pequeño incentivo por sobrevivir
        reward += 0.01
        
        return self._get_state(), reward, self.game_over
    
    def render(self, current_score=None, total_score=None, episodes_left=None, epsilon=None, avg_q=None):
        """Renderiza el estado actual del juego"""
        if not self.render_mode:
            return
            
        self.screen.fill(BLACK)
        
        # Dibujar cuadrícula de fondo para mejor visualización si DEBUG_MODE está activado
        if DEBUG_MODE:
            for x in range(0, SCREEN_WIDTH, GRID_SIZE):
                pygame.draw.line(self.screen, GRID_COLOR, (x, 0), (x, SCREEN_HEIGHT), 1)
            for y in range(0, SCREEN_HEIGHT, GRID_SIZE):
                pygame.draw.line(self.screen, GRID_COLOR, (0, y), (SCREEN_WIDTH, y), 1)
        
        # Dibujar paredes verticales
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH + 1):
                if self.vertical_walls[y][x]:
                    pygame.draw.rect(self.screen, BRIGHT_GRAY, 
                                    (x * GRID_SIZE - 3, y * GRID_SIZE, 
                                    6, GRID_SIZE))
        
        # Dibujar paredes horizontales
        for y in range(GRID_HEIGHT + 1):
            for x in range(GRID_WIDTH):
                if self.horizontal_walls[y][x]:
                    pygame.draw.rect(self.screen, BRIGHT_GRAY, 
                                    (x * GRID_SIZE, y * GRID_SIZE - 3,
                                    GRID_SIZE, 6))
        
        # Dibujar la comida
        pygame.draw.rect(self.screen, RED, 
                        (self.food[0] * GRID_SIZE + 2, self.food[1] * GRID_SIZE + 2, 
                         GRID_SIZE - 4, GRID_SIZE - 4))
        
        # Dibujar la serpiente
        for i, (x, y) in enumerate(self.snake):
            color = GREEN if i > 0 else BLUE  # Cabeza en azul, cuerpo en verde
            pygame.draw.rect(self.screen, color, 
                            (x * GRID_SIZE + 1, y * GRID_SIZE + 1, 
                             GRID_SIZE - 2, GRID_SIZE - 2))
            
            # Dibujar ojos en la cabeza
            if i == 0:
                eye_size = GRID_SIZE // 5
                if self.direction == UP:
                    pygame.draw.circle(self.screen, WHITE, (x * GRID_SIZE + GRID_SIZE // 3, y * GRID_SIZE + GRID_SIZE // 3), eye_size)
                    pygame.draw.circle(self.screen, WHITE, (x * GRID_SIZE + 2*GRID_SIZE // 3, y * GRID_SIZE + GRID_SIZE // 3), eye_size)
                elif self.direction == DOWN:
                    pygame.draw.circle(self.screen, WHITE, (x * GRID_SIZE + GRID_SIZE // 3, y * GRID_SIZE + 2*GRID_SIZE // 3), eye_size)
                    pygame.draw.circle(self.screen, WHITE, (x * GRID_SIZE + 2*GRID_SIZE // 3, y * GRID_SIZE + 2*GRID_SIZE // 3), eye_size)
                elif self.direction == LEFT:
                    pygame.draw.circle(self.screen, WHITE, (x * GRID_SIZE + GRID_SIZE // 3, y * GRID_SIZE + GRID_SIZE // 3), eye_size)
                    pygame.draw.circle(self.screen, WHITE, (x * GRID_SIZE + GRID_SIZE // 3, y * GRID_SIZE + 2*GRID_SIZE // 3), eye_size)
                elif self.direction == RIGHT:
                    pygame.draw.circle(self.screen, WHITE, (x * GRID_SIZE + 2*GRID_SIZE // 3, y * GRID_SIZE + GRID_SIZE // 3), eye_size)
                    pygame.draw.circle(self.screen, WHITE, (x * GRID_SIZE + 2*GRID_SIZE // 3, y * GRID_SIZE + 2*GRID_SIZE // 3), eye_size)
        
        # Panel lateral de información
        if current_score is not None:
            panel_x = SCREEN_WIDTH - 260
            panel_y = 10
            line_height = 25
            
            # Rectángulo semi-transparente para el panel
            panel_surface = pygame.Surface((250, 250))
            panel_surface.set_alpha(180)  # Semi-transparente
            panel_surface.fill((30, 30, 30))  # Gris oscuro
            self.screen.blit(panel_surface, (panel_x, panel_y))
            
            # Mostrar información
            score_text = self.font.render(f"Score Actual: {current_score}", True, WHITE)
            
            self.screen.blit(score_text, (panel_x + 10, panel_y + 0 * line_height))
            
            if total_score is not None:
                total_score_text = self.font.render(f"Score Acumulado: {total_score}", True, WHITE)
                self.screen.blit(total_score_text, (panel_x + 10, panel_y + 1 * line_height))
            
            if episodes_left is not None:
                episodes_text = self.font.render(f"Episodios Faltantes: {episodes_left}", True, WHITE)
                self.screen.blit(episodes_text, (panel_x + 10, panel_y + 2 * line_height))
            
            if epsilon is not None:
                epsilon_text = self.font.render(f"Epsilon: {epsilon:.3f}", True, WHITE)
                self.screen.blit(epsilon_text, (panel_x + 10, panel_y + 3 * line_height))
            
            if avg_q is not None:
                avg_q_text = self.font.render(f"Q-Value Promedio: {avg_q:.3f}", True, WHITE)
                self.screen.blit(avg_q_text, (panel_x + 10, panel_y + 4 * line_height))
            
            steps_text = self.font.render(f"Pasos sin comer: {self.steps_without_food}", True, WHITE)
            self.screen.blit(steps_text, (panel_x + 10, panel_y + 5 * line_height))
            
            if self.num_workers:
                workers_text = self.font.render(f"Entrenamientos paralelos: {self.num_workers}", True, YELLOW)
                self.screen.blit(workers_text, (panel_x + 10, panel_y + 6 * line_height))
            
            difficulty_name = "Sin laberinto" if self.difficulty == 0 else "Simple" if self.difficulty == 1 else "Medio"
            difficulty_text = self.font.render(f"Dificultad: {self.difficulty} ({difficulty_name})", True, YELLOW)
            self.screen.blit(difficulty_text, (panel_x + 10, panel_y + 7 * line_height))
            
            # Instrucciones de control
            controls_text = self.font.render("ESC: Salir | M: Más Rápido | N: Más Lento", True, (200, 200, 100))
            self.screen.blit(controls_text, (panel_x + 10, panel_y + 9 * line_height))
        
        pygame.display.flip()
        self.clock.tick(FPS)
    
    def handle_events(self):
        """Maneja eventos de Pygame."""
        if not self.render_mode:
            return True
            
        global FPS
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                return False
            # Teclas para ajustar velocidad
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_m:  # M = Más rápido
                    FPS = min(60, FPS + 5)
                    print(f"Velocidad aumentada a {FPS} FPS")
                elif event.key == pygame.K_n:  # N = Más lento
                    FPS = max(1, FPS - 5)
                    print(f"Velocidad reducida a {FPS} FPS")
                    
        return True
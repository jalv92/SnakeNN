import torch
import gc
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque, namedtuple
import math
import copy
import time
try:
    import psutil
except ImportError:
    print("psutil no disponible. El monitoreo de memoria será limitado.")

# ------------------- Utilidades para Manejo de Tensores -------------------
def validate_tensor_shape(tensor, expected_shape, tensor_name="tensor"):
    """
    Verifica que un tensor tenga la forma esperada o compatible.
    
    Args:
        tensor (torch.Tensor): El tensor a verificar
        expected_shape (tuple): Forma esperada o mínima (los None son comodines)
        tensor_name (str): Nombre para mensajes de error
        
    Returns:
        bool: True si el tensor es válido
        
    Raises:
        ValueError: Si el tensor no tiene la forma esperada
    """
    if not torch.is_tensor(tensor):
        raise TypeError(f"{tensor_name} debe ser un tensor PyTorch, no {type(tensor)}")
    
    if len(tensor.shape) != len(expected_shape):
        raise ValueError(f"{tensor_name} tiene {len(tensor.shape)} dimensiones, se esperaban {len(expected_shape)}")
    
    for i, (actual, expected) in enumerate(zip(tensor.shape, expected_shape)):
        if expected is not None and actual != expected:
            raise ValueError(f"Dimensión {i} de {tensor_name} es {actual}, se esperaba {expected}")
    
    return True

def safe_cat(tensors, dim=0, tensor_name="tensores"):
    """
    Realiza una concatenación segura de tensores, con verificaciones y manejo de errores.
    
    Args:
        tensors (list): Lista de tensores a concatenar
        dim (int): Dimensión en la que concatenar
        tensor_name (str): Nombre para mensajes de error
        
    Returns:
        torch.Tensor: Tensor concatenado o tensor de respaldo en caso de error
    """
    if not tensors:
        return None
    
    try:
        # Verificar que todos los tensores tengan la misma dimensionalidad
        dims = set(t.dim() for t in tensors)
        if len(dims) > 1:
            # Ajustar dimensiones
            min_dim = min(dims)
            max_dim = max(dims)
            adjusted_tensors = []
            
            for t in tensors:
                if t.dim() < max_dim:
                    # Añadir dimensiones faltantes
                    missing_dims = max_dim - t.dim()
                    adjusted_t = t
                    for _ in range(missing_dims):
                        adjusted_t = adjusted_t.unsqueeze(0)
                    adjusted_tensors.append(adjusted_t)
                else:
                    adjusted_tensors.append(t)
            
            tensors = adjusted_tensors
        
        # Verificar dimensiones compatibles para concatenación
        shapes = [t.shape for t in tensors]
        
        # Verificar que todas las dimensiones excepto 'dim' sean iguales
        for i in range(max(dims)):
            if i != dim:
                sizes = set(s[i] if i < len(s) else 1 for s in shapes)
                if len(sizes) > 1:
                    # Las dimensiones no coinciden, redimensionar
                    size = min(sizes)  # Usar el tamaño mínimo
                    adjusted_tensors = []
                    
                    for t in tensors:
                        if i < t.dim() and t.shape[i] > size:
                            # Recortar la dimensión al tamaño mínimo
                            slices = [slice(None)] * t.dim()
                            slices[i] = slice(0, size)
                            adjusted_t = t[tuple(slices)]
                            adjusted_tensors.append(adjusted_t)
                        else:
                            adjusted_tensors.append(t)
                    
                    tensors = adjusted_tensors
        
        # Intentar concatenar
        return torch.cat(tensors, dim=dim)
    
    except Exception as e:
        print(f"Error al concatenar {tensor_name}: {e}")
        # Dimensión de respaldo
        backup_shape = list(tensors[0].shape)
        backup_shape[dim] = sum(t.shape[dim] for t in tensors)
        backup_tensor = torch.zeros(backup_shape, device=tensors[0].device)
        
        # Intentar copiar datos
        try:
            offset = 0
            for t in tensors:
                slice_indices = [slice(None)] * len(backup_shape)
                slice_indices[dim] = slice(offset, offset + t.shape[dim])
                backup_tensor[tuple(slice_indices)] = t
                offset += t.shape[dim]
        except:
            pass  # Si falla la copia, al menos tenemos un tensor de ceros
        
        return backup_tensor

class MemoryManager:
    """Administrador de memoria para garantizar una gestión eficiente de recursos CUDA."""
    def __init__(self, agent=None):
        self.agent = agent
        self.episode_counter = 0
        self.last_clear_time = time.time()
        self.clear_interval = 30  # segundos entre limpiezas profundas
        self.memory_stats = []
        
    def register_agent(self, agent):
        """Registra un agente para monitorear."""
        self.agent = agent
        
    def after_episode_cleanup(self):
        """Ejecuta limpieza después de cada episodio."""
        self.episode_counter += 1
        
        # Limpiar el recolector de basura de Python
        gc.collect()
        
        # Si hay CUDA disponible, limpiar caché
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
            # Cada 5 episodios, mostrar estadísticas de memoria
            if self.episode_counter % 5 == 0:
                allocated = torch.cuda.memory_allocated() / 1024**2
                reserved = torch.cuda.memory_reserved() / 1024**2
                self.memory_stats.append((allocated, reserved))
                print(f"Memoria CUDA - Asignada: {allocated:.2f} MB, Reservada: {reserved:.2f} MB")
                
                # Si estamos acumulando demasiada memoria reservada, forzar liberación
                if reserved > 1000:  # Si reserva más de 1GB
                    self._force_memory_release()
                    
        # Cada cierto tiempo, hacer limpieza profunda
        current_time = time.time()
        if current_time - self.last_clear_time > self.clear_interval:
            self._deep_cleanup()
            self.last_clear_time = current_time
            
    def _force_memory_release(self):
        """Fuerza liberación de memoria CUDA."""
        if torch.cuda.is_available():
            # Guardar model state_dict
            if self.agent:
                module_states = {}
                for name, module in self.agent.modules.items():
                    module_states[name] = copy.deepcopy(module.state_dict())
                
                # Mover modelos a CPU temporalmente
                for name, module in self.agent.modules.items():
                    self.agent.modules[name] = module.to('cpu')
                
                # Limpiar caché CUDA completamente
                torch.cuda.empty_cache()
                
                # Volver a mover modelos a GPU
                for name in self.agent.modules:
                    self.agent.modules[name] = self.agent.modules[name].to(self.agent.device)
                    self.agent.modules[name].load_state_dict(module_states[name])
                
                print("Forzada liberación de memoria CUDA")
    
    def _deep_cleanup(self):
        """Realiza limpieza profunda de recursos."""
        print("Iniciando limpieza profunda...")
        
        # Limpiar estado interno de PyTorch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Limpiar memoria episódica si es demasiado grande
        if self.agent and hasattr(self.agent, 'episodic_memory'):
            if len(self.agent.episodic_memory.episodes) > 20:
                # Mantener solo los episodios más importantes
                importances = [(i, ep.importance) for i, ep in enumerate(self.agent.episodic_memory.episodes)]
                importances.sort(key=lambda x: x[1], reverse=True)
                
                # Mantener los 15 mejores episodios y descartar el resto
                keep_indices = [x[0] for x in importances[:15]]
                self.agent.episodic_memory.episodes = [self.agent.episodic_memory.episodes[i] for i in keep_indices]
                print(f"Reducida memoria episódica de >20 a {len(self.agent.episodic_memory.episodes)} episodios")
        
        print("Limpieza profunda completada")
        
    def monitor_and_log(self):
        """Monitorea y registra uso de memoria."""
        if torch.cuda.is_available():
            # Registrar estadísticas de uso de memoria
            stats = {
                'cuda_allocated': torch.cuda.memory_allocated() / 1024**2,
                'cuda_reserved': torch.cuda.memory_reserved() / 1024**2,
                'episodes': self.episode_counter,
                'timestamp': time.time()
            }
            
            # Añadir a memoria
            self.memory_stats.append(stats)
            
            # Limitar tamaño del historial
            if len(self.memory_stats) > 100:
                self.memory_stats = self.memory_stats[-100:]

def optimize_cuda_settings():
    """Optimiza la configuración de CUDA para mejor rendimiento"""
    if torch.cuda.is_available():
        print("\n===== Configuración de GPU =====")
        print(f"GPU disponible: {torch.cuda.get_device_name(0)}")
        
        # Mostrar y limpiar caché inicial
        print(f"Memoria CUDA inicial: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        torch.cuda.empty_cache()
        print(f"Después de limpiar caché: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        
        # Optimizaciones de rendimiento
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # Configurar allocator para reducir fragmentación
        try:
            torch.cuda.memory.set_per_process_memory_fraction(0.7)  # Usar hasta 70% de memoria GPU
            print("Limitado uso de memoria GPU al 70%")
        except:
            print("No se pudo limitar memoria de GPU - PyTorch versión antigua")
        
        # Optimizar tipos de precisión
        torch.set_default_dtype(torch.float32)  # Usar float32 en lugar de float64
        
        print("Optimizaciones de GPU activadas")
        print("==============================\n")
        return True
    else:
        print("\n⚠️ ALERTA: CUDA no disponible. El entrenamiento será lento en CPU.")
        print("Verifica instalación de PyTorch con soporte CUDA")
        print("==============================\n")
        return False

# Configurar gestión de memoria para reducir fugas de memoria
def config_memory_management():
    """Configura gestión de memoria para PyTorch"""
    # Recolectar basura de Python
    gc.collect()
    
    # Si CUDA está disponible, limpiar caché
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def check_gpu_compatibility():
    """Verifica la compatibilidad de GPU y muestra información detallada"""
    print("\n===== Diagnóstico de GPU =====")
    
    if not torch.cuda.is_available():
        print("❌ CUDA no disponible. Posibles causas:")
        print("  - PyTorch instalado sin soporte CUDA")
        print("  - Drivers NVIDIA desactualizados")
        print("  - GPU no compatible con CUDA")
        print("\nSolución: Reinstalar PyTorch con soporte CUDA:")
        print("  pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 --index-url https://download.pytorch.org/whl/cu118")
        return False
        
    print(f"✓ CUDA disponible: versión {torch.version.cuda}")
    print(f"✓ GPU detectada: {torch.cuda.get_device_name(0)}")
    print(f"✓ Propiedades de memoria:")
    print(f"  - Total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"  - Reservada: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    print(f"  - Asignada: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    # Crear tensor de prueba para verificar operaciones
    try:
        test_tensor = torch.rand(1000, 1000, device='cuda')
        test_result = torch.matmul(test_tensor, test_tensor)
        del test_tensor, test_result
        print("✓ Operaciones de tensor en GPU funcionando correctamente")
    except Exception as e:
        print(f"❌ Error en operaciones de tensor: {e}")
        
    print("\n✓ Configuración cuDNN:")
    print(f"  - Disponible: {torch.backends.cudnn.is_available()}")
    print(f"  - Versión: {torch.backends.cudnn.version()}")
    print(f"  - Benchmark: {torch.backends.cudnn.benchmark}")
    
    print("\n==== Recomendaciones ====")
    print("1. Reducir tamaño de batch si hay problemas de memoria")
    print("2. Reducir complejidad del modelo si el rendimiento sigue lento")
    print("3. Cerrar otras aplicaciones que consuman GPU")
    print("4. Monitorear temperatura de GPU para evitar throttling")
    print("==============================\n")
    
    return True

# ------------------- Constants -------------------
# Hyperparameters for Dynamic Neural Growth
GROWTH_THRESHOLD = 0.8  # Activation threshold for neuron addition
PRUNE_THRESHOLD = 0.1   # Activity threshold for pruning
MAX_NEURONS_PER_LAYER = 256  # Upper limit for dynamic growth
HEBBIAN_LEARNING_RATE = 0.01  # Rate for Hebbian learning adjustments
GROWTH_CHECK_INTERVAL = 100   # Episodes between growth checks

# Hyperparameters for Meta-Learning
META_LEARNING_RATE = 0.001    # Learning rate for meta-controller
META_UPDATE_INTERVAL = 50     # Episodes between meta-updates
DIFFICULTY_ADJUST_THRESHOLD = 0.7  # Success rate threshold for increasing difficulty

# Hyperparameters for Episodic Memory
EPISODE_MEMORY_SIZE = 1000    # Number of episodes to store
REPLAY_IMPORTANCE_THRESHOLD = 0.6  # Threshold for important experiences
CAUSAL_MEMORY_CAPACITY = 100  # Capacity for causal memory entries

# Hyperparameters for Curiosity
CURIOSITY_WEIGHT = 0.1        # Weight of intrinsic reward
CURIOSITY_DECAY = 0.99        # Decay rate for curiosity
NOVELTY_THRESHOLD = 0.2       # Threshold for novelty detection

# Hyperparameters for Knowledge Transfer
TRANSFER_LEARNING_RATE = 0.0005  # Learning rate for transfer learning
SKILL_LIBRARY_SIZE = 20       # Number of skills to store in library

# ------------------- Dynamic Neural Module -------------------
class DynamicLayer(nn.Module):
    """Neural layer with dynamic growth capabilities"""
    def __init__(self, input_size, output_size, growth_rate=0.1):
        super(DynamicLayer, self).__init__()
        self.input_size = input_size
        self.current_output_size = output_size
        self.max_output_size = MAX_NEURONS_PER_LAYER
        self.growth_rate = growth_rate
        
        # Initialize main weight matrix
        self.weight = nn.Parameter(torch.Tensor(output_size, input_size))
        self.bias = nn.Parameter(torch.Tensor(output_size))
        
        # Initialize using Kaiming initialization
        nn.init.kaiming_normal_(self.weight)
        nn.init.zeros_(self.bias)
        
        # Activity tracking for Hebbian learning and pruning
        self.register_buffer('activation_history', torch.zeros(output_size))
        self.register_buffer('connection_activity', torch.zeros(output_size, input_size))
        self.register_buffer('usage_count', torch.zeros(output_size))
        
        # Hebbian learning state - también registrado como buffer para asegurar mismo dispositivo
        self.register_buffer('hebbian_traces', torch.zeros(output_size, input_size))
        
        # Pruning mask (1 for active connections, 0 for pruned)
        self.register_buffer('prune_mask', torch.ones(output_size, input_size))
        
    def forward(self, x):
        # Apply pruning mask during forward pass
        effective_weight = self.weight * self.prune_mask
        output = F.linear(x, effective_weight, self.bias)
        
        # Update activation history for each neuron (moving average)
        with torch.no_grad():
            # Get batch activations
            activation = torch.sigmoid(output).detach()
            batch_activation = activation.mean(dim=0)
            
            # Update historical activation (as moving average)
            self.activation_history = 0.9 * self.activation_history + 0.1 * batch_activation
            
            # Update usage counters
            active_neurons = (activation > 0.5).float().sum(dim=0)
            self.usage_count += active_neurons
            
            # Update connection activity for Hebbian learning
            if x.dim() > 1:  # If we have a batch
                # For each active output neuron, which input neurons contributed
                for i in range(self.current_output_size):
                    # Calculate correlation between input and this output
                    self.connection_activity[i] = 0.9 * self.connection_activity[i] + \
                                               0.1 * torch.abs(torch.mean(x * activation[:, i].unsqueeze(1), dim=0))
        
        return output
    
    def grow_neurons(self, activation_threshold=GROWTH_THRESHOLD):
        """Add new neurons if existing ones are consistently highly activated"""
        if self.current_output_size >= self.max_output_size:
            return False
        
        # Check if neurons are consistently highly activated
        high_activation_neurons = (self.activation_history > activation_threshold).sum().item()
        growth_neurons = int(high_activation_neurons * self.growth_rate)
        
        # Limit growth to available space
        growth_neurons = min(growth_neurons, self.max_output_size - self.current_output_size)
        
        if growth_neurons <= 0:
            return False
            
        # Grow the layer by adding new neurons
        with torch.no_grad():
            # Create new weight and bias tensors
            new_weight = torch.zeros(self.current_output_size + growth_neurons, self.input_size, 
                                   device=self.weight.device)
            new_bias = torch.zeros(self.current_output_size + growth_neurons,
                                  device=self.bias.device)
            
            # Copy existing weights and biases
            new_weight[:self.current_output_size] = self.weight
            new_bias[:self.current_output_size] = self.bias
            
            # Initialize new neurons with variation of existing successful neurons
            # Take most successful neurons as templates
            _, top_indices = torch.topk(self.activation_history, min(5, self.current_output_size))
            
            for i in range(growth_neurons):
                # Pick a random top neuron as template
                template_idx = top_indices[random.randint(0, len(top_indices) - 1)]
                # Copy with small random variations
                new_weight[self.current_output_size + i] = self.weight[template_idx] + \
                                                         torch.randn_like(self.weight[template_idx]) * 0.1
                new_bias[self.current_output_size + i] = self.bias[template_idx] + \
                                                      torch.randn(1).item() * 0.1
            
            # Update weights and biases
            self.weight = nn.Parameter(new_weight)
            self.bias = nn.Parameter(new_bias)
            
            # Expand activation history and usage tracking
            new_activation_history = torch.zeros(self.current_output_size + growth_neurons,
                                               device=self.activation_history.device)
            new_activation_history[:self.current_output_size] = self.activation_history
            self.activation_history = new_activation_history
            
            new_usage_count = torch.zeros(self.current_output_size + growth_neurons,
                                        device=self.usage_count.device)
            new_usage_count[:self.current_output_size] = self.usage_count
            self.usage_count = new_usage_count
            
            # Expand connection activity tracking
            new_connection_activity = torch.zeros(self.current_output_size + growth_neurons, self.input_size,
                                                device=self.connection_activity.device)
            new_connection_activity[:self.current_output_size] = self.connection_activity
            self.connection_activity = new_connection_activity
            
            # Expand Hebbian traces - corregir para usar el mismo dispositivo
            new_hebbian_traces = torch.zeros(self.current_output_size + growth_neurons, self.input_size,
                                           device=self.hebbian_traces.device)
            new_hebbian_traces[:self.current_output_size] = self.hebbian_traces
            self.register_buffer('hebbian_traces', new_hebbian_traces, persistent=True)
            
            # Expand pruning mask
            new_prune_mask = torch.ones(self.current_output_size + growth_neurons, self.input_size,
                                      device=self.prune_mask.device)
            new_prune_mask[:self.current_output_size] = self.prune_mask
            self.prune_mask = new_prune_mask
            
            # Update current size
            self.current_output_size += growth_neurons
            
            print(f"Layer grew by {growth_neurons} neurons, new size: {self.current_output_size}")
            return True
    
    def apply_hebbian_learning(self, learning_rate=HEBBIAN_LEARNING_RATE):
        """Apply Hebbian learning rule: neurons that fire together, wire together"""
        with torch.no_grad():
            # Ensure hebbian_traces is on the same device as connection_activity
            if self.hebbian_traces.device != self.connection_activity.device:
                self.hebbian_traces = self.hebbian_traces.to(self.connection_activity.device)
            
            # Normalize connection activity for Hebbian update
            if torch.max(self.connection_activity) > 0:
                # Update Hebbian traces based on co-activation
                self.hebbian_traces = self.hebbian_traces * 0.9 + self.connection_activity * 0.1
                
                # Apply Hebbian updates to weights where connections are not pruned
                hebbian_updates = learning_rate * self.hebbian_traces * self.prune_mask
                self.weight.data += hebbian_updates
                
                # Renormalize weights to prevent explosive growth
                for i in range(self.current_output_size):
                    weight_norm = torch.norm(self.weight[i])
                    if weight_norm > 3.0:  # Threshold for normalization
                        self.weight[i] = 3.0 * self.weight[i] / weight_norm

    def get_output_size(self):
        """Return current number of output neurons"""
        return self.current_output_size
    
    def prune_connections(self):
        """Prune inactive connections"""
        with torch.no_grad():
            # Calculate connection strength as absolute weight value times activity
            connection_strength = torch.abs(self.weight) * self.connection_activity
            
            # Find weak connections (bottom 10%)
            flat_strength = connection_strength.view(-1)
            if len(flat_strength) > 10:  # Only if we have enough connections
                threshold = torch.kthvalue(flat_strength, max(1, int(0.1 * len(flat_strength)))).values
                
                # Update pruning mask - set weak connections to 0
                weak_connections = connection_strength < threshold
                self.prune_mask[weak_connections] = 0
                
                # Count pruned connections
                pruned_count = weak_connections.sum().item()
                if pruned_count > 0:
                    print(f"Pruned {pruned_count} weak connections")
                    return True
        return False

# ------------------- Modular Neural Architecture -------------------
class PerceptionModule(nn.Module):
    """Module for processing sensory information from the environment"""
    def __init__(self, input_size, output_size):
        super(PerceptionModule, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        
        # Process state features using dynamic layers
        self.feature_extractor = nn.Sequential(
            DynamicLayer(input_size, 64),
            nn.ReLU(),
            DynamicLayer(64, output_size),
            nn.ReLU()
        )
        
        # Feature attention mechanism
        self.attention = nn.Parameter(torch.ones(output_size) / output_size)  # Initial uniform attention
        
    def forward(self, state):
        features = self.feature_extractor(state)
        # Apply attention - highlight important features
        weighted_features = features * F.softmax(self.attention, dim=0)
        return weighted_features
    
    def update_attention(self, importance_signals):
        """Update attention weights based on what features were most useful"""
        with torch.no_grad():
            # Asegurar que importance_signals esté en el mismo dispositivo que attention
            if isinstance(importance_signals, torch.Tensor):
                if importance_signals.device != self.attention.device:
                    importance_signals = importance_signals.to(self.attention.device)
            else:
                # Si no es un tensor, convertirlo a tensor en el dispositivo correcto
                importance_signals = torch.tensor(importance_signals, device=self.attention.device)
                
            # Adjust attention based on importance feedback
            self.attention.data += 0.1 * importance_signals
            # Normalize attention
            self.attention.data = F.softmax(self.attention.data, dim=0)

class NavigationModule(nn.Module):
    """Module specialized in planning routes and obstacle avoidance"""
    def __init__(self, input_size, output_size):
        super(NavigationModule, self).__init__()
        
        # A-star like navigation system using neural approximation
        self.pathfinder = nn.Sequential(
            DynamicLayer(input_size, 64),
            nn.ReLU(),
            DynamicLayer(64, 32),
            nn.ReLU(),
            DynamicLayer(32, output_size)
        )
        
        # Directional bias for different movement options
        self.direction_bias = nn.Parameter(torch.zeros(output_size))
        
    def forward(self, features, position, target):
        # Concat features with position and target info
        navigation_input = torch.cat([features, position, target], dim=1)
        return self.pathfinder(navigation_input)
    
    def update_bias(self, successful_actions, learning_rate=0.05):
        """Update directional bias based on successful actions"""
        with torch.no_grad():
            for action in successful_actions:
                self.direction_bias.data[action] += learning_rate

class PredictionModule(nn.Module):
    """Module for predicting outcomes of actions"""
    def __init__(self, state_size, action_size):
        super(PredictionModule, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        
        # Forward model to predict next state given current state and action
        self.forward_model = nn.Sequential(
            DynamicLayer(state_size + action_size, 64),
            nn.ReLU(),
            DynamicLayer(64, 64),
            nn.ReLU(),
            DynamicLayer(64, state_size)
        )
        
        # Reward prediction model
        self.reward_predictor = nn.Sequential(
            DynamicLayer(state_size + action_size, 32),
            nn.ReLU(),
            DynamicLayer(32, 1)
        )
        
        # Uncertainty estimation (for intrinsic motivation)
        self.uncertainty_estimator = nn.Sequential(
            DynamicLayer(state_size + action_size, 32),
            nn.ReLU(),
            DynamicLayer(32, 1),
            nn.Sigmoid()  # Output between 0-1
        )
        
    def predict_next_state(self, state, action):
        """Predict the next state given current state and action with manejo robusto de errores"""
        # Validación defensiva
        if not torch.is_tensor(state):
            state = torch.FloatTensor(state).to(self.forward_model[0].weight.device)
        if state.dim() == 1:
            state = state.unsqueeze(0)  # Añadir dimensión de batch
        
        # Convert action to one-hot if not already
        if not torch.is_tensor(action):
            action = torch.tensor([action], device=state.device)
        
        # Intentar inferir action_size correctamente
        action_size = None
        try:
            # Primero intentar desde el modelo
            action_size = self.forward_model[0].input_size - state.size(1)
        except:
            # Fallback para casos especiales
            action_size = self.action_size if hasattr(self, 'action_size') else 3
        
        # Convertir a one-hot con manejo de errores
        try:
            if action.dim() == 1:
                action_onehot = F.one_hot(action.long(), action_size).float().to(state.device)
            else:
                action_onehot = action
        except Exception as e:
            # Fallback: crear tensor de ceros con un 1 en la posición correcta
            print(f"Error en one-hot encoding: {e}")
            action_onehot = torch.zeros(state.size(0), action_size, device=state.device)
            if action.numel() == 1:  # Si es un solo valor
                action_val = action.item() if torch.is_tensor(action) else action
                if 0 <= action_val < action_size:
                    action_onehot[:, action_val] = 1.0
        
        # Intentar concatenar con manejo de errores
        try:
            x = torch.cat([state, action_onehot], dim=1)
        except Exception as e:
            print(f"Error al concatenar en predict_next_state: {e}")
            # Intentar solucionar problemas de dimensión
            if state.size(0) != action_onehot.size(0):
                # Ajustar batch sizes
                if state.size(0) == 1:
                    state = state.expand(action_onehot.size(0), -1)
                elif action_onehot.size(0) == 1:
                    action_onehot = action_onehot.expand(state.size(0), -1)
                else:
                    # Usar el batch size más pequeño
                    min_batch = min(state.size(0), action_onehot.size(0))
                    state = state[:min_batch]
                    action_onehot = action_onehot[:min_batch]
            
            # Reintentar la concatenación
            try:
                x = torch.cat([state, action_onehot], dim=1)
            except:
                # Último recurso: crear entrada de tamaño correcto
                x = torch.zeros(state.size(0), self.forward_model[0].input_size, device=state.device)
        
        # Pasar por el modelo con manejo de errores
        try:
            return self.forward_model(x)
        except Exception as e:
            print(f"Error en forward model: {e}")
            # Retornar salida de emergencia con forma correcta
            return torch.zeros(state.size(0), self.forward_model[-1].current_output_size, device=state.device)
    
    def predict_reward(self, state, action):
        """Predict reward for taking action in state"""
        # Defensive handling similar to predict_next_state
        try:
            # Convert action to one-hot if needed
            if action.dim() == 1:
                action_size = self.reward_predictor[0].input_size - state.size(1)
                action_onehot = F.one_hot(action.long(), action_size).float()
            else:
                action_onehot = action
                
            x = torch.cat([state, action_onehot], dim=1)
            return self.reward_predictor(x)
        except Exception as e:
            print(f"Error en predict_reward: {e}")
            # Return safe fallback
            return torch.zeros(state.size(0), 1, device=state.device)
    
    def estimate_uncertainty(self, state, action):
        """Estimate uncertainty of prediction (for intrinsic reward)"""
        # Defensive handling
        try:
            # Convert action to one-hot if needed
            if action.dim() == 1:
                action_size = self.uncertainty_estimator[0].input_size - state.size(1)
                action_onehot = F.one_hot(action.long(), action_size).float()
            else:
                action_onehot = action
                
            x = torch.cat([state, action_onehot], dim=1)
            return self.uncertainty_estimator(x)
        except Exception as e:
            print(f"Error en estimate_uncertainty: {e}")
            # Return medium uncertainty as fallback
            return torch.ones(state.size(0), 1, device=state.device) * 0.5
    
    def calculate_prediction_error(self, predicted_state, actual_state):
        """Calculate prediction error between predicted and actual state"""
        try:
            return F.mse_loss(predicted_state, actual_state)
        except Exception as e:
            print(f"Error en calculate_prediction_error: {e}")
            # Return medium error as fallback
            return torch.tensor(0.5, device=predicted_state.device)

class ExecutiveModule(nn.Module):
    """Coordinates other modules and makes final decisions"""
    def __init__(self, perception_size, navigation_size, prediction_size, action_size):
        super(ExecutiveModule, self).__init__()
        
        combined_input_size = perception_size + navigation_size + prediction_size
        
        # Integration network
        self.integrator = nn.Sequential(
            DynamicLayer(combined_input_size, 64),
            nn.ReLU(),
            DynamicLayer(64, 32),
            nn.ReLU()
        )
        
        # Action selection
        self.action_selector = DynamicLayer(32, action_size)
        
        # Module attention weights (how much to weight each module's output)
        self.module_attention = nn.Parameter(torch.tensor([0.4, 0.4, 0.2]))  # Initial weights
        
    def forward(self, perception_out, navigation_out, prediction_out):
        """
        Coordina los módulos de percepción, navegación y predicción para tomar una decisión
        
        Args:
            perception_out: Salida del módulo de percepción [batch_size, perception_features]
            navigation_out: Salida del módulo de navegación [batch_size, navigation_features]
            prediction_out: Salida del módulo de predicción que puede tener diferentes formas
        """
        # Apply attention to each module's output
        attn = F.softmax(self.module_attention, dim=0)
        
        # Weight each module's contribution
        weighted_perception = perception_out * attn[0]
        weighted_navigation = navigation_out * attn[1]
        
        # Preparar prediction_out - crucial para evitar errores de dimensiones
        if prediction_out.dim() == 3:  # Si es [batch, actions, features]
            # Convertir siempre a [batch, features_flattened]
            batch_size = prediction_out.size(0)
            prediction_out = prediction_out.reshape(batch_size, -1)
        elif prediction_out.dim() == 2 and perception_out.dim() == 2:
            # Si perception tiene forma [batch, features] pero prediction [features, x]
            if prediction_out.size(0) != perception_out.size(0):
                # Asegurar que la dimensión de batch coincida
                if prediction_out.size(0) == 1:
                    # Si prediction tiene batch=1, repetirlo para coincidir
                    prediction_out = prediction_out.repeat(perception_out.size(0), 1)
                elif perception_out.size(0) == 1:
                    # O reducir batch de perception si prediction tiene más ejemplos
                    weighted_perception = weighted_perception.repeat(prediction_out.size(0), 1)
                    weighted_navigation = weighted_navigation.repeat(prediction_out.size(0), 1)
        
        weighted_prediction = prediction_out * attn[2]
        
        # Verificar dimensiones antes de concatenar
        if weighted_perception.dim() != weighted_prediction.dim():
            # Ajustar dimensiones para que coincidan (preferiblemente a 2D)
            if weighted_perception.dim() == 1:
                weighted_perception = weighted_perception.unsqueeze(0)
            if weighted_navigation.dim() == 1:
                weighted_navigation = weighted_navigation.unsqueeze(0)
            if weighted_prediction.dim() == 1:
                weighted_prediction = weighted_prediction.unsqueeze(0)
        
        # Verificar tamaños de batch y ajustar si es necesario
        sizes = [t.size(0) for t in [weighted_perception, weighted_navigation, weighted_prediction]]
        if len(set(sizes)) > 1:  # Si hay diferentes tamaños de batch
            min_size = min(sizes)
            weighted_perception = weighted_perception[:min_size]
            weighted_navigation = weighted_navigation[:min_size]
            weighted_prediction = weighted_prediction[:min_size]
        
        try:
            # Combine all inputs
            combined = torch.cat([weighted_perception, weighted_navigation, weighted_prediction], dim=1)
        except RuntimeError as e:
            # Si aún falla, último recurso: reshape todo a vectores planos
            print(f"Error en concatenación, usando método alternativo: {e}")
            print(f"Formas: P={weighted_perception.shape}, N={weighted_navigation.shape}, PR={weighted_prediction.shape}")
            
            # Forzar que todo sea 2D con el mismo batch size
            batch_size = min(t.size(0) for t in [weighted_perception, weighted_navigation, weighted_prediction])
            combined = torch.cat([
                weighted_perception[:batch_size].reshape(batch_size, -1),
                weighted_navigation[:batch_size].reshape(batch_size, -1),
                weighted_prediction[:batch_size].reshape(batch_size, -1)
            ], dim=1)
        
        # Integrate and select action
        integrated = self.integrator(combined)
        action_values = self.action_selector(integrated)
        
        return action_values
    
    def update_attention(self, performance_metrics):
        """Update module attention based on which modules were most useful"""
        with torch.no_grad():
            # Asegurar que performance_metrics sea un tensor en el mismo dispositivo que module_attention
            metrics_tensor = torch.tensor(performance_metrics, device=self.module_attention.device)
            
            # Adjust attention weights based on performance metrics
            self.module_attention.data += 0.1 * metrics_tensor
            # Ensure attention weights are positive
            self.module_attention.data = torch.clamp(self.module_attention.data, min=0.1)
            # Re-normalize
            self.module_attention.data = F.softmax(self.module_attention.data, dim=0)
# ------------------- Meta-Learning System -------------------
class MetaController:
    """Meta-learning system that optimizes hyperparameters and learning strategies"""
    def __init__(self, agent, learning_rate=META_LEARNING_RATE):
        self.agent = agent
        self.learning_rate = learning_rate
        
        # Performance history
        self.performance_history = deque(maxlen=100)
        self.reward_history = deque(maxlen=100)
        
        # Hyperparameter ranges
        self.hyper_ranges = {
            'learning_rate': (0.0001, 0.01),
            'gamma': (0.9, 0.999),
            'epsilon_decay': (0.95, 0.999),
            'batch_size': (32, 128)
        }
        
        # Strategy options
        self.strategies = ['exploit', 'explore', 'diversify']
        self.current_strategy = 'explore'  # Start with exploration
        
        # Epsilon for hyperparameter adjustments
        self.meta_epsilon = 1.0
        self.meta_epsilon_decay = 0.99
        
        # Architecture modification state
        self.architecture_frozen = False
        
        # Curriculum learning state
        self.curriculum_level = 0
        self.curriculum_performance = 0.0
        self.curriculum_episodes = 0
        
    def record_performance(self, episode_reward, episode_score):
        """Record performance metrics for an episode"""
        self.performance_history.append(episode_score)
        self.reward_history.append(episode_reward)
    
    def adjust_hyperparameters(self):
        """Adjust agent hyperparameters based on recent performance"""
        if len(self.performance_history) < 10:
            return  # Not enough data
            
        # Calculate recent performance trend
        recent_perf = list(self.performance_history)[-10:]
        performance_trend = sum(recent_perf[-5:]) / 5 - sum(recent_perf[:5]) / 5
        
        # Random exploration vs greedy exploitation for hyperparameter adjustment
        if random.random() < self.meta_epsilon:
            # Explore: Random adjustments
            self._random_hyperparameter_adjustment()
        else:
            # Exploit: Targeted adjustments based on performance
            self._targeted_hyperparameter_adjustment(performance_trend)
            
        # Decay meta-epsilon
        self.meta_epsilon *= self.meta_epsilon_decay
        self.meta_epsilon = max(0.1, self.meta_epsilon)  # Keep some exploration
    
    def _random_hyperparameter_adjustment(self):
        """Make random adjustments to hyperparameters for exploration"""
        # Pick a random hyperparameter to adjust
        param = random.choice(list(self.hyper_ranges.keys()))
        
        if param == 'batch_size':
            # Batch size is discrete
            min_val, max_val = self.hyper_ranges[param]
            current = getattr(self.agent, param)
            new_val = current * random.choice([0.5, 0.75, 1.0, 1.25, 1.5])
            new_val = max(min_val, min(max_val, int(new_val)))
            new_val = 2 ** int(math.log2(new_val))  # Make it a power of 2
            setattr(self.agent, param, new_val)
            print(f"Meta-controller randomly adjusted {param}: {current} -> {new_val}")
        else:
            # Continuous hyperparameters
            min_val, max_val = self.hyper_ranges[param]
            current = getattr(self.agent, param)
            # Random adjustment within ±20%
            adjustment = random.uniform(0.8, 1.2)
            new_val = current * adjustment
            new_val = max(min_val, min(max_val, new_val))
            setattr(self.agent, param, new_val)
            print(f"Meta-controller randomly adjusted {param}: {current:.6f} -> {new_val:.6f}")
    
    def _targeted_hyperparameter_adjustment(self, performance_trend):
        """Make targeted adjustments based on performance trends"""
        if performance_trend > 0:
            # Performance improving - make smaller adjustments
            adjustment_size = 0.05
            print("Performance improving, making fine-tuning adjustments")
        else:
            # Performance stagnant or declining - make larger adjustments
            adjustment_size = 0.2
            print("Performance stagnant, making larger adjustments")
        
        # Learning rate adjustment
        if abs(performance_trend) < 0.5:
            # Fine-tuning phase - reduce learning rate
            current_lr = self.agent.learning_rate
            new_lr = max(self.hyper_ranges['learning_rate'][0], 
                        current_lr * (1 - adjustment_size))
            self.agent.learning_rate = new_lr
            print(f"Adjusted learning rate: {current_lr:.6f} -> {new_lr:.6f}")
            
            # Update optimizer learning rate
            for param_group in self.agent.optimizer.param_groups:
                param_group['lr'] = new_lr
        
        # Adaptive batch sizing - keep between 32 and 64
        current_batch = self.agent.batch_size
        if performance_trend > 0.5:  # Good performance improvement
            # If performance is improving, try increasing batch size for efficiency
            if current_batch < 64:
                new_batch = min(64, int(current_batch * 1.5))
                # Ensure it's a power of 2
                new_batch = 2 ** int(math.log2(new_batch) + 0.5)
                if new_batch != current_batch:
                    self.agent.batch_size = new_batch
                    print(f"Increased batch size: {current_batch} -> {new_batch}")
        elif performance_trend < -0.3:  # Performance declining
            # If performance is declining, try decreasing batch size for better learning
            if current_batch > 32:
                new_batch = max(32, int(current_batch * 0.75))
                # Ensure it's a power of 2
                new_batch = 2 ** int(math.log2(new_batch) + 0.5)
                if new_batch != current_batch:
                    self.agent.batch_size = new_batch
                    print(f"Decreased batch size: {current_batch} -> {new_batch}")
        
        # Exploration-exploitation balance
        if performance_trend < -1.0:
            # Significant performance decline - increase exploration
            current_epsilon = self.agent.epsilon
            new_epsilon = min(0.5, current_epsilon * (1 + adjustment_size))
            self.agent.epsilon = new_epsilon
            print(f"Increased exploration (ε): {current_epsilon:.4f} -> {new_epsilon:.4f}")
        
        # Discount factor adjustment based on performance stability
        reward_variance = np.var(list(self.reward_history)[-10:])
        if reward_variance > 100:  # High variance
            # Reduce gamma to focus more on immediate rewards
            current_gamma = self.agent.gamma
            new_gamma = max(0.9, current_gamma * (1 - adjustment_size * 0.1))
            self.agent.gamma = new_gamma
            print(f"Adjusted discount factor (γ): {current_gamma:.4f} -> {new_gamma:.4f}")
    
    def adjust_architecture(self):
        """Decide whether to grow or prune the neural network"""
        if self.architecture_frozen:
            return  # Architecture changes are frozen
            
        # Check recent performance trend to decide if architecture changes are needed
        if len(self.performance_history) < 20:
            return  # Not enough data
            
        recent_scores = list(self.performance_history)[-20:]
        first_half = sum(recent_scores[:10]) / 10
        second_half = sum(recent_scores[10:]) / 10
        
        performance_trend = second_half - first_half
        
        if performance_trend < -0.5:
            # Performance declining - try growing the network
            self._grow_network()
        elif performance_trend > 2.0:
            # Significant improvement - try pruning to prevent overfitting
            self._prune_network()
        elif random.random() < 0.1:
            # Occasionally try architecture changes regardless of performance
            if random.random() < 0.7:
                self._grow_network()
            else:
                self._prune_network()
    
    def _grow_network(self):
        """Signal the agent to grow its neural network"""
        growth_occurred = False
        
        # Try to grow each dynamic layer in the modules
        for name, module in self.agent.modules.items():
            if isinstance(module, nn.Module):
                for child in module.modules():
                    if isinstance(child, DynamicLayer):
                        if child.grow_neurons():
                            growth_occurred = True
        
        if growth_occurred:
            # Reinitialize optimizer since parameter shapes changed
            self.agent.setup_optimizer()
            print("Meta-controller grew network architecture")
    
    def _prune_network(self):
        """Signal the agent to prune unused connections"""
        # Try to prune each dynamic layer in the modules
        for name, module in self.agent.modules.items():
            if isinstance(module, nn.Module):
                for child in module.modules():
                    if isinstance(child, DynamicLayer):
                        child.prune_connections()
        
        print("Meta-controller pruned network connections")
    
    def adjust_learning_strategy(self):
        """Adjust the learning strategy based on performance"""
        if len(self.performance_history) < 20:
            return  # Not enough data
            
        # Analyze recent performance
        recent_scores = list(self.performance_history)[-20:]
        avg_score = sum(recent_scores) / len(recent_scores)
        score_variance = np.var(recent_scores)
        
        # Decide on strategy change
        if avg_score < 2 and self.current_strategy != 'explore':
            # Low scores - switch to exploration
            self.current_strategy = 'explore'
            self.agent.epsilon = min(0.5, self.agent.epsilon * 2)  # Increase exploration
            print(f"Meta-controller switched to EXPLORE strategy (ε={self.agent.epsilon:.4f})")
        elif avg_score > 10 and score_variance < 5 and self.current_strategy != 'exploit':
            # High consistent scores - switch to exploitation
            self.current_strategy = 'exploit'
            self.agent.epsilon = max(0.01, self.agent.epsilon * 0.5)  # Reduce exploration
            print(f"Meta-controller switched to EXPLOIT strategy (ε={self.agent.epsilon:.4f})")
        elif score_variance > 20 and self.current_strategy != 'diversify':
            # High variance - switch to diversification
            self.current_strategy = 'diversify'
            self.agent.epsilon = 0.2  # Balanced exploration
            # Increase batch diversity
            self.agent.batch_size = min(128, self.agent.batch_size * 2)
            print(f"Meta-controller switched to DIVERSIFY strategy (ε={self.agent.epsilon:.4f}, batch={self.agent.batch_size})")
    
    def update_curriculum(self, env):
        """Update the curriculum difficulty based on agent performance"""
        if len(self.performance_history) < 10:
            return  # Not enough data
            
        # Calculate success rate on current level
        recent_scores = list(self.performance_history)[-10:]
        success_threshold = 5  # Consider score >= 5 as success for curriculum
        success_rate = sum(1 for score in recent_scores if score >= success_threshold) / len(recent_scores)
        
        self.curriculum_performance = 0.8 * self.curriculum_performance + 0.2 * success_rate
        self.curriculum_episodes += 1
        
        # Check if it's time to adjust difficulty
        if self.curriculum_episodes >= 100 and self.curriculum_performance >= DIFFICULTY_ADJUST_THRESHOLD:
            # Good enough performance - increase difficulty
            self.curriculum_level = min(2, self.curriculum_level + 1)  # Max difficulty is 2
            env.difficulty = self.curriculum_level
            
            # Reset curriculum tracking
            self.curriculum_performance = 0.0
            self.curriculum_episodes = 0
            
            print(f"Meta-controller increased curriculum difficulty to level {self.curriculum_level}")
            return True
        elif self.curriculum_episodes >= 200 and self.curriculum_performance < 0.3:
            # Very poor performance for extended time - decrease difficulty
            self.curriculum_level = max(0, self.curriculum_level - 1)
            env.difficulty = self.curriculum_level
            
            # Reset curriculum tracking
            self.curriculum_performance = 0.0
            self.curriculum_episodes = 0
            
            print(f"Meta-controller decreased curriculum difficulty to level {self.curriculum_level}")
            return True
            
        return False  # No changes made

# ------------------- Episodic Memory System -------------------
class Episode:
    """Container for a complete game episode"""
    def __init__(self, max_length=1000):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.importance = 0.0  # Importance score for this episode
        self.score = 0  # Final game score
        self.max_length = max_length
    
    def add(self, state, action, reward, next_state, done):
        """Add a step to the episode"""
        if len(self.states) < self.max_length:
            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)
            self.next_states.append(next_state)
            self.dones.append(done)
            
            # Update running score
            if reward > 5:  # Approximate threshold for eating food
                self.score += 1
    
    def calculate_importance(self):
        """Calculate importance of this episode for replay"""
        # Base importance on final score
        score_importance = min(1.0, self.score / 10)
        
        # High reward moments
        high_reward_moments = sum(1 for r in self.rewards if r > 5)
        reward_importance = min(1.0, high_reward_moments / 5)
        
        # Unusual transitions (high surprisal)
        state_differences = []
        for i in range(len(self.states) - 1):
            if not self.dones[i]:
                diff = np.mean(np.abs(np.array(self.next_states[i]) - np.array(self.states[i+1])))
                state_differences.append(diff)
        
        surprisal_importance = min(1.0, sum(state_differences) / (len(state_differences) + 1e-5) / 10)
        
        # Combine importance factors
        self.importance = 0.5 * score_importance + 0.3 * reward_importance + 0.2 * surprisal_importance
        return self.importance
    
    def get_transitions(self):
        """Get all transitions in this episode"""
        return list(zip(self.states, self.actions, self.rewards, self.next_states, self.dones))
    
    def get_random_batch(self, batch_size):
        """Get a random batch of transitions from this episode"""
        if len(self.states) <= batch_size:
            return self.get_transitions()
        
        indices = random.sample(range(len(self.states)), batch_size)
        return [(self.states[i], self.actions[i], self.rewards[i], 
                 self.next_states[i], self.dones[i]) for i in indices]
    
    def get_sequence(self, start_idx, length):
        """Get a consecutive sequence of transitions starting at start_idx"""
        end_idx = min(start_idx + length, len(self.states))
        return [(self.states[i], self.actions[i], self.rewards[i], 
                 self.next_states[i], self.dones[i]) for i in range(start_idx, end_idx)]

class EpisodicMemory:
    """Enhanced memory system with episodic storage and causal reasoning"""
    def __init__(self, capacity=EPISODE_MEMORY_SIZE):
        self.episodes = deque(maxlen=capacity)
        self.current_episode = Episode()
        self.causal_memory = deque(maxlen=CAUSAL_MEMORY_CAPACITY)
        
        # Importance threshold for causal memory
        self.importance_threshold = REPLAY_IMPORTANCE_THRESHOLD
    
    def add(self, state, action, reward, next_state, done):
        """Add a transition to the current episode"""
        self.current_episode.add(state, action, reward, next_state, done)
        
        # If episode ends, store it and start a new one
        if done:
            self.end_episode()
    
    def end_episode(self):
        """End the current episode and store it if valuable"""
        if len(self.current_episode.states) > 0:
            importance = self.current_episode.calculate_importance()
            
            # Store the episode
            self.episodes.append(self.current_episode)
            
            # If important episode, analyze for causal patterns
            if importance > self.importance_threshold:
                self._analyze_causal_patterns(self.current_episode)
            
            # Start a new episode
            self.current_episode = Episode()
    
    def _analyze_causal_patterns(self, episode):
        """Identify causal patterns in important episodes"""
        # Find high-reward moments
        high_reward_indices = [i for i, r in enumerate(episode.rewards) if r > 5]
        
        for idx in high_reward_indices:
            # Look at the sequence leading to high reward (up to 10 steps back)
            start_idx = max(0, idx - 10)
            causal_sequence = episode.get_sequence(start_idx, idx - start_idx + 1)
            
            # Store this causal sequence
            self.causal_memory.append({
                'sequence': causal_sequence,
                'outcome_reward': episode.rewards[idx],
                'context': episode.states[start_idx],  # Initial context
                'frequency': 1  # How often we've seen this pattern
            })
            
            # Optionally: Consolidate similar causal patterns
            self._consolidate_causal_patterns()
    
    def _consolidate_causal_patterns(self):
        """Consolidate similar causal patterns to identify common causes"""
        if len(self.causal_memory) <= 1:
            return
            
        # Compare latest pattern to others
        latest = self.causal_memory[-1]
        
        for i, pattern in enumerate(list(self.causal_memory)[:-1]):
            # Check if contexts are similar
            context_similarity = self._calculate_state_similarity(
                latest['context'], pattern['context'])
            
            # Check if outcomes are similar
            reward_similarity = abs(latest['outcome_reward'] - pattern['outcome_reward']) < 1.0
            
            if context_similarity > 0.8 and reward_similarity:
                # These patterns are similar - consolidate
                pattern['frequency'] += 1
                
                # If this identical pattern already exists, remove the new one
                if context_similarity > 0.95:
                    self.causal_memory.pop()  # Remove the newest duplicate
                    break
    
    def _calculate_state_similarity(self, state1, state2):
        """Calculate similarity between two states"""
        # Convert to numpy arrays if they aren't already
        s1 = np.array(state1)
        s2 = np.array(state2)
        
        # Normalize the states
        s1_norm = np.linalg.norm(s1)
        s2_norm = np.linalg.norm(s2)
        
        if s1_norm == 0 or s2_norm == 0:
            return 0.0
            
        # Compute cosine similarity
        similarity = np.dot(s1, s2) / (s1_norm * s2_norm)
        return max(0, similarity)  # Ensure non-negative
    
    def sample_batch(self, batch_size):
        """Sample a batch of transitions with emphasis on important episodes"""
        if not self.episodes:
            return None
            
        # Select episodes weighted by importance
        importances = np.array([ep.importance for ep in self.episodes])
        if sum(importances) == 0:
            episode_probs = None  # Uniform
        else:
            episode_probs = importances / sum(importances)
            
        # Determinar cuántos episodios seleccionar (no más que los episodios con prob > 0)
        max_episodes = len(self.episodes)
        if episode_probs is not None:
            # Contar elementos con probabilidad no-cero
            non_zero_probs = np.count_nonzero(episode_probs)
            max_episodes = min(max_episodes, non_zero_probs)
        
        # Asegurarnos de no intentar seleccionar más episodios de los disponibles
        episodes_to_select = min(5, max_episodes)
        
        # Obtener episodios aleatorios si no hay suficientes
        if episodes_to_select < 1:
            # Fallback a selección aleatoria uniforme si no hay suficientes episodios
            selected_episodes = [random.randint(0, len(self.episodes) - 1)]
        else:
            # Seleccionar episodios con las probabilidades calculadas
            selected_episodes = np.random.choice(
                len(self.episodes), 
                episodes_to_select,
                p=episode_probs, 
                replace=False
            )
        
        # Calculate transitions per episode
        transitions_per_episode = batch_size // len(selected_episodes)
        
        # Get transitions from selected episodes
        batch = []
        for ep_idx in selected_episodes:
            episode = self.episodes[ep_idx]
            batch.extend(episode.get_random_batch(transitions_per_episode))
            
        # If we need more transitions, get them randomly
        if len(batch) < batch_size:
            additional_needed = batch_size - len(batch)
            random_episode = random.choice(self.episodes)
            batch.extend(random_episode.get_random_batch(additional_needed))
            
        return batch[:batch_size]  # Ensure we return exactly batch_size transitions
    
    def sample_causal_batch(self, state, batch_size):
        """Sample relevant causal sequences based on current state"""
        if not self.causal_memory:
            return self.sample_batch(batch_size)  # Fall back to regular sampling
            
        # Find causal patterns with similar contexts to current state
        similarities = [self._calculate_state_similarity(state, pattern['context']) 
                       for pattern in self.causal_memory]
        
        # Select most relevant patterns
        if max(similarities) < 0.5:
            return self.sample_batch(batch_size)  # No relevant patterns, use regular sampling
            
        # Weight patterns by similarity and frequency
        weights = [sim * pattern['frequency'] for sim, pattern in zip(similarities, self.causal_memory)]
        probs = np.array(weights) / sum(weights)
        
        # Sample patterns
        selected_indices = np.random.choice(
            len(self.causal_memory),
            min(3, len(self.causal_memory)),
            p=probs,
            replace=False
        )
        
        # Collect transitions from selected patterns
        batch = []
        for idx in selected_indices:
            pattern = self.causal_memory[idx]
            batch.extend(pattern['sequence'])
            
        # If we need more transitions, get them randomly
        if len(batch) < batch_size:
            additional = self.sample_batch(batch_size - len(batch))
            if additional:
                batch.extend(additional)
                
        # Shuffle the batch to prevent sequence bias in learning
        random.shuffle(batch)
        return batch[:batch_size]  # Ensure we return exactly batch_size transitions
    
    def perform_counterfactual_analysis(self, state, action, reward, next_state):
        """Analyze what might have happened with different actions"""
        if not self.causal_memory:
            return None
            
        # Find similar states in causal memory
        similar_patterns = []
        for pattern in self.causal_memory:
            for i, (s, a, r, ns, _) in enumerate(pattern['sequence']):
                similarity = self._calculate_state_similarity(state, s)
                if similarity > 0.8 and a != action:
                    # Found similar state where a different action was taken
                    similar_patterns.append({
                        'similarity': similarity,
                        'action': a,
                        'reward': r,
                        'next_state': ns,
                        'pattern_idx': i
                    })
                    
        if not similar_patterns:
            return None
            
        # Sort by similarity
        similar_patterns.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Return the most similar alternative
        return similar_patterns[0]

# ------------------- Curiosity-Based Learning -------------------
class CuriosityModule:
    """Module for intrinsic motivation based on novelty and surprise"""
    def __init__(self, state_size, action_size, device='cpu'):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        
        # Model to predict next state
        self.forward_model = nn.Sequential(
            nn.Linear(state_size + action_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, state_size)
        ).to(device)
        
        # State novelty tracking
        self.state_counts = {}  # Tracks visitation counts
        self.state_visit_decay = 0.99  # Decay factor for counts
        
        # Surprise tracking
        self.surprise_history = deque(maxlen=1000)
        self.max_surprise = 1.0  # For normalization
        
        # Optimizer
        self.optimizer = optim.Adam(self.forward_model.parameters(), lr=0.001)
        
        # Intrinsic reward scaling
        self.curiosity_weight = CURIOSITY_WEIGHT
        self.curiosity_decay = CURIOSITY_DECAY
        
    def hash_state(self, state):
        """Create a discrete hash for a continuous state"""
        # Quantize state values for hashing
        quantized = tuple(round(float(x) * 5) / 5 for x in state)
        return hash(quantized)
    
    def compute_novelty_reward(self, state):
        """Compute reward based on state novelty"""
        state_hash = self.hash_state(state)
        
        # Get visit count (default 0)
        count = self.state_counts.get(state_hash, 0)
        
        # Compute novelty (inversely proportional to visits)
        novelty = 1.0 / (count + 1)
        
        # Update visit count
        self.state_counts[state_hash] = count + 1
        
        # Periodically decay all counts to maintain exploration
        if random.random() < 0.001:  # 0.1% chance each step
            self._decay_counts()
            
        return novelty
    
    def _decay_counts(self):
        """Decay visit counts to encourage revisiting states"""
        for state_hash in self.state_counts:
            self.state_counts[state_hash] *= self.state_visit_decay
    
    def _prepare_action_tensor(self, action):
        """Helper method to prepare action tensor correctly"""
        # Determine if action is already a tensor
        if isinstance(action, torch.Tensor):
            # If it's a tensor, make sure it's properly shaped
            if action.dim() == 0:  # If it's a scalar tensor
                action_tensor = action.unsqueeze(0).long()
            else:
                action_tensor = action.long()
        else:
            # If it's not a tensor, convert it
            action_tensor = torch.tensor([action], dtype=torch.long).to(self.device)
            
        # Convert to one-hot
        return F.one_hot(action_tensor, num_classes=self.action_size).float().to(self.device)
    
    def compute_surprise_reward(self, state, action, next_state):
        """Compute reward based on prediction error (surprise)"""
        try:
            # Convert to tensors
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action_onehot = self._prepare_action_tensor(action)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            
            # Predict next state
            with torch.no_grad():
                pred_input = torch.cat([state_tensor, action_onehot], dim=1)
                predicted_next_state = self.forward_model(pred_input)
                
                # Compute prediction error
                prediction_error = F.mse_loss(predicted_next_state, next_state_tensor).item()
            
            # Update max surprise for normalization
            if prediction_error > self.max_surprise:
                self.max_surprise = prediction_error
                
            # Normalize and store surprise
            normalized_surprise = min(1.0, prediction_error / self.max_surprise)
            self.surprise_history.append(normalized_surprise)
            
            return normalized_surprise
        except Exception as e:
            print(f"Error en compute_surprise_reward: {e}")
            # Fallback: retornar sorpresa media
            return 0.5
    
    def update_forward_model(self, state, action, next_state):
        """Update the forward model to improve predictions"""
        try:
            # Convert to tensors
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action_onehot = self._prepare_action_tensor(action)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            
            # Training step
            self.optimizer.zero_grad()
            pred_input = torch.cat([state_tensor, action_onehot], dim=1)
            predicted_next_state = self.forward_model(pred_input)
            loss = F.mse_loss(predicted_next_state, next_state_tensor)
            loss.backward()
            self.optimizer.step()
            
            return loss.item()
        except Exception as e:
            print(f"Error en update_forward_model: {e}")
            return 0.0
    
    def compute_intrinsic_reward(self, state, action, next_state):
        """Compute combined intrinsic reward"""
        try:
            novelty = self.compute_novelty_reward(state)
            surprise = self.compute_surprise_reward(state, action, next_state)
            
            # Combined reward
            intrinsic_reward = self.curiosity_weight * (0.5 * novelty + 0.5 * surprise)
            
            # Decay curiosity weight over time
            self.curiosity_weight *= self.curiosity_decay
            self.curiosity_weight = max(0.01, self.curiosity_weight)
            
            return intrinsic_reward
        except Exception as e:
            print(f"Error en compute_intrinsic_reward: {e}")
            return 0.01  # Valor pequeño por defecto
    
    def is_novel_state(self, state, threshold=NOVELTY_THRESHOLD):
        """Check if a state is novel enough for exploration"""
        try:
            novelty = self.compute_novelty_reward(state)
            return novelty > threshold
        except Exception as e:
            print(f"Error en is_novel_state: {e}")
            return False

# ------------------- Knowledge Transfer System -------------------
class SkillLibrary:
    """Library of reusable skills learned during training"""
    def __init__(self, state_size, action_size, capacity=SKILL_LIBRARY_SIZE):
        self.state_size = state_size
        self.action_size = action_size
        self.capacity = capacity
        
        # Skill storage
        self.skills = []
        
        # Skill prototypes - network snippets for different skills
        self.skill_prototypes = {}
    
    def add_skill(self, skill_name, skill_network, skill_context, performance):
        """Add a new skill to the library"""
        # Check if we already have this skill
        for skill in self.skills:
            if skill['name'] == skill_name:
                # Update existing skill if new one is better
                if performance > skill['performance']:
                    skill['network'] = skill_network
                    skill['context'] = skill_context
                    skill['performance'] = performance
                    print(f"Updated skill '{skill_name}' (perf: {performance:.2f})")
                return
        
        # Add new skill if we have space
        if len(self.skills) < self.capacity:
            self.skills.append({
                'name': skill_name,
                'network': skill_network,
                'context': skill_context,
                'performance': performance,
                'uses': 0
            })
            print(f"Added new skill '{skill_name}' to library (perf: {performance:.2f})")
        else:
            # Replace worst performing skill
            worst_skill = min(self.skills, key=lambda x: x['performance'])
            if performance > worst_skill['performance']:
                worst_idx = self.skills.index(worst_skill)
                self.skills[worst_idx] = {
                    'name': skill_name,
                    'network': skill_network,
                    'context': skill_context,
                    'performance': performance,
                    'uses': 0
                }
                print(f"Replaced skill '{worst_skill['name']}' with '{skill_name}' (perf: {performance:.2f})")
    
    def extract_navigation_skill(self, agent, env, performance):
        """Extract navigation skill from agent's network"""
        # Create a snapshot of the navigation module
        navigation_state = copy.deepcopy(agent.modules['navigation'].state_dict())
        
        # Store context about the environment
        context = {
            'difficulty': env.difficulty,
            'avg_wall_density': self._calculate_wall_density(env),
            'extracted_from_episode': agent.episodes_trained
        }
        
        # Add to library
        self.add_skill(f"navigation_diff{env.difficulty}", navigation_state, context, performance)
    
    def extract_food_seeking_skill(self, agent, env, performance):
        """Extract food seeking skill from agent's network"""
        # Create a snapshot of relevant modules
        perception_state = copy.deepcopy(agent.modules['perception'].state_dict())
        executive_state = copy.deepcopy(agent.modules['executive'].state_dict())
        
        # Combined network state
        combined_state = {
            'perception': perception_state,
            'executive': executive_state
        }
        
        # Store context
        context = {
            'difficulty': env.difficulty,
            'extracted_from_episode': agent.episodes_trained
        }
        
        # Add to library
        self.add_skill(f"food_seeking_diff{env.difficulty}", combined_state, context, performance)
    
    def _calculate_wall_density(self, env):
        """Calculate wall density in the environment"""
        # Count walls
        vertical_walls = sum(sum(row) for row in env.vertical_walls)
        horizontal_walls = sum(sum(row) for row in env.horizontal_walls)
        total_walls = vertical_walls + horizontal_walls
        
        # Calculate maximum possible walls
        max_vertical = len(env.vertical_walls) * len(env.vertical_walls[0])
        max_horizontal = len(env.horizontal_walls) * len(env.horizontal_walls[0])
        max_walls = max_vertical + max_horizontal
        
        # Density ratio
        return total_walls / max_walls
    
    def find_matching_skill(self, skill_type, context):
        """Find a matching skill for the current context"""
        matching_skills = []
        
        for skill in self.skills:
            if skill_type in skill['name']:
                # Check context similarity
                difficulty_match = skill['context']['difficulty'] == context['difficulty']
                
                # For navigation, also check wall density
                if 'navigation' in skill_type and 'avg_wall_density' in skill['context']:
                    density_diff = abs(skill['context']['avg_wall_density'] - context['avg_wall_density'])
                    density_match = density_diff < 0.2  # Within 20% similarity
                else:
                    density_match = True
                
                if difficulty_match and density_match:
                    matching_skills.append(skill)
        
        if not matching_skills:
            return None
            
        # Return the best performing matching skill
        best_skill = max(matching_skills, key=lambda x: x['performance'])
        best_skill['uses'] += 1
        return best_skill
    
    def apply_skill(self, agent, skill_name, context):
        """Apply a skill from the library to the agent"""
        # Find the skill
        matching_skill = None
        for skill in self.skills:
            if skill['name'] == skill_name:
                matching_skill = skill
                break
                
        if not matching_skill:
            print(f"Skill '{skill_name}' not found in library")
            return False
            
        # Check if context matches
        if matching_skill['context']['difficulty'] != context['difficulty']:
            print(f"Skill '{skill_name}' doesn't match current difficulty")
            return False
            
        # Apply the skill
        if 'navigation' in skill_name:
            # Apply just to navigation module
            agent.modules['navigation'].load_state_dict(matching_skill['network'])
            print(f"Applied navigation skill '{skill_name}' to agent")
        elif 'food_seeking' in skill_name:
            # Apply to perception and executive
            agent.modules['perception'].load_state_dict(matching_skill['network']['perception'])
            agent.modules['executive'].load_state_dict(matching_skill['network']['executive'])
            print(f"Applied food seeking skill '{skill_name}' to agent")
        else:
            print(f"Unknown skill type '{skill_name}'")
            return False
            
        matching_skill['uses'] += 1
        return True
    
    def create_skill_prototypes(self):
        """Create prototype networks for different skills by averaging top performers"""
        skill_types = ['navigation', 'food_seeking']
        difficulties = [0, 1, 2]
        
        for skill_type in skill_types:
            for diff in difficulties:
                # Find all skills of this type and difficulty
                matching_skills = [s for s in self.skills 
                                 if skill_type in s['name'] and s['context']['difficulty'] == diff]
                
                if len(matching_skills) >= 2:
                    # Average the top 2 performers
                    top_skills = sorted(matching_skills, key=lambda x: x['performance'], reverse=True)[:2]
                    
                    # Create prototype depending on skill type
                    if skill_type == 'navigation':
                        prototype = self._average_state_dicts([s['network'] for s in top_skills])
                    else:  # food_seeking
                        perception_dicts = [s['network']['perception'] for s in top_skills]
                        executive_dicts = [s['network']['executive'] for s in top_skills]
                        
                        prototype = {
                            'perception': self._average_state_dicts(perception_dicts),
                            'executive': self._average_state_dicts(executive_dicts)
                        }
                    
                    # Store prototype
                    prototype_name = f"{skill_type}_diff{diff}_prototype"
                    self.skill_prototypes[prototype_name] = {
                        'network': prototype,
                        'difficulty': diff,
                        'performance': np.mean([s['performance'] for s in top_skills])
                    }
                    
                    print(f"Created skill prototype '{prototype_name}'")
    
    def _average_state_dicts(self, state_dicts):
        """Average multiple state dictionaries together"""
        if not state_dicts:
            return None
            
        avg_dict = {}
        for key in state_dicts[0].keys():
            # Skip buffers that shouldn't be averaged
            if 'activation_history' in key or 'usage_count' in key:
                avg_dict[key] = state_dicts[0][key].clone()
                continue
                
            # Average parameter tensors
            avg_dict[key] = sum(sd[key] for sd in state_dicts) / len(state_dicts)
            
        return avg_dict
    
    def apply_prototype(self, agent, skill_type, difficulty):
        """Apply a skill prototype to the agent"""
        prototype_name = f"{skill_type}_diff{difficulty}_prototype"
        
        if prototype_name not in self.skill_prototypes:
            print(f"Prototype '{prototype_name}' not found")
            return False
            
        prototype = self.skill_prototypes[prototype_name]
        
        # Apply prototype based on skill type
        if skill_type == 'navigation':
            agent.modules['navigation'].load_state_dict(prototype['network'])
        elif skill_type == 'food_seeking':
            agent.modules['perception'].load_state_dict(prototype['network']['perception'])
            agent.modules['executive'].load_state_dict(prototype['network']['executive'])
        
        print(f"Applied skill prototype '{prototype_name}' to agent")
        return True

class ConceptAbstraction:
    """System for abstracting high-level concepts from experience"""
    def __init__(self, state_size):
        self.state_size = state_size
        
        # Abstract concepts
        self.concepts = {
            'obstacle_avoidance': {
                'examples': [],
                'prototype': None,
                'performance': 0.0
            },
            'food_seeking': {
                'examples': [],
                'prototype': None,
                'performance': 0.0
            },
            'dead_end_detection': {
                'examples': [],
                'prototype': None,
                'performance': 0.0
            },
            'corridor_navigation': {
                'examples': [],
                'prototype': None,
                'performance': 0.0
            }
        }
        
        # Feature importance for each concept
        self.feature_importance = {concept: np.zeros(state_size) for concept in self.concepts}
        
        # Concept recognizer - identifies which concept is relevant in a state
        self.concept_recognizer = None
    
    def add_example(self, concept, state, action, result):
        """Add an example of a concept in action"""
        if concept not in self.concepts:
            return
            
        # Store example
        self.concepts[concept]['examples'].append({
            'state': state,
            'action': action,
            'result': result  # Success or failure
        })
        
        # Limit size
        if len(self.concepts[concept]['examples']) > 100:
            # Keep most recent examples
            self.concepts[concept]['examples'] = self.concepts[concept]['examples'][-100:]
    
    def update_feature_importance(self, concept):
        """Update feature importance for a concept based on examples"""
        if concept not in self.concepts or not self.concepts[concept]['examples']:
            return
            
        # Get successful examples
        successful = [ex for ex in self.concepts[concept]['examples'] if ex['result'] == 'success']
        failed = [ex for ex in self.concepts[concept]['examples'] if ex['result'] == 'failure']
        
        if not successful or not failed:
            return
            
        # Convert to arrays
        successful_states = np.array([ex['state'] for ex in successful])
        failed_states = np.array([ex['state'] for ex in failed])
        
        # Calculate mean difference between successful and failed states
        if len(successful_states) > 0 and len(failed_states) > 0:
            successful_mean = np.mean(successful_states, axis=0)
            failed_mean = np.mean(failed_states, axis=0)
            
            # Features with larger differences are more important for this concept
            importance = np.abs(successful_mean - failed_mean)
            
            # Normalize
            if np.sum(importance) > 0:
                importance = importance / np.sum(importance)
                
            self.feature_importance[concept] = importance
    
    def create_concept_prototype(self, concept):
        """Create a prototype for a concept based on successful examples"""
        if concept not in self.concepts:
            return
            
        # Get successful examples
        successful = [ex for ex in self.concepts[concept]['examples'] if ex['result'] == 'success']
        
        if not successful:
            return
            
        # Create prototype by averaging successful states
        states = np.array([ex['state'] for ex in successful])
        prototype = np.mean(states, axis=0)
        
        # Store prototype
        self.concepts[concept]['prototype'] = prototype
        
        # Calculate performance - ratio of successful examples
        total = len(self.concepts[concept]['examples'])
        success_count = len(successful)
        if total > 0:
            self.concepts[concept]['performance'] = success_count / total
    
    def identify_relevant_concept(self, state):
        """Identify which concept is most relevant for a given state"""
        if not any(c['prototype'] is not None for c in self.concepts.values()):
            return None
            
        # Calculate similarity to each concept prototype
        similarities = {}
        for concept, data in self.concepts.items():
            if data['prototype'] is not None:
                # Weight state features by importance for this concept
                weighted_state = np.array(state) * self.feature_importance[concept]
                weighted_prototype = data['prototype'] * self.feature_importance[concept]
                
                # Calculate similarity
                similarity = 1.0 - np.mean(np.abs(weighted_state - weighted_prototype))
                similarities[concept] = similarity
        
        if not similarities:
            return None
            
        # Return most similar concept
        return max(similarities.items(), key=lambda x: x[1])
    
    def get_concept_embedding(self, concept):
        """Get a compact embedding representing a concept"""
        if concept not in self.concepts or self.concepts[concept]['prototype'] is None:
            return None
            
        # Create compact embedding by keeping only the most important features
        importance = self.feature_importance[concept]
        
        # Get indices of 5 most important features
        top_indices = np.argsort(importance)[-5:]
        
        # Create embedding with just these features
        prototype = self.concepts[concept]['prototype']
        embedding = {i: prototype[i] for i in top_indices}
        
        return embedding

# ------------------- Advanced Neural Agent -------------------
class ModularAdvancedAgent:
    """Main agent class integrating all advanced neural systems"""
    def __init__(self, state_size, action_size):
        # Llamar a optimización de CUDA primero
        optimize_cuda_settings()
        
        self.state_size = state_size
        self.action_size = action_size
        
        # Device (CPU/GPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"ModularAdvancedAgent initialized using: {self.device}")
        
        # Base hyperparameters (now adjustable via meta-learning)
        self.gamma = 0.99
        self.learning_rate = 0.0005
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 64
        
        # Create modular neural architecture
        self._init_modular_architecture()
        
        # Initialize optimizer
        self.setup_optimizer()
        
        # Initialize episodic memory
        self.episodic_memory = EpisodicMemory()
        
        # Initialize curiosity module
        self.curiosity = CuriosityModule(state_size, action_size, self.device)
        
        # Initialize skill library
        self.skill_library = SkillLibrary(state_size, action_size)
        
        # Initialize concept abstraction
        self.concept_abstraction = ConceptAbstraction(state_size)
        
        # Meta-controller for adaptive learning
        self.meta_controller = MetaController(self)
        
        # Training metrics
        self.episodes_trained = 0
        self.losses = []
        self.rewards = []
        self.q_values = []
        
        # Recent states/actions for short-term memory
        self.recent_states = deque(maxlen=10)
        self.recent_actions = deque(maxlen=10)
        
        # Buffer for storing transition until episode ends
        self.transition_buffer = []
        
        # Performance metrics for adaptive batch sizing
        self.performance_metrics = {}
        
        print("Advanced neural agent initialized with modular architecture")
    
    def to_gpu(self):
        """Envía explícitamente todos los módulos a GPU y verifica la transferencia"""
        if not torch.cuda.is_available():
            print("GPU no disponible, usando CPU")
            return
        
        # Update device
        self.device = torch.device("cuda")
        
        # Move all modules to GPU
        for name, module in self.modules.items():
            self.modules[name] = module.to(self.device)
            
            # Verificar que el módulo está en la GPU
            for param in module.parameters():
                if param.device.type != 'cuda':
                    print(f"ADVERTENCIA: El módulo {name} contiene parámetros que no están en la GPU")
        
        # Move curiosity module components to GPU
        if hasattr(self, 'curiosity'):
            self.curiosity.device = self.device
            self.curiosity.forward_model = self.curiosity.forward_model.to(self.device)
        
        # Verificar asignación de memoria
        print(f"Memoria GPU después de transferir módulos: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        
        return
    
    def _init_modular_architecture(self):
        """Initialize the modular neural architecture"""
        # Define module sizes
        perception_out_size = 64
        navigation_out_size = 32
        prediction_out_size = 2
        
        # Create modules
        self.modules = {
            'perception': PerceptionModule(self.state_size, perception_out_size).to(self.device),
            'navigation': NavigationModule(perception_out_size + 4 + 4, navigation_out_size).to(self.device),
            'prediction': PredictionModule(self.state_size, self.action_size).to(self.device),
            'executive': ExecutiveModule(perception_out_size, navigation_out_size, 
                                       prediction_out_size, self.action_size).to(self.device)
        }
    
    def setup_optimizer(self):
        """Setup optimizer with all trainable parameters"""
        # Collect all parameters from all modules
        parameters = []
        for name, module in self.modules.items():
            parameters.extend(module.parameters())
            
        # Create optimizer
        self.optimizer = optim.Adam(parameters, lr=self.learning_rate)
    
    def remember(self, state, action, reward, next_state, done):
        """Store transition in memory"""
        # Store in episodic memory
        self.episodic_memory.add(state, action, reward, next_state, done)
        
        # Store in transition buffer
        self.transition_buffer.append((state, action, reward, next_state, done))
        
        # Update recent states/actions
        self.recent_states.append(state)
        self.recent_actions.append(action)
        
        # If episode ends, process the complete episode
        if done:
            self.episodic_memory.end_episode()
            self.transition_buffer = []
            
            # Update concept abstractions
            self._update_concept_examples()
    
    def _update_concept_examples(self):
        """Update concept examples based on the recent episode"""
        if len(self.transition_buffer) < 2:
            return
            
        for i in range(len(self.transition_buffer) - 1):
            state, action, reward, next_state, _ = self.transition_buffer[i]
            
            # Detect obstacle avoidance
            if any(state[0:3]):  # Danger signals in state
                result = 'success' if reward > -1 else 'failure'
                self.concept_abstraction.add_example('obstacle_avoidance', state, action, result)
            
            # Detect food seeking
            if any(state[7:11]):  # Food direction signals
                # Success if we got closer to food
                old_dist = state[11] * (self.state_size + self.state_size)  # Denormalize
                new_dist = next_state[11] * (self.state_size + self.state_size)
                result = 'success' if new_dist < old_dist else 'failure'
                self.concept_abstraction.add_example('food_seeking', state, action, result)
            
            # Detect dead end detection
            if sum(state[0:3]) >= 2:  # At least 2 danger signals (potential dead end)
                result = 'success' if reward > -5 else 'failure'
                self.concept_abstraction.add_example('dead_end_detection', state, action, result)
            
            # Detect corridor navigation
            corridor_pattern = [1, 0, 1]  # Danger on left and right, but not straight
            if list(state[0:3]) == corridor_pattern:  # Convert to list for comparison
                # Success if we continue forward in corridor
                result = 'success' if action == 0 and reward > 0 else 'failure'
                self.concept_abstraction.add_example('corridor_navigation', state, action, result)
        
        # Update prototypes and feature importance
        for concept in self.concept_abstraction.concepts:
            self.concept_abstraction.update_feature_importance(concept)
            self.concept_abstraction.create_concept_prototype(concept)
    
    def act(self, state):
        """Select an action based on current state with manejo robusto de errores"""
        # Exploration vs exploitation
        if np.random.rand() <= self.epsilon:
            # Explore: Choose random action
            # But first check if we should be curious about particular actions
            try:
                if self.curiosity.is_novel_state(state) and random.random() < 0.5:
                    # For novel states, try all actions in prediction model
                    best_action = self._find_most_curious_action(state)
                    return best_action
            except Exception as e:
                print(f"Error en exploración de curiosidad: {e}")
            
            return random.randrange(self.action_size)
        
        # Convert state to tensor con manejo seguro
        try:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        except Exception as e:
            print(f"Error convirtiendo estado a tensor: {e}")
            # Fallback: usar tensor de ceros
            state_tensor = torch.zeros(1, self.state_size, device=self.device)
        
        # Exploit: Use neural modules to select best action
        with torch.no_grad():
            try:
                # 1. Perception module processes raw state
                perception_features = self.modules['perception'](state_tensor)
            except Exception as e:
                print(f"Error en módulo de percepción: {e}")
                # Tensor de respaldo
                perception_features = torch.zeros(1, self.modules['perception'].output_size, device=self.device)
            
            try:
                # 2. Extract position and target from state for navigation
                position = torch.FloatTensor([[state[3], state[4], state[5], state[6]]]).to(self.device)  # Direction one-hot
                target = torch.FloatTensor([[state[7], state[8], state[9], state[10]]]).to(self.device)   # Food direction
            except Exception as e:
                print(f"Error extrayendo posición/objetivo: {e}")
                # Tensores de respaldo
                position = torch.zeros(1, 4, device=self.device)
                target = torch.zeros(1, 4, device=self.device)
            
            try:
                # 3. Navigation module suggests actions
                navigation_output = self.modules['navigation'](perception_features, position, target)
            except Exception as e:
                print(f"Error en módulo de navegación: {e}")
                # Tensor de respaldo
                navigation_output = torch.zeros(1, self.modules['navigation'].pathfinder[-1].current_output_size, 
                                            device=self.device)
            
            # 4. Prediction module con manejo de errores
            prediction_output = None
            try:
                # Collect all action predictions
                action_tensors = []
                prediction_outputs = []
                
                for a in range(self.action_size):
                    try:
                        # Create one-hot action
                        action_onehot = F.one_hot(torch.tensor([a]), self.action_size).float().to(self.device)
                        action_tensors.append(action_onehot)
                        
                        # Get predictions
                        pred_next_state = self.modules['prediction'].predict_next_state(state_tensor, action_onehot)
                        pred_reward = self.modules['prediction'].predict_reward(state_tensor, action_onehot)
                        uncertainty = self.modules['prediction'].estimate_uncertainty(state_tensor, action_onehot)
                        
                        # Combine predictions
                        pred_output = torch.cat([pred_reward, uncertainty], dim=1)
                        prediction_outputs.append(pred_output)
                    except Exception as e:
                        # Si falla para una acción, usar tensor de ceros
                        print(f"Error prediciendo acción {a}: {e}")
                        prediction_outputs.append(torch.zeros(1, 2, device=self.device))
                
                # Combinar las salidas de predicción con manejo de excepciones
                if prediction_outputs:
                    try:
                        # Intentar combinar como una matriz 3D [batch, actions, features]
                        if all(p.dim() == prediction_outputs[0].dim() for p in prediction_outputs):
                            prediction_output = torch.stack(prediction_outputs, dim=0)
                            if prediction_output.dim() == 2:
                                prediction_output = prediction_output.unsqueeze(0)
                        else:
                            # Si hay inconsistencia en dimensiones, concatenar en un vector plano
                            prediction_output = torch.cat([p.flatten().unsqueeze(0) for p in prediction_outputs], dim=1)
                            
                    except Exception as e:
                        print(f"Error combinando predicciones: {e}")
                        # Tensor de respaldo
                        prediction_output = torch.zeros(1, self.action_size * 2, device=self.device)
                else:
                    # No se pudo generar ninguna predicción
                    prediction_output = torch.zeros(1, self.action_size * 2, device=self.device)
                    
            except Exception as e:
                print(f"Error general en módulo de predicción: {e}")
                # Tensor de respaldo
                prediction_output = torch.zeros(1, self.action_size * 2, device=self.device)
            
            try:
                # 5. Executive module integrates all signals and selects final action
                q_values = self.modules['executive'](perception_features, navigation_output, prediction_output)
                
                # Register average Q-value for monitoring
                self.q_values.append(float(torch.mean(q_values).cpu().numpy()))
                
                # Return action with highest Q-value
                return int(torch.argmax(q_values).item())
            except Exception as e:
                print(f"Error en módulo ejecutivo: {e}")
                # Acción aleatoria como fallback
                return random.randrange(self.action_size)
    
    def _find_most_curious_action(self, state):
        """Find the action that would lead to the most curious outcome"""
        try:
            # Try all actions in the prediction model
            curiosity_scores = []
            for action in range(self.action_size):
                # Use the prediction module to predict next state
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                action_onehot = F.one_hot(torch.tensor([action]), self.action_size).float().to(self.device)
                
                with torch.no_grad():
                    # Predict next state
                    pred_next_state = self.modules['prediction'].predict_next_state(state_tensor, action_onehot)
                    # Get uncertainty
                    uncertainty = self.modules['prediction'].estimate_uncertainty(state_tensor, action_onehot)
                    
                    # Higher uncertainty = more curious
                    curiosity_scores.append(float(uncertainty.item()))
            
            # Return action with highest curiosity score
            return int(np.argmax(curiosity_scores))
        except Exception as e:
            print(f"Error en _find_most_curious_action: {e}")
            # Return random action as fallback
            return random.randrange(self.action_size)
    
    def monitor_performance(self, start_time):
        """Monitor performance metrics for batch size adaptation"""
        # Calculate training time
        training_time = time.time() - start_time
        
        # Log performance data periodically
        if self.episodes_trained % 50 == 0:
            try:
                import psutil
                process = psutil.Process()
                memory_info = process.memory_info()
                memory_mb = memory_info.rss / (1024 * 1024)
                
                print(f"Performance Metrics:")
                print(f"  Training Time: {training_time:.2f} ms")
                print(f"  Memory Usage: {memory_mb:.2f} MB")
                print(f"  Current Batch Size: {self.batch_size}")
                
                # Store metrics for meta-controller
                self.performance_metrics = {
                    'training_time': training_time,
                    'memory_usage': memory_mb,
                    'batch_size': self.batch_size
                }
                
            except ImportError:
                print("psutil not installed. Install with: pip install psutil")
                self.performance_metrics = {
                    'training_time': training_time,
                    'batch_size': self.batch_size
                }
        
        return training_time
    
    def process_batch_safely(self, states, actions, rewards, next_states, dones):
        """
        Procesa un batch de experiencias con manejo robusto de excepciones
        y dimensiones de tensores.
        
        Esto mejora la estabilidad del entrenamiento ante datos inesperados.
        """
        # Conversión segura a tensores con dimensiones consistentes
        try:
            states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
            actions = torch.tensor(actions, dtype=torch.long, device=self.device)
            rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
            next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
            dones = torch.tensor(dones, dtype=torch.float32, device=self.device)
        except Exception as e:
            print(f"Error en conversión de batch a tensores: {e}")
            # Si falla, reducir el tamaño del batch y reintentar
            try:
                max_safe_size = min(len(states), len(actions), len(rewards), len(next_states), len(dones))
                safe_size = max(1, max_safe_size // 2)  # Reducir a la mitad para mayor seguridad
                
                states = torch.tensor(np.array(states[:safe_size]), dtype=torch.float32, device=self.device)
                actions = torch.tensor(actions[:safe_size], dtype=torch.long, device=self.device)
                rewards = torch.tensor(rewards[:safe_size], dtype=torch.float32, device=self.device)
                next_states = torch.tensor(np.array(next_states[:safe_size]), dtype=torch.float32, device=self.device)
                dones = torch.tensor(dones[:safe_size], dtype=torch.float32, device=self.device)
            except Exception as e2:
                print(f"Error crítico en procesamiento de batch: {e2}")
                return None  # Devolver None indica que el batch no se puede procesar
        
        # Usar bloques try/except para cada etapa del procesamiento
        try:
            # 1. Current state processing
            with torch.no_grad():
                perception_features = self.modules['perception'](states)
                
                # Extract position and target information with manejo defensivo
                try:
                    positions = torch.stack([
                        torch.tensor([s[3], s[4], s[5], s[6]], dtype=torch.float32, device=self.device) 
                        for s in states
                    ])
                    
                    targets = torch.stack([
                        torch.tensor([s[7], s[8], s[9], s[10]], dtype=torch.float32, device=self.device)
                        for s in states
                    ])
                except Exception as e:
                    print(f"Error extrayendo posiciones/objetivos: {e}")
                    # Crear tensores de respaldo
                    positions = torch.zeros(states.size(0), 4, device=self.device)
                    targets = torch.zeros(states.size(0), 4, device=self.device)
            
            # Procesar navegación
            try:
                navigation_output = self.modules['navigation'](perception_features, positions, targets)
            except Exception as e:
                print(f"Error en módulo de navegación: {e}")
                navigation_output = torch.zeros(states.size(0), 
                                            self.modules['navigation'].pathfinder[-1].current_output_size, 
                                            device=self.device)
            
            # For prediction module, we need the actual actions taken
            try:
                action_onehot = F.one_hot(actions, self.action_size).float()
            except Exception as e:
                print(f"Error en codificación one-hot: {e}")
                action_onehot = torch.zeros(states.size(0), self.action_size, device=self.device)
                for i, a in enumerate(actions):
                    if 0 <= a < self.action_size:
                        action_onehot[i, a] = 1.0
            
            # Procesar predicciones con gestión de errores para cada paso
            prediction_outputs = []
            for i in range(len(states)):
                try:
                    # Use actual state and action
                    state_i = states[i:i+1]
                    action_i = action_onehot[i:i+1]
                    
                    # Get prediction outputs con manejo de errores
                    with torch.no_grad():
                        pred_reward = self.modules['prediction'].predict_reward(state_i, action_i)
                        uncertainty = self.modules['prediction'].estimate_uncertainty(state_i, action_i)
                    
                    prediction_output = torch.cat([pred_reward, uncertainty], dim=1)
                    prediction_outputs.append(prediction_output)
                except Exception as e:
                    # Si falla para este ejemplo, crear un tensor de respaldo
                    print(f"Error en predicción para elemento {i}: {e}")
                    prediction_outputs.append(torch.zeros(1, 2, device=self.device))
            
            # Combinar predicciones de manera segura
            try:
                prediction_output = torch.cat(prediction_outputs, dim=0)
            except Exception as e:
                print(f"Error combinando predicciones: {e}")
                # Tensor de respaldo
                prediction_output = torch.zeros(states.size(0), 2, device=self.device)
            
            # Executive module integration
            try:
                current_q_values = self.modules['executive'](perception_features, navigation_output, prediction_output)
                q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
            except Exception as e:
                print(f"Error en módulo ejecutivo: {e}")
                # Devolver valores Q de respaldo
                return None
            
            # Next state processing para valores objetivo
            with torch.no_grad():
                try:
                    # Procesar estados siguientes
                    next_perception_features = self.modules['perception'](next_states)
                    
                    # Información de posición y objetivo con manejo de errores
                    try:
                        next_positions = torch.stack([
                            torch.tensor([s[3], s[4], s[5], s[6]], dtype=torch.float32, device=self.device)
                            for s in next_states
                        ])
                        
                        next_targets = torch.stack([
                            torch.tensor([s[7], s[8], s[9], s[10]], dtype=torch.float32, device=self.device)
                            for s in next_states
                        ])
                    except Exception as e:
                        print(f"Error extrayendo posiciones/objetivos siguientes: {e}")
                        next_positions = torch.zeros(next_states.size(0), 4, device=self.device)
                        next_targets = torch.zeros(next_states.size(0), 4, device=self.device)
                    
                    next_navigation_output = self.modules['navigation'](next_perception_features, next_positions, next_targets)
                    
                    # Simplificar predicciones para estados siguientes
                    prediction_output_zero = torch.zeros(len(states), 2, device=self.device)
                    next_q_values = self.modules['executive'](next_perception_features, next_navigation_output, prediction_output_zero)
                    
                    # Double DQN approach
                    best_actions = next_q_values.argmax(dim=1)
                    max_next_q_values = next_q_values.gather(1, best_actions.unsqueeze(1)).squeeze(1)
                    
                    # Calcular valores Q objetivo
                    targets = rewards + (1 - dones) * self.gamma * max_next_q_values
                except Exception as e:
                    print(f"Error en procesamiento de estado siguiente: {e}")
                    # Usar recompensas como objetivo si falla el cálculo completo
                    targets = rewards
            
            # Todo el procesamiento fue exitoso, devolver valores
            return q_values, targets
        
        except Exception as e:
            print(f"Error general en procesamiento de batch: {e}")
            return None
    
    def train(self):
        """Versión mejorada de train con mejor manejo de errores y excepciones"""
        # Check if we have enough samples
        if len(self.episodic_memory.episodes) == 0:
            return 0
        
        # Record start time for performance monitoring
        start_time = time.time()
        
        # Verificar si necesitamos entrenar (reducir frecuencia de entrenamiento)
        if self.episodes_trained % 2 != 0 and self.episodes_trained > 100:
            # Entrenar cada 2 episodios después de cierto aprendizaje
            return 0
        
        # Sample from episodic memory with current batch size
        if len(self.recent_states) > 0:
            # Use current state for context
            current_state = self.recent_states[-1]
            batch = self.episodic_memory.sample_causal_batch(current_state, self.batch_size)
        else:
            # Fall back to regular sampling
            batch = self.episodic_memory.sample_batch(self.batch_size)
        
        if not batch:
            return 0
        
        # Unpack batch
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Procesar batch de manera segura
        result = self.process_batch_safely(states, actions, rewards, next_states, dones)
        
        if result is None:
            # El procesamiento falló, omitir este batch
            print("Omitiendo batch debido a errores de procesamiento")
            return 0
        
        q_values, targets = result
        
        # Calculate loss
        loss = F.smooth_l1_loss(q_values, targets)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients to prevent explosion
        torch.nn.utils.clip_grad_norm_(
            [p for module in self.modules.values() for p in module.parameters()], 
            1.0
        )
        self.optimizer.step()
        
        # Reducir frecuencia de aplicación de aprendizaje hebbiano
        if self.episodes_trained % 3 == 0:  # Aplicar cada 3 episodios
            # Apply Hebbian learning to dynamic layers
            for module in self.modules.values():
                for layer in module.modules():
                    if isinstance(layer, DynamicLayer):
                        try:
                            layer.apply_hebbian_learning()
                        except Exception as e:
                            print(f"Error en aprendizaje hebbiano: {e}")
        
        # Reducir actualizaciones del modelo de predicción
        if self.episodes_trained % 5 == 0:  # Actualizar cada 5 episodios
            try:
                # Seleccionar un subconjunto de transiciones para actualizar el modelo
                update_indices = np.random.choice(len(states), min(10, len(states)), replace=False)
                
                for idx in update_indices:
                    try:
                        state = states[idx].cpu().numpy()
                        action = int(actions[idx].item())
                        next_state = next_states[idx].cpu().numpy()
                        
                        # Update forward model in curiosity module
                        self.curiosity.update_forward_model(state, action, next_state)
                    except Exception as e:
                        continue
            except Exception as e:
                print(f"Error actualizando modelo de predicción: {e}")
        
        # Monitor performance metrics
        self.monitor_performance(start_time)
        
        # Return loss for monitoring
        self.losses.append(float(loss.item()))
        return float(loss.item())
    
    def compute_intrinsic_reward(self, state, action, next_state):
        """Compute intrinsic reward based on curiosity"""
        return self.curiosity.compute_intrinsic_reward(state, action, next_state)
    
    def update_modules(self, episode_stats):
        """Update module attention based on performance stats"""
        # Extract performance metrics for each module
        perception_perf = episode_stats.get('perception_utility', 0.3)
        navigation_perf = episode_stats.get('navigation_utility', 0.3)
        prediction_perf = episode_stats.get('prediction_utility', 0.3)
        
        # Update module attention
        self.modules['executive'].update_attention([perception_perf, navigation_perf, prediction_perf])
        
        # Update perception module attention
        if 'feature_importance' in episode_stats:
            self.modules['perception'].update_attention(
                torch.FloatTensor(episode_stats['feature_importance']).to(self.device)
            )
        
        # Update navigation bias based on successful actions
        if 'successful_actions' in episode_stats:
            self.modules['navigation'].update_bias(episode_stats['successful_actions'])
    
    def extract_skills(self, env, episode_stats):
        """Extract reusable skills based on successful episodes"""
        if episode_stats.get('score', 0) < 5:
            return  # Only extract skills from successful episodes
            
        # Calculate overall performance metric
        performance = episode_stats.get('score', 0) + 0.1 * episode_stats.get('reward', 0)
        
        # Extract navigation skill if good performance in maze environment
        if env.difficulty > 0 and episode_stats.get('navigation_utility', 0) > 0.6:
            self.skill_library.extract_navigation_skill(self, env, performance)
            
        # Extract food seeking skill if good at finding food
        if episode_stats.get('food_efficiency', 0) > 0.7:
            self.skill_library.extract_food_seeking_skill(self, env, performance)
            
        # Periodically create skill prototypes
        if self.episodes_trained % 100 == 0 and len(self.skill_library.skills) >= 3:
            self.skill_library.create_skill_prototypes()
    
    def transfer_knowledge(self, env):
        """Transfer knowledge when environment changes"""
        # Create context
        context = {
            'difficulty': env.difficulty,
            'avg_wall_density': sum(sum(row) for row in env.vertical_walls) / (len(env.vertical_walls) * len(env.vertical_walls[0])) +
                             sum(sum(row) for row in env.horizontal_walls) / (len(env.horizontal_walls) * len(env.horizontal_walls[0]))
        }
        
        # Try to find matching navigation skill
        navigation_skill = self.skill_library.find_matching_skill('navigation', context)
        if navigation_skill:
            # Apply the navigation skill
            self.modules['navigation'].load_state_dict(navigation_skill['network'])
            print(f"Transferred navigation knowledge from skill '{navigation_skill['name']}'")
            return True
            
        # Try to use prototype if no exact match
        return self.skill_library.apply_prototype(self, 'navigation', env.difficulty)
    
    def update_meta_learning(self, env, episode_reward, episode_score):
        """Update meta-learning system after each episode"""
        self.episodes_trained += 1
        self.rewards.append(episode_reward)
        
        # Record performance
        self.meta_controller.record_performance(episode_reward, episode_score)
        
        # Every 10 episodes, adjust hyperparameters
        if self.episodes_trained % 10 == 0:
            self.meta_controller.adjust_hyperparameters()
            
        # Every 50 episodes, consider architecture changes
        if self.episodes_trained % 50 == 0:
            self.meta_controller.adjust_architecture()
            
        # Every 20 episodes, adjust learning strategy
        if self.episodes_trained % 20 == 0:
            self.meta_controller.adjust_learning_strategy()
            
        # Update curriculum difficulty
        return self.meta_controller.update_curriculum(env)
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save(self, filename):
        """Save the agent's state"""
        # Prepare modules state dicts
        modules_state = {name: module.state_dict() for name, module in self.modules.items()}
        
        # Prepare the full state
        state = {
            'modules': modules_state,
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'episodes_trained': self.episodes_trained,
            'losses': self.losses,
            'rewards': self.rewards,
            'q_values': self.q_values,
            'skill_library': self.skill_library.skills,
            'concept_prototypes': {
                concept: {
                    'prototype': data['prototype'],
                    'feature_importance': self.concept_abstraction.feature_importance[concept]
                }
                for concept, data in self.concept_abstraction.concepts.items()
                if data['prototype'] is not None
            },
            'meta_controller': {
                'curriculum_level': self.meta_controller.curriculum_level,
                'strategy': self.meta_controller.current_strategy
            }
        }
        
        torch.save(state, filename)
        print(f"Agent saved to {filename}")
    
    def load(self, filename):
        """Load the agent's state"""
        state = torch.load(filename, map_location=self.device)
        
        # Load modules
        for name, module_state in state['modules'].items():
            if name in self.modules:
                self.modules[name].load_state_dict(module_state)
        
        # Load optimizer
        self.optimizer.load_state_dict(state['optimizer'])
        
        # Load other attributes
        self.epsilon = state['epsilon']
        self.episodes_trained = state['episodes_trained']
        self.losses = state['losses']
        self.rewards = state['rewards']
        self.q_values = state['q_values']
        
        # Load skill library
        if 'skill_library' in state:
            self.skill_library.skills = state['skill_library']
        
        # Load concept prototypes
        if 'concept_prototypes' in state:
            for concept, data in state['concept_prototypes'].items():
                if concept in self.concept_abstraction.concepts:
                    self.concept_abstraction.concepts[concept]['prototype'] = data['prototype']
                    self.concept_abstraction.feature_importance[concept] = data['feature_importance']
        
        # Load meta-controller state
        if 'meta_controller' in state:
            self.meta_controller.curriculum_level = state['meta_controller']['curriculum_level']
            self.meta_controller.current_strategy = state['meta_controller']['strategy']
        
        print(f"Agent loaded from {filename}")
        return self.episodes_trained

# ------------------- Integration Helper Functions -------------------
def create_advanced_agent(state_size, action_size):
    """Create an instance of the advanced neural agent with optimized GPU usage"""
    agent = ModularAdvancedAgent(state_size, action_size)
    
    # Forzar el uso de CUDA si está disponible
    if torch.cuda.is_available():
        # Asegurar que todos los módulos estén en la GPU
        agent.to_gpu()
        
        # Verificar asignación de memoria CUDA
        print(f"CUDA disponible: {torch.cuda.is_available()}")
        print(f"Dispositivo actual: {agent.device}")
        print(f"Memoria CUDA reservada: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"Memoria CUDA reservada (caché): {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
        
        # Optimizar rendimiento CUDA
        torch.backends.cudnn.benchmark = True
    
    return agent

def process_state_action_reward(agent, state, action, reward, next_state, done):
    """Process a state-action-reward transition with the advanced agent"""
    # Add intrinsic reward based on curiosity
    intrinsic_reward = agent.compute_intrinsic_reward(state, action, next_state)
    combined_reward = reward + intrinsic_reward
    
    # Remember the transition with combined reward
    agent.remember(state, action, combined_reward, next_state, done)
    
    return combined_reward

def train_advanced_agent(agent):
    """Train the advanced agent on a batch of experiences"""
    return agent.train()

def get_action(agent, state):
    """Get an action from the agent for the given state"""
    return agent.act(state)

def update_after_episode(agent, env, episode_reward, episode_score, episode_stats=None):
    """Update agent after an episode completes"""
    # Default stats if none provided
    if episode_stats is None:
        episode_stats = {
            'perception_utility': 0.33,
            'navigation_utility': 0.33,
            'prediction_utility': 0.33,
            'score': episode_score,
            'reward': episode_reward,
            'food_efficiency': min(1.0, episode_score / 10)
        }
    
    # Update modules based on performance
    agent.update_modules(episode_stats)
    
    # Extract reusable skills
    agent.extract_skills(env, episode_stats)
    
    # Update meta-learning
    curriculum_changed = agent.update_meta_learning(env, episode_reward, episode_score)
    
    # Decay exploration rate
    agent.decay_epsilon()
    
    return curriculum_changed

def save_advanced_agent(agent, filename):
    """Save the advanced agent's state"""
    agent.save(filename)

def load_advanced_agent(agent, filename):
    """Load the advanced agent's state"""
    return agent.load(filename)

def transfer_to_new_environment(agent, env):
    """Transfer knowledge when moving to a new environment"""
    return agent.transfer_knowledge(env)

# ------------------- Visualization Functions -------------------
def visualize_module_structure(agent, output_file="module_structure.png"):
    """Visualize the structure of neural modules"""
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
        
        # Create graph
        G = nx.DiGraph()
        
        # Add nodes for input and output
        G.add_node("Input State", pos=(0, 0), color='green')
        G.add_node("Action Output", pos=(4, 0), color='red')
        
        # Add nodes for each module
        G.add_node("Perception", pos=(1, 1), color='blue')
        G.add_node("Navigation", pos=(2, 2), color='blue')
        G.add_node("Prediction", pos=(2, 0), color='blue')
        G.add_node("Executive", pos=(3, 1), color='purple')
        
        # Add edges
        G.add_edge("Input State", "Perception")
        G.add_edge("Input State", "Prediction")
        G.add_edge("Perception", "Navigation")
        G.add_edge("Perception", "Executive")
        G.add_edge("Navigation", "Executive")
        G.add_edge("Prediction", "Executive")
        G.add_edge("Executive", "Action Output")
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Draw the graph
        pos = nx.get_node_attributes(G, 'pos')
        colors = [G.nodes[n]['color'] for n in G.nodes]
        nx.draw(G, pos, with_labels=True, node_color=colors, node_size=2000, 
               font_size=10, font_weight='bold', arrows=True, 
               arrowsize=20, arrowstyle='->')
        
        # Save figure
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()
        
        print(f"Module structure visualization saved to {output_file}")
        return True
    except ImportError:
        print("Visualization requires matplotlib and networkx")
        return False

def visualize_training_metrics(agent, output_file="training_metrics.png"):
    """Visualize training metrics over time"""
    try:
        import matplotlib.pyplot as plt
        
        # Create figure with subplots
        fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        
        # Plot losses
        if agent.losses:
            axs[0].plot(agent.losses)
            axs[0].set_title('Loss Over Time')
            axs[0].set_ylabel('Loss')
        
        # Plot rewards
        if agent.rewards:
            axs[1].plot(agent.rewards)
            axs[1].set_title('Episode Rewards')
            axs[1].set_ylabel('Reward')
        
        # Plot Q-values
        if agent.q_values:
            axs[2].plot(agent.q_values)
            axs[2].set_title('Average Q-Value')
            axs[2].set_ylabel('Q-Value')
            axs[2].set_xlabel('Training Steps')
        
        # Save figure
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()
        
        print(f"Training metrics visualization saved to {output_file}")
        return True
    except ImportError:
        print("Visualization requires matplotlib")
        return False

def visualize_dynamic_growth(agent, output_file="neural_growth.png"):
    """Visualize the dynamic growth of neural layers"""
    try:
        import matplotlib.pyplot as plt
        
        # Count neurons in each dynamic layer
        layer_sizes = []
        layer_names = []
        
        for name, module in agent.modules.items():
            layer_idx = 0
            for child in module.modules():
                if isinstance(child, DynamicLayer):
                    layer_sizes.append(child.current_output_size)
                    layer_names.append(f"{name}_{layer_idx}")
                    layer_idx += 1
        
        # Create bar chart
        plt.figure(figsize=(12, 6))
        bars = plt.bar(layer_names, layer_sizes)
        
        # Add annotations
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{int(height)}', ha='center', va='bottom')
        
        plt.title('Neural Network Dynamic Growth')
        plt.ylabel('Number of Neurons')
        plt.xlabel('Layer')
        plt.xticks(rotation=45)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()
        
        print(f"Neural growth visualization saved to {output_file}")
        return True
    except ImportError:
        print("Visualization requires matplotlib")
        return False

# ------------------- Función de Integración -------------------
def integrate_with_snakeRL(episodes=1000, save_path="snake_agent.pth"):
    """Integra el agente neuronal avanzado con el entorno SnakeRL y lo entrena."""
    from SnakeRL import SnakeGame  # Importar el entorno desde SnakeRL.py
    env = SnakeGame(render=True, difficulty=0)  # Puedes ajustar la dificultad
    
    state_size = len(env._get_state())  # Obtener el tamaño del estado
    action_size = 3  # Acciones: 0 (recto), 1 (izquierda), 2 (derecha)
    
    agent = create_advanced_agent(state_size, action_size)
    
    # OPTIMIZACIÓN: Verificar uso de GPU
    check_gpu_compatibility()
    
    # Crear gestor de memoria
    memory_manager = MemoryManager(agent)
    
    try:
        for episode in range(episodes):
            state = env.reset()
            done = False
            total_reward = 0
            episode_score = 0
            steps = 0
            
            # Monitorear memoria al inicio del episodio
            memory_manager.monitor_and_log()
            
            while not done:
                action = get_action(agent, state)
                next_state, reward, done = env.step(action)
                combined_reward = process_state_action_reward(agent, state, action, reward, next_state, done)
                
                # Evitar entrenar en cada paso para reducir carga computacional
                if steps % 3 == 0:  # Entrenar cada 3 pasos
                    train_advanced_agent(agent)
                
                state = next_state
                total_reward += combined_reward
                if reward > 0:  # Suponiendo que reward > 0 indica que comió comida
                    episode_score += 1
                steps += 1
                
                # Renderizar el entorno
                env.render(current_score=episode_score, total_score=agent.episodes_trained, 
                       episodes_left=episodes - episode - 1, epsilon=agent.epsilon, 
                       avg_q=np.mean(agent.q_values[-100:]) if agent.q_values else 0)
                
                # Manejar eventos para permitir salida manual
                if not env.handle_events():
                    print("Entrenamiento interrumpido por el usuario.")
                    save_advanced_agent(agent, save_path)
                    env.close()
                    return
            
            # Actualizar después del episodio
            update_after_episode(agent, env, total_reward, episode_score)
            
            # NUEVA OPTIMIZACIÓN: Limpieza después de cada episodio
            memory_manager.after_episode_cleanup()
            
            # Imprimir progreso cada 10 episodios
            if (episode + 1) % 10 == 0:
                print(f"Episodio {episode + 1}/{episodes} - Score: {episode_score} - Reward: {total_reward:.2f} - Steps: {steps}")
                
                # Guardar checkpoint cada 50 episodios
                if (episode + 1) % 50 == 0:
                    save_advanced_agent(agent, f"{save_path}.checkpoint")
        
        # Guardar el agente entrenado
        save_advanced_agent(agent, save_path)
        print(f"Entrenamiento completado. Agente guardado en {save_path}")
        
    finally:
        # Asegurar que se liberen los recursos
        env.close()
        
        # Limpieza final de memoria
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Guardar el agente entren
def integrate_with_snakeRL(episodes=1000, save_path="snake_agent.pth"):
    """Integra el agente neuronal avanzado con el entorno SnakeRL y lo entrena."""
    from SnakeRL import SnakeGame  # Importar el entorno desde SnakeRL.py
    env = SnakeGame(render=True, difficulty=0)  # Puedes ajustar la dificultad
    
    state_size = len(env._get_state())  # Obtener el tamaño del estado
    action_size = 3  # Acciones: 0 (recto), 1 (izquierda), 2 (derecha)
    
    agent = create_advanced_agent(state_size, action_size)
    
    # OPTIMIZACIÓN: Verificar uso de GPU
    check_gpu_compatibility()
    
    # Crear gestor de memoria
    memory_manager = MemoryManager(agent)
    
    try:
        for episode in range(episodes):
            state = env.reset()
            done = False
            total_reward = 0
            episode_score = 0
            steps = 0
            
            # Monitorear memoria al inicio del episodio
            memory_manager.monitor_and_log()
            
            while not done:
                action = get_action(agent, state)
                next_state, reward, done = env.step(action)
                combined_reward = process_state_action_reward(agent, state, action, reward, next_state, done)
                
                # Evitar entrenar en cada paso para reducir carga computacional
                if steps % 3 == 0:  # Entrenar cada 3 pasos
                    train_advanced_agent(agent)
                
                state = next_state
                total_reward += combined_reward
                if reward > 0:  # Suponiendo que reward > 0 indica que comió comida
                    episode_score += 1
                steps += 1
                
                # Renderizar el entorno
                env.render(current_score=episode_score, total_score=agent.episodes_trained, 
                       episodes_left=episodes - episode - 1, epsilon=agent.epsilon, 
                       avg_q=np.mean(agent.q_values[-100:]) if agent.q_values else 0)
                
                # Manejar eventos para permitir salida manual
                if not env.handle_events():
                    print("Entrenamiento interrumpido por el usuario.")
                    save_advanced_agent(agent, save_path)
                    env.close()
                    return
            
            # Actualizar después del episodio
            update_after_episode(agent, env, total_reward, episode_score)
            
            # NUEVA OPTIMIZACIÓN: Limpieza después de cada episodio
            memory_manager.after_episode_cleanup()
            
            # Imprimir progreso cada 10 episodios
            if (episode + 1) % 10 == 0:
                print(f"Episodio {episode + 1}/{episodes} - Score: {episode_score} - Reward: {total_reward:.2f} - Steps: {steps}")
                
                # Guardar checkpoint cada 50 episodios
                if (episode + 1) % 50 == 0:
                    save_advanced_agent(agent, f"{save_path}.checkpoint")
        
        # Guardar el agente entrenado
        save_advanced_agent(agent, save_path)
        print(f"Entrenamiento completado. Agente guardado en {save_path}")
        
    finally:
        # Asegurar que se liberen los recursos
        env.close()
        
        # Limpieza final de memoria
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    integrate_with_snakeRL()
    # Ejecutar verificación de GPU al importar el módulo
    check_gpu_compatibility()

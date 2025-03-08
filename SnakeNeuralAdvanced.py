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
    def __init__(self, input_size, output_size, growth_rate=0.1, on_resize_callback=None):
        super(DynamicLayer, self).__init__()
        self.input_size = input_size
        self.current_output_size = output_size
        self.max_output_size = MAX_NEURONS_PER_LAYER
        self.growth_rate = growth_rate
        self.on_resize_callback = on_resize_callback
        
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
            old_size = self.current_output_size
            self.current_output_size += growth_neurons
            
            print(f"Layer grew by {growth_neurons} neurons, new size: {self.current_output_size}")
            
            # Notify about the resize if callback is provided
            if self.on_resize_callback is not None:
                try:
                    self.on_resize_callback(old_size, self.current_output_size)
                except Exception as e:
                    print(f"Error en callback de cambio de tamaño: {e}")
                    
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
    
    def set_resize_callback(self, callback):
        """Establece un callback que será llamado cuando cambie el tamaño de la capa"""
        self.on_resize_callback = callback
    
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
        
        # Listeners for size changes
        self.output_size_listeners = []
        
        # Register callbacks for dynamic layers
        self._register_layer_callbacks()
        
    def _register_layer_callbacks(self):
        """Registra callbacks para las capas dinámicas"""
        # Register callbacks for all DynamicLayers
        for i, module in enumerate(self.feature_extractor):
            if isinstance(module, DynamicLayer):
                # Last dynamic layer affects output size
                if i == 2:  # El segundo DynamicLayer es el que afecta la salida
                    module.set_resize_callback(self._handle_output_size_change)
    
    def _handle_output_size_change(self, old_size, new_size):
        """Maneja cambios en el tamaño de salida y notifica a los módulos dependientes"""
        print(f"PerceptionModule output size change: {old_size} -> {new_size}")
        
        # Update output size
        self.output_size = new_size
        
        # Resize attention parameter
        old_attention = self.attention.data
        new_attention = torch.ones(new_size, device=self.attention.device) / new_size
        
        # Copy old values where possible
        if old_size <= new_size:
            new_attention[:old_size] = old_attention
        else:
            new_attention = old_attention[:new_size]
        
        # Normalize attention
        new_attention = F.softmax(new_attention, dim=0)
        
        # Replace the parameter
        self.attention = nn.Parameter(new_attention)
        
        # Notify all listeners about the size change
        for listener in self.output_size_listeners:
            try:
                listener(old_size, new_size)
            except Exception as e:
                print(f"Error al notificar cambio de tamaño: {e}")
                
    def register_output_size_listener(self, callback):
        """Registra un callback para ser notificado cuando cambia el tamaño de salida"""
        if callback not in self.output_size_listeners:
            self.output_size_listeners.append(callback)
            
    def get_current_output_size(self):
        """Devuelve el tamaño de salida actual"""
        return self.output_size
        
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
    
    def load_state_dict(self, state_dict, strict=True):
        """Sobrecarga para manejar cambios en las dimensiones durante la carga"""
        # Verificar si hay diferencias en las dimensiones
        if any('feature_extractor.0.weight' in k or 'feature_extractor.2.weight' in k for k in state_dict.keys()):
            # Primera capa dinámica
            if 'feature_extractor.0.weight' in state_dict:
                saved_first_layer_size = state_dict['feature_extractor.0.weight'].shape[0]
                current_first_layer_size = self.feature_extractor[0].weight.shape[0]
                
                if saved_first_layer_size != current_first_layer_size:
                    print(f"Adaptando primera capa de PerceptionModule: {current_first_layer_size} -> {saved_first_layer_size}")
                    # Crear nueva capa dinámica con el tamaño correcto
                    new_layer = DynamicLayer(self.input_size, saved_first_layer_size)
                    new_layer.to(self.feature_extractor[0].weight.device)
                    
                    # Reemplazar la capa
                    self.feature_extractor[0] = new_layer
            
            # Segunda capa dinámica
            if 'feature_extractor.2.weight' in state_dict:
                saved_second_layer_size = state_dict['feature_extractor.2.weight'].shape[0]
                current_second_layer_size = self.feature_extractor[2].weight.shape[0]
                
                # Si la primera capa cambió, necesitamos actualizar la entrada de la segunda
                first_layer_output = self.feature_extractor[0].current_output_size
                
                if saved_second_layer_size != current_second_layer_size or self.feature_extractor[2].input_size != first_layer_output:
                    print(f"Adaptando segunda capa de PerceptionModule: {current_second_layer_size} -> {saved_second_layer_size}")
                    # Crear nueva capa con el tamaño correcto
                    new_layer = DynamicLayer(first_layer_output, saved_second_layer_size)
                    new_layer.to(self.feature_extractor[2].weight.device)
                    
                    # Reemplazar la capa
                    self.feature_extractor[2] = new_layer
            
            # Actualizar el tamaño de salida y attention
            if 'feature_extractor.2.weight' in state_dict:
                new_output_size = state_dict['feature_extractor.2.weight'].shape[0]
                
                if new_output_size != self.output_size:
                    print(f"Actualizando output_size de PerceptionModule: {self.output_size} -> {new_output_size}")
                    self.output_size = new_output_size
                    
                    # Actualizar attention
                    if 'attention' in state_dict:
                        # Usar el attention guardado
                        new_attention = state_dict['attention']
                    else:
                        # Crear nuevo attention
                        new_attention = torch.ones(new_output_size, device=self.attention.device) / new_output_size
                    
                    self.attention = nn.Parameter(new_attention)
            
            # Volver a registrar callbacks después de los cambios
            self._register_layer_callbacks()
        
        # Ahora que las dimensiones coinciden, cargar el estado
        result = super().load_state_dict(state_dict, strict)
        
        # Notificar a todos los listeners sobre el cambio de tamaño
        for listener in self.output_size_listeners:
            try:
                listener(0, self.output_size)  # 0 como valor ficticio para old_size
            except Exception as e:
                print(f"Error al notificar cambio de tamaño: {e}")
                
        return result

class NavigationModule(nn.Module):
    """Module specialized in planning routes and obstacle avoidance"""
    def __init__(self, input_size, output_size):
        super(NavigationModule, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        
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
        
        # Listeners for size changes
        self.output_size_listeners = []
        
        # Register callbacks for dynamic layers
        self._register_layer_callbacks()
    
    def _register_layer_callbacks(self):
        """Registra callbacks para las capas dinámicas"""
        for i, module in enumerate(self.pathfinder):
            if isinstance(module, DynamicLayer):
                # Last dynamic layer affects output size
                if i == 4:  # El último DynamicLayer es el que afecta la salida
                    module.set_resize_callback(self._handle_output_size_change)
    
    def _handle_output_size_change(self, old_size, new_size):
        """Maneja cambios en el tamaño de salida y notifica a los módulos dependientes"""
        print(f"NavigationModule output size change: {old_size} -> {new_size}")
        
        # Update output size
        self.output_size = new_size
        
        # Resize directional bias
        old_bias = self.direction_bias.data
        new_bias = torch.zeros(new_size, device=self.direction_bias.device)
        
        # Copy old values where possible
        if old_size <= new_size:
            new_bias[:old_size] = old_bias
        else:
            new_bias = old_bias[:new_size]
        
        # Replace the parameter
        self.direction_bias = nn.Parameter(new_bias)
        
        # Notify all listeners about the size change
        for listener in self.output_size_listeners:
            try:
                listener(old_size, new_size)
            except Exception as e:
                print(f"Error al notificar cambio de tamaño en NavigationModule: {e}")
    
    def handle_input_size_change(self, old_size, new_size, position_in_input=0):
        """Maneja cambios en el tamaño de entrada"""
        print(f"NavigationModule procesando cambio de tamaño de entrada: {old_size} -> {new_size} en posición {position_in_input}")
        
        # Calcular nuevo tamaño de entrada total
        if position_in_input == 0:  # Si el cambio está en la primera parte del tensor concatenado
            new_total_input = new_size + 8  # Asumiendo que position y target son vectores de 4
        else:
            new_total_input = self.input_size - old_size + new_size
        
        # Crear nuevas capas con pesos adaptados a las nuevas dimensiones
        old_first_layer = self.pathfinder[0]
        
        # Crear un nuevo DynamicLayer con el nuevo tamaño de entrada
        new_layer = DynamicLayer(new_total_input, old_first_layer.current_output_size)
        
        # Copiar pesos anteriores donde sea posible
        with torch.no_grad():
            # Determinar qué parte de los pesos copiar según la posición del cambio
            if position_in_input == 0:
                # El cambio está en la primera parte (features)
                if new_size >= old_size:
                    # Entrada creció
                    new_layer.weight.data[:, :old_size] = old_first_layer.weight.data[:, :old_size]
                    new_layer.weight.data[:, old_size:new_size] = 0.01 * torch.randn_like(new_layer.weight.data[:, old_size:new_size])
                    new_layer.weight.data[:, new_size:] = old_first_layer.weight.data[:, old_size:]
                else:
                    # Entrada se redujo
                    new_layer.weight.data[:, :new_size] = old_first_layer.weight.data[:, :new_size]
                    new_layer.weight.data[:, new_size:] = old_first_layer.weight.data[:, old_size:]
            else:
                # El cambio está en otra parte (position o target)
                # Simplemente ajustar la matriz de pesos según corresponda
                if new_size >= old_size:
                    new_layer.weight.data = F.pad(old_first_layer.weight.data, 
                                              (0, new_size - old_size), 
                                              "constant", 0)
                else:
                    new_layer.weight.data = old_first_layer.weight.data[:, :(old_first_layer.weight.data.shape[1] - (old_size - new_size))]
            
            # Copiar bias
            new_layer.bias.data = old_first_layer.bias.data
        
        # Reemplazar la capa en el modelo
        self.pathfinder[0] = new_layer
        self.input_size = new_total_input
        
        # Registrar callback para la nueva capa
        new_layer.set_resize_callback(lambda old_s, new_s: self._handle_middle_layer_resize(0, old_s, new_s))
        
        print(f"NavigationModule redimensionado a input_size={new_total_input}")
        
    def _handle_middle_layer_resize(self, layer_idx, old_size, new_size):
        """Maneja cambios en el tamaño de las capas intermedias"""
        if layer_idx + 2 >= len(self.pathfinder):
            print(f"Error: Índice de capa {layer_idx} fuera de rango")
            return
            
        print(f"NavigationModule: Capa intermedia {layer_idx} cambió de {old_size} a {new_size}")
        
        # Obtener la siguiente capa dinámica
        next_layer_idx = layer_idx + 2  # Saltar ReLU
        next_layer = self.pathfinder[next_layer_idx]
        
        if not isinstance(next_layer, DynamicLayer):
            print(f"Error: Capa en índice {next_layer_idx} no es DynamicLayer")
            return
            
        # Crear una nueva capa con el tamaño de entrada actualizado
        new_layer = DynamicLayer(new_size, next_layer.current_output_size)
        
        # Copiar pesos existentes donde sea posible
        with torch.no_grad():
            if new_size >= old_size:
                # Entrada creció
                new_layer.weight.data[:, :old_size] = next_layer.weight.data[:, :old_size]
                new_layer.weight.data[:, old_size:] = 0.01 * torch.randn_like(new_layer.weight.data[:, old_size:])
            else:
                # Entrada se redujo
                new_layer.weight.data[:, :new_size] = next_layer.weight.data[:, :new_size]
            
            # Copiar bias
            new_layer.bias.data = next_layer.bias.data
        
        # Reemplazar la capa
        self.pathfinder[next_layer_idx] = new_layer
        
        # Registrar callback para la nueva capa si no es la última
        if next_layer_idx < len(self.pathfinder) - 1:
            new_layer.set_resize_callback(lambda old_s, new_s: self._handle_middle_layer_resize(next_layer_idx, old_s, new_s))
        else:
            new_layer.set_resize_callback(self._handle_output_size_change)
    
    def register_output_size_listener(self, callback):
        """Registra un callback para ser notificado cuando cambia el tamaño de salida"""
        if callback not in self.output_size_listeners:
            self.output_size_listeners.append(callback)
    
    def get_current_output_size(self):
        """Devuelve el tamaño de salida actual"""
        return self.output_size
        
    def forward(self, features, position, target):
        # Concat features with position and target info
        navigation_input = torch.cat([features, position, target], dim=1)
        return self.pathfinder(navigation_input)
    
    def update_bias(self, successful_actions, learning_rate=0.05):
        """Update directional bias based on successful actions"""
        with torch.no_grad():
            for action in successful_actions:
                self.direction_bias.data[action] += learning_rate
    
    def load_state_dict(self, state_dict, strict=True):
        """Sobrecarga para manejar cambios en las dimensiones durante la carga"""
        # Verificar si hay diferencias en las dimensiones para cada capa dinámica
        layers_to_check = [
            ('pathfinder.0.weight', 0),
            ('pathfinder.2.weight', 2),
            ('pathfinder.4.weight', 4)
        ]
        
        for key, idx in layers_to_check:
            if key in state_dict:
                saved_shape = state_dict[key].shape
                current_shape = self.pathfinder[idx].weight.shape
                
                if saved_shape != current_shape:
                    print(f"Adaptando capa {idx} de NavigationModule: {current_shape} -> {saved_shape}")
                    
                    # Determinar los tamaños de entrada y salida
                    if idx == 0:
                        # Primera capa
                        input_size = self.input_size
                        output_size = saved_shape[0]
                    elif idx == 2:
                        # Segunda capa
                        input_size = self.pathfinder[0].current_output_size
                        output_size = saved_shape[0]
                    elif idx == 4:
                        # Tercera capa
                        input_size = self.pathfinder[2].current_output_size
                        output_size = saved_shape[0]
                    
                    # Crear nueva capa con el tamaño correcto
                    new_layer = DynamicLayer(input_size, output_size)
                    new_layer.to(self.pathfinder[idx].weight.device)
                    
                    # Reemplazar la capa
                    self.pathfinder[idx] = new_layer
                    
                    # Si es la última capa, actualizar también output_size y direction_bias
                    if idx == 4 and output_size != self.output_size:
                        print(f"Actualizando output_size de NavigationModule: {self.output_size} -> {output_size}")
                        self.output_size = output_size
                        
                        # Actualizar direction_bias
                        if 'direction_bias' in state_dict:
                            # Usar el bias guardado
                            self.direction_bias = nn.Parameter(state_dict['direction_bias'])
                        else:
                            # Crear nuevo bias
                            self.direction_bias = nn.Parameter(torch.zeros(output_size, device=self.direction_bias.device))
        
        # Volver a registrar callbacks después de los cambios
        self._register_layer_callbacks()
        
        # Ahora que las dimensiones coinciden, cargar el estado
        result = super().load_state_dict(state_dict, strict)
        
        # Notificar a todos los listeners sobre el cambio de tamaño
        for listener in self.output_size_listeners:
            try:
                listener(0, self.output_size)  # 0 como valor ficticio para old_size
            except Exception as e:
                print(f"Error al notificar cambio de tamaño en NavigationModule: {e}")
                
        return result

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
        
        self.perception_size = perception_size
        self.navigation_size = navigation_size
        self.prediction_size = prediction_size
        self.action_size = action_size
        
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
        
        # Register callbacks for dynamic layers
        self._register_layer_callbacks()
    
    def _register_layer_callbacks(self):
        """Registra callbacks para las capas dinámicas"""
        for i, module in enumerate(self.integrator):
            if isinstance(module, DynamicLayer):
                module.set_resize_callback(lambda old_s, new_s, idx=i: self._handle_integrator_resize(idx, old_s, new_s))
        
        # Registrar callback para action_selector
        self.action_selector.set_resize_callback(self._handle_action_selector_resize)
    
    def _handle_integrator_resize(self, layer_idx, old_size, new_size):
        """Maneja cambios en el tamaño de las capas del integrador"""
        print(f"ExecutiveModule: Capa integradora {layer_idx} cambió de {old_size} a {new_size}")
        
        # Si no es la última capa, actualizar la siguiente
        if layer_idx + 2 < len(self.integrator):
            next_layer_idx = layer_idx + 2  # Saltar ReLU
            next_layer = self.integrator[next_layer_idx]
            
            if isinstance(next_layer, DynamicLayer):
                # Crear una nueva capa con el tamaño de entrada actualizado
                new_layer = DynamicLayer(new_size, next_layer.current_output_size)
                
                # Copiar pesos existentes donde sea posible
                with torch.no_grad():
                    if new_size >= old_size:
                        # Entrada creció
                        new_layer.weight.data[:, :old_size] = next_layer.weight.data[:, :old_size]
                        new_layer.weight.data[:, old_size:] = 0.01 * torch.randn_like(new_layer.weight.data[:, old_size:])
                    else:
                        # Entrada se redujo
                        new_layer.weight.data[:, :new_size] = next_layer.weight.data[:, :new_size]
                    
                    # Copiar bias
                    new_layer.bias.data = next_layer.bias.data
                
                # Reemplazar la capa
                self.integrator[next_layer_idx] = new_layer
                
                # Registrar callback para la nueva capa
                new_layer.set_resize_callback(lambda old_s, new_s: self._handle_integrator_resize(next_layer_idx, old_s, new_s))
        else:
            # Si es la última capa del integrador, actualizar el selector de acciones
            with torch.no_grad():
                # Crear un nuevo selector con el tamaño de entrada actualizado
                new_selector = DynamicLayer(new_size, self.action_size)
                
                # Copiar pesos existentes donde sea posible
                if new_size >= old_size:
                    # Entrada creció
                    new_selector.weight.data[:, :old_size] = self.action_selector.weight.data[:, :old_size]
                    new_selector.weight.data[:, old_size:] = 0.01 * torch.randn_like(new_selector.weight.data[:, old_size:])
                else:
                    # Entrada se redujo
                    new_selector.weight.data[:, :new_size] = self.action_selector.weight.data[:, :new_size]
                
                # Copiar bias
                new_selector.bias.data = self.action_selector.bias.data
                
                # Reemplazar el selector
                self.action_selector = new_selector
                
                # Registrar callback
                new_selector.set_resize_callback(self._handle_action_selector_resize)
    
    def _handle_action_selector_resize(self, old_size, new_size):
        """Maneja cambios en el tamaño del selector de acciones"""
        print(f"ExecutiveModule: Action selector cambió de {old_size} a {new_size}")
        # No necesitamos hacer nada aquí ya que esto cambiaría el número de acciones,
        # que no es algo que queramos cambiar dinámicamente
    
    def handle_perception_size_change(self, old_size, new_size):
        """Maneja cambios en el tamaño del módulo de percepción"""
        print(f"ExecutiveModule procesando cambio en percepción: {old_size} -> {new_size}")
        
        # Calcular nuevo tamaño de entrada combinada
        new_combined_size = new_size + self.navigation_size + self.prediction_size
        
        # Actualizar el tamaño de percepción
        self.perception_size = new_size
        
        # Crear una nueva primera capa con el tamaño de entrada actualizado
        old_first_layer = self.integrator[0]
        new_layer = DynamicLayer(new_combined_size, old_first_layer.current_output_size)
        
        # Copiar pesos existentes donde sea posible
        with torch.no_grad():
            if new_size >= old_size:
                # Percepción creció
                new_layer.weight.data[:, :old_size] = old_first_layer.weight.data[:, :old_size]
                new_layer.weight.data[:, old_size:new_size] = 0.01 * torch.randn_like(new_layer.weight.data[:, old_size:new_size])
                new_layer.weight.data[:, new_size:] = old_first_layer.weight.data[:, old_size:]
            else:
                # Percepción se redujo
                new_layer.weight.data[:, :new_size] = old_first_layer.weight.data[:, :new_size]
                new_layer.weight.data[:, new_size:] = old_first_layer.weight.data[:, old_size:]
            
            # Copiar bias
            new_layer.bias.data = old_first_layer.bias.data
        
        # Reemplazar la capa
        self.integrator[0] = new_layer
        
        # Registrar callback para la nueva capa
        new_layer.set_resize_callback(lambda old_s, new_s: self._handle_integrator_resize(0, old_s, new_s))
    
    def handle_navigation_size_change(self, old_size, new_size):
        """Maneja cambios en el tamaño del módulo de navegación"""
        print(f"ExecutiveModule procesando cambio en navegación: {old_size} -> {new_size}")
        
        # Calcular nuevo tamaño de entrada combinada
        new_combined_size = self.perception_size + new_size + self.prediction_size
        
        # Actualizar el tamaño de navegación
        self.navigation_size = new_size
        
        # Crear una nueva primera capa con el tamaño de entrada actualizado
        old_first_layer = self.integrator[0]
        new_layer = DynamicLayer(new_combined_size, old_first_layer.current_output_size)
        
        # Copiar pesos existentes donde sea posible
        with torch.no_grad():
            perception_offset = self.perception_size
            
            if new_size >= old_size:
                # Navegación creció
                # Copiar pesos para percepción
                new_layer.weight.data[:, :perception_offset] = old_first_layer.weight.data[:, :perception_offset]
                
                # Copiar pesos existentes para navegación
                new_layer.weight.data[:, perception_offset:perception_offset+old_size] = \
                    old_first_layer.weight.data[:, perception_offset:perception_offset+old_size]
                
                # Inicializar nuevos pesos para navegación
                new_layer.weight.data[:, perception_offset+old_size:perception_offset+new_size] = \
                    0.01 * torch.randn_like(new_layer.weight.data[:, perception_offset+old_size:perception_offset+new_size])
                
                # Copiar pesos para predicción
                new_layer.weight.data[:, perception_offset+new_size:] = \
                    old_first_layer.weight.data[:, perception_offset+old_size:]
            else:
                # Navegación se redujo
                # Copiar pesos para percepción
                new_layer.weight.data[:, :perception_offset] = old_first_layer.weight.data[:, :perception_offset]
                
                # Copiar pesos reducidos para navegación
                new_layer.weight.data[:, perception_offset:perception_offset+new_size] = \
                    old_first_layer.weight.data[:, perception_offset:perception_offset+new_size]
                
                # Copiar pesos para predicción
                new_layer.weight.data[:, perception_offset+new_size:] = \
                    old_first_layer.weight.data[:, perception_offset+old_size:]
            
            # Copiar bias
            new_layer.bias.data = old_first_layer.bias.data
        
        # Reemplazar la capa
        self.integrator[0] = new_layer
        
        # Registrar callback para la nueva capa
        new_layer.set_resize_callback(lambda old_s, new_s: self._handle_integrator_resize(0, old_s, new_s))
    
    def handle_prediction_size_change(self, old_size, new_size):
        """Maneja cambios en el tamaño del módulo de predicción"""
        print(f"ExecutiveModule procesando cambio en predicción: {old_size} -> {new_size}")
        
        # Calcular nuevo tamaño de entrada combinada
        new_combined_size = self.perception_size + self.navigation_size + new_size
        
        # Actualizar el tamaño de predicción
        self.prediction_size = new_size
        
        # Crear una nueva primera capa con el tamaño de entrada actualizado
        old_first_layer = self.integrator[0]
        new_layer = DynamicLayer(new_combined_size, old_first_layer.current_output_size)
        
        # Copiar pesos existentes donde sea posible
        with torch.no_grad():
            prediction_offset = self.perception_size + self.navigation_size
            
            if new_size >= old_size:
                # Predicción creció
                # Copiar pesos para percepción y navegación
                new_layer.weight.data[:, :prediction_offset] = old_first_layer.weight.data[:, :prediction_offset]
                
                # Copiar pesos existentes para predicción
                new_layer.weight.data[:, prediction_offset:prediction_offset+old_size] = \
                    old_first_layer.weight.data[:, prediction_offset:prediction_offset+old_size]
                
                # Inicializar nuevos pesos para predicción
                new_layer.weight.data[:, prediction_offset+old_size:] = \
                    0.01 * torch.randn_like(new_layer.weight.data[:, prediction_offset+old_size:])
            else:
                # Predicción se redujo
                # Copiar pesos para percepción y navegación
                new_layer.weight.data[:, :prediction_offset] = old_first_layer.weight.data[:, :prediction_offset]
                
                # Copiar pesos reducidos para predicción
                new_layer.weight.data[:, prediction_offset:] = \
                    old_first_layer.weight.data[:, prediction_offset:prediction_offset+new_size]
            
            # Copiar bias
            new_layer.bias.data = old_first_layer.bias.data
        
        # Reemplazar la capa
        self.integrator[0] = new_layer
        
        # Registrar callback para la nueva capa
        new_layer.set_resize_callback(lambda old_s, new_s: self._handle_integrator_resize(0, old_s, new_s))
    
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
    
    def load_state_dict(self, state_dict, strict=True):
        """Sobrecarga para manejar cambios en las dimensiones durante la carga"""
        # Verificar capas del integrador
        layers_to_check = [
            ('integrator.0.weight', 0),
            ('integrator.2.weight', 2)
        ]
        
        # Variable para seguir si necesitamos actualizar action_selector
        update_action_selector = False
        integrator_output_size = 32  # Valor predeterminado
        
        for key, idx in layers_to_check:
            if key in state_dict:
                saved_shape = state_dict[key].shape
                current_shape = self.integrator[idx].weight.shape
                
                if saved_shape != current_shape:
                    print(f"Adaptando capa {idx} de ExecutiveModule: {current_shape} -> {saved_shape}")
                    
                    # Determinar los tamaños de entrada y salida
                    if idx == 0:
                        # Primera capa
                        # Revisar si necesitamos actualizar combined_input_size
                        saved_combined_size = saved_shape[1]
                        current_combined_size = self.perception_size + self.navigation_size + self.prediction_size
                        
                        if saved_combined_size != current_combined_size:
                            print(f"Advertencia: La dimensión de entrada combinada ha cambiado: {current_combined_size} -> {saved_combined_size}")
                        
                        input_size = saved_combined_size
                        output_size = saved_shape[0]
                    elif idx == 2:
                        # Segunda capa
                        input_size = self.integrator[0].current_output_size
                        output_size = saved_shape[0]
                        integrator_output_size = output_size
                        update_action_selector = True
                    
                    # Crear nueva capa con el tamaño correcto
                    new_layer = DynamicLayer(input_size, output_size)
                    new_layer.to(self.integrator[idx].weight.device)
                    
                    # Reemplazar la capa
                    self.integrator[idx] = new_layer
        
        # Verificar action_selector
        if 'action_selector.weight' in state_dict:
            saved_shape = state_dict['action_selector.weight'].shape
            current_shape = self.action_selector.weight.shape
            
            if saved_shape != current_shape or update_action_selector:
                print(f"Adaptando action_selector de ExecutiveModule: {current_shape} -> {saved_shape}")
                
                # Crear nuevo action_selector con las dimensiones correctas
                new_selector = DynamicLayer(integrator_output_size, self.action_size)
                new_selector.to(self.action_selector.weight.device)
                
                # Reemplazar el selector
                self.action_selector = new_selector
        
        # Volver a registrar callbacks después de los cambios
        self._register_layer_callbacks()
        
        # Ahora que las dimensiones coinciden, cargar el estado
        try:
            result = super().load_state_dict(state_dict, strict)
        except Exception as e:
            print(f"Error al cargar estado para ExecutiveModule: {e}")
            # Intento de carga parcial
            result = None
            for name, param in state_dict.items():
                if name in self.state_dict():
                    try:
                        if isinstance(param, torch.nn.Parameter):
                            param = param.data
                        self.state_dict()[name].copy_(param)
                    except Exception as e:
                        print(f"  Error al cargar parámetro {name}: {e}")
        
        return result

# ------------------- Curiosity Module -------------------
class CuriosityModule:
    """Módulo de curiosidad para generar recompensas intrínsecas"""
    def __init__(self, state_size, action_size, device='cpu'):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        
        # Contador de novelty para estados visitados
        self.state_counts = {}
        self.decay_factor = CURIOSITY_DECAY  # Factor de decaimiento de curiosidad
        
        # Modelo predictivo simple para estimar sorpresa
        self.forward_model = nn.Sequential(
            nn.Linear(state_size + action_size, 64),
            nn.ReLU(),
            nn.Linear(64, state_size)
        ).to(device)
        
        # Optimizer para el modelo forward
        self.optimizer = optim.Adam(self.forward_model.parameters(), lr=0.001)
        
        # Factor para balancear novelty vs surprise
        self.novelty_weight = 0.6
        self.surprise_weight = 0.4
    
    def hash_state(self, state):
        """Convierte un estado en un hash discreto para contar visitas"""
        if torch.is_tensor(state):
            state_np = state.cpu().numpy().flatten()
        else:
            state_np = np.array(state).flatten()
        
        # Discretizar estado para hacer hash
        discrete_state = tuple(np.round(state_np * 5).astype(int))
        return hash(discrete_state)
    
    def compute_novelty_reward(self, state):
        """Calcula recompensa de novedad basada en conteo de estados"""
        state_hash = self.hash_state(state)
        
        # Incrementar contador para este estado
        if state_hash in self.state_counts:
            self.state_counts[state_hash] += 1
        else:
            self.state_counts[state_hash] = 1
        
        # Recompensa inversamente proporcional al conteo
        count = self.state_counts[state_hash]
        novelty_reward = 1.0 / np.sqrt(count)
        
        # Limitar recompensa
        return min(1.0, novelty_reward)
    
    def _decay_counts(self):
        """Decae gradualmente los contadores de estados para renovar interés"""
        for state_hash in self.state_counts:
            self.state_counts[state_hash] *= self.decay_factor
    
    def _prepare_action_tensor(self, action):
        """Prepara un tensor de acción para el modelo forward"""
        if not torch.is_tensor(action):
            if isinstance(action, (int, np.integer)):
                # Convertir acción escalar a one-hot
                action_tensor = torch.zeros(self.action_size, device=self.device)
                action_tensor[int(action)] = 1.0
            else:
                # Usar acción directamente si ya es un vector
                action_tensor = torch.FloatTensor(action).to(self.device)
        else:
            action_tensor = action
            
            # Asegurar forma correcta para one-hot
            if action_tensor.dim() == 0:
                # Convertir acción escalar a one-hot
                action_idx = int(action_tensor.item())
                action_tensor = torch.zeros(self.action_size, device=self.device)
                action_tensor[action_idx] = 1.0
        
        return action_tensor
    
    def compute_surprise_reward(self, state, action, next_state):
        """Calcula recompensa de sorpresa basada en error de predicción"""
        # Preparar entrada para el modelo
        if not torch.is_tensor(state):
            state_tensor = torch.FloatTensor(state).to(self.device)
        else:
            state_tensor = state.to(self.device)
            
        # Asegurar dimensiones correctas
        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)
            
        # Preparar acción
        action_tensor = self._prepare_action_tensor(action)
        if action_tensor.dim() == 1:
            action_tensor = action_tensor.unsqueeze(0)
            
        # Preparar siguiente estado
        if not torch.is_tensor(next_state):
            next_state_tensor = torch.FloatTensor(next_state).to(self.device)
        else:
            next_state_tensor = next_state.to(self.device)
            
        if next_state_tensor.dim() == 1:
            next_state_tensor = next_state_tensor.unsqueeze(0)
        
        # Concatenar estado y acción
        model_input = torch.cat([state_tensor, action_tensor], dim=1)
        
        # Predecir siguiente estado
        with torch.no_grad():
            predicted_next_state = self.forward_model(model_input)
        
        # Calcular error de predicción (sorpresa)
        prediction_error = F.mse_loss(predicted_next_state, next_state_tensor)
        
        # Convertir error a recompensa (mayor error = mayor sorpresa = mayor recompensa)
        surprise_reward = torch.tanh(prediction_error).item()
        
        return surprise_reward
    
    def update_forward_model(self, state, action, next_state):
        """Actualiza el modelo predictivo basado en transiciones observadas"""
        # Preparar datos
        if not torch.is_tensor(state):
            state_tensor = torch.FloatTensor(state).to(self.device)
        else:
            state_tensor = state.to(self.device)
            
        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)
            
        action_tensor = self._prepare_action_tensor(action)
        if action_tensor.dim() == 1:
            action_tensor = action_tensor.unsqueeze(0)
            
        if not torch.is_tensor(next_state):
            next_state_tensor = torch.FloatTensor(next_state).to(self.device)
        else:
            next_state_tensor = next_state.to(self.device)
            
        if next_state_tensor.dim() == 1:
            next_state_tensor = next_state_tensor.unsqueeze(0)
        
        # Entrada del modelo
        model_input = torch.cat([state_tensor, action_tensor], dim=1)
        
        # Entrenar modelo
        self.optimizer.zero_grad()
        predicted_next_state = self.forward_model(model_input)
        loss = F.mse_loss(predicted_next_state, next_state_tensor)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def compute_intrinsic_reward(self, state, action, next_state):
        """Combina recompensas de novedad y sorpresa"""
        # Cada 100 llamadas, decaer contadores
        if random.random() < 0.01:  # Aprox. cada 100 llamadas
            self._decay_counts()
        
        # Calcular componentes de recompensa intrínseca
        novelty = self.compute_novelty_reward(state)
        surprise = self.compute_surprise_reward(state, action, next_state)
        
        # Combinar componentes
        intrinsic_reward = (
            self.novelty_weight * novelty + 
            self.surprise_weight * surprise
        )
        
        # Actualizar modelo forward
        self.update_forward_model(state, action, next_state)
        
        # Aplicar peso global de curiosidad
        return CURIOSITY_WEIGHT * intrinsic_reward
    
    def is_novel_state(self, state, threshold=NOVELTY_THRESHOLD):
        """Determina si un estado es suficientemente novedoso"""
        state_hash = self.hash_state(state)
        
        # Si el estado no ha sido visto o ha sido visto pocas veces
        if state_hash not in self.state_counts or self.state_counts[state_hash] < 3:
            return True
            
        # Calcular novedad
        count = self.state_counts[state_hash]
        novelty = 1.0 / np.sqrt(count)
        
        return novelty > threshold

# ------------------- Skill Library -------------------
class SkillLibrary:
    """Biblioteca de habilidades aprendidas que pueden ser reutilizadas"""
    def __init__(self, state_size, action_size, capacity=SKILL_LIBRARY_SIZE):
        self.state_size = state_size
        self.action_size = action_size
        self.capacity = capacity
        self.skills = {}  # Diccionario de habilidades
    
    def add_skill(self, skill_name, skill_network, skill_context, performance):
        """Añade una habilidad a la biblioteca"""
        # Limitar tamaño de la biblioteca
        if len(self.skills) >= self.capacity:
            # Eliminar la habilidad con peor rendimiento
            worst_skill = min(self.skills.items(), key=lambda x: x[1]['performance'])
            del self.skills[worst_skill[0]]
        
        # Almacenar la habilidad
        self.skills[skill_name] = {
            'network': skill_network.state_dict(),
            'context': skill_context,
            'performance': performance
        }
        
        print(f"Habilidad '{skill_name}' añadida a la biblioteca con rendimiento {performance:.2f}")
        return True
    
    def extract_navigation_skill(self, agent, env, performance):
        """Extrae una habilidad específica de navegación"""
        # Simplificado para demostración
        if performance < 0.5:
            return False  # Rendimiento insuficiente
        
        # Crear un contexto de habilidad
        context = {
            'difficulty': env.difficulty,
            'wall_density': self._calculate_wall_density(env),
            'avg_path_length': performance * 10  # Estimación simple
        }
        
        # Crear un nombre único para la habilidad
        skill_name = f"nav_diff{env.difficulty}_perf{performance:.2f}"
        
        # Extraer la red de navegación del agente
        nav_network = agent.modules['navigation']
        
        # Añadir a la biblioteca
        return self.add_skill(skill_name, nav_network, context, performance)
    
    def extract_food_seeking_skill(self, agent, env, performance):
        """Extrae una habilidad de búsqueda de comida"""
        # Simplificado para demostración
        if performance < 0.6:
            return False  # Rendimiento insuficiente
        
        # Crear un contexto de habilidad
        context = {
            'difficulty': env.difficulty,
            'food_eaten': int(performance * 10),
            'efficiency': performance
        }
        
        # Crear un nombre único para la habilidad
        skill_name = f"food_diff{env.difficulty}_perf{performance:.2f}"
        
        # Extraer la red ejecutiva del agente
        exec_network = agent.modules['executive']
        
        # Añadir a la biblioteca
        return self.add_skill(skill_name, exec_network, context, performance)
    
    def _calculate_wall_density(self, env):
        """Calcula la densidad de paredes en el entorno"""
        # Implementación simplificada
        try:
            # Contar paredes en el mapa
            wall_count = 0
            for row in env.vertical_walls:
                wall_count += sum(row)
            for row in env.horizontal_walls:
                wall_count += sum(row)
            
            # Calcular densidad
            total_cells = env.GRID_WIDTH * env.GRID_HEIGHT
            density = wall_count / (2 * total_cells)  # Normalizar
            
            return density
        except:
            # Si hay algún error, retornar un valor predeterminado
            return 0.5
    
    def find_matching_skill(self, skill_type, context):
        """Encuentra una habilidad que coincida con el contexto dado"""
        best_match = None
        best_score = 0.0
        
        for name, skill in self.skills.items():
            # Verificar si la habilidad es del tipo correcto
            if skill_type not in name:
                continue
            
            # Calcular puntuación de coincidencia
            match_score = 0.0
            
            if 'difficulty' in context and 'difficulty' in skill['context']:
                # Mayor puntuación para misma dificultad
                if context['difficulty'] == skill['context']['difficulty']:
                    match_score += 0.5
                else:
                    diff_delta = abs(context['difficulty'] - skill['context']['difficulty'])
                    match_score += 0.5 * (1 - min(1.0, diff_delta / 2))
            
            # Otras métricas de contexto específicas por tipo
            if skill_type == 'nav' and 'wall_density' in context and 'wall_density' in skill['context']:
                density_match = 1.0 - abs(context['wall_density'] - skill['context']['wall_density'])
                match_score += 0.3 * density_match
            
            elif skill_type == 'food' and 'efficiency' in skill['context']:
                # Preferir habilidades con alta eficiencia
                match_score += 0.3 * skill['context']['efficiency']
            
            # Factor de rendimiento
            match_score += 0.2 * skill['performance']
            
            # Actualizar mejor coincidencia
            if match_score > best_score:
                best_score = match_score
                best_match = name
        
        if best_match and best_score > 0.6:  # Umbral mínimo de coincidencia
            return best_match
        return None
    
    def apply_skill(self, agent, skill_name, context):
        """Aplica una habilidad almacenada al agente"""
        if skill_name not in self.skills:
            print(f"Habilidad '{skill_name}' no encontrada")
            return False
        
        skill = self.skills[skill_name]
        
        try:
            # Determinar a qué módulo aplicar la habilidad
            if 'nav' in skill_name:
                target_module = 'navigation'
            elif 'food' in skill_name:
                target_module = 'executive'
            else:
                # Para otros tipos de habilidades
                if 'network_type' in skill['context']:
                    target_module = skill['context']['network_type']
                else:
                    # Elegir basado en tamaño y estructura del state_dict
                    # Simplificado: asumir ejecutivo
                    target_module = 'executive'
            
            # Obtener el módulo objetivo
            module = agent.modules[target_module]
            
            # Crear respaldo del estado actual
            backup = copy.deepcopy(module.state_dict())
            
            # Aplicar la habilidad (mezcla de pesos)
            current_state = module.state_dict()
            
            # Combinar pesos (75% habilidad, 25% actual)
            with torch.no_grad():
                for key in current_state:
                    if key in skill['network']:
                        # Verificar dimensiones
                        if current_state[key].shape == skill['network'][key].shape:
                            # Mezclar pesos
                            skill_tensor = skill['network'][key].to(current_state[key].device)
                            current_state[key] = 0.75 * skill_tensor + 0.25 * current_state[key]
            
            # Aplicar estado actualizado
            module.load_state_dict(current_state)
            
            print(f"Habilidad '{skill_name}' aplicada a módulo '{target_module}'")
            return True
            
        except Exception as e:
            print(f"Error al aplicar habilidad '{skill_name}': {e}")
            # En caso de error, intentar restaurar el estado anterior
            if 'backup' in locals():
                try:
                    module.load_state_dict(backup)
                    print("Restaurado estado anterior del módulo")
                except:
                    pass
            return False
    
    def create_skill_prototypes(self):
        """Crea prototipos para cada tipo de habilidad combinando habilidades existentes"""
        # Agrupar habilidades por tipo
        skill_types = {}
        
        for name, skill in self.skills.items():
            # Determinar tipo de habilidad por el nombre
            if 'nav' in name:
                skill_type = 'navigation'
            elif 'food' in name:
                skill_type = 'food_seeking'
            else:
                skill_type = 'general'
            
            if skill_type not in skill_types:
                skill_types[skill_type] = []
            
            skill_types[skill_type].append((name, skill))
        
        # Crear prototipos para cada tipo
        prototypes = {}
        
        for skill_type, skills in skill_types.items():
            if len(skills) < 2:
                continue  # Necesitamos al menos 2 habilidades para crear un prototipo
            
            # Ordenar por rendimiento
            skills.sort(key=lambda x: x[1]['performance'], reverse=True)
            
            # Tomar las mejores habilidades (máximo 3)
            best_skills = skills[:min(3, len(skills))]
            
            # Crear contexto promedio
            avg_context = {}
            for _, skill in best_skills:
                for key, value in skill['context'].items():
                    if isinstance(value, (int, float)):
                        if key not in avg_context:
                            avg_context[key] = 0
                        avg_context[key] += value / len(best_skills)
            
            # Crear state_dict promedio
            state_dicts = [s[1]['network'] for s in best_skills]
            avg_state_dict = self._average_state_dicts(state_dicts)
            
            if avg_state_dict:
                # Calcular rendimiento promedio
                avg_performance = sum(s[1]['performance'] for s in best_skills) / len(best_skills)
                
                # Crear nombre para el prototipo
                prototype_name = f"{skill_type}_prototype_p{avg_performance:.2f}"
                
                # Almacenar prototipo
                prototypes[prototype_name] = {
                    'network': avg_state_dict,
                    'context': avg_context,
                    'performance': avg_performance,
                    'type': skill_type
                }
                
                print(f"Creado prototipo '{prototype_name}' a partir de {len(best_skills)} habilidades")
        
        # Añadir prototipos a la biblioteca
        for name, prototype in prototypes.items():
            self.skills[name] = prototype
        
        return list(prototypes.keys())
    
    def _average_state_dicts(self, state_dicts):
        """Promedia varios state_dicts con verificación de compatibilidad"""
        if not state_dicts:
            return None
        
        # Usar el primer state_dict como referencia
        result = {}
        reference = state_dicts[0]
        
        try:
            # Para cada clave en el state_dict de referencia
            for key in reference:
                # Verificar si todos los state_dicts tienen esta clave y dimensiones compatibles
                compatible = all(
                    key in sd and sd[key].shape == reference[key].shape
                    for sd in state_dicts
                )
                
                if compatible:
                    # Acumular tensores
                    accumulated = torch.zeros_like(reference[key])
                    for sd in state_dicts:
                        accumulated += sd[key]
                    
                    # Calcular promedio
                    result[key] = accumulated / len(state_dicts)
                else:
                    # Si no son compatibles, usar el valor del mejor (primero)
                    result[key] = reference[key].clone()
                    
            return result
        except Exception as e:
            print(f"Error al promediar state_dicts: {e}")
            return None
    
    def apply_prototype(self, agent, skill_type, difficulty):
        """Aplica un prototipo de habilidad para una dificultad específica"""
        # Buscar prototipo apropiado
        prototype_name = None
        
        for name in self.skills:
            if skill_type in name and 'prototype' in name:
                prototype_name = name
                break
        
        if not prototype_name:
            print(f"No se encontró prototipo para tipo '{skill_type}'")
            return False
        
        # Crear contexto para la dificultad actual
        context = {'difficulty': difficulty}
        
        # Aplicar prototipo
        return self.apply_skill(agent, prototype_name, context)

# ------------------- Concept Abstraction System -------------------
class ConceptAbstraction:
    """Sistema para abstraer y generalizar conceptos de juego"""
    def __init__(self, state_size):
        self.state_size = state_size
        
        # Diccionario de conceptos
        self.concepts = {
            'food': {'examples': [], 'prototype': None},
            'wall': {'examples': [], 'prototype': None},
            'self_collision': {'examples': [], 'prototype': None},
            'open_space': {'examples': [], 'prototype': None}
        }
        
        # Importancia de características para cada concepto
        self.feature_importance = {
            'food': torch.ones(state_size) / state_size,
            'wall': torch.ones(state_size) / state_size,
            'self_collision': torch.ones(state_size) / state_size,
            'open_space': torch.ones(state_size) / state_size
        }
        
        # Auto-encoder simple para extraer características
        self.encoder = nn.Sequential(
            nn.Linear(state_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, state_size)
        )
        
        # Optimizer
        self.optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=0.001
        )
    
    def add_example(self, concept, state, action, result):
        """Añade un ejemplo de un concepto"""
        if concept not in self.concepts:
            self.concepts[concept] = {'examples': [], 'prototype': None}
            self.feature_importance[concept] = torch.ones(self.state_size) / self.state_size
        
        # Almacenar ejemplo
        if torch.is_tensor(state):
            state_tensor = state.detach().clone()
        else:
            state_tensor = torch.FloatTensor(state)
        
        # Limitar el número de ejemplos
        max_examples = 100
        if len(self.concepts[concept]['examples']) >= max_examples:
            # Reemplazar un ejemplo aleatorio
            idx = random.randint(0, max_examples - 1)
            self.concepts[concept]['examples'][idx] = (state_tensor, action, result)
        else:
            self.concepts[concept]['examples'].append((state_tensor, action, result))
        
        # Actualizar prototipo si hay suficientes ejemplos
        if len(self.concepts[concept]['examples']) >= 10:
            self.update_feature_importance(concept)
            self.create_concept_prototype(concept)
    
    def update_feature_importance(self, concept):
        """Actualiza la importancia de características para el concepto"""
        if concept not in self.concepts or not self.concepts[concept]['examples']:
            return
        
        examples = self.concepts[concept]['examples']
        states = torch.stack([ex[0] for ex in examples])
        
        # Calcular varianza de cada característica
        variances = torch.var(states, dim=0)
        
        # Alta varianza → baja importancia; baja varianza → alta importancia
        # Porque características con baja varianza son más consistentes para el concepto
        importance = 1.0 / (variances + 1e-6)
        
        # Normalizar
        importance = importance / importance.sum()
        
        # Actualizar importancia con suavizado
        self.feature_importance[concept] = 0.8 * self.feature_importance[concept] + 0.2 * importance
        
        # Reentrenar autoencoder
        for _ in range(5):  # Número reducido de epochs
            self.optimizer.zero_grad()
            
            # Forward pass con ponderación de características
            weighted_states = states * self.feature_importance[concept].unsqueeze(0)
            encoded = self.encoder(weighted_states)
            decoded = self.decoder(encoded)
            
            # Loss
            loss = F.mse_loss(decoded, states)
            loss.backward()
            self.optimizer.step()
    
    def create_concept_prototype(self, concept):
        """Crea un prototipo para un concepto basado en los ejemplos"""
        if concept not in self.concepts or not self.concepts[concept]['examples']:
            return None
        
        examples = self.concepts[concept]['examples']
        states = torch.stack([ex[0] for ex in examples])
        
        # Ponderar características por importancia
        weighted_states = states * self.feature_importance[concept].unsqueeze(0)
        
        # Calcular prototipo como el estado promedio
        prototype = torch.mean(weighted_states, dim=0)
        
        # Almacenar prototipo
        self.concepts[concept]['prototype'] = prototype
        
        return prototype
    
    def identify_relevant_concept(self, state):
        """Identifica qué concepto es más relevante para el estado actual"""
        if torch.is_tensor(state):
            state_tensor = state.detach()
        else:
            state_tensor = torch.FloatTensor(state)
        
        best_concept = None
        best_similarity = -1.0
        
        for concept, data in self.concepts.items():
            if data['prototype'] is not None:
                # Ponderar tanto el estado como el prototipo
                weighted_state = state_tensor * self.feature_importance[concept]
                weighted_prototype = data['prototype'] * self.feature_importance[concept]
                
                # Calcular similitud coseno
                similarity = F.cosine_similarity(
                    weighted_state.view(1, -1),
                    weighted_prototype.view(1, -1)
                ).item()
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_concept = concept
        
        # Solo retornar si la similitud es suficiente
        if best_similarity > 0.7:
            return best_concept, best_similarity
        return None, 0.0
    
    def get_concept_embedding(self, concept):
        """Obtiene una representación compacta (embedding) del concepto"""
        if concept not in self.concepts or self.concepts[concept]['prototype'] is None:
            return None
        
        # Usar el encoder para obtener una representación compacta
        with torch.no_grad():
            prototype = self.concepts[concept]['prototype']
            weighted_prototype = prototype * self.feature_importance[concept]
            embedding = self.encoder(weighted_prototype.unsqueeze(0))
        
        return embedding.squeeze(0)

# ------------------- Meta-Learning System -------------------

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
        
        # Configurar los listeners para propagar cambios de tamaño entre módulos
        self._setup_size_change_listeners()
    
    def _setup_size_change_listeners(self):
        """Configura listeners para propagar cambios de tamaño entre módulos"""
        # Cuando el tamaño de salida de percepción cambia, actualizar navegación y ejecutivo
        perception = self.modules['perception']
        navigation = self.modules['navigation']
        executive = self.modules['executive']
        
        # Registrar listeners para cambios en el módulo de percepción
        perception.register_output_size_listener(
            lambda old_size, new_size: navigation.handle_input_size_change(old_size, new_size, 0)
        )
        perception.register_output_size_listener(
            lambda old_size, new_size: executive.handle_perception_size_change(old_size, new_size)
        )
        
        # Registrar listeners para cambios en el módulo de navegación
        navigation.register_output_size_listener(
            lambda old_size, new_size: executive.handle_navigation_size_change(old_size, new_size)
        )
    
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
        
        # Verificar la estructura del lote para evitar errores de descompresión
        valid_batch = []
        for transition in batch:
            # Verificar que cada elemento del lote tenga exactamente 5 elementos
            if isinstance(transition, tuple) and len(transition) == 5:
                valid_batch.append(transition)
            else:
                print(f"Advertencia: Elemento de lote inválido detectado: {transition}")
        
        if not valid_batch:
            print("Error: No hay elementos válidos en el lote.")
            return 0
        
        # Unpack batch usando solo los elementos válidos
        states, actions, rewards, next_states, dones = zip(*valid_batch)
        
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
        
        # Primero, ajustar las dimensiones de las capas dinámicas
        self._adapt_dynamic_layers_to_state(state)
        
        # Ahora cargar los módulos con las dimensiones ya ajustadas
        for name, module_state in state['modules'].items():
            if name in self.modules:
                try:
                    self.modules[name].load_state_dict(module_state)
                except Exception as e:
                    print(f"Error al cargar el estado del módulo {name}: {e}")
                    # Si todavía hay error, intentar una carga parcial
                    self._partial_load_module(self.modules[name], module_state)
        
        # Load optimizer
        try:
            self.optimizer.load_state_dict(state['optimizer'])
        except Exception as e:
            print(f"Error al cargar el optimizer, reconstruyendo: {e}")
            self.setup_optimizer()
        
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
        
        print(f"Agente cargado desde {filename}")
        return self.episodes_trained
    
    def _adapt_dynamic_layers_to_state(self, state):
        """Ajusta las dimensiones de las capas dinámicas para que coincidan con el estado guardado"""
        print("Adaptando dimensiones de capas dinámicas al modelo guardado...")
        
        try:
            # 1. Ajustar PerceptionModule
            if 'perception' in state['modules'] and 'perception' in self.modules:
                perception_state = state['modules']['perception']
                perception_module = self.modules['perception']
                
                # Primera capa dinámica
                if 'feature_extractor.0.weight' in perception_state:
                    saved_shape = perception_state['feature_extractor.0.weight'].shape
                    current_shape = perception_module.feature_extractor[0].weight.shape
                    
                    if saved_shape != current_shape:
                        print(f"Adaptando primera capa de Percepción: {current_shape} -> {saved_shape}")
                        # Crear nueva capa con dimensiones correctas
                        new_layer = DynamicLayer(perception_module.input_size, saved_shape[0])
                        new_layer.to(self.device)
                        perception_module.feature_extractor[0] = new_layer
                
                # Segunda capa dinámica
                if 'feature_extractor.2.weight' in perception_state:
                    saved_shape = perception_state['feature_extractor.2.weight'].shape
                    current_shape = perception_module.feature_extractor[2].weight.shape
                    
                    if saved_shape != current_shape:
                        print(f"Adaptando segunda capa de Percepción: {current_shape} -> {saved_shape}")
                        first_layer_output = perception_module.feature_extractor[0].current_output_size
                        new_layer = DynamicLayer(first_layer_output, saved_shape[0])
                        new_layer.to(self.device)
                        perception_module.feature_extractor[2] = new_layer
                
                # Actualizar output_size y atención
                if 'feature_extractor.2.weight' in perception_state:
                    new_output_size = perception_state['feature_extractor.2.weight'].shape[0]
                    perception_module.output_size = new_output_size
                    
                    if 'attention' in perception_state:
                        attention_shape = perception_state['attention'].shape
                        if attention_shape != perception_module.attention.shape:
                            new_attention = torch.ones(new_output_size, device=self.device) / new_output_size
                            perception_module.attention = nn.Parameter(new_attention)
            
            # 2. Ajustar NavigationModule
            if 'navigation' in state['modules'] and 'navigation' in self.modules:
                navigation_state = state['modules']['navigation']
                navigation_module = self.modules['navigation']
                
                # Actualizar dimensiones de las capas
                layers_to_check = [
                    ('pathfinder.0.weight', 0),
                    ('pathfinder.2.weight', 2),
                    ('pathfinder.4.weight', 4)
                ]
                
                for key, idx in layers_to_check:
                    if key in navigation_state:
                        saved_shape = navigation_state[key].shape
                        current_shape = navigation_module.pathfinder[idx].weight.shape
                        
                        if saved_shape != current_shape:
                            print(f"Adaptando capa {idx} de Navegación: {current_shape} -> {saved_shape}")
                            
                            # Determinar tamaños de entrada
                            if idx == 0:
                                input_size = navigation_module.input_size
                            elif idx == 2:
                                input_size = navigation_module.pathfinder[0].current_output_size
                            elif idx == 4:
                                input_size = navigation_module.pathfinder[2].current_output_size
                            
                            # Crear nueva capa
                            new_layer = DynamicLayer(input_size, saved_shape[0])
                            new_layer.to(self.device)
                            navigation_module.pathfinder[idx] = new_layer
                
                # Actualizar output_size y direction_bias
                if 'pathfinder.4.weight' in navigation_state:
                    new_output_size = navigation_state['pathfinder.4.weight'].shape[0]
                    navigation_module.output_size = new_output_size
                    
                    if 'direction_bias' in navigation_state:
                        bias_shape = navigation_state['direction_bias'].shape
                        if bias_shape != navigation_module.direction_bias.shape:
                            new_bias = torch.zeros(new_output_size, device=self.device)
                            navigation_module.direction_bias = nn.Parameter(new_bias)
            
            # 3. Ajustar ExecutiveModule
            if 'executive' in state['modules'] and 'executive' in self.modules:
                executive_state = state['modules']['executive']
                executive_module = self.modules['executive']
                
                # Actualizar dimensiones del integrador
                layers_to_check = [
                    ('integrator.0.weight', 0),
                    ('integrator.2.weight', 2)
                ]
                
                for key, idx in layers_to_check:
                    if key in executive_state:
                        saved_shape = executive_state[key].shape
                        current_shape = executive_module.integrator[idx].weight.shape
                        
                        if saved_shape != current_shape:
                            print(f"Adaptando capa {idx} de Ejecutivo: {current_shape} -> {saved_shape}")
                            
                            # Determinar tamaños de entrada
                            if idx == 0:
                                input_size = saved_shape[1]  # Usar el tamaño guardado
                                # Actualizar las dimensiones combinadas
                                combined_size = executive_module.perception_size + executive_module.navigation_size + executive_module.prediction_size
                                if combined_size != input_size:
                                    print(f"Advertencia: Discrepancia en tamaño de entrada combinado: {combined_size} (actual) vs {input_size} (guardado)")
                            elif idx == 2:
                                input_size = executive_module.integrator[0].current_output_size
                            
                            # Crear nueva capa
                            new_layer = DynamicLayer(input_size, saved_shape[0])
                            new_layer.to(self.device)
                            executive_module.integrator[idx] = new_layer
                
                # Actualizar action_selector
                if 'action_selector.weight' in executive_state:
                    saved_shape = executive_state['action_selector.weight'].shape
                    current_shape = executive_module.action_selector.weight.shape
                    
                    if saved_shape != current_shape:
                        print(f"Adaptando selector de acciones: {current_shape} -> {saved_shape}")
                        # El tamaño de entrada debe coincidir con la salida de la última capa del integrador
                        input_size = executive_module.integrator[2].current_output_size
                        new_layer = DynamicLayer(input_size, executive_module.action_size)
                        new_layer.to(self.device)
                        executive_module.action_selector = new_layer
            
            # 4. Ajustar PredictionModule si es necesario
            # (similar a los anteriores, depende de la estructura exacta)
            
        except Exception as e:
            print(f"Error durante la adaptación de dimensiones: {e}")
            import traceback
            traceback.print_exc()
        
        # Reconfiguramos los listeners para mantener la sincronización
        self._setup_size_change_listeners()
        print("Adaptación de dimensiones completada")
    
    def _adapt_module_dynamic_layers(self, module, module_state):
        """Recorre el módulo buscando capas dinámicas y ajusta sus dimensiones"""
        # Para manejar módulos tipo Sequential
        if isinstance(module, nn.Sequential):
            for i, layer in enumerate(module):
                layer_key = f"{i}"
                if isinstance(layer, DynamicLayer):
                    # Buscar las dimensiones correspondientes en el state_dict
                    weight_key = f"{layer_key}.weight"
                    if weight_key in module_state:
                        saved_shape = module_state[weight_key].shape
                        current_shape = layer.weight.shape
                        
                        if saved_shape != current_shape:
                            print(f"  Ajustando capa: {saved_shape} (guardada) -> {current_shape} (actual)")
                            # Calcular cuánto debe crecer la capa
                            growth_neurons = saved_shape[0] - current_shape[0]
                            
                            if growth_neurons > 0:
                                # Hacer crecer la capa manualmente, similar a grow_neurons pero sin las verificaciones
                                with torch.no_grad():
                                    # Crear nuevos tensores con las dimensiones apropiadas
                                    new_weight = torch.zeros(saved_shape, device=layer.weight.device)
                                    new_bias = torch.zeros(saved_shape[0], device=layer.bias.device)
                                    
                                    # Copiar los pesos y bias existentes
                                    new_weight[:current_shape[0], :] = layer.weight
                                    new_bias[:current_shape[0]] = layer.bias
                                    
                                    # Inicializar aleatoriamente las nuevas neuronas
                                    new_weight[current_shape[0]:, :] = 0.01 * torch.randn(
                                        growth_neurons, saved_shape[1], device=layer.weight.device)
                                    
                                    # Asignar los nuevos pesos y bias
                                    layer.weight = nn.Parameter(new_weight)
                                    layer.bias = nn.Parameter(new_bias)
                                    
                                    # Actualizar buffers necesarios con nuevas dimensiones
                                    for buffer_name, buffer in list(layer._buffers.items()):
                                        if buffer is not None:
                                            buffer_shape = buffer.shape
                                            
                                            # Determinar las nuevas dimensiones del buffer
                                            if len(buffer_shape) == 1 and buffer_shape[0] == current_shape[0]:
                                                # Buffers 1D como activation_history, usage_count
                                                new_buffer = torch.zeros(saved_shape[0], device=buffer.device)
                                                new_buffer[:current_shape[0]] = buffer
                                            elif len(buffer_shape) == 2 and buffer_shape[0] == current_shape[0]:
                                                # Buffers 2D como connection_activity, hebbian_traces
                                                new_buffer = torch.zeros(saved_shape, device=buffer.device)
                                                new_buffer[:current_shape[0], :] = buffer
                                            else:
                                                # Mantener el buffer original
                                                continue
                                                
                                            # Actualizar el buffer
                                            layer.register_buffer(buffer_name, new_buffer)
                                    
                                    # Actualizar el tamaño actual
                                    layer.current_output_size = saved_shape[0]
                                    print(f"    Capa redimensionada a {saved_shape[0]} neuronas")
                
                # Procesar recursivamente las sublayers si las hay
                if hasattr(layer, "_adapt_module_dynamic_layers"):
                    sub_state = {k.split('.', 1)[1]: v for k, v in module_state.items() 
                               if k.startswith(f"{layer_key}.") and len(k.split('.')) > 1}
                    layer._adapt_module_dynamic_layers(layer, sub_state)
        
        # Manejar atributos específicos que son capas dinámicas
        for name, child in module.__dict__.items():
            if isinstance(child, DynamicLayer):
                # Buscar claves que correspondan a este atributo
                weight_key = f"{name}.weight"
                attribute_prefix = f"{name}."
                
                sub_state = {k: v for k, v in module_state.items() if k.startswith(attribute_prefix)}
                
                if weight_key in module_state:
                    saved_shape = module_state[weight_key].shape
                    current_shape = child.weight.shape
                    
                    if saved_shape != current_shape:
                        print(f"  Ajustando capa {name}: {saved_shape} (guardada) -> {current_shape} (actual)")
                        # Calcular cuánto debe crecer la capa
                        growth_neurons = saved_shape[0] - current_shape[0]
                        
                        if growth_neurons > 0:
                            # Hacer crecer la capa manualmente
                            with torch.no_grad():
                                # Crear nuevos tensores con las dimensiones apropiadas
                                new_weight = torch.zeros(saved_shape, device=child.weight.device)
                                new_bias = torch.zeros(saved_shape[0], device=child.bias.device)
                                
                                # Copiar los pesos y bias existentes
                                new_weight[:current_shape[0], :] = child.weight
                                new_bias[:current_shape[0]] = child.bias
                                
                                # Inicializar aleatoriamente las nuevas neuronas
                                new_weight[current_shape[0]:, :] = 0.01 * torch.randn(
                                    growth_neurons, saved_shape[1], device=child.weight.device)
                                
                                # Asignar los nuevos pesos y bias
                                child.weight = nn.Parameter(new_weight)
                                child.bias = nn.Parameter(new_bias)
                                
                                # Actualizar buffers necesarios con nuevas dimensiones
                                for buffer_name, buffer in list(child._buffers.items()):
                                    if buffer is not None:
                                        buffer_shape = buffer.shape
                                        
                                        # Determinar las nuevas dimensiones del buffer
                                        if len(buffer_shape) == 1 and buffer_shape[0] == current_shape[0]:
                                            # Buffers 1D como activation_history, usage_count
                                            new_buffer = torch.zeros(saved_shape[0], device=buffer.device)
                                            new_buffer[:current_shape[0]] = buffer
                                        elif len(buffer_shape) == 2 and buffer_shape[0] == current_shape[0]:
                                            # Buffers 2D como connection_activity, hebbian_traces
                                            new_buffer = torch.zeros(saved_shape, device=buffer.device)
                                            new_buffer[:current_shape[0], :] = buffer
                                        else:
                                            # Mantener el buffer original
                                            continue
                                            
                                        # Actualizar el buffer
                                        child.register_buffer(buffer_name, new_buffer)
                                
                                # Actualizar el tamaño actual
                                child.current_output_size = saved_shape[0]
                                print(f"    Capa redimensionada a {saved_shape[0]} neuronas")
    
    def _partial_load_module(self, module, state_dict):
        """Intenta cargar parcialmente un módulo, excluyendo parámetros con discrepancia de tamaño"""
        print(f"Intentando carga parcial para {type(module).__name__}")
        own_state = module.state_dict()
        
        for name, param in state_dict.items():
            if name in own_state:
                try:
                    if isinstance(param, torch.nn.Parameter):
                        param = param.data
                    own_state[name].copy_(param)
                except Exception as e:
                    print(f"  Saltando parámetro {name}: {e}")
        
        print("Carga parcial completada")

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
def integrate_with_snakeRL(episodes=1000, save_path="snake_agent.pth", load_existing=True, difficulty=0):
    """Integra el agente neuronal avanzado con el entorno SnakeRL y lo entrena."""
    import os
    from SnakeRL import SnakeGame  # Importar el entorno desde SnakeRL.py
    env = SnakeGame(render=True, difficulty=difficulty)  # Usar el parámetro de dificultad
    
    state_size = len(env._get_state())  # Obtener el tamaño del estado
    action_size = 3  # Acciones: 0 (recto), 1 (izquierda), 2 (derecha)
    
    # Crear un nuevo agente
    agent = create_advanced_agent(state_size, action_size)
    
    # Cargar el agente existente si se especifica
    episodes_completed = 0
    if load_existing and os.path.exists(save_path):
        print(f"Cargando agente existente desde {save_path}")
        try:
            episodes_completed = load_advanced_agent(agent, save_path)
            print(f"Continuando entrenamiento desde el episodio {episodes_completed}")
        except Exception as e:
            print(f"Error al cargar agente desde {save_path}: {e}")
            print("Usando un agente nuevo en su lugar.")
            # Reiniciar agente
            agent = create_advanced_agent(state_size, action_size)
    else:
        if load_existing and not os.path.exists(save_path):
            print(f"No se encontró el archivo del modelo {save_path}. Iniciando con un agente nuevo.")
        else:
            print("Iniciando entrenamiento con un agente nuevo")
    
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
        
        print("Limpieza de recursos completada")

# ------------------- Simplified Episode and Memory Classes -------------------
class Episode:
    """Representa un episodio de juego con transiciones"""
    def __init__(self, max_length=1000):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.importance = 1.0  # Importancia predeterminada
        self.max_length = max_length
    
    def add(self, state, action, reward, next_state, done):
        """Añade una transición al episodio"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        
        # Mantener tamaño máximo
        if len(self.states) > self.max_length:
            self.states.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
            self.next_states.pop(0)
            self.dones.pop(0)
    
    def calculate_importance(self):
        """Calcula la importancia del episodio basado en las recompensas"""
        if not self.rewards:
            return 0.0
        
        # Importancia basada en recompensa total y máxima
        total_reward = sum(self.rewards)
        max_reward = max(self.rewards) if self.rewards else 0
        
        # Combinar factores para calcular importancia
        self.importance = 0.7 * total_reward + 0.3 * max_reward
        return self.importance
    
    def get_transitions(self):
        """Devuelve todas las transiciones del episodio"""
        return list(zip(self.states, self.actions, self.rewards, self.next_states, self.dones))
    
    def get_random_batch(self, batch_size):
        """Devuelve un lote aleatorio de transiciones"""
        if not self.states or batch_size <= 0:
            return [], [], [], [], []
        
        indices = np.random.choice(len(self.states), min(batch_size, len(self.states)), replace=False)
        
        return (
            [self.states[i] for i in indices],
            [self.actions[i] for i in indices],
            [self.rewards[i] for i in indices],
            [self.next_states[i] for i in indices],
            [self.dones[i] for i in indices]
        )
    
    def get_sequence(self, start_idx, length):
        """Devuelve una secuencia de transiciones a partir del índice dado"""
        end_idx = min(start_idx + length, len(self.states))
        return (
            self.states[start_idx:end_idx],
            self.actions[start_idx:end_idx],
            self.rewards[start_idx:end_idx],
            self.next_states[start_idx:end_idx],
            self.dones[start_idx:end_idx]
        )

class EpisodicMemory:
    """Memoria de episodios con muestreo basado en importancia"""
    def __init__(self, capacity=100):
        self.episodes = []
        self.capacity = capacity
        self.current_episode = Episode()
        
        # Variables para el análisis causal
        self.causal_patterns = {}
        self.state_visit_counts = {}
    
    def add(self, state, action, reward, next_state, done):
        """Añade una transición a la memoria"""
        # Añadir a episodio actual
        self.current_episode.add(state, action, reward, next_state, done)
        
        # Actualizar contadores de visita de estados
        state_key = self._get_state_key(state)
        self.state_visit_counts[state_key] = self.state_visit_counts.get(state_key, 0) + 1
    
    def end_episode(self):
        """Finaliza el episodio actual y lo almacena en memoria"""
        # Solo almacenar episodios no vacíos
        if len(self.current_episode.states) > 0:
            # Calcular importancia
            self.current_episode.calculate_importance()
            
            # Analizar patrones causales
            self._analyze_causal_patterns(self.current_episode)
            
            # Añadir a la lista de episodios
            self.episodes.append(self.current_episode)
            
            # Mantener capacidad
            if len(self.episodes) > self.capacity:
                # Eliminar episodio menos importante
                min_idx = min(range(len(self.episodes)), key=lambda i: self.episodes[i].importance)
                self.episodes.pop(min_idx)
            
            # Consolidar patrones causales periódicamente
            if len(self.episodes) % 10 == 0:
                self._consolidate_causal_patterns()
        
        # Crear nuevo episodio
        self.current_episode = Episode()
    
    def _analyze_causal_patterns(self, episode):
        """Analiza patrones causales en el episodio"""
        # Versión simplificada para identificar acciones que llevaron a recompensas
        for i in range(len(episode.states) - 1):
            if episode.rewards[i] > 0:
                # Identificar estados y acciones que llevaron a recompensas
                state_key = self._get_state_key(episode.states[i])
                action = episode.actions[i]
                
                # Registrar en patrones causales
                if state_key not in self.causal_patterns:
                    self.causal_patterns[state_key] = {}
                
                if action not in self.causal_patterns[state_key]:
                    self.causal_patterns[state_key][action] = {
                        'count': 0,
                        'reward': 0.0
                    }
                
                self.causal_patterns[state_key][action]['count'] += 1
                self.causal_patterns[state_key][action]['reward'] += episode.rewards[i]
    
    def _consolidate_causal_patterns(self):
        """Consolida patrones causales para reducir ruido"""
        # Eliminar patrones poco frecuentes o con baja recompensa
        keys_to_remove = []
        
        for state_key, actions in self.causal_patterns.items():
            action_keys_to_remove = []
            
            for action, data in actions.items():
                if data['count'] < 3 or data['reward'] / data['count'] < 0.2:
                    action_keys_to_remove.append(action)
            
            for action in action_keys_to_remove:
                del actions[action]
            
            if not actions:
                keys_to_remove.append(state_key)
        
        for state_key in keys_to_remove:
            del self.causal_patterns[state_key]
    
    def _calculate_state_similarity(self, state1, state2):
        """Calcula la similitud entre dos estados"""
        # Implementación simplificada
        if not torch.is_tensor(state1):
            state1 = torch.FloatTensor(state1)
        if not torch.is_tensor(state2):
            state2 = torch.FloatTensor(state2)
            
        # Asegurar misma dimensión
        if state1.dim() != state2.dim():
            if state1.dim() == 1:
                state1 = state1.unsqueeze(0)
            if state2.dim() == 1:
                state2 = state2.unsqueeze(0)
        
        # Distancia coseno (mayor valor = más similar)
        dot_product = torch.sum(state1 * state2)
        norm1 = torch.norm(state1)
        norm2 = torch.norm(state2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def sample_batch(self, batch_size):
        """Muestrea un lote de transiciones de la memoria"""
        if not self.episodes:
            return [], [], [], [], []
            
        # Seleccionar episodios con probabilidad proporcional a su importancia
        episode_importances = [ep.importance for ep in self.episodes]
        total_importance = sum(episode_importances)
        
        if total_importance <= 0:
            # Si todas las importancias son cero, usar distribución uniforme
            probs = None
        else:
            probs = [imp / total_importance for imp in episode_importances]
        
        # Muestrear episodios
        sampled_episodes = np.random.choice(
            self.episodes, 
            min(len(self.episodes), batch_size // 10 + 1),
            replace=False,
            p=probs
        )
        
        # Obtener transiciones de cada episodio
        states, actions, rewards, next_states, dones = [], [], [], [], []
        
        transitions_per_episode = max(1, batch_size // len(sampled_episodes))
        
        for episode in sampled_episodes:
            ep_states, ep_actions, ep_rewards, ep_next_states, ep_dones = episode.get_random_batch(transitions_per_episode)
            
            states.extend(ep_states)
            actions.extend(ep_actions)
            rewards.extend(ep_rewards)
            next_states.extend(ep_next_states)
            dones.extend(ep_dones)
        
        # Limitar al tamaño del lote
        if len(states) > batch_size:
            indices = np.random.choice(len(states), batch_size, replace=False)
            states = [states[i] for i in indices]
            actions = [actions[i] for i in indices]
            rewards = [rewards[i] for i in indices]
            next_states = [next_states[i] for i in indices]
            dones = [dones[i] for i in indices]
        
        return states, actions, rewards, next_states, dones
    
    def sample_causal_batch(self, state, batch_size):
        """Muestrea transiciones causalmente relacionadas con el estado dado"""
        if not self.episodes or batch_size <= 0:
            return self.sample_batch(batch_size)
        
        # Buscar estados similares
        similar_transitions = []
        
        for episode in self.episodes:
            for i, ep_state in enumerate(episode.states):
                similarity = self._calculate_state_similarity(state, ep_state)
                
                if similarity > 0.8:  # Umbral de similitud
                    similar_transitions.append((
                        episode.states[i],
                        episode.actions[i],
                        episode.rewards[i],
                        episode.next_states[i],
                        episode.dones[i],
                        similarity
                    ))
        
        # Si no hay suficientes transiciones similares, completar con muestreo aleatorio
        if len(similar_transitions) < batch_size:
            regular_batch = self.sample_batch(batch_size - len(similar_transitions))
            
            # Combinar con transiciones similares
            states = [t[0] for t in similar_transitions] + regular_batch[0]
            actions = [t[1] for t in similar_transitions] + regular_batch[1]
            rewards = [t[2] for t in similar_transitions] + regular_batch[2]
            next_states = [t[3] for t in similar_transitions] + regular_batch[3]
            dones = [t[4] for t in similar_transitions] + regular_batch[4]
            
            return states, actions, rewards, next_states, dones
        
        # Ordenar por similitud (mayor primero)
        similar_transitions.sort(key=lambda x: x[5], reverse=True)
        
        # Tomar las más similares hasta el tamaño del lote
        selected = similar_transitions[:batch_size]
        
        return (
            [t[0] for t in selected],
            [t[1] for t in selected],
            [t[2] for t in selected],
            [t[3] for t in selected],
            [t[4] for t in selected]
        )
    
    def perform_counterfactual_analysis(self, state, action, reward, next_state):
        """Realiza análisis contrafactual para estimar el valor de acciones alternativas"""
        # Versión simplificada
        counterfactual_estimates = {}
        
        # Obtener clave de estado
        state_key = self._get_state_key(state)
        
        # Buscar en patrones causales
        if state_key in self.causal_patterns:
            for alt_action, data in self.causal_patterns[state_key].items():
                if alt_action != action and data['count'] > 0:
                    # Estimar recompensa para acción alternativa
                    counterfactual_estimates[alt_action] = data['reward'] / data['count']
        
        return counterfactual_estimates
    
    def _get_state_key(self, state):
        """Obtiene una clave de hash para identificar estados similares"""
        # Implementación simplificada
        if torch.is_tensor(state):
            # Redondear a primer decimal para agrupar estados similares
            rounded = torch.round(state * 10) / 10
            # Convertir a tupla para poder usar como clave de diccionario
            return tuple(rounded.flatten().tolist())
        else:
            # Si no es tensor, convertir a tupla de valores redondeados
            return tuple(round(x * 10) / 10 for x in np.array(state).flatten())

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
        """Adjust hyperparameters based on recent performance"""
        # Only adjust if we have enough history
        if len(self.performance_history) < 10:
            return
        
        # Determine if we're improving or declining
        recent_perf = list(self.performance_history)[-10:]
        if len(recent_perf) < 10:
            return
            
        first_half = sum(recent_perf[:5]) / 5
        second_half = sum(recent_perf[5:]) / 5
        
        # Calculate trend
        performance_trend = (second_half - first_half) / max(1, first_half)
        
        # Decide whether to explore or exploit
        if random.random() < self.meta_epsilon:
            # Random exploration of hyperparameters
            self._random_hyperparameter_adjustment()
        else:
            # Targeted adjustments based on performance trend
            self._targeted_hyperparameter_adjustment(performance_trend)
        
        # Decay meta-epsilon
        self.meta_epsilon *= self.meta_epsilon_decay
        self.meta_epsilon = max(0.1, self.meta_epsilon)
    
    def _random_hyperparameter_adjustment(self):
        """Make a random adjustment to one hyperparameter"""
        # Select a random hyperparameter to adjust
        param = random.choice(list(self.hyper_ranges.keys()))
        
        # Get current value and range
        if param == 'learning_rate':
            current = self.agent.learning_rate
        elif param == 'gamma':
            current = self.agent.gamma
        elif param == 'epsilon_decay':
            current = self.agent.epsilon_decay
        elif param == 'batch_size':
            current = self.agent.batch_size
        
        min_val, max_val = self.hyper_ranges[param]
        
        # Generate new value
        if param == 'batch_size':
            # Batch size should be an integer and a power of 2
            options = [32, 64, 96, 128]
            new_value = random.choice(options)
        else:
            # Random adjustment within range
            adjustment = (random.random() - 0.5) * 0.2  # ±10% adjustment
            new_value = current * (1 + adjustment)
            new_value = max(min_val, min(max_val, new_value))
        
        # Apply new value
        if param == 'learning_rate':
            self.agent.learning_rate = new_value
            # Update optimizer
            for param_group in self.agent.optimizer.param_groups:
                param_group['lr'] = new_value
        elif param == 'gamma':
            self.agent.gamma = new_value
        elif param == 'epsilon_decay':
            self.agent.epsilon_decay = new_value
        elif param == 'batch_size':
            self.agent.batch_size = new_value
        
        print(f"Meta-adjustment: {param} = {new_value}")
    
    def _targeted_hyperparameter_adjustment(self, performance_trend):
        """Make targeted adjustments based on performance trend"""
        if performance_trend > 0.1:
            # Significant improvement - stay the course but slightly increase learning
            if random.random() < 0.5:
                self.agent.learning_rate *= 1.05
                self.agent.learning_rate = min(self.hyper_ranges['learning_rate'][1], self.agent.learning_rate)
                # Update optimizer
                for param_group in self.agent.optimizer.param_groups:
                    param_group['lr'] = self.agent.learning_rate
                print(f"Meta: Increasing learning rate to {self.agent.learning_rate:.6f}")
            
            # Slightly slower epsilon decay to exploit more
            self.agent.epsilon_decay = max(self.agent.epsilon_decay * 0.98, self.hyper_ranges['epsilon_decay'][0])
            print(f"Meta: Adjusted epsilon decay to {self.agent.epsilon_decay:.4f}")
            
            # Change strategy to exploit
            self.current_strategy = 'exploit'
        
        elif performance_trend < -0.1:
            # Significant decline - make larger adjustments
            
            # Reduce learning rate
            self.agent.learning_rate *= 0.8
            self.agent.learning_rate = max(self.hyper_ranges['learning_rate'][0], self.agent.learning_rate)
            # Update optimizer
            for param_group in self.agent.optimizer.param_groups:
                param_group['lr'] = self.agent.learning_rate
            print(f"Meta: Decreasing learning rate to {self.agent.learning_rate:.6f}")
            
            # Faster epsilon decay to encourage exploration
            self.agent.epsilon_decay = min(self.agent.epsilon_decay * 1.02, self.hyper_ranges['epsilon_decay'][1])
            print(f"Meta: Adjusted epsilon decay to {self.agent.epsilon_decay:.4f}")
            
            # Change strategy to explore
            self.current_strategy = 'explore'
        
        else:
            # Minor or no change - make small random adjustments
            param = random.choice(['learning_rate', 'gamma'])
            
            if param == 'learning_rate':
                adjustment = (random.random() - 0.5) * 0.1  # ±5% adjustment
                self.agent.learning_rate *= (1 + adjustment)
                self.agent.learning_rate = max(self.hyper_ranges['learning_rate'][0], 
                                            min(self.hyper_ranges['learning_rate'][1], self.agent.learning_rate))
                # Update optimizer
                for param_group in self.agent.optimizer.param_groups:
                    param_group['lr'] = self.agent.learning_rate
                print(f"Meta: Fine-tuning learning rate to {self.agent.learning_rate:.6f}")
                
            elif param == 'gamma':
                adjustment = (random.random() - 0.5) * 0.02  # ±1% adjustment
                self.agent.gamma *= (1 + adjustment)
                self.agent.gamma = max(self.hyper_ranges['gamma'][0], 
                                    min(self.hyper_ranges['gamma'][1], self.agent.gamma))
                print(f"Meta: Fine-tuning gamma to {self.agent.gamma:.4f}")
            
            # Occasionally change strategy
            if random.random() < 0.2:
                self.current_strategy = random.choice(self.strategies)
                print(f"Meta: Switching strategy to {self.current_strategy}")
    
    def adjust_architecture(self):
        """Adapt neural network architecture based on performance"""
        # Don't adjust if architecture is frozen or we don't have enough history
        if self.architecture_frozen or len(self.performance_history) < 20:
            return
        
        # Get performance trend
        recent_perf = list(self.performance_history)[-20:]
        first_half = sum(recent_perf[:10]) / 10
        second_half = sum(recent_perf[10:]) / 10
        performance_trend = (second_half - first_half) / max(1, first_half)
        
        if performance_trend < -0.05:
            # Performance is declining - try growing the network
            if random.random() < 0.7:  # 70% chance to grow
                self._grow_network()
            else:
                self._prune_network()  # Sometimes pruning helps too
        
        elif performance_trend > 0.05:
            # Performance is improving - try pruning for efficiency
            if random.random() < 0.3:  # 30% chance to grow
                self._grow_network()
            else:
                self._prune_network()  # Focus on pruning when things are good
        
        else:
            # Stable performance - occasional random adjustments
            if random.random() < 0.5:
                self._grow_network()
            else:
                self._prune_network()
    
    def _grow_network(self):
        """Trigger growth in one of the neural modules"""
        # Select a random module
        modules = list(self.agent.modules.keys())
        module_name = random.choice(modules)
        module = self.agent.modules[module_name]
        
        # Find dynamic layers in the module
        dynamic_layers = []
        
        # Handle different module structures
        if module_name == 'perception':
            # Look in feature_extractor
            for i, layer in enumerate(module.feature_extractor):
                if isinstance(layer, DynamicLayer):
                    dynamic_layers.append((i, layer))
        
        elif module_name == 'navigation':
            # Look in pathfinder
            for i, layer in enumerate(module.pathfinder):
                if isinstance(layer, DynamicLayer):
                    dynamic_layers.append((i, layer))
        
        elif module_name == 'executive':
            # Look in integrator and action_selector
            for i, layer in enumerate(module.integrator):
                if isinstance(layer, DynamicLayer):
                    dynamic_layers.append((i, layer))
            dynamic_layers.append((-1, module.action_selector))
        
        # Try to grow a random dynamic layer
        if dynamic_layers:
            idx, layer = random.choice(dynamic_layers)
            # Only grow if current size is less than 80% of max to avoid over-growing
            if layer.current_output_size < 0.8 * layer.max_output_size:
                success = layer.grow_neurons()
                if success:
                    print(f"Meta: Grew layer in {module_name} module")
    
    def _prune_network(self):
        """Trigger pruning in one of the neural modules"""
        # Select a random module
        modules = list(self.agent.modules.keys())
        module_name = random.choice(modules)
        module = self.agent.modules[module_name]
        
        # Find dynamic layers in the module
        dynamic_layers = []
        
        # Handle different module structures (similar to _grow_network)
        if module_name == 'perception':
            for i, layer in enumerate(module.feature_extractor):
                if isinstance(layer, DynamicLayer):
                    dynamic_layers.append(layer)
        
        elif module_name == 'navigation':
            for i, layer in enumerate(module.pathfinder):
                if isinstance(layer, DynamicLayer):
                    dynamic_layers.append(layer)
        
        elif module_name == 'executive':
            for i, layer in enumerate(module.integrator):
                if isinstance(layer, DynamicLayer):
                    dynamic_layers.append(layer)
            dynamic_layers.append(module.action_selector)
        
        # Try to prune a random dynamic layer
        if dynamic_layers:
            layer = random.choice(dynamic_layers)
            success = layer.prune_connections()
            if success:
                print(f"Meta: Pruned connections in {module_name} module")
    
    def adjust_learning_strategy(self):
        """Adjust learning strategy based on performance and environment"""
        # Only adjust if we have enough history
        if len(self.performance_history) < 10:
            return
        
        # Calculate average performance
        avg_performance = sum(self.performance_history) / len(self.performance_history)
        
        # Strategies:
        # - exploit: Focus on exploitation (low epsilon, high gamma)
        # - explore: Focus on exploration (high epsilon, balanced gamma)
        # - diversify: Balance between exploration and exploitation
        
        if self.current_strategy == 'exploit':
            # Lower epsilon, higher gamma
            self.agent.epsilon = max(self.agent.epsilon_min, self.agent.epsilon * 0.9)
            self.agent.gamma = min(0.99, self.agent.gamma * 1.01)
            
        elif self.current_strategy == 'explore':
            # Higher epsilon, balanced gamma
            self.agent.epsilon = min(0.3, self.agent.epsilon * 1.1)
            self.agent.gamma = 0.95  # Mid-range value
            
        elif self.current_strategy == 'diversify':
            # Balanced values
            self.agent.epsilon = 0.1
            self.agent.gamma = 0.97
        
        # Log current strategy
        print(f"Learning strategy: {self.current_strategy} (ε={self.agent.epsilon:.2f}, γ={self.agent.gamma:.2f})")
    
    def update_curriculum(self, env):
        """Update curriculum learning parameters based on performance"""
        # Track episodes at current difficulty
        self.curriculum_episodes += 1
        
        # Compute average performance over recent episodes
        avg_performance = 0.0
        if self.performance_history:
            avg_performance = sum(self.performance_history) / len(self.performance_history)
        
        # Update curriculum metrics
        self.curriculum_performance = 0.9 * self.curriculum_performance + 0.1 * avg_performance
        
        # Consider changing difficulty after sufficient episodes
        if self.curriculum_episodes >= 20:
            # If performing well, consider increasing difficulty
            if self.curriculum_performance > DIFFICULTY_ADJUST_THRESHOLD:
                # Increase difficulty (max 2)
                if env.difficulty < 2:
                    env.difficulty += 1
                    print(f"Curriculum: Increased difficulty to {env.difficulty}")
                    
                    # Reset curriculum tracking for new difficulty
                    self.curriculum_episodes = 0
                    self.curriculum_performance = 0.0
                    
                    # Adjust agent parameters for new difficulty
                    self.agent.epsilon = min(0.3, self.agent.epsilon * 1.2)  # Increase exploration
                    
                    # Apply appropriate skill prototype if available
                    if env.difficulty == 1:
                        self.agent.skill_library.apply_prototype(self.agent, 'navigation', 1)
                    elif env.difficulty == 2:
                        self.agent.skill_library.apply_prototype(self.agent, 'navigation', 2)
                        
            # If performing poorly for a long time, consider decreasing difficulty
            elif self.curriculum_performance < DIFFICULTY_ADJUST_THRESHOLD * 0.5 and self.curriculum_episodes > 50:
                # Decrease difficulty (min 0)
                if env.difficulty > 0:
                    env.difficulty -= 1
                    print(f"Curriculum: Decreased difficulty to {env.difficulty}")
                    
                    # Reset curriculum tracking for new difficulty
                    self.curriculum_episodes = 0
                    self.curriculum_performance = 0.0

# ------------------- Meta-Learning System -------------------

if __name__ == "__main__":
    import argparse
    
    # Crear un parser de argumentos para permitir opciones por línea de comandos
    parser = argparse.ArgumentParser(description='Entrenamiento de agente neuronal avanzado para Snake')
    parser.add_argument('--episodes', type=int, default=1000, help='Número de episodios para entrenar')
    parser.add_argument('--save-path', type=str, default='snake_agent.pth', help='Ruta para guardar el agente')
    parser.add_argument('--new-agent', action='store_true', help='Crear un nuevo agente en lugar de cargar uno existente')
    parser.add_argument('--difficulty', type=int, default=0, choices=[0, 1, 2], 
                        help='Dificultad del entorno (0: sin laberinto, 1: laberinto simple, 2: laberinto complejo)')
    
    args = parser.parse_args()
    
    # Iniciar entrenamiento con los argumentos proporcionados
    integrate_with_snakeRL(
        episodes=args.episodes, 
        save_path=args.save_path, 
        load_existing=not args.new_agent,
        difficulty=args.difficulty
    )

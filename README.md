# 🐍 SnakeNN: The Brainiest Snake You'll Ever Meet! 🧠✨

Welcome to **SnakeNN**, where the classic Snake game gets a futuristic makeover with AI! This isn't just a snake chasing apples—it's a neural-network-powered genius learning to slither smarter with every bite.

## 🎉 What's SnakeNN All About?

SnakeNN is the classic Snake game enhanced with artificial intelligence. Using **PyTorch**, **reinforcement learning**, and a sprinkle of curiosity, SnakeNN evolves as it plays. It dodges walls, navigates mazes, and hunts food like a pro—all thanks to its dynamic neural network that grows and prunes itself on the fly.

## 🛠️ Requirements

- **Python 3.7+** 🐍
- **Git** 🌿
- **Optional**: An NVIDIA GPU with CUDA support for faster AI training

## 📥 Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/SnakeNN.git
   cd SnakeNN
   ```

2. **Install Dependencies**
   ```bash
   pip install pygame numpy torch
   ```
   - **Pygame**: Renders the game environment
   - **NumPy**: Handles numerical operations
   - **PyTorch**: Powers the neural network

   *For GPU users: Install the CUDA version of PyTorch from the official website for better performance*

3. **Verify Installation**
   ```bash
   python -c "import pygame, numpy, torch; print('Ready to roll! 🐍')"
   ```

4. **Check GPU Availability** (optional)
   ```bash
   python -c "import torch; print('GPU Ready!' if torch.cuda.is_available() else 'No GPU, but we'll manage!')"
   ```

## 🎮 How to Play

### Manual Mode
Run the game and control the snake yourself:
```bash
python SnakeRL.py
```

**Controls:**
- Arrow Keys ⬆️⬇️⬅️➡️ – Move the snake
- ESC – Exit the game
- M – Increase speed
- N – Decrease speed

### AI Mode
Let the neural network play and learn:
```bash
python SnakeNeuralAdvanced.py
```
- Watch the snake learn and adapt its strategy
- Press ESC to exit

## 🌟 Key Features

- **Dynamic Neural Network**: The network grows and shrinks as it learns
- **Curiosity-Driven Exploration**: The AI explores based on novelty and surprise
- **Maze Navigation**: Successfully handles simple and complex mazes
- **Skill Transfer**: Applies learned strategies to new challenges
- **Progress Visualization**: Optional graphs to monitor AI learning

## 🐛 Troubleshooting

### Performance Issues
- **CPU Running Slow**: Try smaller mazes or fewer episodes
- **GPU Not Utilized**: Verify CUDA setup with the test command above

### Installation Problems
- **Missing Libraries**: Run `pip install pygame numpy torch` again
- **Version Conflicts**: Use a virtual environment:
  ```bash
  python -m venv venv
  source venv/bin/activate  # On Windows: venv\Scripts\activate
  pip install pygame numpy torch
  ```

### Crashes
- Ensure Python version is 3.7+ (`python --version`)
- Check terminal output for error messages

## 🤝 Contributing

- Open issues or submit pull requests on GitHub
- Help make SnakeNN even smarter!

## 📜 License & Credits

- Licensed under the MIT License
- Created by Javier Lora
- Powered by Pygame, NumPy, and PyTorch
- Inspired by the classic Snake game

Happy coding and snake charming! 🍎🐍
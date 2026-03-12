# 📌 Suggested Improvements

## 1. Add visuals for CNN & RNN architecture

## 2. Saving models and training checkpoints

## 3. Note that tank detector example is apocryphal

- https://gwern.net/tank

## 4. **Data Preparation Workflow**

**Improvement**: Add a step-by-step visual workflow for data preparation (normalization → handling missing values → feature engineering).  
**Content Outline**:  

1. **Input**: Raw data.  
2. **Cleaning**: Impute missing values.  
3. **Transformation**: Normalize.  
4. **Feature Engineering**: Create polynomial terms.  

---

## 5. **CNN vs. RNN Comparison**

**Improvement**: Add a side-by-side table contrasting CNNs (spatial hierarchies) and RNNs (temporal dependencies).  
**Content Outline**:  

| Feature | CNN | RNN |  
|--------|-----|-----|  
| Use Case | Images | Time series |  
| Key Layer | Convolutional | Recurrent |  
| Strength | Local pattern detection | Sequence memory |  

---

## 6. **Transformer Architecture Deep Dive** (NOTE: next lecture, DO NOT ADD)

**Improvement**: Add a simplified diagram of the transformer encoder-decoder structure with attention heads.  
**Content Outline**:  

- Label key components: Input Embeddings → Positional Encoding → Multi-Head Attention → Feed-Forward → Output.  

## 8. **Ethics & Hallucination Case Study** (NOTE: next lecture, DO NOT ADD)

**Improvement**: Add a real-world example of LLM hallucination (e.g., fake legal citations) and mitigation strategies.  
**Content Outline**:  

- Scenario: A law firm uses an LLM that cites non-existent precedents.  
- Mitigation: "Add fact-checking layer" and "train with curated data."  

## 10. **TensorBoard Integration**

**Improvement**: Add a short tutorial on using TensorBoard with Keras/PyTorch to visualize training metrics.  
**Content Outline**:  

```python
# Keras
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")
model.fit(x_train, y_train, callbacks=[tensorboard_callback])
```

- Screenshot of TensorBoard interface showing loss  

---

## 11. **Vanishing Gradient Visualization** (DO NOT ADD FOR NOW)

**Improvement**: Add a PyTorch example demonstrating how ReLU mitigates vanishing gradients compared to sigmoid.  
**Content Outline**:  

```python
# Simple RNN gradient flow visualization
import torch
from torch import nn

class GradientTest(nn.Module):
    def __init__(self):
        super().__init__()
        self.l = 5
        self.stacks = nn.ModuleList([nn.Linear(100, 100) for _ in range(self.l)])
        
    def forward(self, x):
        for i, layer in enumerate(self.stacks):
            x = torch.relu(layer(x))
            x.retain_grad()
        return x, [param.grad for param in self.parameters()]
```

- Explain how this connects to the biological neuron analogy in slide #26

## 13. **4-Step Debugging Framework**  

**Improvement**: Create checklist for troubleshooting neural networks  
**Content Outline**:  

```
🔍 INSPECT  
1. Data pipeline: print shapes, value ranges  
2. Weight initialization: should be ~0 mean  
3. Loss components: individual contributions  
4. Gradient flow: use torchviz to visualize
```

- Example code for gradient visualization:

```python
import torchviz
output = model(data)
loss = loss_function(output, target)
graph = torchviz.make_dot(loss)
graph.render("network")

# Maximum Likelihood Estimation & EM Algorithm - Assignment 3

![Python](https://img.shields.io/badge/Python-3.13-blue.svg)
![Status](https://img.shields.io/badge/Status-Complete-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 📋 Descrição do Projeto

Este projeto implementa algoritmos de **Estimação de Máxima Verossimilhança (MLE)** e **Expectation-Maximization (EM)** para análise de dados Gaussianos com dados faltantes. O trabalho compara diferentes abordagens de estimação de parâmetros e demonstra o impacto de dados missing na qualidade das estimativas.

## 🎯 Objetivos

- ✅ Implementar MLE para distribuições Gaussianas (1D, 2D, 3D)
- ✅ Desenvolver algoritmo EM para dados com valores faltantes
- ✅ Comparar resultados entre dados completos e incompletos
- ✅ Visualizar clusters e convergência dos algoritmos
- ✅ Analisar impacto de diferentes estratégias de inicialização

## 📊 Datasets

### Categoria ω₁ (Omega 1)
- **10 pontos tridimensionais** com características [x₁, x₂, x₃]
- **Dados faltantes**: x₃ missing nos pontos pares (2, 4, 6, 8, 10)
- **Taxa de missing**: 50% na dimensão x₃

### Categoria ω₂ (Omega 2)  
- **10 pontos tridimensionais** completos
- **Usado para**: Modelo separável com matriz de covariância diagonal

## 🚀 Principais Funcionalidades

### 1. **MLE Tradicional** (`mle_omega1.py`)
- Estimação univariada para cada característica
- Análise bivariada para pares de características
- Análise trivariada completa (3D)
- Comparação de estimativas de média e variância

### 2. **Algoritmo EM** (`em_algorithm.py` / `em_algorithm_ascii.py`)
- Implementação completa do EM para dados faltantes
- Duas estratégias de inicialização:
  - Inicialização zero: x₃ = 0
  - Inicialização média: x₃ = (x₁ + x₂)/2
- Comparação com dados completos (ground truth)
- Análise de convergência detalhada

### 3. **Visualizações Avançadas**
- Gráficos 3D dos clusters
- Análise de convergência
- Comparações side-by-side
- Matrizes de correlação
- Análise de erros por característica

## 📁 Estrutura do Projeto

```
assignment3/
├── 🔧 Código Principal
│   ├── mle_omega1.py              # MLE tradicional (dados completos)
│   ├── em_algorithm.py            # Algoritmo EM (versão Unicode)
│   └── em_algorithm_ascii.py      # Algoritmo EM (versão ASCII)
│
├── 📊 Visualização e Análise
│   ├── cluster_visualization.py   # Visualizações abrangentes
│   ├── results_summary.py         # Resumo visual dos resultados
│   └── show_plots.py             # Visualizador de gráficos
│
├── 📈 Resultados (Gráficos)
│   ├── em_convergence_analysis.png
│   ├── em_complete_comparison.png
│   └── cluster_analysis_comprehensive.png
│
├── 📚 Documentação
│   ├── README.md                  # Este arquivo
│   ├── complete_vs_missing_comparison.md
│   ├── em_summary_report.md
│   └── requirements.txt           # Dependências
│
└── 📁 Outros
    ├── em_output.txt             # Logs de execução
    └── __pycache__/              # Cache Python
```

## 🛠️ Instalação e Uso

### Pré-requisitos
```bash
Python 3.13+
pip (gerenciador de pacotes Python)
```

### 1. Clone o repositório
```bash
git clone https://github.com/seu-usuario/mle-em-algorithm-assignment3.git
cd mle-em-algorithm-assignment3
```

### 2. Instale as dependências
```bash
pip install -r requirements.txt
```

### 3. Execute os algoritmos

#### MLE Tradicional (dados completos):
```bash
python mle_omega1.py
```

#### Algoritmo EM (dados faltantes):
```bash
python em_algorithm.py
# ou versão ASCII compatível:
python em_algorithm_ascii.py
```

#### Visualizar resultados:
```bash
python show_plots.py
python results_summary.py
```

## 📊 Principais Resultados

### ✅ **Sucessos do Algoritmo EM**
- **Recuperação perfeita** das dimensões observadas (x₁, x₂)
- **Erro zero** em μ₁, μ₂, σ₁², σ₂²
- **Convergência robusta** em 16 iterações
- **Independência da inicialização**

### ❌ **Limitações Identificadas**  
- **Viés sistemático** na dimensão faltante (x₃)
- **Erro de 1.684 unidades** em μ₃ (185% erro relativo)
- **Subestimação de 61%** em σ₃²
- **Compressão do cluster** (54% redução no volume)

### 🎯 **Insights Principais**
1. **MLE preserva informação** nas dimensões observadas perfeitamente
2. **Padrão de missing data** afeta significativamente estimativas
3. **EM é robusto** mas introduz viés previsível
4. **Estrutura de cluster** é parcialmente recuperável

## 📈 Visualizações Geradas

### 1. **Análise de Convergência**
- Evolução da log-likelihood
- Comparação entre estratégias de inicialização
- Demonstração de convergência idêntica

### 2. **Comparação Completo vs Missing**
- Side-by-side dos parâmetros estimados
- Análise de erros por característica
- Impacto visual dos dados faltantes

### 3. **Análise Abrangente de Clusters**
- Visualização 3D dos dados
- Projeções 2D
- Matrizes de correlação
- Análise detalhada de erros

## 🔬 Fundamentos Teóricos

### **MLE (Maximum Likelihood Estimation)**
```
μ̂ = (1/n) × Σᵢ xᵢ
Σ̂ = (1/n) × Σᵢ (xᵢ - μ̂)(xᵢ - μ̂)ᵀ
```

### **Algoritmo EM**
- **E-step**: E[X₃|X₁,X₂] = μ₃ + Σ₃₁Σ₁₁⁻¹(X₁₂ - μ₁₂)
- **M-step**: Atualização dos parâmetros com dados "completos"
- **Convergência**: Baseada na log-likelihood

## 🤝 Contribuições

Contribuições são bem-vindas! Por favor:

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## 📝 Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

## 👥 Autores

- **Seu Nome** - *Trabalho inicial* - [SeuGitHub](https://github.com/seu-usuario)

## 🙏 Agradecimentos

- Implementação baseada em conceitos de Machine Learning e Estatística
- Algoritmos fundamentados em teoria de Estimação de Máxima Verossimilhança
- Visualizações inspiradas em práticas de Data Science

## 📚 Referências

1. Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977). Maximum likelihood from incomplete data via the EM algorithm.
2. Bishop, C. M. (2006). Pattern Recognition and Machine Learning.
3. Murphy, K. P. (2012). Machine Learning: A Probabilistic Perspective.

---

⭐ **Se este projeto foi útil para você, considere dar uma estrela!** ⭐
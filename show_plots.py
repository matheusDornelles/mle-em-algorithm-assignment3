"""
Script para visualizar os gráficos gerados pelo algoritmo EM
"""

import matplotlib.pyplot as plt
from PIL import Image
import os

def show_plots():
    """
    Exibe todos os gráficos gerados pelo algoritmo EM
    """
    # Caminhos dos arquivos de imagem
    convergence_plot = "em_convergence_analysis.png"
    comparison_plot = "em_complete_comparison.png"
    comprehensive_plot = "cluster_analysis_comprehensive.png"
    
    # Lista de gráficos disponíveis
    plots = [
        (convergence_plot, "Análise de Convergência do EM"),
        (comparison_plot, "Comparação: Dados Completos vs EM"),
        (comprehensive_plot, "Análise Abrangente dos Clusters")
    ]
    
    print("=" * 80)
    print("VISUALIZAÇÃO DOS GRÁFICOS DO ALGORITMO EM")
    print("=" * 80)
    
    # Verificar quais gráficos existem
    available_plots = []
    for plot_file, title in plots:
        if os.path.exists(plot_file):
            available_plots.append((plot_file, title))
            print(f"✅ Encontrado: {plot_file}")
        else:
            print(f"❌ Não encontrado: {plot_file}")
    
    if not available_plots:
        print("\n⚠️  Nenhum gráfico encontrado!")
        return
    
    # Exibir cada gráfico disponível
    for i, (plot_file, title) in enumerate(available_plots, 1):
        try:
            print(f"\n📊 Exibindo Gráfico {i}: {title}")
            print("-" * 60)
            
            # Carregar e exibir a imagem
            img = Image.open(plot_file)
            
            # Criar figura matplotlib para exibição
            plt.figure(figsize=(12, 8))
            plt.imshow(img)
            plt.axis('off')
            plt.title(title, fontsize=16, fontweight='bold', pad=20)
            
            # Mostrar o gráfico
            plt.tight_layout()
            plt.show()
            
            print(f"✅ Gráfico exibido com sucesso!")
            
            # Informações sobre o arquivo
            file_size = os.path.getsize(plot_file) / 1024  # KB
            print(f"📁 Arquivo: {plot_file}")
            print(f"📏 Tamanho: {file_size:.1f} KB")
            print(f"🖼️  Dimensões: {img.size[0]} x {img.size[1]} pixels")
            
        except Exception as e:
            print(f"❌ Erro ao carregar {plot_file}: {str(e)}")
    
    print(f"\n" + "=" * 80)
    print("DESCRIÇÃO DOS GRÁFICOS")
    print("=" * 80)
    
    print("""
📊 GRÁFICO 1: Análise de Convergência do EM
   - Painel esquerdo: Evolução da log-likelihood durante as iterações
   - Painel direito: Comparação das estimativas μ₃ entre estratégias
   - Mostra que ambas estratégias convergem para a mesma solução
   
📊 GRÁFICO 2: Comparação Dados Completos vs EM  
   - Comparação direta entre parâmetros verdadeiros e estimados
   - Visualiza o impacto dos dados faltantes
   - Destaca quais parâmetros são perfeitamente recuperados
   
📊 GRÁFICO 3: Análise Abrangente dos Clusters
   - Visualização 3D dos pontos de dados e centros dos clusters
   - Projeções 2D das características
   - Análise de correlações e matrizes de covariância
   - Comparação de erros por característica
   
🎯 PRINCIPAIS INSIGHTS:
   ✅ μ₁ e μ₂: Recuperação perfeita (erro = 0)
   ✅ σ₁² e σ₂²: Estimação exata das variâncias observadas  
   ❌ μ₃: Erro sistemático de 1.684 unidades
   ❌ σ₃²: Subestimação de 61% na variância
   🔄 Convergência: 16 iterações para ambas estratégias
    """)

if __name__ == "__main__":
    show_plots()
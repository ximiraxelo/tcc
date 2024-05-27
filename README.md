# Trabalho de Conclusão de Curso 📄

> Estimação de parâmetros de Sistemas Lineares a Parâmetros Variantes no Tempo utilizando técnicas de aprendizado de máquina

**👦🏻 Aluno:** [Esdras Battosti da Silva](http://lattes.cnpq.br/5361064829624642)
**👩🏻‍🏫 Orientador:** [Profª. Drª. Glaucia Maria Bressan  ](http://lattes.cnpq.br/2648513655629475)
**👨🏻‍🏫 Coorientador:** [Prof. Dr. Cristiano Marcos Agulhari](http://lattes.cnpq.br/4935395556663775)

---
Trabalho de Conclusão de Curso de Graduação apresentado como requisito para obtenção do título de Bacharel em Engenharia de Controle e Automação do Curso de Bacharelado em Engenharia de Controle e Automação da Universidade Tecnológica Federal do Paraná

---

## Projeto 🐍

O projeto foi construído no Python 3.12.2 em conjunto com uma série de bibliotecas. Após clonar o repositório execute:

```
pip install -r requirements.txt
```

---

## Workflow 📄

O projeto se baseia nos seguintes passos:

1. O arquivo `main/dataset.py` é utilizado para gerar o banco de dados. Neste arquivo é possível definir os parâmetros utilizados na construção do dataset e modificar o sistema em estudo.
2. No arquivo `main/data_cleaning.py` é feita a limpeza dos dados, removendo todas as duplicatas em relação à $(X, y)$.
3. No arquivo `main/feature_selection.py` é realizada a primeira etapa da seleção de atributos, baseada na correlação de Pearson. Pode-se modificar neste arquivo o *threshold* para o critério de seleção.
4. No arquivo `main/feature_sequential.py` é realizada a segunda etapa da seleção de atributos, baseada no algoritmo SFS com XGBoost.
   * No arquivo `plots/sfs_plot.py` pode-se gerar o gráfico da relação entre os $k$ atributos selecionados pelo SFS e o coeficiente de determinação $(R^2)$
5. O arquivo `plots/exploratory_plots.py` é responsável por gerar os gráficos da análise exploratória.
    * A análise da performance dos modelos com e sem os atributos altamente correlacionados é realizada no arquivo `main/feature_analysis.py`
6. A otimização de hiperparâmetros é realizada no arquivo `main/hyperparameter.ipynb`. Neste arquivo, pode-se definir os espaços de busca para cada modelo bem como outros parâmetros da otimização.
7. Os resultados obtidos na etapa anterior são exibidos no arquivo `main/models.ipynb`
8. A etapa de validação dos modelos em alguns casos de teste é realizada no arquivo `main/validation.ipynb`

---

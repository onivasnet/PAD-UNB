# PAD-UNB

Dashboard em Streamlit para analisar coerencia entre a atuacao legislativa de deputados federais e o perfil medio de seus partidos, usando arquivos locais dos Dados Abertos da Camara dos Deputados.

## O que o dashboard calcula

- Classificacao tematica de proposicoes por TF-IDF com vocabulario-semente.
- Vetores de atuacao por deputado combinando projetos, votacoes e frentes/grupos.
- Vetor medio por partido, sem classificacao ideologica manual.
- Coerencia por similaridade do cosseno entre deputado e partido.
- PCA e K-Means para identificar agrupamentos de atuacao.
- Graficos comparativos por deputado, partido, tema, cluster e votacao.

## Como executar

1. Instale as dependencias:

```bash
python -m pip install -r requirements.txt
```

2. Coloque os arquivos CSV/JSON/XLSX da Camara dos Deputados na mesma pasta do projeto (`trabalho1`) ou informe outra pasta na barra lateral do dashboard.

3. Execute:

```bash
python -m streamlit run py.py
```

O app abre em `http://localhost:8501`.

## Arquivos esperados

O programa detecta automaticamente arquivos com nomes como:

- `deputados.csv`
- `proposicoes-2026.csv`
- `proposicoesAutores-2026.csv`
- `votacoesVotos-2026.csv`
- `votacoesObjetos-2026.csv`
- `frentesDeputados.csv`
- `gruposMembros.csv`

O seletor lateral permite escolher os anos disponiveis nos arquivos.

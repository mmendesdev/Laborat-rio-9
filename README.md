# LAB 09 – Arquitetura RAG Avançada: HNSW, HyDE e Cross-Encoders

Assistente de busca em manuais médicos que combina **indexação hierárquica (HNSW)**, **transformação semântica de queries (HyDE)** e **re-ranking por atenção cruzada (Cross-Encoder)** para superar o gap semântico entre a linguagem coloquial do paciente e o jargão técnico dos manuais clínicos.

---

## Arquitetura do Pipeline

```
Query coloquial ("dor de cabeça latejante e luz incomodando")
        │
        ▼
[PASSO 2] HyDE — LLM gera documento técnico hipotético
        │  "Cefaleia pulsátil unilateral com fotofobia..."
        ▼
[PASSO 3] Bi-Encoder — vetor HyDE busca Top-10 no índice HNSW
        │  Recuperação rápida O(log N)
        ▼
[PASSO 4] Cross-Encoder — re-ranqueia Top-10, retorna Top-3
        │  Atenção cruzada [CLS] query [SEP] doc [SEP]
        ▼
Top-3 documentos injetados no contexto do LLM gerador
```

---

## Análise dos Hiperparâmetros HNSW vs KNN Exato

### KNN Exato (brute-force)

| Aspecto | Fórmula | Exemplo (N=1M, d=1536) |
|---------|---------|------------------------|
| RAM (vetores) | `N × d × 4 bytes` | ~5,9 GB |
| RAM total | igual aos vetores | ~5,9 GB |
| Complexidade de busca | O(N × d) | linear — lento |

### HNSW

| Aspecto | Fórmula | Exemplo (N=1M, d=1536, M=32) |
|---------|---------|------------------------------|
| RAM (vetores) | `N × d × 4 bytes` | ~5,9 GB |
| RAM (grafo) | `N × 2M × 4 bytes` | ~256 MB (+4%) |
| RAM total | vetores + grafo | ~6,15 GB |
| Complexidade de busca | O(log N) | sub-linear — rápido |

### Efeito de `M` (vizinhos por nó)

- Cada nó armazena até `2×M` links inteiros (int32), logo a RAM do grafo cresce linearmente com M.
- `M=16` → RAM mínima, recall moderado. `M=64` → RAM dobrada, recall alto.
- **Impacto na RAM:** direto e proporcional — dobrar M ≈ dobrar o overhead do grafo.

### Efeito de `ef_construction`

- Controla o beam-search durante a **construção** do grafo.
- **Não altera o tamanho final do índice em RAM/disco** — apenas a qualidade das arestas.
- Valor maior → build mais lento, grafo com vizinhos mais relevantes, melhor recall na busca.
- Valor típico: 100–400 para produção.

### Conclusão

O overhead de memória do HNSW em relação ao KNN é de apenas ~5–10% para vetores de alta dimensão (d≥512), mas oferece busca em O(log N) contra O(N) do KNN — tornando o trade-off amplamente favorável para bases com mais de alguns milhares de documentos.

---

## Instalação

```bash
# 1. Clone o repositório e entre na pasta
git clone <url-do-repo>
cd lab-9

# 2. Crie e ative um ambiente virtual
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# 3. Instale as dependências
pip install -r requirements.txt
```

> **Nota Windows:** caso `faiss-cpu` falhe na instalação, tente `pip install faiss-cpu --extra-index-url https://download.pytorch.org/whl/cpu`

---

## Configuração

Copie `.env.example` para `.env` e insira sua chave da OpenAI:

```bash
cp .env.example .env
# edite .env e adicione sua OPENAI_API_KEY
```

---

## Execução

```bash
python rag_pipeline.py
```

O script executa os 4 passos e imprime no console:

1. Confirmação do índice HNSW construído (24 documentos).
2. O documento hipotético técnico gerado pelo LLM via HyDE.
3. Tabela com os **Top-10 documentos** recuperados pelo bi-encoder (scores de cosseno).
4. Os **Top-3 documentos finais** re-ranqueados pelo cross-encoder, prontos para injeção no LLM gerador.

### Exemplo de saída esperada

```
Query coloquial do usuário: "dor de cabeça latejante e luz incomodando"

PASSO 2 – Documento Hipotético:
  "A cefaleia pulsátil unilateral, associada a fotofobia e fonofobia,
   caracteriza a síndrome migrânosa..."

PASSO 3 – Top-10 (Bi-Encoder):
  #1  0.8821  Cefaleia Migrânosa – Diagnóstico Clínico
  #2  0.8714  Enxaqueca com Aura – Sintomas Neurológicos Focais
  ...

PASSO 4 – Top-3 (Cross-Encoder):
  #1 CE: 9.32  Cefaleia Migrânosa – Diagnóstico Clínico
  #2 CE: 8.17  Enxaqueca com Aura – Sintomas Neurológicos Focais
  #3 CE: 5.44  Cefaleia Tensional – Diagnóstico Diferencial
```

---

## Estrutura do Projeto

```
lab-9/
├── rag_pipeline.py   # Pipeline completo (4 passos)
├── requirements.txt  # Dependências Python
├── .env.example      # Template de variáveis de ambiente
├── .gitignore
└── README.md
```

---

## Declaração de Uso de IA

> Partes deste laboratório foram geradas/complementadas com IA, revisadas e validadas por [Seu Nome].

---

## Referências

- Lewis et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. NeurIPS.
- Gao et al. (2022). *Precise Zero-Shot Dense Retrieval without Relevance Labels (HyDE)*. ACL.
- Malkov & Yashunin (2018). *Efficient and Robust Approximate Nearest Neighbor Search Using HNSW Graphs*. IEEE TPAMI.
- Reimers & Gurevych (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*. EMNLP.

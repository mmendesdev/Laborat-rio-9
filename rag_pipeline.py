"""
LAB 09 – Arquitetura RAG Avançada: HNSW + HyDE + Cross-Encoder
Assistente de Busca em Manuais Médicos
"""

from __future__ import annotations

import os
import sys
import textwrap
import numpy as np
import faiss
from openai import OpenAI
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv

load_dotenv()

# ────────────────────────────────────────────────────────────────────────────────
# Configurações globais
# ────────────────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL      = "text-embedding-3-small"
LLM_MODEL            = "gpt-4o-mini"
CROSS_ENCODER_MODEL  = "cross-encoder/ms-marco-MiniLM-L-6-v2"
EMBEDDING_DIM        = 1536

HNSW_M               = 32    # nº de vizinhos por nó – afeta recall e RAM
HNSW_EF_CONSTRUCTION = 200   # qualidade da construção do grafo
HNSW_EF_SEARCH       = 50    # amplitude da busca em tempo de consulta

TOP_K_RETRIEVE       = 10    # funil largo (Bi-Encoder)
TOP_K_RERANK         = 3     # funil fino  (Cross-Encoder)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ────────────────────────────────────────────────────────────────────────────────
# Base de Conhecimento: 24 fragmentos de manuais médicos técnicos
# ────────────────────────────────────────────────────────────────────────────────
MEDICAL_FRAGMENTS: list[dict] = [
    {
        "id": 0,
        "title": "Cefaleia Migrânosa – Diagnóstico Clínico",
        "text": (
            "A cefaleia pulsátil unilateral, associada a fotofobia, fonofobia e náuseas, "
            "caracteriza a migrânea sem aura. O diagnóstico é clínico, baseado em pelo menos "
            "5 episódios com duração de 4 a 72 horas. A dor piora com atividade física "
            "rotineira e pode ser incapacitante. Critérios diagnósticos seguem a ICHD-3."
        ),
    },
    {
        "id": 1,
        "title": "Enxaqueca com Aura – Sintomas Neurológicos Focais",
        "text": (
            "Migrânea com aura caracteriza-se por sintomas neurológicos focais reversíveis que "
            "precedem ou acompanham a cefaleia pulsátil. A aura visual típica inclui escotomas "
            "cintilantes, fotópsia e teicópsias com progressão gradual em 20 a 60 minutos. "
            "Parestesias unilaterais e disfasia transitória também podem ocorrer."
        ),
    },
    {
        "id": 2,
        "title": "Cefaleia Tensional – Características e Diagnóstico Diferencial",
        "text": (
            "A cefaleia do tipo tensional manifesta-se como dor em pressão ou aperto bilateral, "
            "sem pulsatilidade, de intensidade leve a moderada. Não há agravamento pela atividade "
            "física. Pode ocorrer fotofobia OU fonofobia, mas não ambas simultaneamente. "
            "Frequentemente associada a estresse, bruxismo e tensão muscular cervical e pericraniana."
        ),
    },
    {
        "id": 3,
        "title": "Hipertensão Arterial Sistêmica – Classificação e Manejo",
        "text": (
            "Hipertensão arterial sistêmica (HAS) é definida por pressão sistólica ≥140 mmHg "
            "ou diastólica ≥90 mmHg em medições repetidas em consultório. A cefaleia occipital "
            "matinal pode ser sintoma em crises hipertensivas (PA>180/120 mmHg). Tratamento "
            "inclui IECA, BRA, bloqueadores de canal de cálcio e tiazídicos conforme o risco cardiovascular."
        ),
    },
    {
        "id": 4,
        "title": "Diabetes Mellitus Tipo 2 – Critérios Diagnósticos",
        "text": (
            "Diabetes mellitus tipo 2 é diagnosticado por glicemia de jejum ≥126 mg/dL em duas "
            "ocasiões, TOTG 2h ≥200 mg/dL, HbA1c ≥6,5% ou glicemia casual ≥200 mg/dL com "
            "sintomas clássicos. Poliúria, polidipsia e visão turva são manifestações comuns. "
            "Metformina é a primeira linha terapêutica; GLP-1 e SGLT-2 para comorbidades."
        ),
    },
    {
        "id": 5,
        "title": "Insuficiência Cardíaca Congestiva – Fisiopatologia e Tratamento",
        "text": (
            "Insuficiência cardíaca congestiva (ICC) resulta de disfunção sistólica (FE<40%) ou "
            "diastólica do ventrículo esquerdo. Dispneia aos esforços, ortopneia, estertores "
            "pulmonares e edema de membros inferiores são achados cardinais. BNP elevado confirma "
            "diagnóstico. Tratamento: IECA, betabloqueadores, diuréticos e antagonistas da aldosterona."
        ),
    },
    {
        "id": 6,
        "title": "Asma Brônquica – Diagnóstico e Classificação de Gravidade",
        "text": (
            "Asma brônquica é doença inflamatória crônica das vias aéreas com obstrução reversível. "
            "Sibilos expiratórios, tosse noturna, dispneia e aperto no peito são sintomas cardinais. "
            "Espirometria com VEF1/CVF <0,70 e resposta broncodilatadora ≥12% confirmam o diagnóstico. "
            "Classificação em intermitente, leve persistente, moderada e grave persistente."
        ),
    },
    {
        "id": 7,
        "title": "DPOC – Doença Pulmonar Obstrutiva Crônica e Estadiamento GOLD",
        "text": (
            "DPOC é definida por obstrução do fluxo aéreo não completamente reversível, causada "
            "principalmente pelo tabagismo crônico. Tosse produtiva, dispneia progressiva e "
            "hiperinsuflação pulmonar são características. Espirometria pós-broncodilatador com "
            "VEF1/CVF <0,70 confirma. Estadiamento GOLD I (leve) a IV (muito grave) conforme VEF1%."
        ),
    },
    {
        "id": 8,
        "title": "Infarto Agudo do Miocárdio com Supradesnivelamento de ST",
        "text": (
            "IAMCSST caracteriza-se por dor precordial opressiva irradiada para membro superior "
            "esquerdo, mandíbula ou dorso, com duração superior a 20 minutos. ECG evidencia "
            "supradesnivelamento de ST ≥1mm em ≥2 derivações contíguas. Troponina ultrassensível "
            "eleva-se em 1 a 3 horas. Reperfusão por angioplastia primária é a conduta de escolha."
        ),
    },
    {
        "id": 9,
        "title": "Acidente Vascular Cerebral Isquêmico – Reconhecimento e Tratamento",
        "text": (
            "AVC isquêmico resulta da oclusão arterial cerebral por trombose ou embolia. "
            "Hemiparesia contralateral súbita, afasia, hemianopsia homônima e ataxia são "
            "manifestações neurológicas focais. Escala NIHSS quantifica a gravidade. Alteplase IV "
            "até 4,5h do início dos sintomas; trombectomia mecânica até 24h em casos selecionados."
        ),
    },
    {
        "id": 10,
        "title": "Pneumonia Bacteriana Adquirida na Comunidade",
        "text": (
            "Pneumonia bacteriana adquirida na comunidade manifesta-se com febre alta, calafrios, "
            "tosse produtiva purulenta e dor pleurítica. Consolidação lobar ao RX tórax. "
            "Streptococcus pneumoniae é o agente mais comum. PSI/PORT e CURB-65 estratificam risco. "
            "Amoxicilina-clavulanato ou ceftriaxona são opções terapêuticas de primeira linha."
        ),
    },
    {
        "id": 11,
        "title": "Gastrite Erosiva e Úlcera Péptica – Etiopatogenia",
        "text": (
            "Gastrite erosiva e úlcera péptica decorrem de desequilíbrio entre fatores agressores "
            "(ácido clorídrico, H. pylori, AINEs) e protetores da mucosa gástrica. Epigastralgia "
            "em queimação, pior em jejum, com melhora pós-prandial caracteriza a úlcera duodenal. "
            "Endoscopia confirma o diagnóstico e permite biópsia. Tríplice terapia para H. pylori."
        ),
    },
    {
        "id": 12,
        "title": "Doença do Refluxo Gastroesofágico – Diagnóstico e Complicações",
        "text": (
            "DRGE é definida pelo refluxo patológico do conteúdo gástrico para o esôfago, causando "
            "pirose e regurgitação ácida como sintomas típicos. Disfagia, tosse crônica, rouquidão "
            "e erosões dentárias são manifestações atípicas. pH-metria de 24h e impedância são "
            "padrão-ouro. Esôfago de Barrett é complicação pré-maligna com vigilância endoscópica."
        ),
    },
    {
        "id": 13,
        "title": "Apendicite Aguda – Diagnóstico e Conduta Cirúrgica",
        "text": (
            "Apendicite aguda apresenta dor migratória da região periumbilical para a fossa ilíaca "
            "direita (ponto de McBurney), com anorexia, náuseas, febre baixa e leucocitose. Sinal "
            "de Blumberg (descompressão dolorosa) indica peritonismo. TC de abdômen tem alta "
            "sensibilidade. Apendicectomia laparoscópica é o tratamento padrão de referência."
        ),
    },
    {
        "id": 14,
        "title": "Pancreatite Aguda – Critérios de Gravidade e Manejo",
        "text": (
            "Pancreatite aguda caracteriza-se por dor abdominal epigástrica intensa, irradiada em "
            "faixa para o dorso, com elevação de amilase e lipase superiores a 3x o limite normal. "
            "Critérios de Atlanta 2012: leve, moderada e grave. PCR >150 mg/dL nas primeiras 48h "
            "indica gravidade. Hidratação venosa vigorosa é o pilar terapêutico fundamental."
        ),
    },
    {
        "id": 15,
        "title": "Lesão Renal Aguda – Classificação KDIGO e Indicações de Diálise",
        "text": (
            "Lesão renal aguda (LRA) é definida pela classificação KDIGO como: aumento de creatinina "
            "≥0,3 mg/dL em 48h, aumento ≥1,5x o basal em 7 dias, ou débito urinário <0,5 mL/kg/h "
            "por ≥6 horas. Causas: pré-renal, intrínseca (necrose tubular aguda) e pós-renal. "
            "Diálise de urgência em hipercalemia refratária, acidose grave ou edema pulmonar."
        ),
    },
    {
        "id": 16,
        "title": "Infecção do Trato Urinário – Diagnóstico e Antibioticoterapia",
        "text": (
            "Infecção do trato urinário (ITU) não complicada manifesta-se com disúria, polaciúria, "
            "urgência miccional e dor suprapúbica. Urina tipo I com leucocitúria >10.000/mL e "
            "bacteriúria confirmam o diagnóstico. Urocultura orienta antibioticoterapia definitiva. "
            "Nitrofurantoína ou fosfomicina são primeira linha para cistite simples em mulheres."
        ),
    },
    {
        "id": 17,
        "title": "Artrite Reumatoide – Critérios ACR/EULAR 2010 e Terapia Biológica",
        "text": (
            "Artrite reumatoide é doença autoimune sistêmica com sinovite crônica de pequenas "
            "articulações das mãos e pés, rigidez matinal superior a 1 hora, FR e anti-CCP "
            "positivos. Critérios ACR/EULAR 2010 pontuam envolvimento articular, sorologias, "
            "provas inflamatórias e duração ≥6 semanas. Metotrexato é âncora; biológicos para refratários."
        ),
    },
    {
        "id": 18,
        "title": "Fibromialgia – Critérios Diagnósticos e Manejo Multidisciplinar",
        "text": (
            "Fibromialgia caracteriza-se por dor musculoesquelética difusa crônica, fadiga, "
            "distúrbios do sono e disfunção cognitiva (fibro-fog). Critérios ACR 2010 utilizam "
            "Índice de Dor Generalizada (WPI) e Escala de Gravidade de Sintomas (SSS). Tratamento "
            "multimodal: duloxetina, pregabalina, amitriptilina, exercício aeróbico e TCC."
        ),
    },
    {
        "id": 19,
        "title": "Conjuntivite Alérgica – Diagnóstico e Terapêutica Ocular",
        "text": (
            "Conjuntivite alérgica cursa com hiperemia conjuntival bilateral, prurido ocular "
            "intenso, lacrimejamento e secreção serosa. Papilas na conjuntiva tarsal são achado "
            "típico. Frequentemente associada a rinite alérgica e atopia. Tratamento: colírios "
            "anti-histamínicos, estabilizadores de mastócito e, em graves, corticoides tópicos."
        ),
    },
    {
        "id": 20,
        "title": "Rinite Alérgica – Classificação ARIA e Imunoterapia",
        "text": (
            "Rinite alérgica é mediada por IgE, com espirros em salvas, prurido nasal, rinorreia "
            "hialina e obstrução nasal. Classificação ARIA: intermitente ou persistente, leve ou "
            "moderada/grave. IgE específica e teste de puntura confirmam sensibilização alérgica. "
            "Corticoides nasais são primeira linha; imunoterapia é a única modalidade doença-modificadora."
        ),
    },
    {
        "id": 21,
        "title": "Dermatite Atópica – Critérios de Hanifin & Rajka e Terapia Biológica",
        "text": (
            "Dermatite atópica é doença inflamatória cutânea crônica e recidivante com prurido "
            "intenso como critério diagnóstico maior. Lesões eczematosas com distribuição flexural "
            "em adultos e face/extensores em lactentes. Critérios de Hanifin & Rajka incluem "
            "história familiar atópica e xerose. Emolientes, corticoides tópicos e dupilumabe para graves."
        ),
    },
    {
        "id": 22,
        "title": "Colecistite Aguda – Critérios de Tokyo 2018 e Tratamento",
        "text": (
            "Colecistite aguda manifesta-se por dor no hipocôndrio direito irradiada para a "
            "escápula, febre, náuseas e sinal de Murphy positivo à palpação. Ultrassonografia "
            "revela espessamento parietal vesicular >4mm e líquido perivesicular. Critérios de "
            "Tokyo 2018 estratificam a gravidade. Colecistectomia laparoscópica precoce (<72h) é o padrão."
        ),
    },
    {
        "id": 23,
        "title": "Sepse e Choque Séptico – Definição Sepsis-3 e Bundles",
        "text": (
            "Sepse é definida pela Sepsis-3 como disfunção orgânica ameaçadora à vida por resposta "
            "desregulada do hospedeiro à infecção, com aumento do escore SOFA ≥2 pontos. Choque "
            "séptico: vasopressores para manter PAM ≥65 mmHg e lactato >2 mmol/L após ressuscitação "
            "volêmica adequada. Bundle de 1h: hemoculturas, antibióticos, lactato, 30 mL/kg cristaloide."
        ),
    },
]


# ────────────────────────────────────────────────────────────────────────────────
# Utilitário de impressão formatada
# ────────────────────────────────────────────────────────────────────────────────

def _print_block(text: str, indent: str = "  │  ", width: int = 62) -> None:
    for line in textwrap.wrap(text, width=width):
        print(f"{indent}{line}")


# ────────────────────────────────────────────────────────────────────────────────
# Passo 1 – Embeddings e Índice HNSW
# ────────────────────────────────────────────────────────────────────────────────

def get_embeddings(texts: list[str]) -> np.ndarray:
    """Gera embeddings densos via OpenAI text-embedding-3-small (1536 dims)."""
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return np.array([item.embedding for item in response.data], dtype=np.float32)


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-10)


def build_hnsw_index(embeddings: np.ndarray) -> faiss.IndexHNSWFlat:
    """
    Constrói índice HNSW com métrica de produto interno (equivalente à
    Similaridade de Cosseno para vetores normalizados em L2).

    Hiperparâmetros relevantes:
      M               – número de vizinhos por nó nas camadas do grafo. Maior M
                        implica mais arestas, mais RAM e melhor recall.
      efConstruction  – amplitude do beam-search durante a construção. Maior
                        valor = grafo de maior qualidade, build mais lento.
                        NÃO altera o tamanho final do índice em disco/RAM.
      efSearch        – amplitude do beam-search em tempo de consulta. Aumenta
                        recall a custo de maior latência por query.
    """
    dim = embeddings.shape[1]
    index = faiss.IndexHNSWFlat(dim, HNSW_M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
    index.hnsw.efSearch       = HNSW_EF_SEARCH
    index.add(_l2_normalize(embeddings))
    return index


# ────────────────────────────────────────────────────────────────────────────────
# Passo 2 – HyDE: Geração do Documento Hipotético
# ────────────────────────────────────────────────────────────────────────────────

def generate_hypothetical_document(user_query: str) -> str:
    """
    Transforma uma query coloquial em um documento técnico hipotético (HyDE).

    O documento falso gerado pelo LLM serve como âncora geométrica no espaço
    vetorial dos manuais, reduzindo o gap semântico entre a linguagem leiga do
    paciente e o jargão clínico dos textos indexados.
    """
    system_prompt = (
        "Você é um médico especialista redator de manuais clínicos. "
        "Dado um relato de sintoma do paciente, escreva um trecho de 3 a 5 frases "
        "de manual médico em português usando terminologia clínica precisa — "
        "como se fosse um verbete real de um livro de medicina. "
        "Não se dirija ao paciente; escreva no estilo enciclopédico e descritivo."
    )
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": f"Relato do paciente: {user_query}"},
        ],
        temperature=0.7,
        max_tokens=300,
    )
    return response.choices[0].message.content.strip()


# ────────────────────────────────────────────────────────────────────────────────
# Passo 3 – Recuperação via Bi-Encoder (HNSW)
# ────────────────────────────────────────────────────────────────────────────────

def retrieve_top_k(
    hypothetical_doc: str,
    index: faiss.IndexHNSWFlat,
    k: int = TOP_K_RETRIEVE,
) -> list[dict]:
    """
    Recupera os top-k fragmentos mais similares utilizando o vetor do documento
    hipotético como âncora de busca no índice HNSW (funil largo / bi-encoder).
    """
    query_vec = _l2_normalize(get_embeddings([hypothetical_doc]))
    scores, indices = index.search(query_vec, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx >= 0:                               # FAISS retorna -1 em índices inválidos
            doc = dict(MEDICAL_FRAGMENTS[idx])
            doc["bi_score"] = float(score)
            results.append(doc)
    return results


# ────────────────────────────────────────────────────────────────────────────────
# Passo 4 – Re-ranking com Cross-Encoder
# ────────────────────────────────────────────────────────────────────────────────

def rerank_with_cross_encoder(
    original_query: str,
    candidates: list[dict],
    top_n: int = TOP_K_RERANK,
) -> list[dict]:
    """
    Re-ranqueia os candidatos via cross-encoder de atenção profunda.

    O modelo recebe o par concatenado [CLS] query [SEP] documento [SEP] e calcula
    um score de relevância com atenção cruzada bidirecional — muito mais preciso
    que a similaridade de cosseno independente do bi-encoder, porém mais lento.
    """
    cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
    pairs  = [[original_query, doc["text"]] for doc in candidates]
    scores = cross_encoder.predict(pairs)

    for doc, score in zip(candidates, scores):
        doc["ce_score"] = float(score)

    reranked = sorted(candidates, key=lambda d: d["ce_score"], reverse=True)
    return reranked[:top_n]


# ────────────────────────────────────────────────────────────────────────────────
# Pipeline Principal
# ────────────────────────────────────────────────────────────────────────────────

def run_rag_pipeline(user_query: str) -> list[dict]:
    """Executa o pipeline RAG completo em 4 passos e retorna os top-3 documentos."""
    W = 70

    print(f"\n{'═' * W}")
    print("  PIPELINE RAG AVANÇADO — Assistente de Manuais Médicos")
    print(f"{'═' * W}")
    print(f"\n  Query coloquial do usuário: \"{user_query}\"")

    # ── Passo 1: Construção do Índice HNSW ──────────────────────────────────────
    print(f"\n{'─' * W}")
    print("  PASSO 1 » Construção e Indexação do Grafo HNSW")
    print(f"{'─' * W}")

    texts = [frag["text"] for frag in MEDICAL_FRAGMENTS]
    print(f"  Gerando embeddings para {len(texts)} fragmentos (OpenAI {EMBEDDING_MODEL})...")
    embeddings = get_embeddings(texts)
    index      = build_hnsw_index(embeddings)

    print(f"  Índice HNSW construído com sucesso.")
    print(f"  ├─ Documentos indexados  : {index.ntotal}")
    print(f"  ├─ Dimensão dos vetores  : {EMBEDDING_DIM}")
    print(f"  ├─ Parâmetro M           : {HNSW_M}")
    print(f"  ├─ ef_construction       : {HNSW_EF_CONSTRUCTION}")
    print(f"  ├─ ef_search             : {HNSW_EF_SEARCH}")
    print(f"  └─ Métrica               : Inner Product (≡ Cosseno com L2-norm)")

    # ── Passo 2: HyDE – Documento Hipotético ────────────────────────────────────
    print(f"\n{'─' * W}")
    print("  PASSO 2 » Query Transformation via HyDE")
    print(f"{'─' * W}")
    print("  Solicitando ao LLM a geração de um documento hipotético técnico...\n")

    hypothetical_doc = generate_hypothetical_document(user_query)

    print("  Documento Hipotético (âncora semântica gerada):")
    print(f"  ┌{'─' * (W - 4)}")
    _print_block(hypothetical_doc, indent="  │  ", width=W - 8)
    print(f"  └{'─' * (W - 4)}")

    # ── Passo 3: Recuperação via Bi-Encoder ─────────────────────────────────────
    print(f"\n{'─' * W}")
    print(f"  PASSO 3 » Busca Rápida via Bi-Encoder no Índice HNSW (Top-{TOP_K_RETRIEVE})")
    print(f"{'─' * W}")

    candidates = retrieve_top_k(hypothetical_doc, index, k=TOP_K_RETRIEVE)

    print(f"  {'#':<4} {'Score BI':<12} Título")
    print(f"  {'─' * 3:<4} {'─' * 11:<12} {'─' * 48}")
    for rank, doc in enumerate(candidates, 1):
        print(f"  {rank:<4} {doc['bi_score']:<12.4f} {doc['title']}")

    # ── Passo 4: Re-ranking com Cross-Encoder ───────────────────────────────────
    print(f"\n{'─' * W}")
    print(f"  PASSO 4 » Filtro Fino com Cross-Encoder — Top-{TOP_K_RERANK} Finais")
    print(f"{'─' * W}")
    print(f"  Carregando modelo '{CROSS_ENCODER_MODEL}'...")

    top_reranked = rerank_with_cross_encoder(user_query, candidates, top_n=TOP_K_RERANK)

    print(f"\n  Documentos selecionados para injeção no contexto do LLM gerador:\n")
    for rank, doc in enumerate(top_reranked, 1):
        print(f"  ┌─ #{rank} | CE Score: {doc['ce_score']:.4f} | BI Score: {doc['bi_score']:.4f}")
        print(f"  │   {doc['title']}")
        print(f"  │")
        _print_block(doc["text"], indent="  │  ", width=W - 8)
        print(f"  └{'─' * (W - 4)}\n")

    print(f"{'═' * W}")
    print("  Pipeline concluído. Documentos prontos para o contexto do LLM gerador.")
    print(f"{'═' * W}\n")

    return top_reranked


# ────────────────────────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("Erro: OPENAI_API_KEY não definida. Crie um arquivo .env com a chave.")
        sys.exit(1)

    # Query de demonstração: linguagem coloquial → jargão médico esperado
    demo_query = "dor de cabeça latejante e luz incomodando"
    run_rag_pipeline(demo_query)

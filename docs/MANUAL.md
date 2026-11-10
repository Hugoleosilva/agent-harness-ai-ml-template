# Manual da Fábrica Agêntica Solo: Fábrica de Modelos de IA

Repositório: https://github.com/Hugoleosilva/agent-harness-ai-ml-template

---

## 1. Intenção do projeto e benefícios

### O que é este projeto?

Este projeto redefine a "Fábrica de Software" tradicional. Em vez de uma grande equipe humana digitando código, criamos um **Harness (esteira de governança)** modular. Ele permite que um desenvolvedor solo atue como **Diretor de Engenharia**, usando IA (Claude Code, Cursor, Copilot) como força de trabalho rápida, enquanto o controle de qualidade e as decisões finais continuam com o humano.

### Por que ele foi criado?

Ao programar com IA, os problemas mais comuns são:

- **Viés de confirmação:** a IA tende a concordar com o que você pediu e a declarar sucesso sem ter verificado.
- **Perda de contexto:** regras combinadas no início são esquecidas ao longo da conversa.
- **Invenção de funções** (alucinação) e **overengineering** (complexidade além do necessário).

O harness reduz esses riscos com duas camadas:

1. **Instruções:** regras escritas que dizem à IA como se comportar (`AGENTS_CORE.md`).
2. **Verificação independente:** scripts, *hooks* e CI que checam o resultado sem depender da palavra da IA.

A governança (como a IA se comporta) fica separada da tecnologia (linguagem e framework). Trocar de modelo de IA ou de biblioteca exige, em geral, apenas atualizar o `.agentconfig.json`.

### Principais benefícios

- **Velocidade:** reduz o tempo de tarefas repetitivas (limpeza de dados, esqueleto de treino, testes), do tamanho de dias para horas em muitos casos. O ganho real depende da complexidade do problema.
- **Baixo custo de infraestrutura:** o harness é só arquivos de texto e scripts em Python. O custo variável é o uso de tokens da IA, que você controla escolhendo o modelo para cada tarefa.
- **Evolução sem reescrita:** como as regras ficam separadas da tecnologia, uma nova linguagem ou modelo de IA normalmente pede só ajustes de configuração.

> **Limites honestos:** o harness reduz erros, mas não os elimina. A IA ainda pode errar, e por isso existem as checagens automáticas e a aprovação humana.

---

## 2. Arquitetura geral e boas práticas

### O Protocolo Multi-Agente de ML (Criar, Validar, Reparar)

Antes de entregar qualquer script, a IA percorre três papéis e **registra a evidência de cada um** no log da execução. Um papel que não foi registrado é considerado não executado.

1. **O Criador (The Builder):** desenha o pipeline mínimo (transformações de dados, arquitetura, loop de treino) e define o baseline e a métrica-alvo.
2. **O Validador (The Validator):** atua de forma adversária e anexa evidência (saída de comandos, não opinião): shapes e dtypes, data leakage, seeds fixadas, splits persistidos, NaN/Inf, balanceamento de classes, loops não vetorizados e uso de memória.
3. **O Reparador (The Fixer):** aplica a menor correção possível para cada achado. O ciclo Validador → Reparador roda no máximo **3 vezes**; se ainda falhar, a IA para e pede ajuda ao humano.

O humano atua como **Human-in-the-Loop**: avalia as métricas (F1, PR-AUC, RMSE, perda), os logs, e é o único que aprova a exportação dos pesos oficiais.

### Como garantimos que a IA não só "diz" que validou

Dizer que validou sem ter validado é uma falha conhecida de agentes de IA (relato infiel). O harness não depende da palavra do agente:

| Camada | Onde está | O que faz |
|---|---|---|
| Hook de saída (`Stop`) | `.agent-harness/hooks/stop_gate.py` | Roda `ruff`, `pytest` e a checagem de leakage por conta própria. Se algo falhar, **bloqueia** a IA de encerrar a tarefa. |
| Log de auditoria | `.agent-harness/hooks/audit_log.py` → `logs/audit.jsonl` | Registra cada comando e arquivo escrito, para comparar com o que a IA afirmou ter feito. |
| CI | `.github/workflows/ci.yml` | Roda lint e testes a cada push, num ambiente independente da sua máquina. |
| Aprovação humana | Regra "Human Gate" | Exportar pesos ou rodar jobs caros exige o seu OK. |

Opcionalmente, rastreadores de experimento (MLflow, Weights & Biases) comprovam que um treino rodou, com métricas e seed registradas.

### Boas práticas aplicadas

- **Integridade de dados:** o split vem antes de qualquer ajuste; scalers, encoders e imputers são ajustados só no treino; splits por grupo ou por tempo quando as linhas não são independentes; o conjunto de teste é avaliado **uma única vez**, ao final; os índices de split são salvos em disco.
- **Reprodutibilidade:** uma função central `seed_everything` fixa as sementes (`random`, `numpy`, `torch`, cudnn, workers do DataLoader); versões das dependências fixadas; seed, commit e versão dos dados registrados em cada execução.
- **Eficiência de computação:** operações vetorizadas, escolha deliberada de tipos de dado, gerenciamento explícito de dispositivo (CPU/GPU), precisão mista (AMP), gradient accumulation e teste de overfit com um batch antes de treinos longos.
- **Simplicidade (KISS):** sempre construir um baseline trivial primeiro. Uma arquitetura complexa só entra se superar o baseline por margem relevante.
- **Segurança:** o `.gitignore` bloqueia chaves, `.env`, datasets (`.csv`, `.pkl`, `.parquet`...) e pesos (`.pt`, `.safetensors`, `.onnx`...). Segredos vêm só de variáveis de ambiente.

---

## 3. Repositório: Fábrica de Modelos de IA

**Foco:** sistemas probabilísticos (Ciência de Dados, Machine Learning, Deep Learning, engenharia de prompt e análise estatística).

### Estrutura de pastas

```text
.
├── .agent-harness/
│   ├── AGENTS_CORE.md          # O "cérebro": regras de integridade, reprodutibilidade e VRAM (em inglês)
│   ├── SETUP_PROMPT.md         # Prompt (em inglês) que ativa o Setup Wizard de ML
│   ├── hooks/                  # stop_gate.py (bloqueia saída com falhas) e audit_log.py (auditoria)
│   ├── templates/              # seed_everything.py e check_leakage.py
│   ├── mcp/mcp-config.json     # Conexões MCP (opcional)
│   ├── memory/ctx_memory.md    # Hiperparâmetros, linhagem de dados e melhor checkpoint
│   ├── changelog/updates.md    # Histórico de versões e métricas
│   └── logs/
│       ├── mapping/            # Rastreamento de shapes de tensores e matrizes
│       └── archive/            # Histórico de execuções, curvas de perda e falhas
├── .claude/settings.json       # Liga os hooks no Claude Code
├── .github/workflows/ci.yml    # CI: lint e testes
├── src/                        # Esqueleto: data, features, model, train, evaluate
├── tests/                      # Testes do harness e do esqueleto
├── CLAUDE.md / AGENTS.md       # Fazem as ferramentas de IA carregarem o AGENTS_CORE.md
├── .agentconfig.json           # Metadados do projeto, métricas e comandos de validação
├── .env.example                # Modelo de chaves (S3, Hugging Face, W&B...). Nunca coloque valores reais
├── .gitignore                  # Bloqueio de chaves, datasets e pesos
├── pyproject.toml              # Dependências e configuração de lint/testes
└── LICENSE                     # MIT
```

### Função dos arquivos principais

- **AGENTS_CORE.md:** instruções em XML focadas em ML. Orienta o trio de agentes a garantir integridade dos dados (sem vazamento do teste para o treino), reprodutibilidade (seeds globais) e uso responsável de VRAM.
- **.agentconfig.json:** define tarefa, métrica principal, baseline, política de split, hardware e os **comandos de validação** (`ruff`, `pytest`, `check_leakage`) que precisam passar antes de um treino ser dado como concluído.
- **CLAUDE.md / AGENTS.md:** as ferramentas de IA não leem o `AGENTS_CORE.md` sozinhas; estes arquivos apontam para ele.

---

## 4. Instruções de uso (passo a passo)

### Passo 1: Inicializando um novo projeto

1. Copie a estrutura do harness para a raiz do seu repositório de Ciência de Dados e abra no VS Code.
2. Instale as dependências de desenvolvimento: `pip install -r requirements-dev.txt`.
3. Inicie o chat do seu agente de IA.
4. Abra `.agent-harness/SETUP_PROMPT.md`, copie o prompt (em inglês) e envie no chat.

### Passo 2: O assistente interativo (a entrevista)

A IA faz **5 perguntas, uma por vez**: o problema e o tipo de tarefa; os dados disponíveis; como medir sucesso (métrica e baseline); hardware e orçamento; stack e restrições.

> **Nota para iniciantes:** se você não souber qual algoritmo usar, responda: *"Não sei, o que você me recomenda para esse tipo de base de dados?"* A IA atuará como consultora sênior, explicando em português claro as vantagens e desvantagens de cada opção.

Ao fim da entrevista a IA resume o plano e **espera o seu "OK"**. Só então preenche o `.agentconfig.json`, inicia a memória do projeto e propõe o primeiro passo: uma auditoria dos dados e um baseline trivial.

### Passo 3: Solicitando funcionalidades e correções

Daqui em diante, dê ordens em linguagem natural, em português ou inglês. A IA aplica o filtro de simplicidade (KISS), percorre o ciclo Criador → Validador → Reparador, registra o histórico no changelog (com a data) e entrega o resultado **com a evidência das checagens**, para o seu aval final.

---

## 5. Estratégia de idioma e economia de tokens

### O problema

Os grandes modelos são treinados majoritariamente em inglês. Em geral, o mesmo conteúdo em português consome **mais tokens** que em inglês (a diferença varia conforme o modelo e o texto, e costuma ficar na faixa de dezenas de por cento).

### A solução: "idioma híbrido"

Os arquivos-base que a IA lê com frequência (`AGENTS_CORE.md`, `SETUP_PROMPT.md`) ficam **em inglês**. Isso tende a reduzir o custo de tokens de entrada. O ganho exato depende da ferramenta e do uso de cache de prompt, então **meça no seu caso** (por exemplo, com o comando `/cost` do Claude Code) antes de assumir um percentual.

Você continua livre para escrever comandos e regras de negócio em português. A IA é orientada a responder de forma direta, com foco em código.

### Os três níveis de interação

**Nível 1: linguagem natural pura (português).** Ideal para tarefas simples, correções rápidas ou consultoria.

> "Treine um baseline de regressão logística para esta base e me diga se vale a pena usar uma rede neural."

**Nível 2: tags XML + português.** Ideal para passar regras específicas e reduzir a chance de a IA ignorar restrições.

```xml
<instrucao>
Crie a função que separa a base em treino, validação e teste.
</instrucao>

<regras_de_negocio>
- Os clientes aparecem em várias linhas: nunca separe um mesmo cliente em splits diferentes.
- Fixe a seed global e salve os índices em disco.
- Aplique o filtro de simplicidade (KISS): não crie funções genéricas extras.
</regras_de_negocio>
```

**Nível 3: o combo (tags XML + instruções em inglês).** É o nível mais preciso e tende a ser o mais econômico em tokens, mesmo que o código final use termos em português. Tags XML deixam os limites da tarefa claros, e instruções em inglês costumam ser seguidas com mais consistência. Isso melhora a obediência, mas **não garante** respostas perfeitas, e por isso as checagens automáticas continuam valendo.

Benefícios:

1. **Precisão:** a IA entende os limites da tarefa e mistura menos código antigo com novas ordens.
2. **Economia:** texto estruturado em inglês tende a gerar prompts menores e respostas mais diretas.
3. **Independência de idioma:** regras em inglês se aplicam a código com nomes em português ou a qualquer framework.

```xml
<performance_optimization>
Optimize the current dataframe loading sequence to reduce RAM overhead.
</performance_optimization>

<strict_constraints>
- Use strictly vectorized operations instead of iterative loops.
- Explicitly force garbage collection (gc.collect) after matrix splitting.
</strict_constraints>

<current_code>
[Cole aqui o seu script de tratamento de dados atual]
</current_code>
```

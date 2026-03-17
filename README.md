# Проект обучения LLM по книге «Книга Ангелов – Мастер Ключей»

Модульный проект для обучения небольшой decoder-only языковой модели с нуля по датасету на основе книги: обучение токенизатора (SentencePiece Unigram), обучение LM, классификатор ответности (answerability), retrieval по BM25 и инференс-пайплайн с отказом при недостатке данных.

## Требования

- Python 3.10+
- CUDA (для обучения и инференса на GPU)
- Установка зависимостей: `pip install -r requirements.txt`

## Структура проекта

```
project_root/
├── data/                    # Датасет (положить подготовленные файлы)
│   ├── 01_book_clean.txt
│   ├── 02_tokenizer_corpus.txt
│   ├── 03_angels_structured.jsonl
│   ├── 04_retrieval_chunks.jsonl
│   ├── 05_sft_chat_train.jsonl
│   ├── 06_sft_chat_val.jsonl
│   ├── 07_answerability_train.jsonl
│   ├── 08_answerability_val.jsonl
│   ├── 09_system_prompt.txt
│   ├── 10_recommended_training_config.yaml
│   └── stats.json
├── checkpoints/             # Промежуточные чекпоинты
│   ├── tokenizer/
│   ├── lm/
│   └── answerability/
├── model/                   # Финальные артефакты
│   ├── tokenizer/
│   ├── lm/
│   ├── answerability/
│   ├── retrieval/
│   └── inference/
├── configs/
│   ├── train.yaml
│   ├── tokenizer.yaml
│   ├── answerability.yaml
│   ├── retrieval.yaml
│   └── inference.yaml
├── modules/                 # Модули проекта
│   ├── config.py, constants.py, paths.py, logger.py, seed.py, exceptions.py
│   ├── io_utils.py, text_utils.py
│   ├── tokenizer_utils.py, tokenizer_trainer.py
│   ├── model_config.py, modeling_decoder_lm.py, generation.py
│   ├── dataset_sft.py, dataset_answerability.py, collators.py
│   ├── answerability_model.py, answerability_trainer.py
│   ├── retrieval_index.py, retriever.py
│   ├── prompt_builder.py, postcheck.py, inference_pipeline.py
│   ├── checkpointing.py, distributed.py, trainer_lm.py
│   ├── metrics.py, evaluator.py
│   └── ...
├── scripts/
│   ├── train_tokenizer.py
│   ├── train_lm.py
│   ├── train_answerability.py
│   ├── build_retrieval.py
│   ├── evaluate.py
│   └── infer.py
├── tests/
├── train.py                 # Единая точка входа обучения
├── infer.py                 # Инференс
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Подготовка данных

Положите подготовленные файлы датасета в каталог `data/` как указано выше. Форматы:

- `01_book_clean.txt`, `02_tokenizer_corpus.txt` — текст в UTF-8.
- `04_retrieval_chunks.jsonl` — строки JSON с полями `chunk_id` (или `id`) и `text` (или `content`).
- `05_sft_chat_train.jsonl`, `06_sft_chat_val.jsonl` — либо `messages` (список `role`/`content`), либо `prompt`/`completion`.
- `07_answerability_train.jsonl`, `08_answerability_val.jsonl` — `question` и `label` (`answerable` / `not_answerable`).
- `09_system_prompt.txt` — системный промпт для генерации.

## Запуск

### 1. Обучение токенизатора

```bash
python train.py tokenizer
# или
python scripts/train_tokenizer.py --config tokenizer
```

Токенизатор сохраняется в `model/tokenizer/`.

### 2. Обучение LM (2 x GPU через DDP)

```bash
torchrun --nproc-per-node=2 train.py lm
# или
torchrun --nproc-per-node=2 scripts/train_lm.py --config train
```

Чекпоинты — в `checkpoints/lm/`, финальная модель — в `model/lm/`. Возобновление с последнего чекпоинта по умолчанию включено.

### 3. Обучение классификатора ответности

```bash
python train.py answerability
# или
python scripts/train_answerability.py --config answerability
```

Артефакты — в `model/answerability/`.

### 4. Построение индекса BM25

```bash
python scripts/build_retrieval.py
```

Индекс сохраняется в `model/retrieval/`.

### 5. Инференс

Один вопрос из аргумента:

```bash
python infer.py "Ваш вопрос по книге"
```

Интерактивный режим:

```bash
python infer.py --interactive
```

Используются конфиг `configs/inference.yaml`, модель из `model/lm/`, классификатор из `model/answerability/`, retriever из `model/retrieval/`. При недостатке данных возвращается отказ: «В книге нет данных для ответа на этот вопрос.»

### 6. Оценка

```bash
python scripts/evaluate.py
# вывод метрик в stdout; опционально:
python scripts/evaluate.py --output metrics.json
```

## Архитектура

- **Answerability:** классификатор реализован как голова поверх скрытых состояний основной LM (один скрытый слой + линейный слой в 2 класса). Общая представление с генератором, меньше параметров, один раз загружаемая LM.
- **Отказ:** если классификатор считает вопрос неответным по контексту или post-check не подтверждает ответ контекстом, возвращается строка отказа. Поддерживается частичный ответ с явным указанием, что по остальному данных нет.

## Чекпоинты и модель

- Все промежуточные чекпоинты обучения сохраняются в `checkpoints/` (по подзадачам: tokenizer, lm, answerability).
- Готовая модель и артефакты (токенизатор, LM, answerability, retrieval) — в `model/`.

## Тесты

```bash
pytest tests/ -v
```

Тесты используют моки и небольшие фикстуры и не требуют полного датасета.

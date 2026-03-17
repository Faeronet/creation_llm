# Пакет данных для обучения доменной LLM по книге ангелов

Этот пакет собран из загруженного PDF книги, начиная с первого полноценного ангела (Vehuiah) и без вступительного шумового блока.

## Что внутри

1. `01_book_clean.txt`  
   Очищенный корпус книги, начиная с первого ангела.

2. `02_tokenizer_corpus.txt`  
   Корпус для обучения токенизатора: сырой текст книги + все сгенерированные вопросы и ответы.

3. `03_angels_structured.jsonl`  
   Структурированные записи по ангелам. Для каждой записи есть:
   - `angel_id`
   - `name_lat`
   - `name_ru`
   - `date_range`
   - `abode`
   - `specificity`
   - `zodiac`
   - `qualities`, `qualities_items`
   - `distortions`, `distortions_items`
   - `situations`, `situations_items`
   - `physical_manifestation`
   - `emotional_manifestation`
   - `intellectual_manifestation`
   - `dominion`
   - `divine_name`
   - `angelic_order`
   - `divine_embodiment`
   - `evil_genius`
   - `gifts`
   - `raw_text`

4. `04_retrieval_chunks.jsonl`  
   Семантические чанки для BM25 / retrieval-слоя.

5. `05_sft_chat_train.jsonl`  
   Основной chat-SFT train set.

6. `06_sft_chat_val.jsonl`  
   Валидационный chat-SFT set с альтернативными формулировками.

7. `07_answerability_train.jsonl`  
   Train set для бинарной задачи `answerable / not_answerable`.

8. `08_answerability_val.jsonl`  
   Validation set для классификатора отказа.

9. `09_system_prompt.txt`  
   Строгий системный промпт с запретом на домысливание.

10. `10_recommended_training_config.yaml`  
    Рекомендуемый базовый конфиг под 2 x RTX 3090.

## Типы вопросов в SFT-наборе

В наборе есть не один шаблон, а несколько семейств формулировок:
- специализация + обитель
- знак зодиака + физическое проявление
- качества энергии
- искажения энергии
- ситуации и проблемы
- интеллектуальное проявление
- эмоциональное проявление
- дары / способности
- главная роль ангела
- обратные вопросы по дате
- обратные вопросы по времени интеллектуального проявления
- господство
- Божественное имя и ангельский чин
- сведения о злом гении
- частичный ответ + отказ на недостающую часть
- полный отказ, если атрибут в книге не указан

## Размеры наборов

- train SFT: 2890
  - answerable: 2384
  - not_answerable / partial refusal: 506

- val SFT: 1099
  - answerable: 955
  - not_answerable / partial refusal: 144

- ангелов в структурированном наборе: 72
- retrieval chunks: 691

## Важные замечания

1. Это **не просто датасет “вопрос -> ответ”**, а полный минимальный пакет для:
   - токенизатора
   - retrieval
   - SFT
   - классификатора отказа

2. PDF извлечён автоматически, поэтому отдельные OCR/переносные артефакты могли сохраниться в части полей.
   Для прод-качества рекомендована ручная ревизия `03_angels_structured.jsonl`.

3. В train/val уже заложены:
   - прямые вопросы
   - перефразированные вопросы
   - обратные вопросы
   - вопросы с частичным ответом
   - вопросы, на которые модель должна отвечать отказом

4. Для вашей задачи лучше использовать эти файлы вместе:
   - retrieval по `04_retrieval_chunks.jsonl`
   - answerability classifier по `07/08`
   - SFT по `05/06`
   - строгий системный промпт из `09_system_prompt.txt`

## Рекомендуемый порядок

1. Обучить SentencePiece tokenizer на `02_tokenizer_corpus.txt`
2. Обучить scratch decoder-only LM на `05_sft_chat_train.jsonl`
3. Дообучить / обучить отдельно answerability head на `07_answerability_train.jsonl`
4. На инференсе:
   - сначала retrieval
   - затем answerability check
   - затем генерация
   - затем жёсткий отказ при отсутствии подтверждения


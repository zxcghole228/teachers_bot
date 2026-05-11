import json
import chromadb
from sentence_transformers import SentenceTransformer


DATASET_FILE = "ege_economics_dataset.json"
MODEL_NAME = "intfloat/multilingual-e5-large"
COLLECTION_NAME = "ege_economics"


def load_data(filepath):
    """Загружает данные из JSON и готовит их для векторизации."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ ОШИБКА: Файл '{filepath}' не найден в папке проекта.")
        return [], [], []

    documents = []
    metadatas = []
    ids = []
    for category in data['dataset']:
        cat_id = category['category_id']
        cat_desc = category.get('description', '')
        if 'subtypes' in category:
            for subtype in category['subtypes']:
                nuance = subtype['nuance']
                for example in subtype['examples']:
                    doc_text = f"passage: Категория: {cat_desc}. Задача: {example['text']}"

                    documents.append(doc_text)
                    metadatas.append({
                        "category_id": cat_id,
                        "nuance": nuance,
                        "original_text": example['text']
                    })
                    ids.append(example['id'])
        elif 'examples' in category:
            for example in category['examples']:
                doc_text = f"passage: Категория: {cat_desc}. Задача: {example['text']}"
                documents.append(doc_text)
                metadatas.append({
                    "category_id": cat_id,
                    "nuance": example.get('nuance', 'unknown'),
                    "original_text": example['text']
                })
                ids.append(example['id'])

    return documents, metadatas, ids


def main():
    print("⏳ Загрузка модели (может занять время в первый раз)...")
    model = SentenceTransformer(MODEL_NAME)

    print(f"📂 Чтение датасета из {DATASET_FILE}...")
    documents, metadatas, ids = load_data(DATASET_FILE)

    if not documents:
        print("❌ Нет данных для загрузки. Проверьте JSON-файл.")
        return

    print(f"   Найдено {len(documents)} задач.")
    print("🧠 Векторизация данных...")
    embeddings = model.encode(documents)

    print("💾 Сохранение в векторную базу ChromaDB...")
    client = chromadb.Client()
    try:
        client.delete_collection(COLLECTION_NAME)
    except:
        pass

    collection = client.create_collection(name=COLLECTION_NAME)

    collection.add(
        embeddings=embeddings.tolist(),
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    print("✅ База готова! Можно искать.\n")
    print("=========================================")
    print("ВВЕДИТЕ ЗАПРОС (или 'exit' для выхода)")
    print("Примеры: 'завод рабочие', 'Светлана кредит', 'вклад пополняемый'")
    print("=========================================")

    while True:
        query = input("\n🔍 Ваш запрос: ")
        if query.lower() in ['exit', 'quit', 'выход']:
            break

        if not query.strip():
            continue
        emb = model.encode(f"query: {query}")
        query_vec = [emb.tolist()]

        results = collection.query(
            query_embeddings=query_vec,
            n_results=3  # Сколько похожих задач искать
        )

        print(f"\n--- Результаты для: '{query}' ---")
        if not results['ids'] or not results['ids'][0]:
            print("Ничего не найдено.")
            continue

        for i in range(len(results['ids'][0])):
            score = results['distances'][0][i]
            meta = results['metadatas'][0][i]
            doc_id = results['ids'][0][i]
            preview_text = meta['original_text'].replace('\n', ' ')
            if len(preview_text) > 150:
                preview_text = preview_text[:147] + "..."

            print(f"\n[{i + 1}] ID: {doc_id} (Дистанция: {score:.4f})")
            print(f"    Тип: {meta['category_id']} -> {meta['nuance']}")
            print(f"    Текст: {preview_text}")


if __name__ == "__main__":
    main()

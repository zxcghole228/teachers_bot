import json
import chromadb
from sentence_transformers import SentenceTransformer
from rag_system import load_data, DATASET_FILE


TEST_FILE = "test_queries.json"
COLLECTION_NAME = "ege_economics"
MODEL_NAME = "intfloat/multilingual-e5-large"


def main():
    print("⏳ Загрузка модели...")
    model = SentenceTransformer(MODEL_NAME)

    print("♻️  Пересоздание базы...")
    documents, metadatas, ids = load_data(DATASET_FILE)
    embeddings = model.encode(documents)

    client = chromadb.Client()
    try:
        client.delete_collection(COLLECTION_NAME)
    except:
        pass
    collection = client.create_collection(name=COLLECTION_NAME)
    collection.add(embeddings=embeddings.tolist(), documents=documents, metadatas=metadatas, ids=ids)

    print("📂 Чтение тестов...")
    try:
        with open(TEST_FILE, 'r', encoding='utf-8') as f:
            test_cases = json.load(f)
    except FileNotFoundError:
        return

    print(f"🚀 Запуск теста (Category Match)...\n")

    hits = 0
    total = len(test_cases)
    id_to_cat = {}
    for i, doc_id in enumerate(ids):
        id_to_cat[doc_id] = metadatas[i]['category_id']

    print(f"{'QUERY ID':<20} | {'EXPECTED CAT':<20} | {'FOUND CAT':<20} | {'STATUS':<6}")
    print("-" * 75)

    for case in test_cases:
        query_text = case['text']
        query_id = case['query_id']

        expected_cat = id_to_cat.get(query_id, "unknown")

        query_vec = [model.encode(f"query: {query_text}").tolist()]

        results = collection.query(query_embeddings=query_vec, n_results=5)

        found_ids = results['ids'][0]
        found_metas = results['metadatas'][0]

        best_match_cat = None
        for i, fid in enumerate(found_ids):
            if fid != query_id:
                best_match_cat = found_metas[i]['category_id']
                break

        if best_match_cat == expected_cat:
            status = "✅"
            hits += 1
        else:
            status = "❌"

        p_qid = (query_id[:17] + '..') if len(query_id) > 17 else query_id
        p_exp = (expected_cat[:17] + '..') if len(expected_cat) > 17 else expected_cat
        p_fcat = (str(best_match_cat)[:17] + '..') if best_match_cat else "None"

        print(f"{p_qid:<20} | {p_exp:<20} | {p_fcat:<20} | {status:<6}")

    accuracy = (hits / total) * 100

    print("\n" + "=" * 40)
    print(f"CATEGORY ACCURACY: {accuracy:.2f}%")
    print("=" * 40)


if __name__ == "__main__":
    main()

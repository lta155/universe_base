import faiss
import numpy as np
import json
import os
from typing import List, Dict, Optional
import pickle


class FaissVectorStore:
    def __init__(
        self, 
        dimension: int = 1024, 
        index: Optional[faiss.Index] = None,
        kb: Optional[List[Dict]] = None
    ):
        """
        初始化向量存储
        
        Args:
            dimension: 向量维度
            index: 已有的 FAISS 索引（可选）
            kb: 已有的知识库文档列表（可选）
        """
        if index is not None:
            self.index = index
        else:
            self.index = faiss.IndexFlatL2(dimension)
        
        self.kb = kb if kb is not None else []
        self.dimension = dimension
    
    @classmethod
    def from_persist_path(
        cls,
        index_path: str,
        kb_path: str,
        config_path: Optional[str] = None
    ):
        """
        从持久化路径加载向量存储
        
        Args:
            index_path: FAISS 索引文件路径
            kb_path: 知识库 JSON 文件路径
            config_path: 配置信息文件路径（可选）
        """
        if not os.path.exists(index_path):
            raise ValueError(f"No existing index found at {index_path}.")
        index = faiss.read_index(index_path)
        
        if not os.path.exists(kb_path):
            raise ValueError(f"No existing KB found at {kb_path}.")
        with open(kb_path, "r", encoding="utf-8") as f:
            kb = json.load(f)
        
        # 加载配置信息（如果存在）
        config = None
        if config_path and os.path.exists(config_path):
            with open(config_path, "rb") as f:
                config = pickle.load(f)
        
        if config:
            return cls(
                dimension=config.get("dimension", 1024),
                index=index,
                kb=kb
            )
        else:
            # 如果配置不存在，创建一个默认的实例
            instance = cls(index=index, kb=kb)
            # 尝试从索引获取维度
            if hasattr(index, 'd'):
                instance.dimension = index.d
            return instance
    
    def add(self, embedding: List[List[float]], doc: List[dict]):
        """
        添加向量和文档到存储
        
        Args:
            embedding: 向量数组，形状为 (n, dimension)
            doc: 对应的文档列表
        """
        embedding = np.array(embedding)
        if embedding.shape[0] != len(doc):
            raise ValueError(f"Number of embeddings ({embedding.shape[0]}) does not match number of documents ({len(doc)})")
        
        # 检查向量维度是否匹配
        if embedding.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension ({embedding.shape[1]}) does not match index dimension ({self.dimension})")
        self.index.add(embedding)
        self.kb.extend(doc)

    def query(
        self,
        query_emb: List[float],
        top_k: int = 5,
    ) -> List[List[Dict]]:
        """
        查询相似的向量，支持批量查询
        
        Args:
            query_emb: 查询向量，形状为 (1, dim) 或 (n, dim)
            top_k: 每个查询返回的最相似结果数量
            score_threshold: 相似度分数阈值（L2距离越小越相似）
        
        Returns:
            每个查询的相似结果列表，每个结果包含文档和距离
        """
        query_emb = np.array(query_emb)
        if len(query_emb.shape) == 1:
            query_emb = query_emb.reshape(1, -1)
        
        # 检查查询向量维度
        if query_emb.shape[1] != self.dimension:
            raise ValueError(f"Query embedding dimension ({query_emb.shape[1]}) does not match index dimension ({self.dimension})")
        
        # 执行搜索
        distances, indices = self.index.search(query_emb, top_k)
        
        query_results = []

        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx == -1:  # FAISS 返回 -1 表示没有足够的结果
                break
            # 获取对应的文档
            if idx < len(self.kb):
                result = self.kb[idx].copy()  # 避免修改原始数据
                result["distance"] = float(distance)
                result["score"] = 1.0 / (1.0 + distance) if distance > 0 else 1.0  # 将距离转换为相似度分数
                query_results.append(result)
        
        return query_results
    
    
    def persist(
        self,
        index_path: str,
        kb_path: str,
        config_path: Optional[str] = None
    ) -> None:
        """
        持久化保存向量存储
        
        Args:
            index_path: FAISS 索引保存路径
            kb_path: 知识库保存路径
            config_path: 配置信息保存路径（可选）
        """
        # 保存索引
        dirpath = os.path.dirname(index_path)
        if dirpath and not os.path.exists(dirpath):
            os.makedirs(dirpath)
        faiss.write_index(self.index, index_path)
        
        # 保存知识库
        dirpath = os.path.dirname(kb_path)
        if dirpath and not os.path.exists(dirpath):
            os.makedirs(dirpath)
        with open(kb_path, "w", encoding="utf-8") as f:
            json.dump(self.kb, f, indent=4, ensure_ascii=False)
        
        # 保存配置信息
        if config_path:
            config = {
                "dimension": self.dimension,
            }
            dirpath = os.path.dirname(config_path)
            if dirpath and not os.path.exists(dirpath):
                os.makedirs(dirpath)
            with open(config_path, "wb") as f:
                pickle.dump(config, f)


# 使用示例
if __name__ == "__main__":
    from openai import OpenAI
    import json
    from tqdm import tqdm
    from dotenv import load_dotenv
    load_dotenv()
    # 1. 创建向量存储（内置索引创建）
    vector_db = FaissVectorStore(dimension=1024)
    with open("docs.json", "r") as f:
        # type of md_texts: list[dict["text": str, "path": str]]
        md_texts = json.load(f)

    texts = [i["text"] for i in md_texts]
    # 2. 添加数据
    client = OpenAI()
    batch_size = 10
    batch_texts = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]

    embs = []
    for text in tqdm(batch_texts):
        response = client.embeddings.create(
            model="text-embedding-v4",
            input=text,
            dimensions=1024
        )
        embs.extend([d.embedding for d in response.data])
    vector_db.add(embedding=embs, doc=md_texts)
    
    # 3. 查询
    query_emb = client.embeddings.create(
            model="text-embedding-v4",
            input="魔搭社区",
            dimensions=1024
        )
    results = vector_db.query(query_emb.data[0].embedding, top_k=5)
    print(results)
    # 4. 保存
    vector_db.persist(
        index_path="kb/index.faiss",
        kb_path="kb/docs.json",
        config_path="kb/config.pkl"
    )

    # 5. 加载
    vector_db = FaissVectorStore.from_persist_path(
        index_path="kb/index.faiss",
        kb_path="kb/docs.json",
        config_path="kb/config.pkl"
    )

    print("FaissVectorStore 已创建")
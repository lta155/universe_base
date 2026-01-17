import os
import fitz
from typing import List

def get_all_files(directory: str) -> List[str]:
    """
    获取目录下所有文件的完整路径
    
    Args:
        directory: 文件夹路径
    
    Returns:
        文件夹下各子文件路径
    """
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
    return file_paths

def read_files_to_text(file_paths: List[str]) -> List[str]:
    """
    Args:
        file_paths: 文件路径列表
        
    Returns:
        包含文件纯文本的列表
    """
    results = []
    
    for file_path in file_paths:
        if not os.path.exists(file_path):
            continue
        
        file_ext = os.path.splitext(file_path)[1].lower()
        
        # 读取markdown文件
        if file_ext == '.md':
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    results.append(f.read())
            except:
                continue
        
        # 读取PDF文件
        elif file_ext == '.pdf':
            try:
                doc = fitz.open(file_path)
                pdf_text = []
                
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    pdf_text.append(page.get_text())
                
                results.append("\n".join(pdf_text))
                doc.close()
            except:
                continue
    
    return results


def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    将文本切分成指定大小的chunk，支持重叠
    
    Args:
        text: 输入的文本
        chunk_size: 每个chunk的大小（字符数）
        overlap: chunk之间的重叠大小（字符数）
    
    Returns:
        切分好的文本chunk列表
    """
    if not text:
        return []
    
    if chunk_size <= 0:
        raise ValueError("chunk_size必须大于0")
    
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap必须在0到chunk_size-1之间")

    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0    
    while start < len(text):
        # 计算当前chunk的结束位置
        end = start + chunk_size
        
        # 获取当前chunk
        current_chunk = text[start:end]
        chunks.append(current_chunk)
        
        # 如果已经到达文本末尾，停止循环
        if end >= len(text):
            break
        
        # 更新下一个chunk的起始位置，考虑重叠
        start = start + chunk_size - overlap
    
    return chunks
"""
このファイルは、RAG（Retrieval-Augmented Generation）の処理を担当するファイルです。
バッチ実行可能な形式で実装されています。
"""

############################################################
# ライブラリの読み込み
############################################################
import os
import sys
import unicodedata
import logging
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
import constants as ct


############################################################
# 定数定義
############################################################
# ChromaDBの保存先ディレクトリ
CHROMA_DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")


############################################################
# 関数定義
############################################################

def process_rag(persist_directory=CHROMA_DB_DIR):
    """
    RAGの処理を実行するメイン関数
    
    Args:
        persist_directory: ChromaDBの保存先ディレクトリパス
    
    Returns:
        retriever: ベクターストアを検索するRetrieverオブジェクト
    """
    # ロガーを読み込むことで、後続の処理中に発生したエラーなどがログファイルに記録される
    logger = logging.getLogger(ct.LOGGER_NAME)
    
    # ChromaDBの保存先ディレクトリを作成
    os.makedirs(persist_directory, exist_ok=True)
    logger.info(f"ChromaDBの保存先: {persist_directory}")
    
    # RAGの参照先となるデータソースの読み込み
    docs_all = load_data_sources()
    logger.info(f"読み込んだドキュメント数: {len(docs_all)}")

    # OSがWindowsの場合、Unicode正規化と、cp932（Windows用の文字コード）で表現できない文字を除去
    for doc in docs_all:
        doc.page_content = adjust_string(doc.page_content)
        for key in doc.metadata:
            doc.metadata[key] = adjust_string(doc.metadata[key])
    
    # 埋め込みモデルの用意
    embeddings = OpenAIEmbeddings()
    
    # チャンク分割用のオブジェクトを作成
    text_splitter = CharacterTextSplitter(
        chunk_size=ct.CHUNK_SIZE,
        chunk_overlap=ct.CHUNK_OVERLAP,
        separator="\n"
    )

    # チャンク分割を実施
    splitted_docs = text_splitter.split_documents(docs_all)
    logger.info(f"チャンク分割後のドキュメント数: {len(splitted_docs)}")

    try:
        # 既存のベクターストアを削除（存在する場合）
        if os.path.exists(persist_directory):
            import shutil
            shutil.rmtree(persist_directory)
            os.makedirs(persist_directory, exist_ok=True)
            logger.info("既存のベクターストアを削除しました。")

        # ベクターストアの作成（永続化設定を追加）
        db = Chroma.from_documents(
            documents=splitted_docs,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        
        # ベクターストアを永続化
        db.persist()
        logger.info("ベクターストアを永続化しました。")

        # 保存されたファイルの確認
        if os.path.exists(persist_directory):
            files = os.listdir(persist_directory)
            logger.info(f"保存されたファイル: {files}")
        else:
            logger.error("ベクターストアの保存に失敗しました。")

        # ベクターストアを検索するRetrieverの作成
        return db.as_retriever(search_kwargs={"k": ct.SEARCH_K})
    
    except Exception as e:
        logger.error(f"ベクターストアの作成中にエラーが発生しました: {e}")
        raise


def load_data_sources():
    """
    RAGの参照先となるデータソースの読み込み

    Returns:
        読み込んだ通常データソース
    """
    # データソースを格納する用のリスト
    docs_all = []
    # ファイル読み込みの実行（渡した各リストにデータが格納される）
    recursive_file_check(ct.RAG_TOP_FOLDER_PATH, docs_all)

    web_docs_all = []
    # ファイルとは別に、指定のWebページ内のデータも読み込み
    # 読み込み対象のWebページ一覧に対して処理
    for web_url in ct.WEB_URL_LOAD_TARGETS:
        # 指定のWebページを読み込み
        loader = WebBaseLoader(web_url)
        web_docs = loader.load()
        # for文の外のリストに読み込んだデータソースを追加
        web_docs_all.extend(web_docs)
    # 通常読み込みのデータソースにWebページのデータを追加
    docs_all.extend(web_docs_all)

    return docs_all


def recursive_file_check(path, docs_all):
    """
    RAGの参照先となるデータソースの読み込み

    Args:
        path: 読み込み対象のファイル/フォルダのパス
        docs_all: データソースを格納する用のリスト
    """
    # パスがフォルダかどうかを確認
    if os.path.isdir(path):
        # フォルダの場合、フォルダ内のファイル/フォルダ名の一覧を取得
        files = os.listdir(path)
        # 各ファイル/フォルダに対して処理
        for file in files:
            # ファイル/フォルダ名だけでなく、フルパスを取得
            full_path = os.path.join(path, file)
            # フルパスを渡し、再帰的にファイル読み込みの関数を実行
            recursive_file_check(full_path, docs_all)
    else:
        # パスがファイルの場合、ファイル読み込み
        file_load(path, docs_all)


def file_load(path, docs_all):
    """
    ファイル内のデータ読み込み

    Args:
        path: ファイルパス
        docs_all: データソースを格納する用のリスト
    """
    # ファイルの拡張子を取得
    file_extension = os.path.splitext(path)[1]
    # ファイル名（拡張子を含む）を取得
    file_name = os.path.basename(path)

    # 想定していたファイル形式の場合のみ読み込む
    if file_extension in ct.SUPPORTED_EXTENSIONS:
        # CSVファイルの場合、特別な処理を行う
        if file_extension == ".csv":
            loader = ct.SUPPORTED_EXTENSIONS[file_extension](path)
            docs = loader.load()
            # 全行を統合して1つのドキュメントにする
            combined_text = "\n".join([doc.page_content for doc in docs])
            docs_all.append(Document(page_content=combined_text, metadata={"source": path}))
        else:
            # その他のファイル形式は通常通り読み込む
            loader = ct.SUPPORTED_EXTENSIONS[file_extension](path)
            docs = loader.load()
            docs_all.extend(docs)


def adjust_string(s):
    """
    Windows環境でRAGが正常動作するよう調整
    
    Args:
        s: 調整を行う文字列
    
    Returns:
        調整を行った文字列
    """
    # 調整対象は文字列のみ
    if type(s) is not str:
        return s

    # OSがWindowsの場合、Unicode正規化と、cp932（Windows用の文字コード）で表現できない文字を除去
    if sys.platform.startswith("win"):
        s = unicodedata.normalize('NFC', s)
        s = s.encode("cp932", "ignore").decode("cp932")
        return s
    
    # OSがWindows以外の場合はそのまま返す
    return s


if __name__ == "__main__":
    # ログ出力の設定
    os.makedirs(ct.LOG_DIR_PATH, exist_ok=True)
    logger = logging.getLogger(ct.LOGGER_NAME)
    logger.setLevel(logging.INFO)
    
    # ログハンドラーの設定
    log_handler = logging.FileHandler(
        os.path.join(ct.LOG_DIR_PATH, "rag_processor.log"),
        encoding="utf8"
    )
    formatter = logging.Formatter(
        f"[%(levelname)s] %(asctime)s line %(lineno)s, in %(funcName)s: %(message)s"
    )
    log_handler.setFormatter(formatter)
    logger.addHandler(log_handler)
    
    try:
        # RAG処理の実行
        retriever = process_rag()
        logger.info("RAG処理が正常に完了しました。")
    except Exception as e:
        logger.error(f"RAG処理中にエラーが発生しました: {e}")
        raise
    
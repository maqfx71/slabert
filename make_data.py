import pykakasi
import MeCab
import re
import os
import sys

def remove_tags(text):
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

def extract_and_convert(xml_path, output_path):
    # MeCabの初期化
    mecab = MeCab.Tagger("-Owakati")
    # PyKakasiの初期化
    kks = pykakasi.kakasi()
    # # モードの設定
    # kks.setMode("H", "a")  # Hiragana to ASCII
    # kks.setMode("K", "a")  # Katakana to ASCII
    # kks.setMode("J", "a")  # Japanese to ASCII
    # kks.setMode("E", "a")  # English to ASCII (新しい設定)
    with open(xml_path, 'r', encoding='utf-8') as file:
        xml_content = file.read()
    # 不要なタグを取り除く
    clean_text = remove_tags(xml_content)
    # MeCabで分かち書き
    # parsed_text = mecab.parse(clean_text.encode('utf-8')).decode('utf-8')
    parsed_text = mecab.parse(clean_text)
    # print(parsed_text)
    # ローマ字に変換
    converted_text = kks.convert(parsed_text)
    # 結果を標準出力に表示
    # conv = kks.getConverter()
    # converted_text = conv.do(parsed_text)
    # print(converted_text)
    output_text = ""
    for converted_word in converted_text:
        # print(f"{converted_word['hepburn']}", end ="")
        output_text = output_text + converted_word['hepburn']
    # "."の後ろに改行コードを挿入
    output_text = output_text.replace('. ', '.\n')
    # "?"の後ろに改行コードを挿入
    output_text = output_text.replace('? ', '?\n')
    # 結果をテキストファイルに保存
    with open(output_path, 'w', encoding='utf-8') as output_file:
        output_file.write(output_text)
    print(f"学習用テキストファイルを作成しました: {output_path}")

def process_selected_xml_files(directory_path):
    # ディレクトリ内の"OT4"から始まるXMLファイルを取得
    xml_files = [f for f in os.listdir(directory_path) if f.startswith("OT4") and f.endswith(".xml")]
    # XMLファイルごとに処理
    for xml_file in xml_files:
        xml_file_path = os.path.join(directory_path, xml_file)
        # 拡張子を取り除いた部分を基に出力ファイルのパスを生成
        file_name_without_extension = os.path.splitext(xml_file)[0]
        output_path = os.path.join(directory_path, file_name_without_extension + ".txt")
        extract_and_convert(xml_file_path, output_path)

def combine_text_files(directory_path):
    # ディレクトリ内の"OT4"から始まるテキストファイルを取得
    text_files = [f for f in os.listdir(directory_path) if f.startswith("OT4") and f.endswith(".txt")]
    # まとめたテキストファイルのパス
    combined_output_path = os.path.join(directory_path, "combined_output.txt")
    # まとめたテキストファイルを開く
    with open(combined_output_path, "w", encoding="utf-8") as combined_output_file:
        # テキストファイルごとに処理
        for text_file in text_files:
            text_file_path = os.path.join(directory_path, text_file)
            # テキストファイルに書き込む
            with open(text_file_path, "r", encoding="utf-8") as individual_text_file:
                combined_output_file.write(individual_text_file.read())

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 make_data.py <DIRECTORY_PATH>")
        sys.exit(1)
    # コマンドライン引数からディレクトリのパスを取得
    directory_path = sys.argv[1]
    # "OT4"から始まるXMLファイルのみを処理
    # process_selected_xml_files(directory_path)
    combine_text_files(directory_path)

# if __name__ == "__main__":
#     # xml_path = "data/C-OT42_00001.xml"  # 使用するファイルパスに変更
#     # output_path = "data/C-OT42_00001.txt"  # 対応するパスに変更
#     # extract_and_convert(xml_path, output_path)
#     if len(sys.argv) != 2:
#         print("Usage: python script.py <XML_FILE_PATH>")
#         sys.exit(1)
#     # コマンドライン引数からXMLファイルのパスを取得
#     xml_path = sys.argv[1]
#     # 拡張子を取り除いた部分を基に出力ファイルのパスを生成
#     file_name_without_extension = xml_path.split(".")[0]
#     output_path = file_name_without_extension + ".txt"
#     extract_and_convert(xml_path, output_path)

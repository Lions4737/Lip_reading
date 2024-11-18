import os
import shutil

# 元のフォルダのパス
original_folder = 'original_mov_before'
# 動画を格納する先のフォルダ
target_folder = 'original_mov'

# もし 'original_mov' フォルダが存在しなければ作成
if not os.path.exists(target_folder):
    os.makedirs(target_folder)

# フォルダ内のすべてのサブフォルダを調べる
for subfolder in os.listdir(original_folder):
    subfolder_path = os.path.join(original_folder, subfolder)
    
    # '_LEFOI'で終わるサブフォルダを選択
    if os.path.isdir(subfolder_path) and subfolder.endswith('_LFROI'):
        
        # サブフォルダ内のファイルを確認
        for filename in os.listdir(subfolder_path):
            file_path = os.path.join(subfolder_path, filename)
            
            # .mp4ファイルを選択
            if filename.endswith('.mp4') and os.path.isfile(file_path):
                # .mp4ファイルを 'original_mov' フォルダにコピー
                shutil.copy(file_path, os.path.join(target_folder, filename))

print("全ての.mp4ファイルを 'original_mov' フォルダに格納しました。")

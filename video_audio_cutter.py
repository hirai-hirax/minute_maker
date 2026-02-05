import os
import streamlit as st
import tempfile
import pandas as pd
import subprocess
import zipfile
from io import BytesIO
from datetime import timedelta, datetime
import re

def format_time(seconds):
    """Formats float seconds into HH:MM:SS.mmm."""
    seconds = float(seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02}:{minutes:02}:{secs:05.2f}"

def parse_time_to_seconds(time_str):
    """Converts HH:MM:SS or seconds string to total seconds."""
    if ':' in str(time_str):
        parts = list(map(float, str(time_str).split(':')))
        if len(parts) == 3:
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
        elif len(parts) == 2:
            return parts[0] * 60 + parts[1]
        else:
            raise ValueError("Invalid time format. Use HH:MM:SS or MM:SS.")
    else:
        try:
            return float(time_str)
        except ValueError:
            return 0.0

def video_to_audio_cutter_app():
    st.set_page_config(page_title="動画から音声を切り出しMP3で保存", layout="wide")
    
    st.title("動画から音声を切り出しMP3で保存")
    st.write("動画ファイルをアップロードし、切り出したい開始時間と終了時間を指定してください。複数の区間を切り出すことができます。")

    if 'processed_audio_path' not in st.session_state:
        st.session_state.processed_audio_path = None
    if 'current_file_id' not in st.session_state:
        st.session_state.current_file_id = None
    if 'original_file_path' not in st.session_state:
        st.session_state.original_file_path = None
    if 'removed_sections' not in st.session_state:
        st.session_state.removed_sections = []

    uploaded_file = st.file_uploader("編集したいファイルを選択（動画・音声対応）", type=["wav","mp3","mp4", "mov", "avi", "mkv", "webm"])

    if uploaded_file:
        file_id = f"{uploaded_file.name}_{uploaded_file.size}"
        if st.session_state.current_file_id != file_id:
            st.session_state.current_file_id = file_id
            st.session_state.processed_audio_path = None
            st.session_state.removed_sections = []
            
            suffix = f".{uploaded_file.name.split('.')[-1]}"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
                f.write(uploaded_file.getbuffer())
                st.session_state.original_file_path = f.name
            
            with st.spinner("プレビューの読み込み中..."):
                out_path = os.path.join(tempfile.gettempdir(), f"preview_{file_id}.mp3")
                if not os.path.exists(out_path):
                    cmd = ["ffmpeg", "-i", st.session_state.original_file_path, "-vn", "-ab", "192k", "-y", out_path]
                    subprocess.run(cmd, check=True, capture_output=True)
                st.session_state.processed_audio_path = out_path

        # --- STEP 1: 無音削除設定 ---
        st.divider()
        with st.expander("Step 1：無音区間の短縮（沈黙部分をカットする）", expanded=True):
            st.markdown("音声内の不要な沈黙を自動で検出し、指定した長さに短縮します。")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                threshold = st.number_input("無音判定しきい値 (dB)", value=-50, min_value=-100, max_value=-10, step=1, 
                                            help="値が小さい（例：-60）ほど、より静かな音だけを無音とみなします。")
            with col2:
                min_duration = st.number_input("最小無音時間 (秒)", value=1.0, min_value=0.1, max_value=10.0, step=0.1,
                                              help="この時間以上続く沈黙をカットの対象にします。")
            with col3:
                buffer_val = st.number_input("残す無音バッファ (秒)", value=0.3, min_value=0.0, max_value=2.0, step=0.05,
                                            help="カットした後に残す余白時間です。0.3秒程度あると自然に聞こえます。")

            apply_btn = st.button("✨ 無音削除を適用する")
            
            if apply_btn:
                with st.spinner("分析・処理中..."):
                    # 1. 無音区間の検出
                    detect_cmd = [
                        "ffmpeg", "-i", st.session_state.original_file_path,
                        "-af", f"silencedetect=noise={threshold}dB:d={min_duration}",
                        "-f", "null", "-"
                    ]
                    output = subprocess.run(detect_cmd, capture_output=True, text=True, encoding="utf-8").stderr
                    
                    starts = re.findall(r"silence_start: (\d+\.?\d*)", output)
                    ends = re.findall(r"silence_end: (\d+\.?\d*)", output)
                    
                    sections = []
                    for s, e in zip(starts, ends):
                        s_val = float(s)
                        e_val = float(e)
                        cut_start = s_val + buffer_val
                        if cut_start < e_val:
                            sections.append({
                                "開始位置": format_time(s_val),
                                "終了位置": format_time(e_val),
                                "短縮時間": round(e_val - cut_start, 2)
                            })
                    st.session_state.removed_sections = sections

                    # 2. 実際の処理
                    settings_hash = f"{threshold}_{min_duration}_{buffer_val}"
                    clean_path = os.path.join(tempfile.gettempdir(), f"clean_{settings_hash}_{file_id}.mp3")
                    
                    cmd = [
                        "ffmpeg", "-i", st.session_state.original_file_path,
                        "-af", f"silenceremove=stop_periods=-1:stop_duration={buffer_val}:stop_threshold={threshold}dB",
                        "-ab", "192k", "-y", clean_path
                    ]
                    try:
                        subprocess.run(cmd, check=True, capture_output=True, text=True)
                        st.session_state.processed_audio_path = clean_path
                        st.success(f"処理が完了しました。全体の約 {sum(s['短縮時間'] for s in sections):.1f} 秒を短縮しました。")
                    except Exception as e:
                        st.error(f"エラーが発生しました: {e}")

            if st.session_state.removed_sections:
                with st.expander("📝 削除（短縮）された箇所の詳細"):
                    df_removed = pd.DataFrame(st.session_state.removed_sections)
                    st.dataframe(df_removed, use_container_width=True, hide_index=True)

        # --- STEP 2 & 3: プレビューと切り出し ---
        if st.session_state.processed_audio_path:
            st.divider()
            
            # Step 2: 確認再生 (全幅)
            st.subheader("🎧 ステップ2：再生確認")
            st.audio(st.session_state.processed_audio_path)
            
            base_name = os.path.splitext(uploaded_file.name)[0]
            with open(st.session_state.processed_audio_path, "rb") as f:
                st.download_button(
                    label="📥 処理後の音声をダウンロード",
                    data=f,
                    file_name=f"{base_name}_processed.mp3",
                    mime="audio/mpeg",
                    use_container_width=True
                )
            st.caption("※無音カットを適用した場合は、全体の再生時間が短くなります。")

            st.divider()

            # Step 3: 切り出し (全幅)
            st.subheader("✂️ ステップ3：必要な区間の切り出し")
            st.markdown("特定の箇所だけを別ファイルとして保存します（複数指定可）。")
                
            if 'cut_data' not in st.session_state or st.session_state.current_file_id != file_id:
                st.session_state.cut_data = pd.DataFrame([
                    {"開始時間": "00:00:00", "終了時間": "00:00:30", "出力ファイル名": f"{base_name}_clip_1"}
                ])

            edited_df = st.data_editor(
                st.session_state.cut_data,
                num_rows="dynamic",
                use_container_width=True,
                column_config={
                    "開始時間": st.column_config.TextColumn("開始 (HH:MM:SS)", default="00:00:00"),
                    "終了時間": st.column_config.TextColumn("終了 (HH:MM:SS)", default="00:00:30"),
                    "出力ファイル名": st.column_config.TextColumn("保存名", default=f"{base_name}_clip_")
                },
                key="intervals_editor"
            )

            if st.button("📦 指定した全区間を一括保存", use_container_width=True):
                if edited_df.empty:
                    st.warning("切り出し区間が設定されていません。")
                else:
                    output_files = []
                    zip_buffer = BytesIO()
                    
                    progress_bar = st.progress(0)
                    for index, row in edited_df.iterrows():
                        start_str = str(row["開始時間"])
                        end_str = str(row["終了時間"])
                        out_name = str(row["出力ファイル名"]).strip()
                        
                        try:
                            start_sec = parse_time_to_seconds(start_str)
                            end_sec = parse_time_to_seconds(end_str)
                            
                            if start_sec >= end_sec:
                                st.error(f"区間 {index+1}: 時間が正しくありません。")
                                continue
                            
                            final_filename = out_name if out_name else f"clip_{index+1}"
                            if not final_filename.lower().endswith(".mp3"):
                                final_filename += ".mp3"
                            
                            out_full_path = os.path.join(tempfile.gettempdir(), final_filename)
                            cmd = [
                                "ffmpeg", "-i", st.session_state.processed_audio_path,
                                "-ss", format_time(start_sec),
                                "-to", format_time(end_sec),
                                "-ab", "192k", "-y", out_full_path
                            ]
                            subprocess.run(cmd, check=True, capture_output=True)
                            output_files.append(out_full_path)
                        except Exception as e:
                            st.error(f"区間 {index+1} でエラー: {e}")
                        progress_bar.progress((index + 1) / len(edited_df))

                    if output_files:
                        st.divider()
                        st.write("#### 💾 保存の準備ができました")
                        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                            for path in output_files:
                                if os.path.exists(path):
                                    zf.write(path, os.path.basename(path))
                                    st.write(f"- ✅ {os.path.basename(path)}")
                        zip_buffer.seek(0)
                        
                        st.download_button(
                            label="📥 全てをZIP形式でダウンロード",
                            data=zip_buffer,
                            file_name=f"{base_name}_clips.zip",
                            mime="application/zip",
                            use_container_width=True
                        )

if __name__ == "__main__":
    video_to_audio_cutter_app()

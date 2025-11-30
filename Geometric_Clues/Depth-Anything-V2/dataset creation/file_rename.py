# # save as rename_videos_sequential.py
# import os
# from pathlib import Path

# def rename_videos_sequential(folder="real", start=1):
#     video_exts = {".mp4", ".mov", ".mkv", ".avi", ".wmv", ".flv", ".m4v", ".webm"}
#     p = Path(folder)
#     if not p.is_dir():
#         raise FileNotFoundError(f"Folder not found: {p.resolve()}")

#     # Collect only files with video extensions
#     files = [f for f in p.iterdir() if f.is_file() and f.suffix.lower() in video_exts]

#     # Sort for stable order (alphabetical); change key if a different order is desired
#     files.sort(key=lambda x: x.name)

#     # Two-phase rename: first to temporary names to avoid collisions, then final names
#     temp_suffix = ".mp4"
#     temps = []
#     for f in files:
#         tmp = f.with_name(f.name + temp_suffix)
#         f.rename(tmp)
#         temps.append(tmp)

#     # Now assign 1, 2, 3, ... preserving each file's original extension
#     n = start
#     for tmp in temps:
#         new_name = f"{n}{tmp.suffix.replace(temp_suffix, '')}"
#         final_path = tmp.with_name(new_name)
#         # Ensure not overwriting existing files
#         while final_path.exists():
#             n += 1
#             new_name = f"{n}{tmp.suffix.replace(temp_suffix, '')}"
#             final_path = tmp.with_name(new_name)
#         tmp.rename(final_path)
#         n += 1

# if __name__ == "__main__":
#     # Change folder="real" if the directory is elsewhere
#     rename_videos_sequential(folder="real", start=1)

# save as rename_videos_keep_ext.py
from pathlib import Path

def rename_videos_sequential(folder="gen_ai", start=1):
    p = Path(folder)
    files = sorted([f for f in p.iterdir() if f.is_file()], key=lambda x: x.name)

    # temp-rename to avoid collisions
    temp = []
    for f in files:
        t = f.with_name(f.name + ".tmp_renaming")
        f.rename(t)
        temp.append(t)

    n = start
    for t in temp:
        ext = t.suffix.replace(".tmp_renaming", "") or ".mp4"  # fallback if none
        new_path = t.with_name(f"{n}{ext}")
        while new_path.exists():
            n += 1
            new_path = t.with_name(f"{n}{ext}")
        t.rename(new_path)
        n += 1

if __name__ == "__main__":
    rename_videos_sequential("gen_ai", 1)

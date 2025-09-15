"""import pandas as pd

# 1. 正确读取数据（根据你的截图应是空格分隔）
df = pd.read_csv("data/NetEase/user_bundle_test.txt", header=None, sep='\s+')
df.columns = ["userId", "old_bundleId"]

# 2. 创建旧 user → 新 userId 的映射（保留第一次出现的顺序）
unique_users = df["old_bundleId"].drop_duplicates().reset_index(drop=True)
user_map = {old_id: new_id for new_id, old_id in enumerate(unique_users)}

# 3. 替换原 user 为新 userId
df["bundleId"] = df["old_bundleId"].map(user_map)

# 4. 保留原顺序的 [userId, bundleId] 列
df = df[["userId", "bundleId"]]

# 5. 保存（避免覆盖原文件，可更换文件名）
df.to_csv("data/NetEase/user_bundle_test_reindex.txt", index=False, header=False, sep=' ')
"""
import pandas as pd

# 1. 读取数据（空格分隔，无header）
df = pd.read_csv("data/NetEase/user_item.txt", header=None, sep='\s+')

# 2. 重命名列（便于处理）
df.columns = ["old_userId", "bundleId"]

# 3. 创建旧userId → 新userId的映射（从0开始连续编号）
# 获取所有唯一用户ID（按首次出现顺序）
unique_users = df["old_userId"].drop_duplicates().reset_index(drop=True)
# 创建映射字典 {旧ID: 新ID}，新ID从0开始
user_map = {old_id: new_id for new_id, old_id in enumerate(unique_users)}

# 4. 创建新userId列
df["userId"] = df["old_userId"].map(user_map)

# 5. 只保留新userId和原始bundleId（保持原始顺序）
result_df = df[["userId", "bundleId"]]

# 6. 保存文件（无列名，空格分隔）
result_df.to_csv("data/NetEase/user_item_reindex.txt",
                 index=False,
                 header=False,  # 关键修改：不添加列名
                 sep=' ')

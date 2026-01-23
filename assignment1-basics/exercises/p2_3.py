import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
splited_text = re.findall(PAT, "some text that i'll pre-tokenize")
print(splited_text)

# Remark:
# 这种pre-tokenization方式会把空格(' ')放在下一个分词的开头
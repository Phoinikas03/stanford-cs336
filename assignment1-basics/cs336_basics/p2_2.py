test_string = "hello! こんにちは!"

# utf-8
utf8_encoded = test_string.encode("utf-8")
print(utf8_encoded)
print(list(utf8_encoded))

print(len(test_string))
print(len(utf8_encoded))
print(utf8_encoded.decode("utf-8"))

print("#" * 50)
# utf-16
utf16_encoded = test_string.encode("utf-16")
print(utf16_encoded)
print(list(utf16_encoded))

print(len(test_string))
print(len(utf16_encoded))
print(utf16_encoded.decode("utf-16"))

# Remark:
# utf-16相比utf-8会消耗更多的bytes

print("#" * 50)

def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])

utf_8_wrongly_decoded = decode_utf8_bytes_to_str_wrong(utf8_encoded)
try:
    print(utf_8_wrongly_decoded)

except Exception as e:
    print(e)

print('#' * 50)

# Remark:
# UTF-8 中英文字符（ASCII）占 1 个字节，日文、中文等非 ASCII 字符通常占 3 个字节
#
# 例如 "こ"（U+3053）的 UTF-8 编码是：E3 81 93
#
# 其中：
# E3 只是这个字符的第一个字节
# 81 是第二个字节
# 93 是第三个字节
#
# UTF-8 decoder 看到 E3 时知道：
# “这是一个要占 3 个字节的字符，我还需要后面两个 continuation bytes（格式为 10xxxxxx）。”
